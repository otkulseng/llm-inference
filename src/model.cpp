#include "model.h"

#include "bf16.h"
#include "config.h"
#include "kernels.cuh"
#include "tokenizer.h"
#include <cassert>
#include <cmath>
#include <cstdint>

// Bulk weight load. mmap the BF16 blob and cudaMemcpy raw bytes to GPU — the
// on-disk format and the in-GPU format are now identical, so no conversion is
// needed at load time.
static DeviceBuffer<__nv_bfloat16>
bulk_load_to_gpu(LlamaDumpLoader &loader, const string &path,
                 size_t expected_count) {
    auto blob = loader.open_blob(path);
    assert(blob.header.dtype == 2 /*BF16*/);
    assert(blob.header.rows * blob.header.cols == expected_count);
    DeviceBuffer<__nv_bfloat16> d_out(expected_count);
    cudaMemcpy(d_out.data(), blob.raw,
               expected_count * sizeof(__nv_bfloat16),
               cudaMemcpyHostToDevice);
    return d_out;
}

// (s, h_d/2) cos and sin tables on the host as fp32, downcast to BF16, then
// uploaded to the GPU. theta_i = 1 / ROPE_BASE^(2i/h_d) per part2.pdf §3.2.
static std::pair<DeviceBuffer<__nv_bfloat16>, DeviceBuffer<__nv_bfloat16>>
make_rope_tables(int s, int h_d) {
    int half = h_d / 2;
    vector<float> cos_h(s * half), sin_h(s * half);
    for (int i = 0; i < half; ++i) {
        float theta = 1.0f / std::pow(ROPE_BASE, (2.0f * i) / h_d);
        for (int p = 0; p < s; ++p) {
            cos_h[p * half + i] = std::cos(p * theta);
            sin_h[p * half + i] = std::sin(p * theta);
        }
    }
    auto cos_bf = to_bf16_host(cos_h);
    auto sin_bf = to_bf16_host(sin_h);
    return {DeviceBuffer<__nv_bfloat16>(cos_bf),
            DeviceBuffer<__nv_bfloat16>(sin_bf)};
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

Model::Model() : loader_(DumpFloatType::FP32), tokenizer_(TOKENIZER_PATH) {
    // Eager load of every weight. ~16 GB BF16 total, fits in the 20 GB MIG.
    embed_ = bulk_load_to_gpu(loader_,
                              "assets/llama3/blobs/model.embed_tokens.weight",
                              (size_t)VOCAB_SIZE * EMBEDDING_DIM);
    layers_.reserve(N_LAYERS);
    for (int L = 0; L < N_LAYERS; ++L) {
        layers_.push_back(load_layer(L));
    }
    final_norm_ = bulk_load_to_gpu(
        loader_, "assets/llama3/blobs/model.norm.weight", EMBEDDING_DIM);
    lm_head_ = bulk_load_to_gpu(loader_, "assets/llama3/blobs/lm_head.weight",
                                (size_t)VOCAB_SIZE * EMBEDDING_DIM);
}

vector<int> Model::tokenize(const string &input) {
    return tokenizer_.encode(input);
}

string Model::detokenize(const vector<int> &ids) {
    return tokenizer_.decode(ids);
}

vector<float> Model::embed(const vector<int> &ids) {
    // GPU gather on the cached embed table; convert to fp32 for the caller.
    int n = (int)ids.size();
    assert(n <= MAX_SEQ_LEN);
    DeviceBuffer<int> d_ids(ids);
    DeviceBuffer<__nv_bfloat16> d_out(n * EMBEDDING_DIM);
    launch_embed_gather(embed_.data(), d_ids.data(), d_out.data(),
                        n, EMBEDDING_DIM);
    auto out_bf = d_out.to_host();
    return to_fp32_host(out_bf);
}

Model::LayerWeights Model::load_layer(int layer_idx) {
    string p = "assets/llama3/blobs/model.layers." +
               std::to_string(layer_idx) + ".";
    const size_t d  = EMBEDDING_DIM;
    const size_t qd = (size_t)N_HEADS    * H_DIM * d;
    const size_t kd = (size_t)N_KV_HEADS * H_DIM * d;
    const size_t fd = (size_t)D_FF * d;
    return LayerWeights{
        bulk_load_to_gpu(loader_, p + "input_layernorm.weight",          d),
        bulk_load_to_gpu(loader_, p + "self_attn.q_proj.weight",         qd),
        bulk_load_to_gpu(loader_, p + "self_attn.k_proj.weight",         kd),
        bulk_load_to_gpu(loader_, p + "self_attn.v_proj.weight",         kd),
        bulk_load_to_gpu(loader_, p + "self_attn.o_proj.weight",         qd),
        bulk_load_to_gpu(loader_, p + "post_attention_layernorm.weight", d),
        bulk_load_to_gpu(loader_, p + "mlp.gate_proj.weight",            fd),
        bulk_load_to_gpu(loader_, p + "mlp.up_proj.weight",              fd),
        bulk_load_to_gpu(loader_, p + "mlp.down_proj.weight",            fd),
    };
}

DeviceBuffer<__nv_bfloat16> Model::run_decoder_block(
    const DeviceBuffer<__nv_bfloat16> &d_x, const LayerWeights &w,
    const __nv_bfloat16 *d_cos, const __nv_bfloat16 *d_sin, int s) {

    // --- Attention sub-block ---
    DeviceBuffer<__nv_bfloat16> d_xnorm(s * EMBEDDING_DIM);
    launch_rmsnorm(d_x.data(), w.input_norm.data(), d_xnorm.data(),
                   s, EMBEDDING_DIM, RMS_NORM_EPSILON);

    DeviceBuffer<__nv_bfloat16> d_Q(s * N_HEADS    * H_DIM);
    DeviceBuffer<__nv_bfloat16> d_K(s * N_KV_HEADS * H_DIM);
    DeviceBuffer<__nv_bfloat16> d_V(s * N_KV_HEADS * H_DIM);
    launch_matmul(d_xnorm.data(), w.q.data(), d_Q.data(),
                      s, EMBEDDING_DIM, N_HEADS    * H_DIM);
    launch_matmul(d_xnorm.data(), w.k.data(), d_K.data(),
                      s, EMBEDDING_DIM, N_KV_HEADS * H_DIM);
    launch_matmul(d_xnorm.data(), w.v.data(), d_V.data(),
                      s, EMBEDDING_DIM, N_KV_HEADS * H_DIM);

    DeviceBuffer<__nv_bfloat16> d_Q_rope(s * N_HEADS    * H_DIM);
    DeviceBuffer<__nv_bfloat16> d_K_rope(s * N_KV_HEADS * H_DIM);
    launch_rope(d_Q.data(), d_cos, d_sin, d_Q_rope.data(), N_HEADS,    s, H_DIM);
    launch_rope(d_K.data(), d_cos, d_sin, d_K_rope.data(), N_KV_HEADS, s, H_DIM);

    DeviceBuffer<__nv_bfloat16> d_O(s * N_HEADS * H_DIM);
    launch_gqa_attention(d_Q_rope.data(), d_K_rope.data(), d_V.data(),
                         d_O.data(), s);

    DeviceBuffer<__nv_bfloat16> d_attn_out(s * EMBEDDING_DIM);
    launch_matmul(d_O.data(), w.o.data(), d_attn_out.data(),
                      s, N_HEADS * H_DIM, EMBEDDING_DIM);

    DeviceBuffer<__nv_bfloat16> d_x1(s * EMBEDDING_DIM);
    launch_residual_add(d_x.data(), d_attn_out.data(), d_x1.data(),
                        s * EMBEDDING_DIM);

    // --- FFN sub-block ---
    DeviceBuffer<__nv_bfloat16> d_x1_norm(s * EMBEDDING_DIM);
    launch_rmsnorm(d_x1.data(), w.post_norm.data(), d_x1_norm.data(),
                   s, EMBEDDING_DIM, RMS_NORM_EPSILON);

    DeviceBuffer<__nv_bfloat16> d_ffn_out = run_swiglu_core(d_x1_norm, w, s);

    DeviceBuffer<__nv_bfloat16> d_x2(s * EMBEDDING_DIM);
    launch_residual_add(d_x1.data(), d_ffn_out.data(), d_x2.data(),
                        s * EMBEDDING_DIM);

    return d_x2;
}

DeviceBuffer<__nv_bfloat16>
Model::run_swiglu_core(const DeviceBuffer<__nv_bfloat16> &d_x_norm,
                       const LayerWeights &w, int s) {
    DeviceBuffer<__nv_bfloat16> d_gate(s * D_FF);
    DeviceBuffer<__nv_bfloat16> d_up(s * D_FF);
    DeviceBuffer<__nv_bfloat16> d_hidden(s * D_FF);
    DeviceBuffer<__nv_bfloat16> d_out(s * EMBEDDING_DIM);

    launch_matmul(d_x_norm.data(), w.gate.data(), d_gate.data(),
                      s, EMBEDDING_DIM, D_FF);
    launch_matmul(d_x_norm.data(), w.up.data(), d_up.data(),
                      s, EMBEDDING_DIM, D_FF);
    launch_silu_mul(d_gate.data(), d_up.data(), d_hidden.data(), s * D_FF);
    launch_matmul(d_hidden.data(), w.down.data(), d_out.data(),
                      s, D_FF, EMBEDDING_DIM);
    return d_out;
}

vector<float> Model::swiglu_ffn(const vector<float> &x_norm, int layer_idx,
                                int s) {
    assert(s <= MAX_SEQ_LEN);
    DeviceBuffer<__nv_bfloat16> d_x(to_bf16_host(x_norm));
    DeviceBuffer<__nv_bfloat16> d_out =
        run_swiglu_core(d_x, layers_[layer_idx], s);
    return to_fp32_host(d_out.to_host());
}

vector<float> Model::decoder_block(const vector<float> &x, int layer_idx,
                                   int s) {
    assert(s <= MAX_SEQ_LEN);
    const LayerWeights &w = layers_[layer_idx];
    auto [d_cos, d_sin] = make_rope_tables(s, H_DIM);
    DeviceBuffer<__nv_bfloat16> d_x(to_bf16_host(x));

    DeviceBuffer<__nv_bfloat16> d_out =
        run_decoder_block(d_x, w, d_cos.data(), d_sin.data(), s);
    return to_fp32_host(d_out.to_host());
}

int Model::forward_one_step(const vector<int> &token_ids) {
    int s = (int)token_ids.size();
    assert(s <= MAX_SEQ_LEN);

    // 1. Embedding lookup via GPU gather on the cached embed table.
    DeviceBuffer<int> d_ids(token_ids);
    DeviceBuffer<__nv_bfloat16> d_x(s * EMBEDDING_DIM);
    launch_embed_gather(embed_.data(), d_ids.data(), d_x.data(),
                        s, EMBEDDING_DIM);

    // 2. RoPE tables, built once and reused across all 32 layers.
    auto [d_cos, d_sin] = make_rope_tables(s, H_DIM);

    // 3. Decoder loop using cached layer weights.
    for (int L = 0; L < N_LAYERS; ++L) {
        d_x = run_decoder_block(d_x, layers_[L], d_cos.data(), d_sin.data(), s);
    }

    // 4. Final RMSNorm with model.norm.weight.
    DeviceBuffer<__nv_bfloat16> d_x_norm(s * EMBEDDING_DIM);
    launch_rmsnorm(d_x.data(), final_norm_.data(), d_x_norm.data(),
                   s, EMBEDDING_DIM, RMS_NORM_EPSILON);

    // 5. lm_head on the last token only (part2.pdf §4: extract only the last
    //    token before lm_head). 1×d row times W_lm^T (V×d row-major) → (1, V).
    DeviceBuffer<__nv_bfloat16> d_logits(VOCAB_SIZE);
    const __nv_bfloat16 *d_x_last = d_x_norm.data() + (s - 1) * EMBEDDING_DIM;
    launch_matmul(d_x_last, lm_head_.data(), d_logits.data(),
                      1, EMBEDDING_DIM, VOCAB_SIZE);

    // 6. Greedy argmax on host.
    auto logits_bf = d_logits.to_host();
    int best = 0;
    float best_v = bf16_to_fp32(logits_bf[0]);
    for (int v = 1; v < VOCAB_SIZE; ++v) {
        float lv = bf16_to_fp32(logits_bf[v]);
        if (lv > best_v) { best_v = lv; best = v; }
    }
    return best;
}

vector<int> Model::generate(const vector<int> &token_ids, int n_new) {
    assert((int)token_ids.size() + n_new <= MAX_SEQ_LEN);
    vector<int> all = token_ids;
    vector<int> out;
    out.reserve(n_new);
    for (int i = 0; i < n_new; ++i) {
        int next = forward_one_step(all);
        all.push_back(next);
        out.push_back(next);
    }
    return out;
}
