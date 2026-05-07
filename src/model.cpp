#include "model.h"

#include "config.h"
#include "kernels.cuh"
#include "tokenizer.h"
#include <cassert>
#include <cmath>
#include <cstdint>

// Bulk weight load. mmap the blob, copy raw bytes to GPU, convert to FP32 on
// the GPU. Avoids the single-threaded CPU BF16→FP32 loop and halves PCIe
// traffic (BF16 = 2 B/elem). Caller passes expected_count (rows*cols) so we
// can sanity-check the header against the operator's known shape.
static DeviceBuffer<float> bulk_load_to_gpu(LlamaDumpLoader &loader,
                                            const string &path,
                                            size_t expected_count) {
    auto blob = loader.open_blob(path);
    assert(blob.header.rows * blob.header.cols == expected_count);

    if (blob.header.dtype == 1 /*FP32*/) {
        DeviceBuffer<float> d_out(expected_count);
        cudaMemcpy(d_out.data(), blob.raw, expected_count * sizeof(float),
                   cudaMemcpyHostToDevice);
        return d_out;
    }
    if (blob.header.dtype == 2 /*BF16*/) {
        DeviceBuffer<uint16_t> d_raw(expected_count);
        cudaMemcpy(d_raw.data(), blob.raw, expected_count * sizeof(uint16_t),
                   cudaMemcpyHostToDevice);
        DeviceBuffer<float> d_out(expected_count);
        launch_bf16_to_fp32(d_raw.data(), d_out.data(), expected_count);
        return d_out;
    }
    throw runtime_error("bulk_load_to_gpu: unsupported dtype " +
                        std::to_string(blob.header.dtype));
}

// (s, h_d/2) cos and sin tables on the host.  theta_i = 1 / ROPE_BASE^(2i/h_d)
// per part2.pdf §3.2.
static std::pair<vector<float>, vector<float>>
compute_rope_tables(int s, int h_d) {
    int half = h_d / 2;
    vector<float> cos_h(s * half), sin_h(s * half);
    for (int i = 0; i < half; ++i) {
        float theta = 1.0f / std::pow(ROPE_BASE, (2.0f * i) / h_d);
        for (int p = 0; p < s; ++p) {
            cos_h[p * half + i] = std::cos(p * theta);
            sin_h[p * half + i] = std::sin(p * theta);
        }
    }
    return {std::move(cos_h), std::move(sin_h)};
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

Model::Model() : loader_(DumpFloatType::FP32) {}

vector<int> Model::tokenize(const string &input) {
    BPETokenizer tok(TOKENIZER_PATH);
    return tok.encode(input);
}

string Model::detokenize(const vector<int> &ids) {
    BPETokenizer tok(TOKENIZER_PATH);
    return tok.decode(ids);
}

vector<float> Model::embed(const vector<int> &ids) {
    loader_.load_embeddings(EMBEDDING_MATRIX_PATH, EMBEDDING_DIM);
    float_t *raw = loader_.get_embeddings(ids);
    size_t total = ids.size() * EMBEDDING_DIM;
    vector<float> result(raw, raw + total);
    delete[] raw;
    return result;
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

// One decoder block on GPU. Operator order per part2.pdf §3.1:
//   rmsnorm -> q/k/v -> rope -> gqa -> o_proj -> residual ->
//   rmsnorm -> swiglu -> residual.
DeviceBuffer<float> Model::run_decoder_block(
    const DeviceBuffer<float> &d_x, const LayerWeights &w,
    const float *d_cos, const float *d_sin, int s) {

    // --- Attention sub-block ---
    DeviceBuffer<float> d_xnorm(s * EMBEDDING_DIM);
    launch_rmsnorm(d_x.data(), w.input_norm.data(), d_xnorm.data(),
                   s, EMBEDDING_DIM, RMS_NORM_EPSILON);

    // Q/K/V projections produce flat (s, n_heads * h_d). Per part2.pdf §3.1
    // the reshape to (s, n_heads, h_d) is logical — no data movement.
    DeviceBuffer<float> d_Q(s * N_HEADS    * H_DIM);
    DeviceBuffer<float> d_K(s * N_KV_HEADS * H_DIM);
    DeviceBuffer<float> d_V(s * N_KV_HEADS * H_DIM);
    launch_matmul_xwt(d_xnorm.data(), w.q.data(), d_Q.data(),
                      s, EMBEDDING_DIM, N_HEADS    * H_DIM);
    launch_matmul_xwt(d_xnorm.data(), w.k.data(), d_K.data(),
                      s, EMBEDDING_DIM, N_KV_HEADS * H_DIM);
    launch_matmul_xwt(d_xnorm.data(), w.v.data(), d_V.data(),
                      s, EMBEDDING_DIM, N_KV_HEADS * H_DIM);

    DeviceBuffer<float> d_Q_rope(s * N_HEADS    * H_DIM);
    DeviceBuffer<float> d_K_rope(s * N_KV_HEADS * H_DIM);
    launch_rope(d_Q.data(), d_cos, d_sin, d_Q_rope.data(), N_HEADS,    s, H_DIM);
    launch_rope(d_K.data(), d_cos, d_sin, d_K_rope.data(), N_KV_HEADS, s, H_DIM);

    DeviceBuffer<float> d_O(s * N_HEADS * H_DIM);
    launch_gqa_attention(d_Q_rope.data(), d_K_rope.data(), d_V.data(),
                         d_O.data(), N_HEADS, N_KV_HEADS, s, H_DIM);

    DeviceBuffer<float> d_attn_out(s * EMBEDDING_DIM);
    launch_matmul_xwt(d_O.data(), w.o.data(), d_attn_out.data(),
                      s, N_HEADS * H_DIM, EMBEDDING_DIM);

    DeviceBuffer<float> d_x1(s * EMBEDDING_DIM);
    launch_residual_add(d_x.data(), d_attn_out.data(), d_x1.data(),
                        s * EMBEDDING_DIM);

    // --- FFN sub-block ---
    DeviceBuffer<float> d_x1_norm(s * EMBEDDING_DIM);
    launch_rmsnorm(d_x1.data(), w.post_norm.data(), d_x1_norm.data(),
                   s, EMBEDDING_DIM, RMS_NORM_EPSILON);

    DeviceBuffer<float> d_gate(s * D_FF);
    DeviceBuffer<float> d_up(s * D_FF);
    DeviceBuffer<float> d_hidden(s * D_FF);
    DeviceBuffer<float> d_ffn_out(s * EMBEDDING_DIM);
    launch_matmul_xwt(d_x1_norm.data(), w.gate.data(), d_gate.data(),
                      s, EMBEDDING_DIM, D_FF);
    launch_matmul_xwt(d_x1_norm.data(), w.up.data(),   d_up.data(),
                      s, EMBEDDING_DIM, D_FF);
    launch_silu_mul(d_gate.data(), d_up.data(), d_hidden.data(), s * D_FF);
    launch_matmul_xwt(d_hidden.data(), w.down.data(), d_ffn_out.data(),
                      s, D_FF, EMBEDDING_DIM);

    DeviceBuffer<float> d_x2(s * EMBEDDING_DIM);
    launch_residual_add(d_x1.data(), d_ffn_out.data(), d_x2.data(),
                        s * EMBEDDING_DIM);

    return d_x2;
}

vector<float> Model::swiglu_ffn(const vector<float> &x_norm, int layer_idx,
                                int s) {
    // HuggingFace stores projection weights as (out, in) so all three matmuls
    // below are X @ W^T (see part2.pdf §4).
    string prefix = "assets/llama3/blobs/model.layers." +
                    std::to_string(layer_idx) + ".mlp.";

    DeviceBuffer<float> d_x(x_norm);
    DeviceBuffer<float> d_Wg = bulk_load_to_gpu(loader_, prefix + "gate_proj.weight",
                                                (size_t)D_FF * EMBEDDING_DIM);
    DeviceBuffer<float> d_Wu = bulk_load_to_gpu(loader_, prefix + "up_proj.weight",
                                                (size_t)D_FF * EMBEDDING_DIM);
    DeviceBuffer<float> d_Wd = bulk_load_to_gpu(loader_, prefix + "down_proj.weight",
                                                (size_t)EMBEDDING_DIM * D_FF);

    DeviceBuffer<float> d_gate(s * D_FF);
    DeviceBuffer<float> d_up(s * D_FF);
    DeviceBuffer<float> d_hidden(s * D_FF);
    DeviceBuffer<float> d_out(s * EMBEDDING_DIM);

    launch_matmul_xwt(d_x.data(),   d_Wg.data(), d_gate.data(),
                      s, EMBEDDING_DIM, D_FF);
    launch_matmul_xwt(d_x.data(),   d_Wu.data(), d_up.data(),
                      s, EMBEDDING_DIM, D_FF);
    launch_silu_mul(d_gate.data(), d_up.data(), d_hidden.data(), s * D_FF);
    launch_matmul_xwt(d_hidden.data(), d_Wd.data(), d_out.data(),
                      s, D_FF, EMBEDDING_DIM);

    return d_out.to_host();
}

vector<float> Model::decoder_block(const vector<float> &x, int layer_idx,
                                   int s) {
    LayerWeights w = load_layer(layer_idx);

    auto [cos_h, sin_h] = compute_rope_tables(s, H_DIM);
    DeviceBuffer<float> d_cos(cos_h);
    DeviceBuffer<float> d_sin(sin_h);
    DeviceBuffer<float> d_x(x);

    DeviceBuffer<float> d_out =
        run_decoder_block(d_x, w, d_cos.data(), d_sin.data(), s);
    return d_out.to_host();
}

int Model::forward_one_step(const vector<int> &token_ids) {
    int s = (int)token_ids.size();

    // 1. Embedding lookup.
    loader_.load_embeddings(EMBEDDING_MATRIX_PATH, EMBEDDING_DIM);
    float *h_embed = loader_.get_embeddings(token_ids);
    DeviceBuffer<float> d_x(s * EMBEDDING_DIM);
    cudaMemcpy(d_x.data(), h_embed, s * EMBEDDING_DIM * sizeof(float),
               cudaMemcpyHostToDevice);
    delete[] h_embed;

    // 2. RoPE tables, built once and reused across all 32 layers.
    auto [cos_h, sin_h] = compute_rope_tables(s, H_DIM);
    DeviceBuffer<float> d_cos(cos_h);
    DeviceBuffer<float> d_sin(sin_h);

    // 3. Stream layers: load, run, free. Pre-loading all 32 (~28 GB) is a
    //    deferred optimization.
    for (int L = 0; L < N_LAYERS; ++L) {
        LayerWeights w = load_layer(L);
        d_x = run_decoder_block(d_x, w, d_cos.data(), d_sin.data(), s);
    }

    // 4. Final RMSNorm with model.norm.weight.
    DeviceBuffer<float> d_norm = bulk_load_to_gpu(
        loader_, "assets/llama3/blobs/model.norm.weight", EMBEDDING_DIM);
    DeviceBuffer<float> d_x_norm(s * EMBEDDING_DIM);
    launch_rmsnorm(d_x.data(), d_norm.data(), d_x_norm.data(),
                   s, EMBEDDING_DIM, RMS_NORM_EPSILON);

    // 5. lm_head on the last token only (part2.pdf §4: extract only the last
    //    token before lm_head). 1×d row times W_lm^T (V×d row-major) → (1, V).
    DeviceBuffer<float> d_lm = bulk_load_to_gpu(
        loader_, "assets/llama3/blobs/lm_head.weight",
        (size_t)VOCAB_SIZE * EMBEDDING_DIM);
    DeviceBuffer<float> d_logits(VOCAB_SIZE);
    const float *d_x_last = d_x_norm.data() + (s - 1) * EMBEDDING_DIM;
    launch_matmul_xwt(d_x_last, d_lm.data(), d_logits.data(),
                      1, EMBEDDING_DIM, VOCAB_SIZE);

    // 6. Greedy argmax on host.
    vector<float> logits = d_logits.to_host();
    int best = 0;
    float best_v = logits[0];
    for (int v = 1; v < VOCAB_SIZE; ++v) {
        if (logits[v] > best_v) { best_v = logits[v]; best = v; }
    }
    return best;
}

vector<int> Model::generate(const vector<int> &token_ids, int n_new) {
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
