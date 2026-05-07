

#include "test_api.h"
#include "config.h"
#include "device_buffer.h"
#include "kernels.cuh"
#include "loader.h"
#include "tokenizer.h"
#include <cmath>
#include <iostream>

vector<int> TestAPI::tokenize(string input) {
    // Your code for testing the tokenizer goes here.
    // You can use the tokenizer you implemented in src/tokenizer_bpe.cpp and
    // include the header file here. Read the project description for more
    // information.
    BPETokenizer tok(TOKENIZER_PATH);
    std::cout << input << std::endl;

    auto res = tok.encode(input);
    for (auto e : res) {
        std::cout << e << ' ';
    }
    std::cout << std::endl;
    return res;

    // throw runtime_error(
    //     "Not implemented: you need to implement the tokenize function here");
}

vector<float> TestAPI::get_embeddings(vector<int> token_ids) {
    LlamaDumpLoader loader(DumpFloatType::FP32);
    loader.load_embeddings(EMBEDDING_MATRIX_PATH, EMBEDDING_DIM);

    float_t *raw = loader.get_embeddings(token_ids);
    size_t total = token_ids.size() * EMBEDDING_DIM;
    vector<float> result(raw, raw + total);
    delete[] raw;
    return result;
}

vector<float> TestAPI::matmul(const vector<float> &A, const vector<float> &B,
                              int M, int K, int N) {
    DeviceBuffer<float> d_A(A);
    DeviceBuffer<float> d_B(B);
    DeviceBuffer<float> d_C(M * N);

    launch_matmul(d_A.data(), d_B.data(), d_C.data(), M, K, N);

    return d_C.to_host();
}

// ---- Milestone 2/3 stubs (replace with real implementations) ----

string TestAPI::detokenize(vector<int> token_ids) {
    BPETokenizer tok(TOKENIZER_PATH);
    return tok.decode(token_ids);
}

vector<float> TestAPI::rmsnorm(const vector<float> &x,
                               const vector<float> &gamma, int s, int d) {
    DeviceBuffer<float> d_x(x);
    DeviceBuffer<float> d_gamma(gamma);
    DeviceBuffer<float> d_y(s * d);

    launch_rmsnorm(d_x.data(), d_gamma.data(), d_y.data(), s, d, RMS_NORM_EPSILON);

    return d_y.to_host();
}

// Permute (n_heads, s, h_d) <-> (s, n_heads, h_d), both flat row-major.
// The CUDA kernels operate on (s, n_heads, h_d) — the natural post-matmul
// layout used by decoder_block. Test 11/12 fixtures are in (n_heads, s, h_d),
// so we transpose once at the test boundary.
static vector<float> transpose_hsd_to_shd(const vector<float> &in,
                                          int n_heads, int s, int h_d) {
    vector<float> out(in.size());
    for (int h = 0; h < n_heads; ++h)
        for (int p = 0; p < s; ++p)
            for (int d = 0; d < h_d; ++d)
                out[p * (n_heads * h_d) + h * h_d + d] =
                    in[h * (s * h_d) + p * h_d + d];
    return out;
}

static vector<float> transpose_shd_to_hsd(const vector<float> &in,
                                          int n_heads, int s, int h_d) {
    vector<float> out(in.size());
    for (int h = 0; h < n_heads; ++h)
        for (int p = 0; p < s; ++p)
            for (int d = 0; d < h_d; ++d)
                out[h * (s * h_d) + p * h_d + d] =
                    in[p * (n_heads * h_d) + h * h_d + d];
    return out;
}

vector<float> TestAPI::rope(const vector<float> &qk, int n_heads, int s,
                            int h_d) {
    int half = h_d / 2;

    // Pre-compute cos(p * theta_i) and sin(p * theta_i) per part2.pdf §3.2:
    // theta_i = 1 / ROPE_BASE^(2i/h_d), shared across heads.
    vector<float> cos_tab(s * half);
    vector<float> sin_tab(s * half);
    for (int i = 0; i < half; ++i) {
        float theta = 1.0f / std::pow(ROPE_BASE, (2.0f * i) / h_d);
        for (int p = 0; p < s; ++p) {
            cos_tab[p * half + i] = std::cos(p * theta);
            sin_tab[p * half + i] = std::sin(p * theta);
        }
    }

    vector<float> qk_shd = transpose_hsd_to_shd(qk, n_heads, s, h_d);
    DeviceBuffer<float> d_qk(qk_shd);
    DeviceBuffer<float> d_cos(cos_tab);
    DeviceBuffer<float> d_sin(sin_tab);
    DeviceBuffer<float> d_out(qk.size());

    launch_rope(d_qk.data(), d_cos.data(), d_sin.data(), d_out.data(),
                n_heads, s, h_d);

    return transpose_shd_to_hsd(d_out.to_host(), n_heads, s, h_d);
}

vector<float> TestAPI::gqa_attention(const vector<float> &Q,
                                     const vector<float> &K,
                                     const vector<float> &V, int s) {
    // Inputs come as (n, s, h_d). The kernel reads (s, n, h_d). Output is
    // already (s, n_heads, h_d) flat = (s, n_heads * h_d) flat — matches the
    // test 12 fixture, no output transpose needed.
    vector<float> Q_shd = transpose_hsd_to_shd(Q, N_HEADS,    s, H_DIM);
    vector<float> K_shd = transpose_hsd_to_shd(K, N_KV_HEADS, s, H_DIM);
    vector<float> V_shd = transpose_hsd_to_shd(V, N_KV_HEADS, s, H_DIM);

    DeviceBuffer<float> d_Q(Q_shd);
    DeviceBuffer<float> d_K(K_shd);
    DeviceBuffer<float> d_V(V_shd);
    DeviceBuffer<float> d_O(s * N_HEADS * H_DIM);

    launch_gqa_attention(d_Q.data(), d_K.data(), d_V.data(), d_O.data(),
                         N_HEADS, N_KV_HEADS, s, H_DIM);

    return d_O.to_host();
}

vector<float> TestAPI::residual_add(const vector<float> &a,
                                    const vector<float> &b) {
    DeviceBuffer<float> d_a(a);
    DeviceBuffer<float> d_b(b);
    DeviceBuffer<float> d_y(a.size());

    launch_residual_add(d_a.data(), d_b.data(), d_y.data(), a.size());

    return d_y.to_host();
}

vector<float> TestAPI::silu_mul(const vector<float> &gate,
                                const vector<float> &up) {
    DeviceBuffer<float> d_gate(gate);
    DeviceBuffer<float> d_up(up);
    DeviceBuffer<float> d_y(gate.size());

    launch_silu_mul(d_gate.data(), d_up.data(), d_y.data(), gate.size());

    return d_y.to_host();
}

vector<float> TestAPI::swiglu_ffn(const vector<float> &x_norm, int layer_idx,
                                  int s) {
    // Load this layer's MLP weights from disk. HuggingFace stores them as
    // (out, in) so all three matmuls below are X @ W^T (see part2.pdf §4).
    LlamaDumpLoader loader(DumpFloatType::FP32);
    string prefix = "assets/llama3/blobs/model.layers." +
                    std::to_string(layer_idx) + ".mlp.";
    float *W_gate = loader.load_2d(prefix + "gate_proj.weight",
                                   D_FF, EMBEDDING_DIM);
    float *W_up   = loader.load_2d(prefix + "up_proj.weight",
                                   D_FF, EMBEDDING_DIM);
    float *W_down = loader.load_2d(prefix + "down_proj.weight",
                                   EMBEDDING_DIM, D_FF);

    DeviceBuffer<float> d_x(x_norm);
    DeviceBuffer<float> d_Wg(D_FF * EMBEDDING_DIM);
    DeviceBuffer<float> d_Wu(D_FF * EMBEDDING_DIM);
    DeviceBuffer<float> d_Wd(EMBEDDING_DIM * D_FF);
    cudaMemcpy(d_Wg.data(), W_gate, D_FF * EMBEDDING_DIM * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wu.data(), W_up, D_FF * EMBEDDING_DIM * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wd.data(), W_down, EMBEDDING_DIM * D_FF * sizeof(float),
               cudaMemcpyHostToDevice);
    delete[] W_gate;
    delete[] W_up;
    delete[] W_down;

    DeviceBuffer<float> d_gate(s * D_FF);
    DeviceBuffer<float> d_up(s * D_FF);
    DeviceBuffer<float> d_hidden(s * D_FF);
    DeviceBuffer<float> d_out(s * EMBEDDING_DIM);

    // gate = x_norm @ W_gate^T   (s, d) · (d_ff, d)^T -> (s, d_ff)
    launch_matmul_xwt(d_x.data(), d_Wg.data(), d_gate.data(),
                      s, EMBEDDING_DIM, D_FF);
    // up = x_norm @ W_up^T
    launch_matmul_xwt(d_x.data(), d_Wu.data(), d_up.data(),
                      s, EMBEDDING_DIM, D_FF);
    // hidden = SiLU(gate) ⊙ up
    launch_silu_mul(d_gate.data(), d_up.data(), d_hidden.data(), s * D_FF);
    // ffn_out = hidden @ W_down^T   (s, d_ff) · (d, d_ff)^T -> (s, d)
    launch_matmul_xwt(d_hidden.data(), d_Wd.data(), d_out.data(),
                      s, D_FF, EMBEDDING_DIM);

    return d_out.to_host();
}

// ---- Per-layer helpers, shared by decoder_block and forward_one_step ----

struct LayerWeights {
    DeviceBuffer<float> input_norm, q, k, v, o;
    DeviceBuffer<float> post_norm, gate, up, down;
};

static DeviceBuffer<float> load_to_gpu_1d(LlamaDumpLoader &loader,
                                          const string &path, size_t n) {
    float *h = loader.load_1d(path, n);
    DeviceBuffer<float> d(n);
    cudaMemcpy(d.data(), h, n * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h;
    return d;
}

static DeviceBuffer<float> load_to_gpu_2d(LlamaDumpLoader &loader,
                                          const string &path,
                                          size_t r, size_t c) {
    float *h = loader.load_2d(path, r, c);
    DeviceBuffer<float> d(r * c);
    cudaMemcpy(d.data(), h, r * c * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h;
    return d;
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

static LayerWeights load_layer(LlamaDumpLoader &loader, int layer_idx) {
    string p = "assets/llama3/blobs/model.layers." +
               std::to_string(layer_idx) + ".";
    return LayerWeights{
        load_to_gpu_1d(loader, p + "input_layernorm.weight", EMBEDDING_DIM),
        load_to_gpu_2d(loader, p + "self_attn.q_proj.weight",
                       (size_t)N_HEADS * H_DIM, EMBEDDING_DIM),
        load_to_gpu_2d(loader, p + "self_attn.k_proj.weight",
                       (size_t)N_KV_HEADS * H_DIM, EMBEDDING_DIM),
        load_to_gpu_2d(loader, p + "self_attn.v_proj.weight",
                       (size_t)N_KV_HEADS * H_DIM, EMBEDDING_DIM),
        load_to_gpu_2d(loader, p + "self_attn.o_proj.weight",
                       EMBEDDING_DIM, (size_t)N_HEADS * H_DIM),
        load_to_gpu_1d(loader, p + "post_attention_layernorm.weight",
                       EMBEDDING_DIM),
        load_to_gpu_2d(loader, p + "mlp.gate_proj.weight", D_FF, EMBEDDING_DIM),
        load_to_gpu_2d(loader, p + "mlp.up_proj.weight",   D_FF, EMBEDDING_DIM),
        load_to_gpu_2d(loader, p + "mlp.down_proj.weight", EMBEDDING_DIM, D_FF),
    };
}

// One decoder block on GPU.  Operator order per part2.pdf §3.1:
//   rmsnorm -> q/k/v -> rope -> gqa -> o_proj -> residual ->
//   rmsnorm -> swiglu -> residual.
static DeviceBuffer<float> run_decoder_block(
    const DeviceBuffer<float> &d_x, const LayerWeights &w,
    const float *d_cos, const float *d_sin, int s) {

    // --- Attention sub-block ---
    DeviceBuffer<float> d_xnorm(s * EMBEDDING_DIM);
    launch_rmsnorm(d_x.data(), w.input_norm.data(), d_xnorm.data(),
                   s, EMBEDDING_DIM, RMS_NORM_EPSILON);

    // Q/K/V projections produce flat (s, n_heads * h_d).  Per part2.pdf §3.1
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

vector<float> TestAPI::decoder_block(const vector<float> &x, int layer_idx,
                                     int s) {
    LlamaDumpLoader loader(DumpFloatType::FP32);
    LayerWeights w = load_layer(loader, layer_idx);

    auto [cos_h, sin_h] = compute_rope_tables(s, H_DIM);
    DeviceBuffer<float> d_cos(cos_h);
    DeviceBuffer<float> d_sin(sin_h); 
    DeviceBuffer<float> d_x(x);

    DeviceBuffer<float> d_out =
        run_decoder_block(d_x, w, d_cos.data(), d_sin.data(), s);
    return d_out.to_host();
}

int TestAPI::forward_one_step(const vector<int> &token_ids) {
    int s = (int)token_ids.size();
    LlamaDumpLoader loader(DumpFloatType::FP32);

    // 1. Embedding lookup. Reuses the existing get_embeddings() path.
    loader.load_embeddings(EMBEDDING_MATRIX_PATH, EMBEDDING_DIM);
    float *h_embed = loader.get_embeddings(token_ids);
    DeviceBuffer<float> d_x(s * EMBEDDING_DIM);
    cudaMemcpy(d_x.data(), h_embed, s * EMBEDDING_DIM * sizeof(float),
               cudaMemcpyHostToDevice);
    delete[] h_embed;

    // 2. RoPE tables, built once and reused across all 32 layers.
    auto [cos_h, sin_h] = compute_rope_tables(s, H_DIM);
    DeviceBuffer<float> d_cos(cos_h);
    DeviceBuffer<float> d_sin(sin_h);

    // 3. Stream layers: load, run, free.  Peak GPU weight footprint ~865 MB
    //    per iteration. Pre-loading all 32 (~28 GB) is a deferred optimization.
    for (int L = 0; L < N_LAYERS; ++L) {
        LayerWeights w = load_layer(loader, L);
        d_x = run_decoder_block(d_x, w, d_cos.data(), d_sin.data(), s);
    }

    // 4. Final RMSNorm with model.norm.weight.
    DeviceBuffer<float> d_norm = load_to_gpu_1d(
        loader, "assets/llama3/blobs/model.norm.weight", EMBEDDING_DIM);
    DeviceBuffer<float> d_x_norm(s * EMBEDDING_DIM);
    launch_rmsnorm(d_x.data(), d_norm.data(), d_x_norm.data(),
                   s, EMBEDDING_DIM, RMS_NORM_EPSILON);

    // 5. lm_head on the last token only (part2.pdf §4: extract only the last
    //    token before lm_head). 1×d row times W_lm^T (V×d row-major) → (1, V).
    DeviceBuffer<float> d_lm = load_to_gpu_2d(
        loader, "assets/llama3/blobs/lm_head.weight", VOCAB_SIZE, EMBEDDING_DIM);
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

vector<int> TestAPI::generate(const vector<int> &token_ids, int n_new) {
    throw runtime_error("Not implemented: generate");
}
