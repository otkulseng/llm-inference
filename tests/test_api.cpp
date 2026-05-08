#include "test_api.h"

#include "bf16.h"
#include "config.h"
#include "device_buffer.h"
#include "kernels.cuh"
#include "model.h"
#include <cmath>

vector<int> TestAPI::tokenize(string input) {
    return Model{}.tokenize(input);
}

string TestAPI::detokenize(vector<int> token_ids) {
    return Model{}.detokenize(token_ids);
}

vector<float> TestAPI::get_embeddings(vector<int> token_ids) {
    return Model{}.embed(token_ids);
}

// Test wrappers convert fp32 inputs/outputs at the boundary; everything on
// the GPU is BF16 with FP32 register accumulators inside kernels.

vector<float> TestAPI::matmul(const vector<float> &A, const vector<float> &B,
                              int M, int K, int N) {
    // Matmul calculates A @ B^T. Transpose beforehand
    vector<float> Bt(N * K);
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            Bt[n * K + k] = B[k * N + n];

    DeviceBuffer<__nv_bfloat16> d_A(to_bf16_host(A));
    DeviceBuffer<__nv_bfloat16> d_Bt(to_bf16_host(Bt));
    DeviceBuffer<__nv_bfloat16> d_C(M * N);

    launch_matmul(d_A.data(), d_Bt.data(), d_C.data(), M, K, N);

    return to_fp32_host(d_C.to_host());
}

vector<float> TestAPI::rmsnorm(const vector<float> &x,
                               const vector<float> &gamma, int s, int d) {
    DeviceBuffer<__nv_bfloat16> d_x(to_bf16_host(x));
    DeviceBuffer<__nv_bfloat16> d_gamma(to_bf16_host(gamma));
    DeviceBuffer<__nv_bfloat16> d_y(s * d);

    launch_rmsnorm(d_x.data(), d_gamma.data(), d_y.data(), s, d, RMS_NORM_EPSILON);

    return to_fp32_host(d_y.to_host());
}

// Permute (n_heads, s, h_d) <-> (s, n_heads, h_d), both flat row-major.
// The CUDA kernels operate on (s, n_heads, h_d) — the natural post-matmul
// layout used by Model. Test 11/12 fixtures are in (n_heads, s, h_d), so
// we transpose once at the test boundary.
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
    // theta_i = 1 / ROPE_BASE^(2i/h_d), shared across heads. Convert fp32 ->
    // BF16 for the kernel.
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
    DeviceBuffer<__nv_bfloat16> d_qk(to_bf16_host(qk_shd));
    DeviceBuffer<__nv_bfloat16> d_cos(to_bf16_host(cos_tab));
    DeviceBuffer<__nv_bfloat16> d_sin(to_bf16_host(sin_tab));
    DeviceBuffer<__nv_bfloat16> d_out(qk.size());

    launch_rope(d_qk.data(), d_cos.data(), d_sin.data(), d_out.data(),
                n_heads, s, h_d);

    return transpose_shd_to_hsd(to_fp32_host(d_out.to_host()),
                                n_heads, s, h_d);
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

    DeviceBuffer<__nv_bfloat16> d_Q(to_bf16_host(Q_shd));
    DeviceBuffer<__nv_bfloat16> d_K(to_bf16_host(K_shd));
    DeviceBuffer<__nv_bfloat16> d_V(to_bf16_host(V_shd));
    DeviceBuffer<__nv_bfloat16> d_O(s * N_HEADS * H_DIM);

    launch_gqa_attention(d_Q.data(), d_K.data(), d_V.data(), d_O.data(), s);

    return to_fp32_host(d_O.to_host());
}

vector<float> TestAPI::residual_add(const vector<float> &a,
                                    const vector<float> &b) {
    DeviceBuffer<__nv_bfloat16> d_a(to_bf16_host(a));
    DeviceBuffer<__nv_bfloat16> d_b(to_bf16_host(b));
    DeviceBuffer<__nv_bfloat16> d_y(a.size());

    launch_residual_add(d_a.data(), d_b.data(), d_y.data(), a.size());

    return to_fp32_host(d_y.to_host());
}

vector<float> TestAPI::silu_mul(const vector<float> &gate,
                                const vector<float> &up) {
    DeviceBuffer<__nv_bfloat16> d_gate(to_bf16_host(gate));
    DeviceBuffer<__nv_bfloat16> d_up(to_bf16_host(up));
    DeviceBuffer<__nv_bfloat16> d_y(gate.size());

    launch_silu_mul(d_gate.data(), d_up.data(), d_y.data(), gate.size());

    return to_fp32_host(d_y.to_host());
}

vector<float> TestAPI::swiglu_ffn(const vector<float> &x_norm, int layer_idx,
                                  int s) {
    return Model{}.swiglu_ffn(x_norm, layer_idx, s);
}

vector<float> TestAPI::decoder_block(const vector<float> &x, int layer_idx,
                                     int s) {
    return Model{}.decoder_block(x, layer_idx, s);
}

int TestAPI::forward_one_step(const vector<int> &token_ids) {
    return Model{}.forward_one_step(token_ids);
}

vector<int> TestAPI::generate(const vector<int> &token_ids, int n_new) {
    return Model{}.generate(token_ids, n_new);
}
