

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

    DeviceBuffer<float> d_qk(qk);
    DeviceBuffer<float> d_cos(cos_tab);
    DeviceBuffer<float> d_sin(sin_tab);
    DeviceBuffer<float> d_out(qk.size());

    launch_rope(d_qk.data(), d_cos.data(), d_sin.data(), d_out.data(),
                n_heads, s, h_d);

    return d_out.to_host();
}

vector<float> TestAPI::gqa_attention(const vector<float> &Q,
                                     const vector<float> &K,
                                     const vector<float> &V, int s) {
    throw runtime_error("Not implemented: gqa_attention");
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
    throw runtime_error("Not implemented: swiglu_ffn");
}

vector<float> TestAPI::decoder_block(const vector<float> &x, int layer_idx,
                                     int s) {
    throw runtime_error("Not implemented: decoder_block");
}

int TestAPI::forward_one_step(const vector<int> &token_ids) {
    throw runtime_error("Not implemented: forward_one_step");
}

vector<int> TestAPI::generate(const vector<int> &token_ids, int n_new) {
    throw runtime_error("Not implemented: generate");
}
