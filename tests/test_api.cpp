

#include "test_api.h"
#include "config.h"
#include "loader.h"
#include "tokenizer.h"
#include "kernels.cuh"
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
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Copy inputs to device
    cudaMemcpy(d_A, A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), size_B, cudaMemcpyHostToDevice);

    // Launch kernel
    launch_matmul(d_A, d_B, d_C, M, K, N);

    // Copy result back to host
    vector<float> C(M * N);
    cudaMemcpy(C.data(), d_C, size_C, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

// ---- Milestone 2/3 stubs (replace with real implementations) ----

string TestAPI::detokenize(vector<int> token_ids) {
    throw runtime_error("Not implemented: detokenize");
}

vector<float> TestAPI::rmsnorm(const vector<float> &x,
                               const vector<float> &gamma, int s, int d) {
    throw runtime_error("Not implemented: rmsnorm");
}

vector<float> TestAPI::rope(const vector<float> &qk, int n_heads, int s,
                            int h_d) {
    throw runtime_error("Not implemented: rope");
}

vector<float> TestAPI::gqa_attention(const vector<float> &Q,
                                     const vector<float> &K,
                                     const vector<float> &V, int s) {
    throw runtime_error("Not implemented: gqa_attention");
}

vector<float> TestAPI::residual_add(const vector<float> &a,
                                    const vector<float> &b) {
    throw runtime_error("Not implemented: residual_add");
}

vector<float> TestAPI::silu_mul(const vector<float> &gate,
                                const vector<float> &up) {
    throw runtime_error("Not implemented: silu_mul");
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
