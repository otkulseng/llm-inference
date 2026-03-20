#include "test_api.h"
#include "config.h"
#include <cmath>
#include <iostream>

const float EPSILON = 1e-2;

bool test_embeddings() {
    TestAPI api;

    vector<int> token_ids = api.tokenize("Hello world");
    vector<float> emb = api.get_embeddings(token_ids);

    size_t expected_size = token_ids.size() * EMBEDDING_DIM;
    if (emb.size() != expected_size) {
        std::cout << "Embedding size mismatch: expected " << expected_size
                  << " got " << emb.size() << "\n";
        return false;
    }

    // Check that values are not all zero
    float sum = 0.0f;
    for (size_t i = 0; i < emb.size(); i++) {
        sum += emb[i] * emb[i];
    }
    if (sum < EPSILON) {
        std::cout << "Embeddings are all zero\n";
        return false;
    }

    std::cout << "Embedding size: " << emb.size()
              << ", L2 norm of first token: ";
    float norm = 0.0f;
    for (int i = 0; i < 4096; i++) {
        norm += emb[i] * emb[i];
    }
    std::cout << std::sqrt(norm) << "\n";

    return true;
}

bool test_matmul_small() {
    // A = [1, 2]    B = [5, 6]    C = A*B = [19, 22]
    //     [3, 4]        [7, 8]              [43, 50]
    TestAPI api;

    vector<float> A = {1, 2, 3, 4};
    vector<float> B = {5, 6, 7, 8};
    vector<float> C = api.matmul(A, B, 2, 2, 2);

    vector<float> expected = {19, 22, 43, 50};

    if (C.size() != expected.size()) {
        std::cout << "Size mismatch: expected " << expected.size()
                  << " got " << C.size() << "\n";
        return false;
    }

    for (size_t i = 0; i < C.size(); i++) {
        if (std::abs(C[i] - expected[i]) > EPSILON) {
            std::cout << "Mismatch at [" << i << "]: expected " << expected[i]
                      << " got " << C[i] << "\n";
            return false;
        }
    }
    return true;
}

bool test_matmul_nonsquare() {
    // A[2,3] * B[3,2] = C[2,2]
    // A = [1, 2, 3]    B = [1, 4]    C = [14, 32]
    //     [4, 5, 6]        [2, 5]        [32, 77]
    //                      [3, 6]
    TestAPI api;

    vector<float> A = {1, 2, 3, 4, 5, 6};
    vector<float> B = {1, 4, 2, 5, 3, 6};
    vector<float> C = api.matmul(A, B, 2, 3, 2);

    vector<float> expected = {14, 32, 32, 77};

    for (size_t i = 0; i < C.size(); i++) {
        if (std::abs(C[i] - expected[i]) > EPSILON) {
            std::cout << "Mismatch at [" << i << "]: expected " << expected[i]
                      << " got " << C[i] << "\n";
            return false;
        }
    }
    return true;
}

bool test_matmul_large() {
    // Test with dimensions that aren't multiples of TILE_SIZE (32)
    // A[65,33] * B[33,17] = C[65,17]
    // Fill with simple pattern: A[i][j] = i+j, B[i][j] = i-j
    int M = 65, K = 33, N = 17;
    TestAPI api;

    vector<float> A(M * K), B(K * N);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            A[i * K + j] = (float)(i + j);
    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            B[i * N + j] = (float)(i - j);

    // CPU reference
    vector<float> expected(M * N, 0.0f);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < K; k++)
                expected[i * N + j] += A[i * K + k] * B[k * N + j];

    vector<float> C = api.matmul(A, B, M, K, N);

    for (int i = 0; i < M * N; i++) {
        if (std::abs(C[i] - expected[i]) > EPSILON) {
            std::cout << "Mismatch at [" << i << "]: expected " << expected[i]
                      << " got " << C[i] << "\n";
            return false;
        }
    }
    return true;
}

int main() {
    const char *GREEN = "\033[32m";
    const char *RED = "\033[31m";
    const char *RESET = "\033[0m";

    struct {
        const char *name;
        bool (*fn)();
    } tests[] = {
        {"test_embeddings", test_embeddings},
        {"test_matmul_small", test_matmul_small},
        {"test_matmul_nonsquare", test_matmul_nonsquare},
        {"test_matmul_large", test_matmul_large},
    };

    int passed = 0, total = 0;
    for (auto &t : tests) {
        total++;
        std::cout << "Running " << t.name << "... ";
        try {
            if (t.fn()) {
                std::cout << GREEN << "PASSED" << RESET << "\n";
                passed++;
            } else {
                std::cout << RED << "FAILED" << RESET << "\n";
            }
        } catch (const std::exception &e) {
            std::cout << RED << "THREW: " << e.what() << RESET << "\n";
        }
    }

    std::cout << "\n" << passed << "/" << total << " passed\n";
    return (passed == total) ? 0 : 1;
}
