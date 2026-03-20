#include "test_api.h"
#include "config.h"
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

int main() {
    const char *GREEN = "\033[32m";
    const char *RED = "\033[31m";
    const char *RESET = "\033[0m";

    struct {
        const char *name;
        bool (*fn)();
    } tests[] = {
        {"test_embeddings", test_embeddings},
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
