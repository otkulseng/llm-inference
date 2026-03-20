

#include "test_api.h"
#include "config.h"
#include "loader.h"
#include "tokenizer.h"
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
    LlamaDumpLoader loader(DumpFloatType::BF16);
    loader.load_embeddings(EMBEDDING_MATRIX_PATH, EMBEDDING_DIM);

    float_t *raw = loader.get_embeddings(token_ids);
    size_t total = token_ids.size() * EMBEDDING_DIM;
    vector<float> result(raw, raw + total);
    delete[] raw;
    return result;
}

vector<float> TestAPI::matmul(const vector<float> &A, const vector<float> &B,
                              int M, int K, int N) {
    throw runtime_error("Not implemented: you need to implement the "
                        "matmul function here");
}
