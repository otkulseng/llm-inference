// After tokenizer, and dumping your weight files you can start implementing the loader. 
// The loader is responsible for loading the weights from disk and providing them to the operators. 
// You can implement it in a way that it loads the weights on demand, or you can load all the weights at once and keep them in memory. 
// It is up to you to decide how to implement it, but it should be efficient and easy to use for the operators.

#pragma once
#include "prelude.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class DumpFloatType { FP16, BF16, FP32 };

struct BlobHeader {
  uint32_t dtype;
  uint64_t rows;
  uint64_t cols;
  uint64_t nbytes;
};

class LlamaDumpLoader {
  public:
    DumpFloatType float_type;

    // Construct a generic loader with a chosen float type. The path and
    // dimensions are provided per-call to loading functions.
    explicit LlamaDumpLoader(DumpFloatType float_type);

    ~LlamaDumpLoader();

    // Embedding-specific helpers (2D table: [vocab, embedding_dim]).
    // Provide dump directory or file path, and the expected embedding_dim.
    // These functions will mmap on first use (or remap if the file changes).
    size_t vocab_size(const std::string &dump_path, int embedding_dim);

    bool load_embeddings(const std::string &dump_path, int embedding_dim);

    // Convert only requested token rows into a contiguous float_t buffer.
    // Returns newly allocated buffer of size token_ids.size() * embedding_dim.
    // Caller frees with delete[].
    // There is a memory leak here
    float_t *get_embeddings(const std::vector<int> &token_ids);


    float_t *load_1d(const std::string &dump_file, const size_t &dim0);
    float_t *load_2d(const std::string &dump_file, const size_t &dim0, const size_t &dim1);

  private:
    // mmap state for the embeddings table
    void *emb_mapped = nullptr;
    size_t emb_mapped_size = 0;
    BlobHeader emb_header = {};

    // Helper: mmap a blob file and parse its header
    void *mmap_blob(const std::string &path, size_t &out_mapped_size,
                    BlobHeader &out_header);

    // Helper: convert raw on-disk data to float_t
    void convert_to_float(const void *src, float_t *dst, size_t count,
                          uint32_t dtype_code);
};
