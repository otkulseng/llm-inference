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
  uint64_t dtype;
  uint64_t rows;
  uint64_t cols;
  uint64_t nbytes;
};

class LlamaDumpLoader {
  public:
    DumpFloatType float_type;

    // RAII handle for an mmap'd blob. Move-only. Caller reads the raw payload
    // bytes via `raw` and the header metadata via `header`. Destructor munmaps.
    // Bulk weight loads use this directly; the bytes are then memcpy'd to GPU
    // and converted there (see launch_bf16_to_fp32 in kernel/kernels.cuh).
    struct Blob {
        const void *raw = nullptr;     // points just past the header inside mmap_base_
        BlobHeader header = {};

        Blob() = default;
        Blob(void *base, size_t size, BlobHeader h);
        ~Blob();
        Blob(Blob &&o) noexcept;
        Blob &operator=(Blob &&o) noexcept;
        Blob(const Blob &) = delete;
        Blob &operator=(const Blob &) = delete;

      private:
        void *mmap_base_ = nullptr;
        size_t mmap_size_ = 0;
    };

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

    // Open + mmap a blob, parse its header. Returns a move-only RAII handle.
    // Throws on I/O or format error. Caller pulls raw bytes via blob.raw and
    // converts (e.g. on the GPU) as needed.
    Blob open_blob(const std::string &dump_file);

  private:
    // mmap state for the embeddings table
    void *emb_mapped = nullptr;
    size_t emb_mapped_size = 0;
    BlobHeader emb_header = {};

    // Helper: mmap a blob file and parse its header
    void *mmap_blob(const std::string &path, size_t &out_mapped_size,
                    BlobHeader &out_header);

    // Helper: convert raw on-disk data to float_t (used by get_embeddings).
    void convert_to_float(const void *src, float_t *dst, size_t count,
                          uint32_t dtype_code);
};
