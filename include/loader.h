// After tokenizer, and dumping your weight files you can start implementing the loader.
// The loader is responsible for loading the weights from disk and providing them to the operators.
// You can implement it in a way that it loads the weights on demand, or you can load all the weights at once and keep them in memory.
// It is up to you to decide how to implement it, but it should be efficient and easy to use for the operators.

#pragma once
#include "prelude.h"
#include <cstddef>
#include <cstdint>
#include <string>

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
    // Bulk weight loads use this directly; the bytes are then memcpy'd to GPU.
    struct Blob {
        const void *raw = nullptr;
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

    explicit LlamaDumpLoader(DumpFloatType float_type);

    // Open + mmap a blob, parse its header. Returns a move-only RAII handle.
    // Throws on I/O or format error.
    Blob open_blob(const std::string &dump_file);

  private:
    // Helper: mmap a blob file and parse its header.
    void *mmap_blob(const std::string &path, size_t &out_mapped_size,
                    BlobHeader &out_header);
};
