// Loader for the dumper-produced binary blobs in assets/llama3/blobs/. One
// LlamaDumpLoader instance owns one mmap'd file: constructor opens + maps +
// parses the header, destructor unmaps. Move-only. Callers read the payload
// via `raw` (which already skips past the header) and shape metadata via
// `header`.

#pragma once
#include "prelude.h"
#include <cstddef>
#include <cstdint>
#include <string>

struct BlobHeader {
    uint64_t dtype;
    uint64_t rows;
    uint64_t cols;
    uint64_t nbytes;
};

class LlamaDumpLoader {
  public:
    const void *raw = nullptr;
    BlobHeader header = {};

    explicit LlamaDumpLoader(const std::string &path);
    ~LlamaDumpLoader();
    LlamaDumpLoader(LlamaDumpLoader &&o) noexcept;
    LlamaDumpLoader &operator=(LlamaDumpLoader &&o) noexcept;
    LlamaDumpLoader(const LlamaDumpLoader &) = delete;
    LlamaDumpLoader &operator=(const LlamaDumpLoader &) = delete;

  private:
    void *mmap_base_ = nullptr;
    size_t mmap_size_ = 0;
};
