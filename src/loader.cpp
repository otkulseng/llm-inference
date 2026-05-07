#include "loader.h"

#include <cassert>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

LlamaDumpLoader::LlamaDumpLoader(DumpFloatType float_type)
    : float_type(float_type) {}

void *LlamaDumpLoader::mmap_blob(const std::string &path,
                                 size_t &out_mapped_size,
                                 BlobHeader &out_header) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw runtime_error("Failed to open blob file: " + path);
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        throw runtime_error("Failed to stat blob file: " + path);
    }

    size_t file_size = static_cast<size_t>(st.st_size);
    if (file_size < sizeof(BlobHeader)) {
        close(fd);
        throw runtime_error("Blob file too small for header: " + path);
    }

    void *mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (mapped == MAP_FAILED) {
        throw runtime_error("mmap failed for: " + path);
    }

    memcpy(&out_header, mapped, sizeof(BlobHeader));
    out_mapped_size = file_size;
    return mapped;
}

// ---------------------------------------------------------------------------
// Blob: RAII handle for an mmap'd blob payload.
// ---------------------------------------------------------------------------

LlamaDumpLoader::Blob::Blob(void *base, size_t size, BlobHeader h)
    : raw(static_cast<const char *>(base) + sizeof(BlobHeader)),
      header(h),
      mmap_base_(base),
      mmap_size_(size) {}

LlamaDumpLoader::Blob::~Blob() {
    if (mmap_base_) munmap(mmap_base_, mmap_size_);
}

LlamaDumpLoader::Blob::Blob(Blob &&o) noexcept
    : raw(o.raw),
      header(o.header),
      mmap_base_(o.mmap_base_),
      mmap_size_(o.mmap_size_) {
    o.raw = nullptr;
    o.mmap_base_ = nullptr;
    o.mmap_size_ = 0;
}

LlamaDumpLoader::Blob &
LlamaDumpLoader::Blob::operator=(Blob &&o) noexcept {
    if (this != &o) {
        if (mmap_base_) munmap(mmap_base_, mmap_size_);
        raw = o.raw;
        header = o.header;
        mmap_base_ = o.mmap_base_;
        mmap_size_ = o.mmap_size_;
        o.raw = nullptr;
        o.mmap_base_ = nullptr;
        o.mmap_size_ = 0;
    }
    return *this;
}

LlamaDumpLoader::Blob LlamaDumpLoader::open_blob(const std::string &dump_file) {
    BlobHeader hdr;
    size_t mapped_size;
    void *mapped = mmap_blob(dump_file, mapped_size, hdr);
    return Blob(mapped, mapped_size, hdr);
}
