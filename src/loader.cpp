#include "loader.h"

#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

LlamaDumpLoader::LlamaDumpLoader(const std::string &path) {
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

    std::memcpy(&header, mapped, sizeof(BlobHeader));
    mmap_base_ = mapped;
    mmap_size_ = file_size;
    raw = static_cast<const char *>(mapped) + sizeof(BlobHeader);
}

LlamaDumpLoader::~LlamaDumpLoader() {
    if (mmap_base_) munmap(mmap_base_, mmap_size_);
}

LlamaDumpLoader::LlamaDumpLoader(LlamaDumpLoader &&o) noexcept
    : raw(o.raw),
      header(o.header),
      mmap_base_(o.mmap_base_),
      mmap_size_(o.mmap_size_) {
    o.raw = nullptr;
    o.mmap_base_ = nullptr;
    o.mmap_size_ = 0;
}

LlamaDumpLoader &LlamaDumpLoader::operator=(LlamaDumpLoader &&o) noexcept {
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
