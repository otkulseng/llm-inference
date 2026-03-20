#include "loader.h"

#include <cassert>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>


static float fp16_to_float(uint16_t h) {
    // IEEE 754 half-precision → single-precision
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        // subnormal or zero
        if (mant == 0) {
            // zero
            uint32_t bits = sign;
            float result;
            memcpy(&result, &bits, 4);
            return result;
        }
        // subnormal: normalize
        exp = 1;
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        mant &= 0x3FF;
        exp = exp + (127 - 15);
    } else if (exp == 31) {
        // inf / nan
        exp = 255;
    } else {
        exp = exp + (127 - 15);
    }

    uint32_t bits = sign | (exp << 23) | (mant << 13);
    float result;
    memcpy(&result, &bits, 4);
    return result;
}

static float bf16_to_float(uint16_t bf) {
    // BF16 is the upper 16 bits of a float32 — just shift left
    uint32_t bits = (uint32_t)bf << 16;
    float result;
    memcpy(&result, &bits, 4);
    return result;
}

// --- LlamaDumpLoader implementation ---------------------------------------

LlamaDumpLoader::LlamaDumpLoader(DumpFloatType float_type)
    : float_type(float_type) {}

LlamaDumpLoader::~LlamaDumpLoader() {
    if (emb_mapped) {
        munmap(emb_mapped, emb_mapped_size);
        emb_mapped = nullptr;
    }
}

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
    close(fd); // safe to close — mapping stays valid

    if (mapped == MAP_FAILED) {
        throw runtime_error("mmap failed for: " + path);
    }

    // Copy out the header from the start of the mapped region
    memcpy(&out_header, mapped, sizeof(BlobHeader));
    out_mapped_size = file_size;

    return mapped;
}

void LlamaDumpLoader::convert_to_float(const void *src, float_t *dst,
                                        size_t count, uint32_t dtype_code) {
    switch (dtype_code) {
    case 0: { // FP16
        const uint16_t *in = static_cast<const uint16_t *>(src);
        for (size_t i = 0; i < count; i++) {
            dst[i] = fp16_to_float(in[i]);
        }
        break;
    }
    case 1: { // FP32
        memcpy(dst, src, count * sizeof(float));
        break;
    }
    case 2: { // BF16
        const uint16_t *in = static_cast<const uint16_t *>(src);
        for (size_t i = 0; i < count; i++) {
            dst[i] = bf16_to_float(in[i]);
        }
        break;
    }
    default:
        throw runtime_error("Unknown dtype code: " + std::to_string(dtype_code));
    }
}

size_t LlamaDumpLoader::vocab_size(const std::string &dump_path,
                                    int embedding_dim) {
    if (!emb_mapped) {
        emb_mapped = mmap_blob(dump_path, emb_mapped_size, emb_header);
        printf("sizeof(BlobHeader) = %zu\n", sizeof(BlobHeader));
        printf("dtype=%u rows=%lu cols=%lu nbytes=%lu\n",
               emb_header.dtype, emb_header.rows, emb_header.cols, emb_header.nbytes);
        assert(emb_header.cols == static_cast<uint64_t>(embedding_dim));
    }
    return emb_header.rows;
}

bool LlamaDumpLoader::load_embeddings(const std::string &dump_path,
                                       int embedding_dim) {
    vocab_size(dump_path, embedding_dim);
    return emb_mapped != nullptr;
}

float_t *LlamaDumpLoader::get_embeddings(const std::vector<int> &token_ids) {
    assert(emb_mapped && "Must call load_embeddings first");

    size_t emb_dim = emb_header.cols;
    size_t n_tokens = token_ids.size();

    // Determine bytes-per-element based on dtype
    size_t elem_size;
    switch (emb_header.dtype) {
    case 0: // FP16
    case 2: // BF16
        elem_size = 2;
        break;
    case 1: // FP32
        elem_size = 4;
        break;
    default:
        throw runtime_error("Unknown dtype in embedding header");
    }

    // Raw data starts right after the header
    // I casst to const char * so that pointer arithmetic pushes 1 byte at a time
    // making my life easier. Const because of the PROT_READ above.
    const char *raw = static_cast<const char *>(emb_mapped) + sizeof(BlobHeader);
    size_t row_bytes = emb_dim * elem_size;

    float_t *out = new float_t[n_tokens * emb_dim];

    for (size_t i = 0; i < n_tokens; i++) {
        int tid = token_ids[i];

        // For debug purposes. Assertion instead of segfault. 
        assert(tid >= 0 && static_cast<uint64_t>(tid) < emb_header.rows);

        const void *row_ptr = raw + tid * row_bytes;
        convert_to_float(row_ptr, out + i * emb_dim, emb_dim, emb_header.dtype);
    }

    return out;
}

// Could read instead, but easiert to use mmap_blob now that I have
// the header logic properly there.
float_t *LlamaDumpLoader::load_1d(const std::string &dump_file,
                                   const size_t &dim0) {
    BlobHeader hdr;
    size_t mapped_size;
    void *mapped = mmap_blob(dump_file, mapped_size, hdr);

    // The dumper unsqueezes 1D tensors to [dim0, 1]
    assert(hdr.rows == dim0 && hdr.cols == 1);

    const void *raw = static_cast<const char *>(mapped) + sizeof(BlobHeader);

    float_t *out = new float_t[dim0];
    convert_to_float(raw, out, dim0, hdr.dtype);

    munmap(mapped, mapped_size);
    return out;
}

float_t *LlamaDumpLoader::load_2d(const std::string &dump_file,
                                   const size_t &dim0, const size_t &dim1) {
    BlobHeader hdr;
    size_t mapped_size;
    void *mapped = mmap_blob(dump_file, mapped_size, hdr);

    assert(hdr.rows == dim0 && hdr.cols == dim1);

    const void *raw = static_cast<const char *>(mapped) + sizeof(BlobHeader);

    float_t *out = new float_t[dim0 * dim1];
    convert_to_float(raw, out, dim0 * dim1, hdr.dtype);

    munmap(mapped, mapped_size);
    return out;
}
