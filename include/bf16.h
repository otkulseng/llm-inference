#pragma once

// Host-side fp32 ↔ bf16 helpers used at the TestAPI boundary. BF16 is the
// upper 16 bits of an fp32 value (same exponent layout, 7-bit mantissa) — the
// conversion is a shift on the bit pattern. Truncation matches the inverse of
// src/loader.cpp's bf16_to_float, which our on-disk weights were dumped with.

#include <cstdint>
#include <cstring>
#include <cuda_bf16.h>
#include <vector>

inline __nv_bfloat16 fp32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    uint16_t bf = (uint16_t)(bits >> 16);
    __nv_bfloat16 out;
    std::memcpy(&out, &bf, 2);
    return out;
}

inline float bf16_to_fp32(__nv_bfloat16 bf) {
    uint16_t bits;
    std::memcpy(&bits, &bf, 2);
    uint32_t f_bits = (uint32_t)bits << 16;
    float f;
    std::memcpy(&f, &f_bits, 4);
    return f;
}

inline std::vector<__nv_bfloat16> to_bf16_host(const std::vector<float> &x) {
    std::vector<__nv_bfloat16> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = fp32_to_bf16(x[i]);
    return out;
}

inline std::vector<float> to_fp32_host(const std::vector<__nv_bfloat16> &x) {
    std::vector<float> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = bf16_to_fp32(x[i]);
    return out;
}
