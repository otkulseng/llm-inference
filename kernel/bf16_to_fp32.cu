#include "kernels.cuh"

// BF16 → FP32 elementwise conversion on the GPU.
//
// BF16 (bfloat16) is the upper 16 bits of an FP32 value: same exponent layout,
// truncated mantissa. Widening BF16 → FP32 is therefore a lossless left-shift
// by 16 (zero-fills the low mantissa bits). This matches the host-side
// bf16_to_float in src/loader.cpp:47-53 bit-for-bit.
//
// One thread per element. We use this in the bulk weight-load path so the
// GPU does the conversion (instead of a single-threaded CPU loop) and we
// transfer half the bytes over PCIe (2 B/elem instead of 4 B/elem).

constexpr int BLOCK_SIZE = 256;

__global__ void bf16_to_fp32_kernel(const uint16_t *in, float *out,
                                    std::size_t n) {
    std::size_t i = (std::size_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) {
        uint32_t bits = (uint32_t)in[i] << 16;
        out[i] = __int_as_float(bits);
    }
}

void launch_bf16_to_fp32(const uint16_t *d_in, float *d_out, std::size_t n) {
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    bf16_to_fp32_kernel<<<grid, block>>>(d_in, d_out, n);
}
