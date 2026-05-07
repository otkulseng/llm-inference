#include "kernels.cuh"

// Elementwise residual add: y[i] = a[i] + b[i]. BF16 in/out, FP32 in registers.
// Per part2.pdf §3.2: "One thread per element is sufficient. Reuse this
// kernel for both residuals in the decoder block."

constexpr int BLOCK_SIZE = 256;

__global__ void residual_add_kernel(const __nv_bfloat16 *a,
                                    const __nv_bfloat16 *b,
                                    __nv_bfloat16 *y, int n) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) {
        float fa = __bfloat162float(a[i]);
        float fb = __bfloat162float(b[i]);
        y[i] = __float2bfloat16(fa + fb);
    }
}

void launch_residual_add(const __nv_bfloat16 *d_a, const __nv_bfloat16 *d_b,
                         __nv_bfloat16 *d_y, int n) {
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    residual_add_kernel<<<grid, block>>>(d_a, d_b, d_y, n);
}
