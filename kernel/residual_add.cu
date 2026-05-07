#include "kernels.cuh"

// Elementwise residual add: y[i] = a[i] + b[i].
// Per part2.pdf §3.2: "One thread per element is sufficient. Reuse this
// kernel for both residuals in the decoder block."

constexpr int BLOCK_SIZE = 256;

__global__ void residual_add_kernel(const float *a, const float *b, float *y,
                                    int n) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) {
        y[i] = a[i] + b[i];
    }
}

void launch_residual_add(const float *d_a, const float *d_b, float *d_y, int n) {
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    residual_add_kernel<<<grid, block>>>(d_a, d_b, d_y, n);
}
