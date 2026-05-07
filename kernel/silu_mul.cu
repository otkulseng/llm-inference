#include "kernels.cuh"

// Elementwise SiLU(gate) * up:  y[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i].
// Per part2.pdf §3.2: "Implement the SiLU(gate) ⊙ up step as a simple CUDA
// kernel with one thread per element."

constexpr int BLOCK_SIZE = 256;

__global__ void silu_mul_kernel(const float *gate, const float *up, float *y,
                                int n) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        y[i] = silu * up[i];
    }
}

void launch_silu_mul(const float *d_gate, const float *d_up, float *d_y, int n) {
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    silu_mul_kernel<<<grid, block>>>(d_gate, d_up, d_y, n);
}
