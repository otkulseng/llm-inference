#include "kernels.cuh"

// Elementwise SiLU(gate) * up:  y[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i].
// Per part2.pdf §3.2: "Implement the SiLU(gate) ⊙ up step as a simple CUDA
// kernel with one thread per element." BF16 in/out, FP32 in registers (exp
// needs FP32 for stability).

constexpr int BLOCK_SIZE = 256;

__global__ void silu_mul_kernel(const __nv_bfloat16 *gate,
                                const __nv_bfloat16 *up,
                                __nv_bfloat16 *y, int n) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) {
        float g = __bfloat162float(gate[i]);
        float u = __bfloat162float(up[i]);
        float silu = g / (1.0f + expf(-g));
        y[i] = __float2bfloat16(silu * u);
    }
}

void launch_silu_mul(const __nv_bfloat16 *d_gate, const __nv_bfloat16 *d_up,
                     __nv_bfloat16 *d_y, int n) {
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    silu_mul_kernel<<<grid, block>>>(d_gate, d_up, d_y, n);
}
