#include "kernels.cuh"

// Row-wise RMSNorm: y[r, i] = x[r, i] * gamma[i] / sqrt(mean(x[r, :]^2) + eps).
// Per part2.pdf §2.2 we follow the four-stage PMPP reduction pattern.
//
// All persistent storage is BF16. Per-thread FP32 accumulators are used for
// the sum-of-squares reduction; the shared-memory tree-reduction array is
// BF16 (per the no-FP32-buffer rule), so each step reads BF16, accumulates
// in float, writes BF16. The final rsqrt is computed once by thread 0.

constexpr int BLOCK_SIZE = 256;

__global__ void rmsnorm_kernel(const __nv_bfloat16 *x,
                               const __nv_bfloat16 *gamma,
                               __nv_bfloat16 *y, int d, float eps) {
    __shared__ __nv_bfloat16 sdata[BLOCK_SIZE];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const __nv_bfloat16 *x_row = x + row * d;
    __nv_bfloat16 *y_row = y + row * d;

    // Stage 1: each thread accumulates partial sum-of-squares over its
    // strided slice of the row in FP32 register, then writes one BF16 to smem.
    float partial = 0.0f;
    for (int i = tid; i < d; i += BLOCK_SIZE) {
        float v = __bfloat162float(x_row[i]);
        partial += v * v;
    }
    sdata[tid] = __float2bfloat16(partial);
    __syncthreads();

    // Stage 2: block-level tree reduction. Each step reads BF16, accumulates
    // in FP32 register, writes BF16 back.
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float a = __bfloat162float(sdata[tid]);
            float b = __bfloat162float(sdata[tid + stride]);
            sdata[tid] = __float2bfloat16(a + b);
        }
        __syncthreads();
    }

    // Stage 3: thread 0 computes the reciprocal RMS and broadcasts via shared
    // memory. Reciprocal lives in BF16 — fine because rsqrt of a sum-of-squares
    // is far from BF16's tiny denormal range.
    __shared__ __nv_bfloat16 rstd_bf16;
    if (tid == 0) {
        float sum = __bfloat162float(sdata[0]);
        float rstd = rsqrtf(sum / d + eps);
        rstd_bf16 = __float2bfloat16(rstd);
    }
    __syncthreads();

    // Stage 4: scaled, gamma-weighted output, same strided pattern.
    float rstd = __bfloat162float(rstd_bf16);
    for (int i = tid; i < d; i += BLOCK_SIZE) {
        float v = __bfloat162float(x_row[i]);
        float g = __bfloat162float(gamma[i]);
        y_row[i] = __float2bfloat16(v * rstd * g);
    }
}

void launch_rmsnorm(const __nv_bfloat16 *d_x, const __nv_bfloat16 *d_gamma,
                    __nv_bfloat16 *d_y, int s, int d, float eps) {
    dim3 grid(s);
    dim3 block(BLOCK_SIZE);
    rmsnorm_kernel<<<grid, block>>>(d_x, d_gamma, d_y, d, eps);
}
