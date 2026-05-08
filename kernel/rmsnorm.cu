#include "kernels.cuh"
constexpr int BLOCK_SIZE = 256;


__global__ void rmsnorm_kernel(const __nv_bfloat16 *x,
                               const __nv_bfloat16 *gamma,
                               __nv_bfloat16 *y, int d, float eps) {
    // Step 1: Use shared memory to divide the computation into BLOCK_SIZE
    __shared__ float sdata[BLOCK_SIZE];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const __nv_bfloat16 *x_row = x + row * d;
    __nv_bfloat16 *y_row = y + row * d;

    float partial = 0.0f;
    for (int i = tid; i < d; i += BLOCK_SIZE) {
        float v = __bfloat162float(x_row[i]);
        partial += v * v;
    }
    sdata[tid] = partial;
    __syncthreads();

    // Step 2: tree-reduce. 
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = sdata[tid] + sdata[tid + stride];
        }
        __syncthreads();
    }

    // Stage 3: thread 0 computes the reciprocal RMS and broadcasts via shared
    // memory.
    __shared__ float rstd;
    if (tid == 0) {
        rstd = rsqrtf(sdata[0] / d + eps);
    }
    __syncthreads();

    // Stage 4: scaled, gamma-weighted output, same strided pattern.
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
