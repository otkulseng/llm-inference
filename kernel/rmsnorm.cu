#include "kernels.cuh"
constexpr int BLOCK_SIZE = 256;

__global__ void rmsnorm_kernel(const float *x, const float *gamma, float *y,
                               int d, float eps) {
    __shared__ float sdata[BLOCK_SIZE];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float *x_row = x + row * d;
    float *y_row = y + row * d;

    // Stage 1: each thread accumulates partial sum-of-squares over its
    // strided slice of the row.
    float partial = 0.0f;
    for (int i = tid; i < d; i += BLOCK_SIZE) {
        float v = x_row[i];
        partial += v * v;
    }
    sdata[tid] = partial;
    __syncthreads();

    // Stage 2: block-level reduction in shared memory. Tree halves the
    // active-thread range each iteration; result lands in sdata[0].
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Stage 3: thread 0 computes the reciprocal RMS and broadcasts it.
    // epsilon is inside the sqrt to keep RMS finite when the row is ~0.
    __shared__ float rstd;
    if (tid == 0) {
        rstd = rsqrtf(sdata[0] / d + eps);
    }
    __syncthreads();

    // Stage 4: scaled, gamma-weighted output, same strided pattern.
    for (int i = tid; i < d; i += BLOCK_SIZE) {
        y_row[i] = x_row[i] * rstd * gamma[i];
    }
}

void launch_rmsnorm(const float *d_x, const float *d_gamma, float *d_y,
                    int s, int d, float eps) {
    dim3 grid(s);
    dim3 block(BLOCK_SIZE);
    rmsnorm_kernel<<<grid, block>>>(d_x, d_gamma, d_y, d, eps);
}
