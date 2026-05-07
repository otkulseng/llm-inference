// CUDA kernel declarations
#pragma once

#include <cuda_runtime.h>

// Tiled matrix multiplication: C[M,N] = A[M,K] * B[K,N]
// All matrices are row-major, flat arrays.
void launch_matmul(const float *d_A, const float *d_B, float *d_C,
                   int M, int K, int N);

// Row-wise RMSNorm: y[r, i] = x[r, i] * gamma[i] / sqrt(mean(x[r, :]^2) + eps).
// x and y are flat (s, d) row-major; gamma is (d,).
void launch_rmsnorm(const float *d_x, const float *d_gamma, float *d_y,
                    int s, int d, float eps);
