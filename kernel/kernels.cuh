// CUDA kernel declarations
#pragma once

#include <cuda_runtime.h>

// Tiled matrix multiplication: C[M,N] = A[M,K] * B[K,N]
// All matrices are row-major, flat arrays.
void launch_matmul(const float *d_A, const float *d_B, float *d_C,
                   int M, int K, int N);
