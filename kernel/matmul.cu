#include "kernels.cuh"

// Tiled matrix multiplication: C[M,N] = A[M,K] * B[K,N], all row-major BF16.
// Per-thread FP32 accumulator over BF16 inputs. Shared-memory tiles stay BF16
// (no FP32 buffers of any kind); the inner loop reads BF16 from smem and
// converts to float in registers per multiply.

#define TILE_SIZE 32

__global__ void matmul_tiled(const __nv_bfloat16 *A, const __nv_bfloat16 *B,
                             __nv_bfloat16 *C, int M, int K, int N) {

    __shared__ __nv_bfloat16 sA[TILE_SIZE][TILE_SIZE];
    __shared__ __nv_bfloat16 sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K) {
            sA[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            sA[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }
        int b_row = t * TILE_SIZE + threadIdx.y;
        if (b_row < K && col < N) {
            sB[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            float a = __bfloat162float(sA[threadIdx.y][i]);
            float b = __bfloat162float(sB[i][threadIdx.x]);
            sum += a * b;
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = __float2bfloat16(sum);
    }
}

void launch_matmul(const __nv_bfloat16 *d_A, const __nv_bfloat16 *d_B,
                   __nv_bfloat16 *d_C, int M, int K, int N) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
}
