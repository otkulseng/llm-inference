#include "kernels.cuh"

#define TILE_SIZE 32
__global__ void matmul_tiled(const float *A, const float *B, float *C,
                             int M, int K, int N) {

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K) {
            sA[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        int b_row = t * TILE_SIZE + threadIdx.y;
        if (b_row < K && col < N) {
            sB[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void launch_matmul(const float *d_A, const float *d_B, float *d_C,
                   int M, int K, int N) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
}
