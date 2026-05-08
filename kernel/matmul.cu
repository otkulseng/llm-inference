#include "kernels.cuh"

#define TILE_SIZE 32

// Tiled coalesced matmul for X · W^T.
__global__ void matmul_tiled(const __nv_bfloat16 *X, const __nv_bfloat16 *W,
                             __nv_bfloat16 *C, int M, int K, int N) {
    __shared__ float sX[TILE_SIZE][TILE_SIZE];
    __shared__ float sW[TILE_SIZE][TILE_SIZE];  // stored as sW[k_local][n_local]

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // m
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // n

    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        // X tile: standard row-major load. sX[m_local][k_local].
        int x_k = t * TILE_SIZE + threadIdx.x;
        sX[threadIdx.y][threadIdx.x] = (row < M && x_k < K)
            ? __bfloat162float(X[row * K + x_k])
            : 0.0f;

        // W tile: W is stored (N, K). To stay coalesced we let lanes (varying
        // threadIdx.x) walk along K — the contiguous stride. We then store
        // transposed into smem so the inner loop reads sW[k_local][n_local]
        // with the same access pattern as a standard matmul.
        int w_n = blockIdx.x * TILE_SIZE + threadIdx.y;
        int w_k = t * TILE_SIZE + threadIdx.x;
        sW[threadIdx.x][threadIdx.y] = (w_n < N && w_k < K)
            ? __bfloat162float(W[w_n * K + w_k])
            : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += sX[threadIdx.y][i] * sW[i][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = __float2bfloat16(sum);
    }
}

void launch_matmul(const __nv_bfloat16 *d_X, const __nv_bfloat16 *d_W,
                   __nv_bfloat16 *d_C, int M, int K, int N) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_tiled<<<grid, block>>>(d_X, d_W, d_C, M, K, N);
}
