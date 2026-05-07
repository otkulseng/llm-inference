#include "kernels.cuh"

// Tiled matmul where B is stored transposed: C[M, N] = A[M, K] · B^T,
// with B given as (N, K) row-major. Equivalent to PyTorch `A @ B.T` for B of
// shape (N, K). All BF16 in/out, FP32 register accumulator. Used pervasively
// in this project because HuggingFace stores projection weights as
// (out_dim, in_dim) — so X @ W.T in math terms is the natural call site
// (see part2.pdf §4).
//
// Same tile/blocking as matmul.cu; only the B load and shared-memory layout
// change. We store B transposed in shared memory (sB[k][n] = B[n, k]) so that
// the inner dot-product loop is identical to the regular matmul. The load
// itself stays coalesced because consecutive threads in a warp (varying
// threadIdx.x) read consecutive K elements within a fixed N row of B.

#define TILE_SIZE 32

__global__ void matmul_xwt_tiled(const __nv_bfloat16 *A, const __nv_bfloat16 *B,
                                 __nv_bfloat16 *C, int M, int K, int N) {
    __shared__ __nv_bfloat16 sA[TILE_SIZE][TILE_SIZE];
    __shared__ __nv_bfloat16 sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // m
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // n

    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        // sA[ty][tx] = A[row, t*TS + tx]
        int a_k = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_k < K) {
            sA[threadIdx.y][threadIdx.x] = A[row * K + a_k];
        } else {
            sA[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }
        // sB[tx][ty] = B[blockIdx.x*TS + ty, t*TS + tx]   (coalesced over tx → k)
        int b_n = blockIdx.x * TILE_SIZE + threadIdx.y;
        int b_k = t * TILE_SIZE + threadIdx.x;
        if (b_n < N && b_k < K) {
            sB[threadIdx.x][threadIdx.y] = B[b_n * K + b_k];
        } else {
            sB[threadIdx.x][threadIdx.y] = __float2bfloat16(0.0f);
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

void launch_matmul_xwt(const __nv_bfloat16 *d_A, const __nv_bfloat16 *d_B,
                       __nv_bfloat16 *d_C, int M, int K, int N) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_xwt_tiled<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
}
