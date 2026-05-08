#include "config.h"
#include "kernels.cuh"
#include <cassert>


//
// Layout (single, hardcoded):
//   Q : flat (s, N_HEADS,    H_DIM) row-major
//   K : flat (s, N_KV_HEADS, H_DIM) row-major
//   V : flat (s, N_KV_HEADS, H_DIM) row-major
//   O : flat (s, N_HEADS,    H_DIM) row-major  (= concat-of-heads per token)
//


constexpr int BLOCK_SIZE = 256;

__global__ void gqa_attention_kernel(const __nv_bfloat16 *Q,
                                     const __nv_bfloat16 *K,
                                     const __nv_bfloat16 *V,
                                     __nv_bfloat16 *O, int s) {
    
    __shared__ float scores[MAX_SEQ_LEN];
    __shared__ float reduce_scratch[BLOCK_SIZE];
    const int tid    = threadIdx.x;
    const float scale = rsqrtf((float)H_DIM);


    // This kernel reads Q[p, head_i] and populates O[p, head_i].
    const int head_i = blockIdx.x;
    const int p      = blockIdx.y;
    const __nv_bfloat16 *q_row = Q + p * (N_HEADS * H_DIM) + head_i * H_DIM;
    __nv_bfloat16 *o_row = O + p * (N_HEADS * H_DIM) + head_i * H_DIM;

    // Group of K/V heads associated to this Q head.
    const int g      = head_i / (N_HEADS / N_KV_HEADS);

    // Stage 0: Read Q[p, head_i] into shared memory
    __shared__ float q_shared[H_DIM];
    for (int d = tid; d < H_DIM; d += BLOCK_SIZE) {
        q_shared[d] = __bfloat162float(q_row[d]);
    }
    __syncthreads();

    // Stage 1: scaled dot-product Q_i[p] · K_g[q]^T and causal mask.
    for (int q = tid; q < s; q += BLOCK_SIZE) {
        const __nv_bfloat16 *k_row = K + q * (N_KV_HEADS * H_DIM) + g * H_DIM;
        float dot = 0.0f;
        for (int d = 0; d < H_DIM; ++d) {
            dot += q_shared[d] * __bfloat162float(k_row[d]);
        }
        float mask = (q > p) ? -1.0e6f : 0.0f; // NB WANT TO ASK ABT THIS ONE!
        scores[q] = dot * scale + mask;
    }
    __syncthreads();

    // Stage 2: find row max using the rmsnorm reduction pattern from earlier
    float local_max = -INFINITY;
    for (int q = tid; q < s; q += BLOCK_SIZE) {
        float v = scores[q];
        if (v > local_max) local_max = v;
    }
    reduce_scratch[tid] = local_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float a = reduce_scratch[tid];
            float b = reduce_scratch[tid + stride];
            reduce_scratch[tid] = a > b ? a : b;
        }
        __syncthreads();
    }


    __syncthreads();
    float row_max = reduce_scratch[0];

    // Stage 3 softmax
    float local_sum = 0.0f;
    for (int q = tid; q < s; q += BLOCK_SIZE) {
        float e = expf(scores[q] - row_max);
        scores[q] = e;
        local_sum += e;
    }
    reduce_scratch[tid] = local_sum;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_scratch[tid] += reduce_scratch[tid + stride];
        }
        __syncthreads();
    }
    __syncthreads();
    float row_sum = reduce_scratch[0];

    // Stage 4: weighted sum O[p, head_i, d] = Σ_q (scores[q] / row_sum) · V[q, g, d].
    // Skip q > p — those scores are effectively zero after the mask + exp.
    const float inv_sum = 1.0f / row_sum;

    for (int d = tid; d < H_DIM; d += BLOCK_SIZE) {
        float acc = 0.0f;
        for (int q = 0; q <= p; ++q) {
            const __nv_bfloat16 *v_row = V + q * (N_KV_HEADS * H_DIM) + g * H_DIM;
            acc += scores[q] * __bfloat162float(v_row[d]);
        }
        o_row[d] = __float2bfloat16(acc * inv_sum);
    }
}

void launch_gqa_attention(const __nv_bfloat16 *d_Q, const __nv_bfloat16 *d_K,
                          const __nv_bfloat16 *d_V, __nv_bfloat16 *d_O,
                          int s) {
    assert(s <= MAX_SEQ_LEN);
    dim3 grid(N_HEADS, s);
    dim3 block(BLOCK_SIZE);
    gqa_attention_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, s);
}
