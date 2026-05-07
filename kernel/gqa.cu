#include "kernels.cuh"

// Grouped Query Attention with causal mask and numerically stable softmax,
// fused into one kernel per (head, query) per part2.pdf §3.1-3.2.
//
// All persistent storage is BF16 (Q, K, V, O, and shared-memory scratch).
// FP32 register accumulators are used for the dot-product, the row-max, the
// row-sum-of-exps, and the per-d weighted sum — these are the precision-
// critical reductions.
//
// Layout (single, hardcoded):
//   Q : flat (s, n_heads,    h_d) row-major
//   K : flat (s, n_kv_heads, h_d) row-major
//   V : flat (s, n_kv_heads, h_d) row-major
//   O : flat (s, n_heads,    h_d) row-major  (= concat-of-heads per token)
//
// One block per (head_i, query position p). Inside a block the score row
// scores[0..s) lives in shared memory as BF16 and is reused as the
// unnormalized alpha row — it never touches HBM.

constexpr int BLOCK_SIZE = 256;

__global__ void gqa_attention_kernel(const __nv_bfloat16 *Q,
                                     const __nv_bfloat16 *K,
                                     const __nv_bfloat16 *V,
                                     __nv_bfloat16 *O,
                                     int n_heads, int n_kv_heads,
                                     int s, int h_d) {
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16 *q_shared       = smem;                  // [h_d]
    __nv_bfloat16 *scores         = q_shared + h_d;        // [s]
    __nv_bfloat16 *reduce_scratch = scores + s;            // [BLOCK_SIZE]

    const int head_i = blockIdx.x;
    const int p      = blockIdx.y;
    const int tid    = threadIdx.x;
    const int g      = head_i / (n_heads / n_kv_heads);
    const float scale = rsqrtf((float)h_d);

    const __nv_bfloat16 *q_row = Q + p * (n_heads * h_d) + head_i * h_d;

    // Stage 0: stage Q[p, head_i, :] into shared memory.
    for (int d = tid; d < h_d; d += BLOCK_SIZE) {
        q_shared[d] = q_row[d];
    }
    __syncthreads();

    // Stage 1: scaled dot-product Q_i[p] · K_g[q]^T and causal mask.
    // q > p positions get -1e6f (≈ -∞ for softmax) per part2.pdf §3.2.
    for (int q = tid; q < s; q += BLOCK_SIZE) {
        if (q > p) {
            scores[q] = __float2bfloat16(-1.0e6f);
        } else {
            const __nv_bfloat16 *k_row = K + q * (n_kv_heads * h_d) + g * h_d;
            float dot = 0.0f;
            for (int d = 0; d < h_d; ++d) {
                dot += __bfloat162float(q_shared[d]) * __bfloat162float(k_row[d]);
            }
            scores[q] = __float2bfloat16(dot * scale);
        }
    }
    __syncthreads();

    // Stage 2: row max via shared-memory tree reduction (PMPP pattern). Per
    // thread we accumulate the local max in an FP32 register, then reduce
    // through BF16 shared memory — each smem read converts to FP32, max in
    // register, write BF16 back.
    float local_max = -INFINITY;
    for (int q = tid; q < s; q += BLOCK_SIZE) {
        float v = __bfloat162float(scores[q]);
        if (v > local_max) local_max = v;
    }
    reduce_scratch[tid] = __float2bfloat16(local_max);
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float a = __bfloat162float(reduce_scratch[tid]);
            float b = __bfloat162float(reduce_scratch[tid + stride]);
            reduce_scratch[tid] = __float2bfloat16(a > b ? a : b);
        }
        __syncthreads();
    }
    __shared__ __nv_bfloat16 row_max_bf16;
    if (tid == 0) row_max_bf16 = reduce_scratch[0];
    __syncthreads();
    float row_max = __bfloat162float(row_max_bf16);

    // Stage 3: exponentiate in place (subtract row max first — required for
    // numerical stability per part2.pdf §4) and reduce to row sum. FP32 exp
    // and FP32 register accumulator; BF16 store back to scores.
    float local_sum = 0.0f;
    for (int q = tid; q < s; q += BLOCK_SIZE) {
        float e = expf(__bfloat162float(scores[q]) - row_max);
        scores[q] = __float2bfloat16(e);
        local_sum += e;
    }
    reduce_scratch[tid] = __float2bfloat16(local_sum);
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float a = __bfloat162float(reduce_scratch[tid]);
            float b = __bfloat162float(reduce_scratch[tid + stride]);
            reduce_scratch[tid] = __float2bfloat16(a + b);
        }
        __syncthreads();
    }
    __shared__ __nv_bfloat16 row_sum_bf16;
    if (tid == 0) row_sum_bf16 = reduce_scratch[0];
    __syncthreads();
    float row_sum = __bfloat162float(row_sum_bf16);

    // Stage 4: weighted sum O[p, head_i, d] = Σ_q (scores[q] / row_sum) · V[q, g, d].
    const float inv_sum = 1.0f / row_sum;
    __nv_bfloat16 *o_row = O + p * (n_heads * h_d) + head_i * h_d;
    for (int d = tid; d < h_d; d += BLOCK_SIZE) {
        float acc = 0.0f;
        for (int q = 0; q <= p; ++q) {
            const __nv_bfloat16 *v_row = V + q * (n_kv_heads * h_d) + g * h_d;
            acc += __bfloat162float(scores[q]) * __bfloat162float(v_row[d]);
        }
        o_row[d] = __float2bfloat16(acc * inv_sum);
    }
}

void launch_gqa_attention(const __nv_bfloat16 *d_Q, const __nv_bfloat16 *d_K,
                          const __nv_bfloat16 *d_V, __nv_bfloat16 *d_O,
                          int n_heads, int n_kv_heads, int s, int h_d) {
    size_t smem_bytes = sizeof(__nv_bfloat16) * (h_d + s + BLOCK_SIZE);
    dim3 grid(n_heads, s);
    dim3 block(BLOCK_SIZE);
    gqa_attention_kernel<<<grid, block, smem_bytes>>>(
        d_Q, d_K, d_V, d_O, n_heads, n_kv_heads, s, h_d);
}
