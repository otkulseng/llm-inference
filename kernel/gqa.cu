#include "kernels.cuh"

// Grouped Query Attention with causal mask and numerically stable softmax,
// fused into one kernel per (head, query) per part2.pdf §3.1-3.2.
//
// Buffer layout (single, hardcoded):
//   Q : flat (s, n_heads,    h_d) row-major
//   K : flat (s, n_kv_heads, h_d) row-major
//   V : flat (s, n_kv_heads, h_d) row-major
//   O : flat (s, n_heads,    h_d) row-major  (= concat-of-heads per token)
//
// This is the natural post-matmul layout — decoder_block consumes it directly
// with no data movement (part2.pdf §3.1: "this is a logical reshape ... and
// requires no data movement"). Test 12's fixture is in (n, s, h_d); the test
// wrapper host-transposes the inputs once.
//
// KV head index for query head i: g = i / (n_heads / n_kv_heads). Each group
// of (n_heads / n_kv_heads) query heads shares one K head and one V head.
//
// One block per (head_i, query position p). Inside a block the score row
// scores[0..s) lives in shared memory and is reused as the unnormalized
// alpha row — it never touches HBM. Reduction stages mirror the PMPP-style
// tree reduction in rmsnorm.cu.

constexpr int BLOCK_SIZE = 256;

__global__ void gqa_attention_kernel(const float *Q, const float *K,
                                     const float *V, float *O,
                                     int n_heads, int n_kv_heads,
                                     int s, int h_d) {
    extern __shared__ float smem[];
    float *q_shared       = smem;                  // [h_d]
    float *scores         = q_shared + h_d;        // [s]
    float *reduce_scratch = scores + s;            // [BLOCK_SIZE]

    const int head_i = blockIdx.x;
    const int p      = blockIdx.y;
    const int tid    = threadIdx.x;
    const int g      = head_i / (n_heads / n_kv_heads);
    const float scale = rsqrtf((float)h_d);

    const float *q_row = Q + p * (n_heads * h_d) + head_i * h_d;

    // Stage 0: stage Q[p, head_i, :] into shared memory.
    for (int d = tid; d < h_d; d += BLOCK_SIZE) {
        q_shared[d] = q_row[d];
    }
    __syncthreads();

    // Stage 1: scaled dot-product Q_i[p] · K_g[q]^T and causal mask.
    // q > p positions get -1e6f (≈ -∞ for softmax) per part2.pdf §3.2.
    for (int q = tid; q < s; q += BLOCK_SIZE) {
        if (q > p) {
            scores[q] = -1.0e6f;
        } else {
            const float *k_row = K + q * (n_kv_heads * h_d) + g * h_d;
            float dot = 0.0f;
            for (int d = 0; d < h_d; ++d) {
                dot += q_shared[d] * k_row[d];
            }
            scores[q] = dot * scale;
        }
    }
    __syncthreads();

    // Stage 2: row max via shared-memory tree reduction (PMPP pattern).
    float local_max = -INFINITY;
    for (int q = tid; q < s; q += BLOCK_SIZE) {
        float v = scores[q];
        if (v > local_max) local_max = v;
    }
    reduce_scratch[tid] = local_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other = reduce_scratch[tid + stride];
            if (other > reduce_scratch[tid]) reduce_scratch[tid] = other;
        }
        __syncthreads();
    }
    __shared__ float row_max;
    if (tid == 0) row_max = reduce_scratch[0];
    __syncthreads();

    // Stage 3: exponentiate in place (subtract row max first — required for
    // numerical stability per part2.pdf §4) and reduce to row sum.
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
    __shared__ float row_sum;
    if (tid == 0) row_sum = reduce_scratch[0];
    __syncthreads();

    // Stage 4: weighted sum O[p, head_i, d] = Σ_q (scores[q] / row_sum) · V[q, g, d].
    const float inv_sum = 1.0f / row_sum;
    float *o_row = O + p * (n_heads * h_d) + head_i * h_d;
    for (int d = tid; d < h_d; d += BLOCK_SIZE) {
        float acc = 0.0f;
        for (int q = 0; q <= p; ++q) {
            const float *v_row = V + q * (n_kv_heads * h_d) + g * h_d;
            acc += scores[q] * v_row[d];
        }
        o_row[d] = acc * inv_sum;
    }
}

void launch_gqa_attention(const float *d_Q, const float *d_K, const float *d_V,
                          float *d_O, int n_heads, int n_kv_heads,
                          int s, int h_d) {
    size_t smem_bytes = sizeof(float) * (h_d + s + BLOCK_SIZE);
    dim3 grid(n_heads, s);
    dim3 block(BLOCK_SIZE);
    gqa_attention_kernel<<<grid, block, smem_bytes>>>(
        d_Q, d_K, d_V, d_O, n_heads, n_kv_heads, s, h_d);
}
