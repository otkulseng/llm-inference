#include "kernels.cuh"

// Rotary Positional Embeddings.
//
// Buffer layout: flat (s, n_heads, h_d) row-major — the natural post-matmul
// layout produced by the Q/K projections, so decoder_block consumes it with
// zero data movement (per part2.pdf §3.1's "logical reshape ... requires no
// data movement"). Test 11's fixture is in (n_heads, s, h_d); the test
// wrapper transposes once on the host.
//
// For each head, each token p, and each pair index i in [0, h_d/2):
//     q'[i]         =  cos(p*theta_i) * q[i]         - sin(p*theta_i) * q[i + h_d/2]
//     q'[i + h_d/2] =  sin(p*theta_i) * q[i]         + cos(p*theta_i) * q[i + h_d/2]
//
// This is the "rotate_half" convention from the HuggingFace Llama
// implementation: pair dim i with dim i+h_d/2, NOT i with i+1.
//
// cos_table and sin_table are pre-computed (s, h_d/2) row-major BF16 (computed
// on host as fp32, downcast at upload). All compute is in FP32 registers; the
// stored q outputs are BF16.

constexpr int BLOCK_SIZE = 256;

__global__ void rope_kernel(const __nv_bfloat16 *qk,
                            const __nv_bfloat16 *cos_table,
                            const __nv_bfloat16 *sin_table,
                            __nv_bfloat16 *out,
                            int n_heads, int s, int h_d) {
    int half = h_d / 2;
    int total = n_heads * s * half;
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= total) return;

    int pair = idx % half;
    int t    = (idx / half) % s;
    int head = idx / (half * s);

    int row_base = t * (n_heads * h_d) + head * h_d;
    int lo = row_base + pair;
    int hi = row_base + pair + half;

    float c  = __bfloat162float(cos_table[t * half + pair]);
    float si = __bfloat162float(sin_table[t * half + pair]);
    float q_lo = __bfloat162float(qk[lo]);
    float q_hi = __bfloat162float(qk[hi]);

    out[lo] = __float2bfloat16(c  * q_lo - si * q_hi);
    out[hi] = __float2bfloat16(si * q_lo + c  * q_hi);
}

void launch_rope(const __nv_bfloat16 *d_qk, const __nv_bfloat16 *d_cos,
                 const __nv_bfloat16 *d_sin, __nv_bfloat16 *d_out,
                 int n_heads, int s, int h_d) {
    int total = n_heads * s * (h_d / 2);
    dim3 grid((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    rope_kernel<<<grid, block>>>(d_qk, d_cos, d_sin, d_out, n_heads, s, h_d);
}
