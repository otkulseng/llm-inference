#include "kernels.cuh"

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
