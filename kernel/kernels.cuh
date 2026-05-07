// CUDA kernel declarations
#pragma once

#include <cuda_runtime.h>

// Tiled matrix multiplication: C[M,N] = A[M,K] * B[K,N]
// All matrices are row-major, flat arrays.
void launch_matmul(const float *d_A, const float *d_B, float *d_C,
                   int M, int K, int N);

// Matmul with B stored transposed: C[M,N] = A[M,K] * B^T, where B is given
// as (N, K) row-major. Equivalent to PyTorch `A @ B.T` for B of shape (N, K).
// Used for X @ W^T where W is stored as (out, in) per the HuggingFace
// checkpoint convention (part2.pdf §4).
void launch_matmul_xwt(const float *d_A, const float *d_B, float *d_C,
                       int M, int K, int N);

// Row-wise RMSNorm: y[r, i] = x[r, i] * gamma[i] / sqrt(mean(x[r, :]^2) + eps).
// x and y are flat (s, d) row-major; gamma is (d,).
void launch_rmsnorm(const float *d_x, const float *d_gamma, float *d_y,
                    int s, int d, float eps);

// Elementwise residual add: y[i] = a[i] + b[i]. All flat, length n.
void launch_residual_add(const float *d_a, const float *d_b, float *d_y, int n);

// Elementwise SiLU(gate) * up: y[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i].
void launch_silu_mul(const float *d_gate, const float *d_up, float *d_y, int n);

// Rotary Positional Embeddings on a flat (n_heads, s, h_d) row-major buffer.
// cos_table and sin_table are precomputed (s, h_d/2) row-major. Pairs dim i
// with dim i + h_d/2 (rotate_half convention).
void launch_rope(const float *d_qk, const float *d_cos, const float *d_sin,
                 float *d_out, int n_heads, int s, int h_d);

// Grouped Query Attention with causal mask and numerically stable softmax.
// Q is (n_heads, s, h_d) row-major; K, V are (n_kv_heads, s, h_d) row-major.
// Output is flat (s, n_heads * h_d) — concatenated head outputs per token.
// Group size = n_heads / n_kv_heads (integer); KV head index for query head i
// is g = i / group_size, per part2.pdf §3.1.
void launch_gqa_attention(const float *d_Q, const float *d_K, const float *d_V,
                          float *d_O, int n_heads, int n_kv_heads, int s, int h_d);
