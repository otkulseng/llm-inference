// CUDA kernel declarations
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// All persistent storage lives as __nv_bfloat16. Per-thread FP32 accumulators
// are used inside kernels for precision (matmul reductions, softmax sums,
// rmsnorm sum-of-squares) but no FP32 buffer ever touches global or shared
// memory.

// Tiled matmul: C[M,N] = X[M,K] · W^T, where W is (N, K) row-major (the
// HuggingFace (out_dim, in_dim) checkpoint layout — see part2.pdf §4). Every
// matmul in the model is of this form (Q/K/V/O projections, gate/up/down,
// lm_head). FP32 accumulator in registers; BF16 store at end.
void launch_matmul(const __nv_bfloat16 *d_X, const __nv_bfloat16 *d_W,
                   __nv_bfloat16 *d_C, int M, int K, int N);

// Row-wise RMSNorm: y[r, i] = x[r, i] * gamma[i] / sqrt(mean(x[r, :]^2) + eps).
// x and y are flat (s, d) row-major; gamma is (d,). All BF16.
void launch_rmsnorm(const __nv_bfloat16 *d_x, const __nv_bfloat16 *d_gamma,
                    __nv_bfloat16 *d_y, int s, int d, float eps);

// Elementwise residual add: y[i] = a[i] + b[i].
void launch_residual_add(const __nv_bfloat16 *d_a, const __nv_bfloat16 *d_b,
                         __nv_bfloat16 *d_y, int n);

// Elementwise SiLU(gate) * up: y[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i].
void launch_silu_mul(const __nv_bfloat16 *d_gate, const __nv_bfloat16 *d_up,
                     __nv_bfloat16 *d_y, int n);

// Rotary Positional Embeddings on a flat (s, n_heads, h_d) row-major buffer.
// cos_table and sin_table are precomputed (s, h_d/2) row-major BF16. Pairs
// dim i with dim i + h_d/2 (rotate_half convention).
void launch_rope(const __nv_bfloat16 *d_qk, const __nv_bfloat16 *d_cos,
                 const __nv_bfloat16 *d_sin, __nv_bfloat16 *d_out,
                 int n_heads, int s, int h_d);

// Grouped Query Attention with causal mask and numerically stable softmax.
// Q is flat (s, N_HEADS, H_DIM); K, V are (s, N_KV_HEADS, H_DIM); output is
// (s, N_HEADS, H_DIM) = (s, N_HEADS * H_DIM). KV head index = i / (N_HEADS/N_KV_HEADS).
// Architecture constants from config.h (N_HEADS=32, N_KV_HEADS=8, H_DIM=128)
// are baked into the kernel; only the runtime-varying s is passed.
void launch_gqa_attention(const __nv_bfloat16 *d_Q, const __nv_bfloat16 *d_K,
                          const __nv_bfloat16 *d_V, __nv_bfloat16 *d_O,
                          int s);

// Embedding lookup: out[t, :] = table[token_ids[t], :]. table is (V, d) BF16,
// out is (n_tokens, d) BF16. token_ids is host-allocated int but copied to GPU.
void launch_embed_gather(const __nv_bfloat16 *d_table, const int *d_token_ids,
                         __nv_bfloat16 *d_out, int n_tokens, int d);
