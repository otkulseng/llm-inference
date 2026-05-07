#include "kernels.cuh"

// Embedding lookup. Replaces the host-side LlamaDumpLoader::get_embeddings
// path now that the embedding table lives on the GPU as BF16. One block per
// token, blockDim.x threads cover the d-dimension in stride.

constexpr int BLOCK_SIZE = 256;

__global__ void embed_gather_kernel(const __nv_bfloat16 *table,
                                    const int *token_ids, __nv_bfloat16 *out,
                                    int n_tokens, int d) {
    int t = blockIdx.x;
    if (t >= n_tokens) return;
    int row = token_ids[t];
    const __nv_bfloat16 *src = table + (size_t)row * d;
    __nv_bfloat16 *dst = out + (size_t)t * d;
    for (int i = threadIdx.x; i < d; i += BLOCK_SIZE) {
        dst[i] = src[i];
    }
}

void launch_embed_gather(const __nv_bfloat16 *d_table, const int *d_token_ids,
                         __nv_bfloat16 *d_out, int n_tokens, int d) {
    dim3 grid(n_tokens);
    dim3 block(BLOCK_SIZE);
    embed_gather_kernel<<<grid, block>>>(d_table, d_token_ids, d_out,
                                          n_tokens, d);
}
