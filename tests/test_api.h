// DO NOT CHANGE THIS FILE, Implement the functions in test_api.cpp
// If you have any signature issues, let us know.

#include "prelude.h"


// Cast it freely :), suboptimal performance is okay for testing. 
// Don't change the API signatures, though

class TestAPI {
  public:
    // ---- Milestone 1 ----

    // dont forget padding with bos token
    vector<int> tokenize(string input);

    // Decode a list of token ids back to text. Leading BOS may be included
    // or stripped; the test compares after BOS stripping.
    string detokenize(vector<int> token_ids);

    // If you do bf16 or fp16 dumping, you will need to convert the embeddings
    // to fp32 before returning them.
    vector<float> get_embeddings(vector<int> token_ids);

    // Matrix multiply (row-major): A[M,K] * B[K,N] -> C[M,N].
    // Inputs are flattened row-major buffers of sizes M*K and K*N.
    // Returns a flattened row-major vector of size M*N.
    // Everthing flat!
    vector<float> matmul(const vector<float> &A, const vector<float> &B, int M,
                         int K, int N);

    // RMSNorm applied row-wise to x of shape (s, d), with learned scale gamma
    // of shape (d,). Returns flat row-major (s, d).
    vector<float> rmsnorm(const vector<float> &x, const vector<float> &gamma,
                          int s, int d);

    // Apply RoPE to a flat (n_heads, s, h_d) row-major buffer in place style
    // (returns a new vector). Use base 500000 and the first-half / second-half
    // pairing convention. Used for both Q (n_heads = H) and K (n_heads = H_K).
    vector<float> rope(const vector<float> &qk, int n_heads, int s, int h_d);

    // Grouped Query Attention with causal mask and stable softmax.
    // Q is flat (H,   s, h_d), K and V are flat (H_K, s, h_d).
    // Returns flat (s, H * h_d) of concatenated head outputs.
    // H, H_K, h_d are fixed by config.h.
    vector<float> gqa_attention(const vector<float> &Q, const vector<float> &K,
                                const vector<float> &V, int s);

    // Elementwise a + b. Both flat, same length.
    vector<float> residual_add(const vector<float> &a, const vector<float> &b);

    // Elementwise SiLU(gate) * up. Both flat, same length.
    vector<float> silu_mul(const vector<float> &gate, const vector<float> &up);

    // Full SwiGLU FFN sub-block on x_norm of shape (s, d), using the FFN
    // weights of the given layer. Returns flat (s, d).
    vector<float> swiglu_ffn(const vector<float> &x_norm, int layer_idx, int s);

    // Run one full decoder block at the given layer index on input x of
    // shape (s, d). Reads weights for that layer from the loaded checkpoint.
    // Returns flat (s, d).
    vector<float> decoder_block(const vector<float> &x, int layer_idx, int s);

    // Full forward pass through all 32 decoder blocks plus the output layer.
    // Returns the greedy next-token id.
    int forward_one_step(const vector<int> &token_ids);

    // Generate n_new tokens autoregressively starting from the given prompt.
    // Returns a vector of length n_new with the newly generated token ids
    // (the prompt is not included). KV-cache implementations are welcome.
    vector<int> generate(const vector<int> &token_ids, int n_new);
};
