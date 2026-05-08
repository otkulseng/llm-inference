#pragma once

#include "device_buffer.h"
#include "prelude.h"
#include "tokenizer.h"
#include <cuda_bf16.h>

// Llama-3-8B inference engine. All persistent storage is __nv_bfloat16; FP32
// is used only in per-thread registers inside kernels for precision-critical
// reductions (matmul accumulators, softmax sums, rmsnorm sum-of-squares).
//
// All weights are loaded eagerly at construction (~16 GB in BF16 — fits in
// the 20 GB MIG slice). No lazy/optional/streaming abstraction; methods read
// the cached members directly.
class Model {
  public:
    Model();

    vector<int> tokenize(const string &input);
    string detokenize(const vector<int> &ids);

    // Embedding lookup via the on-GPU gather kernel. Returns FP32 to match
    // the test fixture format.
    vector<float> embed(const vector<int> &ids);

    vector<float> swiglu_ffn(const vector<float> &x_norm, int layer_idx, int s);
    vector<float> decoder_block(const vector<float> &x, int layer_idx, int s);

    int forward_one_step(const vector<int> &ids);
    vector<int> generate(const vector<int> &ids, int n_new);

  private:
    struct LayerWeights {
        DeviceBuffer<__nv_bfloat16> input_norm, q, k, v, o;
        DeviceBuffer<__nv_bfloat16> post_norm, gate, up, down;
    };

    LayerWeights load_layer(int layer_idx);
    DeviceBuffer<__nv_bfloat16>
    run_decoder_block(const DeviceBuffer<__nv_bfloat16> &d_x,
                      const LayerWeights &w,
                      const __nv_bfloat16 *d_cos, const __nv_bfloat16 *d_sin,
                      int s);
    // Core SwiGLU FFN: gate/up/silu_mul/down on an already-normalized input.
    // Shared by run_decoder_block (FFN sub-block) and swiglu_ffn (test API).
    DeviceBuffer<__nv_bfloat16>
    run_swiglu_core(const DeviceBuffer<__nv_bfloat16> &d_x_norm,
                    const LayerWeights &w, int s);

    BPETokenizer tokenizer_;
    // NOTE: part2.pdf §4 says lm_head and embed_tokens are tied in Llama 3;
    // that's true for Llama 1/2 but the actual Llama-3-8B-Instruct config has
    // tie_word_embeddings: false (verified). The two tensors differ, so we
    // hold both. ~2 GB total.
    DeviceBuffer<__nv_bfloat16> embed_;       // (V, d)
    vector<LayerWeights> layers_;             // 32
    DeviceBuffer<__nv_bfloat16> final_norm_;  // (d,)
    DeviceBuffer<__nv_bfloat16> lm_head_;     // (V, d)
};
