#pragma once

#include "device_buffer.h"
#include "loader.h"
#include "prelude.h"

// Llama-3-8B inference engine. Owns the per-instance loader. Layer weights
// are loaded on demand inside the methods that need them — no eager preload.
//
// Layout:
//   - Tokenizer + embedding: thin wrappers around BPETokenizer / LlamaDumpLoader.
//   - swiglu_ffn / decoder_block: single-layer ops, used by tests 15 and 16.
//   - forward_one_step: full pipeline (32 layers + final norm + lm_head + argmax).
//   - generate: autoregressive loop over forward_one_step.
//
// All compute happens in FP32; weights are stored on disk as BF16 and converted
// to FP32 on the GPU via launch_bf16_to_fp32.
class Model {
  public:
    Model();

    // Tokenizer (constructs BPETokenizer lazily inside each call).
    vector<int> tokenize(const string &input);
    string detokenize(const vector<int> &ids);

    // Embedding lookup. Selective row access via the LlamaDumpLoader's CPU path.
    vector<float> embed(const vector<int> &ids);

    // Single-layer ops.
    vector<float> swiglu_ffn(const vector<float> &x_norm, int layer_idx, int s);
    vector<float> decoder_block(const vector<float> &x, int layer_idx, int s);

    // Full inference.
    int forward_one_step(const vector<int> &ids);
    vector<int> generate(const vector<int> &ids, int n_new);

  private:
    struct LayerWeights {
        DeviceBuffer<float> input_norm, q, k, v, o;
        DeviceBuffer<float> post_norm, gate, up, down;
    };

    LayerWeights load_layer(int layer_idx);
    DeviceBuffer<float> run_decoder_block(const DeviceBuffer<float> &d_x,
                                          const LayerWeights &w,
                                          const float *d_cos,
                                          const float *d_sin, int s);

    LlamaDumpLoader loader_;
};
