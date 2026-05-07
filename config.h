// You can set your configuration here... 
// Read the manual for our assumption for directory of model weights and tokenizer files. 
#pragma once

#include <string>

const std::string EMBEDDING_MATRIX_PATH = "assets/llama3/blobs/model.embed_tokens.weight";
const std::string TOKENIZER_PATH = "assets/llama3/token.model";

const int EMBEDDING_DIM = 4096;
const int N_HEADS = 32;
const int N_KV_HEADS = 8;
const int H_DIM = 128;
const int D_FF = 14336;
const int VOCAB_SIZE = 128256;
const int N_LAYERS = 32;
const float RMS_NORM_EPSILON = 1e-5f;
const float ROPE_BASE = 500000.0f;  // Llama 3 uses 500000, NOT 10000 like the original RoPE paper.
