// You can set your configuration here... 
// Read the manual for our assumption for directory of model weights and tokenizer files. 
#pragma once

#include <string>

const std::string EMBEDDING_MATRIX_PATH = "assets/llama3/blobs/model.embed_tokens.weight";
const std::string TOKENIZER_PATH = "assets/llama3/token.model";

const int EMBEDDING_DIM = 4096;
const float RMS_NORM_EPSILON = 1e-5f;
