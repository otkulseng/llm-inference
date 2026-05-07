// Llama-3-8B inference demo.
//
// Usage:
//   ./bin/llm                          # default prompt and 10 new tokens
//   ./bin/llm "Your prompt here"       # custom prompt, 10 new tokens
//   ./bin/llm "Your prompt here" 25    # custom prompt, 25 new tokens
//
// Tokens are streamed to stdout as they're generated. Each new token currently
// takes ~4 seconds (32-layer disk reload + GPU compute per step).

#include "model.h"
#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
    std::string prompt = (argc > 1) ? argv[1] : "Once upon a time";
    int n_new = (argc > 2) ? std::atoi(argv[2]) : 10;

    Model model;

    std::vector<int> ids = model.tokenize(prompt);

    std::cout << prompt << std::flush;
    for (int i = 0; i < n_new; ++i) {
        int next = model.forward_one_step(ids);
        ids.push_back(next);
        std::cout << model.detokenize({next}) << std::flush;
    }
    std::cout << std::endl;
    return 0;
}
