// Llama-3-8B inference demo.
//
// Usage:
//   ./bin/llm                          # default prompt and 10 new tokens
//   ./bin/llm "Your prompt here"       # custom prompt, 10 new tokens
//   ./bin/llm "Your prompt here" 25    # custom prompt, 25 new tokens
//
// Tokens are streamed to stdout as they're generated. Diagnostic timings are
// written to stderr so they don't mix with the prompt/token output.

#include "model.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

using clock_type = std::chrono::steady_clock;

static double elapsed_s(clock_type::time_point start) {
    auto now = clock_type::now();
    return std::chrono::duration<double>(now - start).count();
}

int main(int argc, char **argv) {
    std::string prompt = (argc > 1) ? argv[1] : "Once upon a time";
    int n_new = (argc > 2) ? std::atoi(argv[2]) : 10;

    auto t_total = clock_type::now();

    std::cerr << "[loading weights]..." << std::flush;
    auto t_load = clock_type::now();
    Model model;
    std::cerr << " done in " << elapsed_s(t_load) << " s\n";

    std::cerr << "[tokenizing] \"" << prompt << "\"..." << std::flush;
    auto t_tok = clock_type::now();
    std::vector<int> ids = model.tokenize(prompt);
    std::cerr << " " << ids.size() << " tokens in " << elapsed_s(t_tok)
              << " s\n";

    std::cerr << "[generating] " << n_new << " tokens (no KV cache, "
              << "each step reprocesses the full sequence)\n";
    auto t_gen = clock_type::now();

    std::cout << prompt << std::flush;
    for (int i = 0; i < n_new; ++i) {
        int next = model.forward_one_step(ids);
        ids.push_back(next);
        std::cout << model.detokenize({next}) << std::flush;
    }
    std::cout << std::endl;

    double gen_s = elapsed_s(t_gen);
    std::cerr << "[done] " << n_new << " new tokens in " << gen_s
              << " s (" << (n_new / gen_s) << " tok/s); "
              << "total wall " << elapsed_s(t_total) << " s\n";
    return 0;
}
