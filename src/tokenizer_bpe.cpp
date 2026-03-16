// This is your first implementation file
// you have to write small snippet missing in the encode_chunk function to implement the BPE merge algorithm.

#include "prelude.h"
#include "tokenizer.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstring>
#include <fstream>
#include <stdexcept>

vector<int> BPETokenizer::encode(const string &text) const {
    return encode_impl(text, true);
}

vector<int> BPETokenizer::encode_no_merge(const string &text) const {
    return encode_impl(text, false);
}

string BPETokenizer::decode(const vector<int> &ids) const {
    string out;
    for (int id : ids) {
        if (id2special.count(id))
            continue;
        if (id >= 0 && id < static_cast<int>(id2tok.size()))
            out += id2tok[id];
    }
    return out;
}

int BPETokenizer::bos_id() const { return bos_id_; }
int BPETokenizer::eos_id() const { return eos_id_; }

BPETokenizer::BPETokenizer() {
    throw std::runtime_error("BPETokenizer must be initialized with a path");
}

BPETokenizer::BPETokenizer(const string &path) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("cannot open " + path);

    string line;
    vector<std::pair<string, int>> entries;
    while (std::getline(f, line)) {
        if (line.empty())
            continue;
        size_t sp = line.find_last_of(' ');
        if (sp == string::npos)
            continue;
        string tok = b64decode(line.substr(0, sp));
        int r = std::stoi(line.substr(sp + 1));
        entries.emplace_back(std::move(tok), r);
    }

    size_t maxid = 0;
    for (auto &kv : entries)
        maxid = std::max(maxid, static_cast<size_t>(kv.second));

    id2tok.assign(maxid + 1, string());
    for (auto &kv : entries) {
        rank[kv.first] = kv.second;
        id2tok[kv.second] = kv.first;
    }

    vector<string> specials = {"<|begin_of_text|>",
                               "<|end_of_text|>",
                               "<|reserved_special_token_0|>",
                               "<|reserved_special_token_1|>",
                               "<|finetune_right_pad_id|>",
                               "<|step_id|>",
                               "<|start_header_id|>",
                               "<|end_header_id|>",
                               "<|eom_id|>",
                               "<|eot_id|>",
                               "<|python_tag|>"};

    int reserved_total = 256;
    for (int i = 2; i < reserved_total - static_cast<int>(specials.size()) + 2;
         i++) {
        specials.push_back("<|reserved_special_token_" + std::to_string(i) +
                           "|>");
    }

    int next = static_cast<int>(id2tok.size());
    for (const auto &s : specials) {
        special2id[s] = next;
        id2special[next] = s;
        ++next;
    }

    bos_id_ = special2id.at("<|begin_of_text|>");
    eos_id_ = special2id.at("<|end_of_text|>");

    for (const auto &kv : special2id)
        specials_sorted.push_back(kv.first);
    std::sort(
        specials_sorted.begin(), specials_sorted.end(),
        [](const string &a, const string &b) { return a.size() > b.size(); });
}

string BPETokenizer::b64decode(const string &s) {
    static int T[256];
    static bool init = false;
    if (!init) {
        std::fill(std::begin(T), std::end(T), -1);
        string a =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; i++)
            T[static_cast<unsigned char>(a[i])] = i;
        init = true;
    }

    string out;
    out.reserve(s.size() * 3 / 4);

    int val = 0, valb = -8;
    for (unsigned char c : s) {
        if (std::isspace(c))
            continue;
        if (c == '=')
            break;
        int d = T[c];
        if (d < 0)
            continue;
        val = (val << 6) + d;
        valb += 6;
        if (valb >= 0) {
            out.push_back(char((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

vector<int> BPETokenizer::encode_impl(const string &text,
                                      bool enable_merge) const {
    vector<int> ids;
    size_t i = 0, n = text.size();

    while (i < n) {
        bool matched = false;

        for (const auto &s : specials_sorted) {
            if (i + s.size() <= n &&
                std::memcmp(text.data() + i, s.data(), s.size()) == 0) {
                ids.push_back(special2id.at(s));
                i += s.size();
                matched = true;
                break;
            }
        }
        if (matched)
            continue;

        size_t j = i + 1;
        while (j < n) {
            bool hit = false;
            for (const auto &sp : specials_sorted) {
                if (j + sp.size() <= n &&
                    std::memcmp(text.data() + j, sp.data(), sp.size()) == 0) {
                    hit = true;
                    break;
                }
            }
            if (hit)
                break;
            ++j;
        }

        string chunk = text.substr(i, j - i);
        auto v = encode_chunk(chunk, enable_merge);
        ids.insert(ids.end(), v.begin(), v.end());
        i = j;
    }

    return ids;
}

vector<int> BPETokenizer::encode_chunk(const string &s,
                                       bool enable_merge) const {
    vector<string> toks;
    toks.reserve(s.size());
    for (unsigned char c : s)
        toks.emplace_back(1, char(c));

    auto pair_rank = [&](const string &a, const string &b) -> int {
        auto it = rank.find(a + b);
        return it == rank.end() ? INT_MAX : it->second;
    };

    if (enable_merge) {
        while (true) {
            int best = INT_MAX;
            int best_idx = -1;
            // throw runtime_error("Not implemented: you need to implement the BPE merge algorithm here");
            // YOUR CODE HERE
            // Find the best pair to merge using pair_rank function
            for (int i = 1; i < toks.size(); i++) {
                int new_val = pair_rank(toks[i-1], toks[i]);
                if (new_val < best) {
                    best = new_val;
                    best_idx = i;
                }
            }
            if (best == INT_MAX) {
                // No matches found.
                break;
            }

            // Merge best_idx-1 and best_idx
            toks[best_idx-1] = toks[best_idx-1] + toks[best_idx];
            toks.erase(toks.begin() + best_idx);
        }
    }

    vector<int> ids;
    ids.reserve(toks.size());
    for (const auto &tkn : toks) {
        auto it = rank.find(tkn);
        if (it != rank.end()) {
            ids.push_back(it->second);
        } else {
            for (unsigned char c : tkn) {
                ids.push_back(rank.at(string(1, char(c))));
            }
        }
    }
    return ids;
}
