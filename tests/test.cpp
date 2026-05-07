

// DO NOT CHANGE THIS FILE, THIS IS FOR OUR TESTING PURPOSES ONLY
// But We include a sample test here for you to see how the we do the final
// testing We use functions in API to implement our tests
#include "config.h"
#include "test_api.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>

const float EPSILON = 1e-2;

const char *DATA_DIR = "tests/data";
const char *SENTENCE =
    "first try give yourself a break don't be so hard on yourself first tries "
    "often fail and it is your first time living";

// -----------------------------------------------------------------------------
// Binary loader helpers
// -----------------------------------------------------------------------------

static std::vector<int> read_int_array(std::ifstream &f, int n) {
    std::vector<int> v(n);
    f.read(reinterpret_cast<char *>(v.data()), n * sizeof(int));
    return v;
}

static std::vector<float> read_float_array(std::ifstream &f, int n) {
    std::vector<float> v(n);
    f.read(reinterpret_cast<char *>(v.data()), n * sizeof(float));
    return v;
}

struct TokenizeFixture {
    std::vector<int> token_ids;
};

struct EmbeddingFixture {
    std::vector<int> token_ids;
    std::vector<float> embeddings; // [num_tokens * EMBEDDING_DIM]
};

struct MatmulFixture {
    int M, K, N;
    std::vector<float> A, B, C;
};

struct DetokFixture {
    std::vector<int> ids;
    std::string text;
};

struct RmsnormFixture {
    int s, d;
    std::vector<float> x, gamma, out;
};

struct RopeFixture {
    int s;
    std::vector<float> Q, Q_rot;
    std::vector<float> K, K_rot;
};

struct GqaFixture {
    int s;
    std::vector<float> Q, K, V, O;
};

struct AddFixture {
    int n;
    std::vector<float> a, b, out;
};

struct SiluMulFixture {
    int n;
    std::vector<float> gate, up, out;
};

struct SwigluFixture {
    int s, layer_idx;
    std::vector<float> x_norm, out;
};

struct DecoderBlockFixture {
    int s, layer_idx;
    std::vector<float> x_in, x_out;
};

struct ForwardOneFixture {
    std::vector<int> ids;
    int next_id;
};

struct GenerateFixture {
    std::vector<int> ids;
    int T;
    std::vector<int> chain;
};

static TokenizeFixture load_tokenize(const char *path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error(std::string("Cannot open ") + path);
    int n;
    f.read(reinterpret_cast<char *>(&n), sizeof(int));
    return {read_int_array(f, n)};
}

static EmbeddingFixture load_embedding(const char *path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error(std::string("Cannot open ") + path);
    int n;
    f.read(reinterpret_cast<char *>(&n), sizeof(int));
    auto ids = read_int_array(f, n);
    auto emb = read_float_array(f, n * EMBEDDING_DIM);
    return {ids, emb};
}

static MatmulFixture load_matmul(const char *path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error(std::string("Cannot open ") + path);
    MatmulFixture fix;
    f.read(reinterpret_cast<char *>(&fix.M), sizeof(int));
    f.read(reinterpret_cast<char *>(&fix.K), sizeof(int));
    f.read(reinterpret_cast<char *>(&fix.N), sizeof(int));
    fix.A = read_float_array(f, fix.M * fix.K);
    fix.B = read_float_array(f, fix.K * fix.N);
    fix.C = read_float_array(f, fix.M * fix.N);
    return fix;
}

static int read_i32(std::ifstream &f) {
    int v;
    f.read(reinterpret_cast<char *>(&v), sizeof(int));
    return v;
}

static DetokFixture load_detok(const char *path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("Cannot open ") + path);
    DetokFixture fix;
    int n = read_i32(f);
    fix.ids = read_int_array(f, n);
    int nbytes = read_i32(f);
    fix.text.resize(nbytes);
    f.read(&fix.text[0], nbytes);
    return fix;
}

static RmsnormFixture load_rmsnorm(const char *path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("Cannot open ") + path);
    RmsnormFixture fix;
    fix.s = read_i32(f);
    fix.d = read_i32(f);
    fix.x = read_float_array(f, fix.s * fix.d);
    fix.gamma = read_float_array(f, fix.d);
    fix.out = read_float_array(f, fix.s * fix.d);
    return fix;
}

static RopeFixture load_rope(const char *path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("Cannot open ") + path);
    RopeFixture fix;
    int nq = read_i32(f); int sq = read_i32(f); int hdq = read_i32(f);
    fix.s = sq;
    fix.Q     = read_float_array(f, nq * sq * hdq);
    fix.Q_rot = read_float_array(f, nq * sq * hdq);
    int nk = read_i32(f); int sk = read_i32(f); int hdk = read_i32(f);
    fix.K     = read_float_array(f, nk * sk * hdk);
    fix.K_rot = read_float_array(f, nk * sk * hdk);
    return fix;
}

static GqaFixture load_gqa(const char *path) {
    const int Hh = 32, Hk = 8, Hd = 128;
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("Cannot open ") + path);
    GqaFixture fix;
    fix.s = read_i32(f);
    fix.Q = read_float_array(f, Hh * fix.s * Hd);
    fix.K = read_float_array(f, Hk * fix.s * Hd);
    fix.V = read_float_array(f, Hk * fix.s * Hd);
    fix.O = read_float_array(f, fix.s * Hh * Hd);
    return fix;
}

static AddFixture load_add(const char *path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("Cannot open ") + path);
    AddFixture fix;
    fix.n = read_i32(f);
    fix.a   = read_float_array(f, fix.n);
    fix.b   = read_float_array(f, fix.n);
    fix.out = read_float_array(f, fix.n);
    return fix;
}

static SiluMulFixture load_silu_mul(const char *path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("Cannot open ") + path);
    SiluMulFixture fix;
    fix.n = read_i32(f);
    fix.gate = read_float_array(f, fix.n);
    fix.up   = read_float_array(f, fix.n);
    fix.out  = read_float_array(f, fix.n);
    return fix;
}

static SwigluFixture load_swiglu(const char *path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("Cannot open ") + path);
    SwigluFixture fix;
    fix.s = read_i32(f);
    fix.layer_idx = read_i32(f);
    fix.x_norm = read_float_array(f, fix.s * EMBEDDING_DIM);
    fix.out    = read_float_array(f, fix.s * EMBEDDING_DIM);
    return fix;
}

static DecoderBlockFixture load_decoder(const char *path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("Cannot open ") + path);
    DecoderBlockFixture fix;
    fix.s = read_i32(f);
    fix.layer_idx = read_i32(f);
    fix.x_in  = read_float_array(f, fix.s * EMBEDDING_DIM);
    fix.x_out = read_float_array(f, fix.s * EMBEDDING_DIM);
    return fix;
}

static ForwardOneFixture load_forward_one(const char *path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("Cannot open ") + path);
    ForwardOneFixture fix;
    int n = read_i32(f);
    fix.ids = read_int_array(f, n);
    fix.next_id = read_i32(f);
    return fix;
}

static GenerateFixture load_generate(const char *path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("Cannot open ") + path);
    GenerateFixture fix;
    int n = read_i32(f);
    fix.ids = read_int_array(f, n);
    fix.T = read_i32(f);
    fix.chain = read_int_array(f, fix.T);
    return fix;
}

// Strip a leading "<|begin_of_text|>" marker if present, so we accept
// either skip_special_tokens=True or False from the student's detokenizer.
static std::string strip_bos_text(const std::string &s) {
    static const std::string BOS = "<|begin_of_text|>";
    if (s.size() >= BOS.size() && s.compare(0, BOS.size(), BOS) == 0)
        return s.substr(BOS.size());
    return s;
}

// -----------------------------------------------------------------------------
// Comparison helper
// -----------------------------------------------------------------------------

static bool check_max_abs(const std::vector<float> &got,
                          const std::vector<float> &expected, float epsilon) {
    if (got.size() != expected.size()) {
        std::cout << "  size mismatch: got " << got.size() << " expected "
                  << expected.size() << "\n";
        return false;
    }
    float max_err = 0.0f;
    int worst = 0;
    for (int i = 0; i < (int)got.size(); i++) {
        float err = std::fabs(got[i] - expected[i]);
        if (err > max_err) {
            max_err = err;
            worst = i;
        }
    }
    if (max_err > epsilon) {
        std::cout << "  max |err|=" << max_err << " at index " << worst
                  << " (got=" << got[worst] << " expected=" << expected[worst]
                  << ") epsilon=" << epsilon << "\n";
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

bool test_1() {
    TestAPI api;

    string input = "Hello world";
    vector<int> token_ids = api.tokenize(input);
    vector<int> expected = {128000, 9906, 1917};

    if (token_ids.size() != expected.size()) {
        std::cout << "Test failed: size mismatch. Expected " << expected.size()
                  << " but got " << token_ids.size() << "\n";
        return false;
    }

    for (int i = 0; i < (int)token_ids.size(); i++) {
        if (token_ids[i] != expected[i]) {
            std::cout << "Test failed: element mismatch. Expected "
                      << expected[i] << " but got " << token_ids[i] << "\n";
            return false;
        }
    }
    return true;
}

bool test_2() {
    char path[256];
    std::snprintf(path, sizeof(path), "%s/test2_tokenize.bin", DATA_DIR);
    auto fix = load_tokenize(path);

    TestAPI api;
    auto got = api.tokenize(SENTENCE);

    if (got.size() != fix.token_ids.size()) {
        std::cout << "  size mismatch: got " << got.size() << " expected "
                  << fix.token_ids.size() << "\n";
        return false;
    }
    for (int i = 0; i < (int)got.size(); i++) {
        if (got[i] != fix.token_ids[i]) {
            std::cout << "  mismatch at position " << i << ": got " << got[i]
                      << " expected " << fix.token_ids[i] << "\n";
            return false;
        }
    }
    return true;
}

bool test_3() {
    char path[256];
    std::snprintf(path, sizeof(path), "%s/test3_embedding.bin", DATA_DIR);
    auto fix = load_embedding(path); 

    TestAPI api;
    auto got = api.get_embeddings(fix.token_ids);

    return check_max_abs(got, fix.embeddings, EPSILON);
}

bool test_4() {
    char path[256];
    std::snprintf(path, sizeof(path), "%s/test4_embedding.bin", DATA_DIR);
    auto fix = load_embedding(path);

    TestAPI api;
    auto got = api.get_embeddings(fix.token_ids);

    return check_max_abs(got, fix.embeddings, EPSILON);
}

static bool run_matmul_test(const char *bin_path, const char *label) {
    auto fix = load_matmul(bin_path);

    TestAPI api;

    auto t0 = std::chrono::high_resolution_clock::now();
    auto got = api.matmul(fix.A, fix.B, fix.M, fix.K, fix.N);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  " << label << " M=" << fix.M << " K=" << fix.K
              << " N=" << fix.N << "  time=" << ms << "ms\n";

    return check_max_abs(got, fix.C, EPSILON);
}

bool test_5() {
    char path[256];
    std::snprintf(path, sizeof(path), "%s/test5_matmul.bin", DATA_DIR);
    return run_matmul_test(path, "seq_len=1  [first]");
}

bool test_6() {
    char path[256];
    std::snprintf(path, sizeof(path), "%s/test6_matmul.bin", DATA_DIR);
    return run_matmul_test(path, "seq_len=10 ");
}

bool test_7() {
    char path[256];
    std::snprintf(path, sizeof(path), "%s/test7_matmul.bin", DATA_DIR);
    return run_matmul_test(path, "seq_len=100 [last] ");
}

// -----------------------------------------------------------------------------
// Milestone 2 / 3 tests (8..18)
// -----------------------------------------------------------------------------

bool test_8() {
    char path[256];
    std::snprintf(path, sizeof(path), "%s/test8_detokenize.bin", DATA_DIR);
    auto fix = load_detok(path);

    TestAPI api;
    std::string got = api.detokenize(fix.ids);
    std::string a = strip_bos_text(got);
    std::string b = strip_bos_text(fix.text);
    if (a != b) {
        std::cout << "  detokenize mismatch\n  got:      [" << a
                  << "]\n  expected: [" << b << "]\n";
        return false;
    }
    return true;
}

static bool run_rmsnorm_test(int s) {
    char path[256];
    std::snprintf(path, sizeof(path), "%s/test9_rmsnorm_s%d.bin", DATA_DIR, s);
    auto fix = load_rmsnorm(path);
    TestAPI api;
    auto got = api.rmsnorm(fix.x, fix.gamma, fix.s, fix.d);
    std::cout << "  s=" << fix.s << " d=" << fix.d << "\n";
    return check_max_abs(got, fix.out, EPSILON);
}

bool test_9() {
    return run_rmsnorm_test(1) && run_rmsnorm_test(10) && run_rmsnorm_test(100);
}

bool test_10() {
    char path[256];
    std::snprintf(path, sizeof(path), "%s/test10_q_matmul.bin", DATA_DIR);
    return run_matmul_test(path, "Q-proj real");
}

static bool run_rope_test(int s) {
    const int Hh = 32, Hk = 8, Hd = 128;
    char path[256];
    std::snprintf(path, sizeof(path), "%s/test11_rope_s%d.bin", DATA_DIR, s);
    auto fix = load_rope(path);
    TestAPI api;
    auto gotQ = api.rope(fix.Q, Hh, fix.s, Hd);
    auto gotK = api.rope(fix.K, Hk, fix.s, Hd);
    std::cout << "  s=" << fix.s << " (Q heads=" << Hh
              << ", K heads=" << Hk << ")\n";
    bool ok = check_max_abs(gotQ, fix.Q_rot, EPSILON);
    if (!ok) std::cout << "  Q rotation mismatch\n";
    bool okK = check_max_abs(gotK, fix.K_rot, EPSILON);
    if (!okK) std::cout << "  K rotation mismatch\n";
    return ok && okK;
}

bool test_11() {
    return run_rope_test(1) && run_rope_test(10) && run_rope_test(100);
}

static bool run_gqa_test(int s) {
    char path[256];
    std::snprintf(path, sizeof(path), "%s/test12_gqa_s%d.bin", DATA_DIR, s);
    auto fix = load_gqa(path);
    TestAPI api;
    auto got = api.gqa_attention(fix.Q, fix.K, fix.V, fix.s);
    std::cout << "  s=" << fix.s << "\n";
    return check_max_abs(got, fix.O, EPSILON);
}

bool test_12() {
    return run_gqa_test(1) && run_gqa_test(10) && run_gqa_test(100);
}

bool test_13() {
    char path[256];
    std::snprintf(path, sizeof(path), "%s/test13_residual_add.bin", DATA_DIR);
    auto fix = load_add(path);
    TestAPI api;
    auto got = api.residual_add(fix.a, fix.b);
    return check_max_abs(got, fix.out, EPSILON);
}

bool test_14() {
    char path[256];
    std::snprintf(path, sizeof(path), "%s/test14_silu_mul.bin", DATA_DIR);
    auto fix = load_silu_mul(path);
    TestAPI api;
    auto got = api.silu_mul(fix.gate, fix.up);
    return check_max_abs(got, fix.out, EPSILON);
}

bool test_15() {
    char path[256];
    std::snprintf(path, sizeof(path), "%s/test15_swiglu_ffn.bin", DATA_DIR);
    auto fix = load_swiglu(path);
    TestAPI api;
    auto got = api.swiglu_ffn(fix.x_norm, fix.layer_idx, fix.s);
    std::cout << "  s=" << fix.s << " layer=" << fix.layer_idx << "\n";
    return check_max_abs(got, fix.out, EPSILON);
}

bool test_16() {
    char path[256];
    std::snprintf(path, sizeof(path), "%s/test16_decoder_block.bin", DATA_DIR);
    auto fix = load_decoder(path);
    TestAPI api;
    auto got = api.decoder_block(fix.x_in, fix.layer_idx, fix.s);
    std::cout << "  s=" << fix.s << " layer=" << fix.layer_idx << "\n";
    return check_max_abs(got, fix.x_out, EPSILON);
}

bool test_17() {
    char path[256];
    std::snprintf(path, sizeof(path), "%s/test17_forward_one.bin", DATA_DIR);
    auto fix = load_forward_one(path);
    TestAPI api;
    int got = api.forward_one_step(fix.ids);
    std::cout << "  got=" << got << "  expected=" << fix.next_id << "\n";
    return got == fix.next_id;
}



bool test_18() {
    char path[256];
    std::snprintf(path, sizeof(path), "%s/test18_generate.bin", DATA_DIR);
    auto fix = load_generate(path);
    TestAPI api;
    auto got = api.generate(fix.ids, fix.T);
    if ((int)got.size() != fix.T) {
        std::cout << "  size mismatch: got " << got.size() << " expected "
                  << fix.T << "\n";
        return false;
    }
    for (int i = 0; i < fix.T; i++) {
        if (got[i] != fix.chain[i]) {
            std::cout << "  divergence at step " << i << ": got " << got[i]
                      << " expected " << fix.chain[i] << "\n";
            return false;
        }
    }
    std::cout << "  T=" << fix.T << " all tokens match\n";
    return true;
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    const char *GREEN = "\033[32m";
    const char *RED = "\033[31m";
    const char *RESET = "\033[0m";

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <test_id>\n";
        return 1;
    }

    int test_id = 0;
    try {
        test_id = std::stoi(argv[1]);
    } catch (...) {
        std::cout << RED << "Invalid test id: " << argv[1] << RESET << "\n";
        return 2;
    }

    bool ok = false;
    try {
        switch (test_id) {
        case 1:
            ok = test_1();
            break;
        case 2:
            ok = test_2();
            break;
        case 3:
            ok = test_3();
            break;
        case 4:
            ok = test_4();
            break;
        case 5:
            ok = test_5();
            break;
        case 6:
            ok = test_6();
            break;
        case 7:
            ok = test_7();
            break;
        case 8:
            ok = test_8();
            break;
        case 9:
            ok = test_9();
            break;
        case 10:
            ok = test_10();
            break;
        case 11:
            ok = test_11();
            break;
        case 12:
            ok = test_12();
            break;
        case 13:
            ok = test_13();
            break;
        case 14:
            ok = test_14();
            break;
        case 15:
            ok = test_15();
            break;
        case 16:
            ok = test_16();
            break;
        case 17:
            ok = test_17();
            break;
        case 18:
            ok = test_18();
            break;
        default:
            std::cout << RED << "Unknown test id: " << test_id << RESET << "\n";
            return 2;
        }
    } catch (const std::exception &e) {
        std::cout << RED << "Test threw: " << e.what() << RESET << "\n";
        ok = false;
    }

    std::cout << (ok ? GREEN : RED) << (ok ? "PASSED" : "FAILED") << RESET
              << "\n";
    return ok ? 0 : 3;
}
