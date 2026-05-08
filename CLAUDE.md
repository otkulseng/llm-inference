ALWAYS READ CLAUDE.md before tool calls.

ALWAYS update CLAUDE.md to keep it up-to-date. For instance, the progress below must be correct.
# CS265 LLM-from-Scratch Project

A from-scratch Llama-3-8B inference engine: C++ controller plus hand-written CUDA
kernels, no PyTorch on the inference path. Three milestones:

- **M1** — tokenization, weight dump + load, embedding lookup, matmul kernel. **Done.**
- **M2** — RMSNorm kernel + Q/K/V projections (matmul reuse). **Done** — tests 9 (rmsnorm) and 10 (Q-projection matmul) pass.
- **M3** — RoPE, GQA with causal mask, SwiGLU FFN, decoder block, 32-layer loop,
  output head, autoregressive token generation. **Done.**
  - `forward_one_step` (test 17): ~3.3 s/token. `generate` (test 18, 5 tokens):
    ~4.6 s. `bin/llm` end-to-end: ~5 s.
  - All weights cached on the GPU at `Model{}` construction (~16 GB total in
    BF16). Eager load, no lazy/optional/streaming abstraction.

The model-specific code lives in `include/model.h` + `src/model.cpp` (a `Model`
class). `tests/test_api.cpp` is just kernel-test wrappers + thin `Model{}.foo()`
pass-throughs — `TestAPI::generate(ids, n) { return Model{}.generate(ids, n); }`.

**BF16 everywhere.** All persistent storage on the GPU is `__nv_bfloat16` (cached
weights, activations, rope tables, attention scores). FP32 lives only in
per-thread registers inside kernels for precision-critical reductions:
matmul accumulators, softmax max/sum, rmsnorm sum-of-squares. No FP32 buffer
ever touches global or shared memory. Host helpers in `include/bf16.h`
convert at the test-API boundary.

Currently 9 of 18 tests fail by small BF16-precision margins (max errors
0.013-0.07 vs the original `EPSILON=0.01`); the course has agreed to relax
those thresholds. The high-level integration tests (17, 18) and most
elementwise/attention tests (1, 2, 3, 4, 8, 12, 13) still pass within 0.01.

The full spec is in `part1.pdf` (M1) and `part2.pdf` (M2 + M3). A PyTorch
reference for every operator is in `reference.py` — compare against it numerically
when debugging.


## Working preferences

- **Simplicity and readability over micro-optimization.** Code is reviewed by
  hand after the tests pass; both correctness and readability are required.
- **Follow the spec PDFs literally.** When `part2.pdf §2.2` describes a
  reduction in four stages, write the kernel in those four stages with comments
  citing the PDF wording. The PDFs encode pedagogical intent, not just an
  outcome.
- **Match course-textbook patterns.** Programming Massively Parallel Processors
  (PMPP) is the course reference. Prefer its shared-memory tree reduction over
  warp-shuffle micro-optimizations when both pass.
- **Don't introduce abstractions until repetition justifies them.** A
  `DeviceBuffer` RAII helper around `cudaMalloc`/`cudaFree` makes sense once
  ~5+ kernel wrappers exist (M3); not before.
- **Verify operators independently** (per `part2.pdf §2.2`). Run tests one ID
  at a time during development.


## Build & test

```bash
module load cuda/12.2.0-fasrc01    # nvcc is NOT on the default PATH
make tests                         # builds bin/tests
./bin/tests <N>                    # run test N (1..18)
make my_tests                      # builds bin/my_tests for scratch tests
make clean                         # rm -rf build/ bin/
```

The `module load` line is required in every fresh shell.

## Architecture constants (Llama-3-8B)

| Symbol | Value | Notes |
|---|---|---|
| `d` | 4096 | embedding dimension |
| `h` | 32 | query heads |
| `h_k` | 8 | KV heads (GQA: `h/h_k = 4` queries per KV) |
| `h_d` | 128 | per-head dimension (`d = h * h_d`) |
| `d_ff` | 14336 | SwiGLU intermediate size |
| `V` | 128256 | vocabulary size |
| layers | 32 | decoder block count |
| RoPE base | **500000** | NOT 10000 — Llama 3 changed this |
| `eps` | 1e-5 | RMSNorm, inside the sqrt |

## Known issues / non-goals

- KV caching and batching are explicitly out of scope per `part1.pdf §3.1`
  (optional bonus only).
- Weights are stored transposed in the checkpoint as `(out_dim, in_dim)`, so
  matmul always computes `X · W^T`. This is a common shape-bug source — see
  `part2.pdf §4`.
