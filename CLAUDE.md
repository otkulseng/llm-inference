ALWAYS READ CLAUDE.md before tool calls.

ALWAYS update CLAUDE.md to keep it up-to-date. For instance, the progress below must be correct.
# CS265 LLM-from-Scratch Project

A from-scratch Llama-3-8B inference engine: C++ controller plus hand-written CUDA
kernels, no PyTorch on the inference path. Three milestones:

- **M1** — tokenization, weight dump + load, embedding lookup, matmul kernel. **Done.**
- **M2** — RMSNorm kernel + Q/K/V projections (matmul reuse). **Done** — tests 9 (rmsnorm) and 10 (Q-projection matmul) pass.
- **M3** — RoPE, GQA with causal mask, SwiGLU FFN, decoder block, 32-layer loop,
  output head, autoregressive token generation. **In progress.**
  - `detokenize` (test 8) — done.
  - `residual_add` (test 13) — done.
  - `silu_mul` (test 14) — done.
  - `rope` (test 11) — done.
  - `gqa_attention` (test 12), `swiglu_ffn` (test 15), `decoder_block` (test 16), `forward_one_step` (test 17), `generate` (test 18) — pending.

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

- Test 1 currently fails: tokenizer doesn't prepend the BOS token (`128000`).
  Pre-existing, separate concern.
- KV caching and batching are explicitly out of scope per `part1.pdf §3.1`
  (optional bonus only).
- Weights are stored transposed in the checkpoint as `(out_dim, in_dim)`, so
  matmul always computes `X · W^T`. This is a common shape-bug source — see
  `part2.pdf §4`.
