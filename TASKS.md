# CS265 Project Tasks (extracted from part1.pdf and part2.pdf)

A from-scratch Llama-3-8B-Instruct inference engine in C++ + CUDA. Single-prompt, no batching, no KV cache (both optional bonuses worth +5% each). Sequence length ≤ 1000 tokens.

---

## Logistics & Constraints

- **Language**: C++ for the controller, CUDA for compute kernels. Python only for weight dumping.
- **`DO NOT CHANGE THIS`**-tagged files must remain unmodified — they are used for evaluation.
- **Model files** must live in `./assets/llama3/`. The grader will not re-download.
- **Dumper script** must be at `tools/dumper.py`.
- **Test entry points** specified in `tests/test_api.h` — implement all listed functions.
- **Required kernel optimizations**: tiling, shared-memory reuse, coalesced HBM accesses. (Mandatory for matmul.)
- **Bonus optimizations**: tensor cores, vectorization, double buffering.
- **Precision**: BF16 encouraged but FP32 allowed.

## Grading

- Final code review: **35%** (mandatory to pass for any test credit)
- Midway check-in: **10%** (M1 due)
- Automated testing: **10%**
- Bonus thought-experiment answers: up to **+5% per milestone**
- Optional: KV cache, batching: each up to +5%

---

## Milestone 1 — Loading, Tokenization, Embeddings, Matmul

**Required functionality:**

1. **Step 1: Download assets.** Use the provided downloader script; weights and tokenizer end up in `assets/llama3/`.
2. **Step 2: Tokenize prompts.** Convert e.g. `"Hello world"` → `[128000, 9906, 1917]`. Special-token handling not required at this milestone.
3. **Step 3: Dump and load weights.** A Python `tools/dumper.py` extracts tensors from the `.safetensors` files into a binary format of your choice; a C++ loader reads them.
4. **Step 4: Embedding lookup.** Given token IDs, produce the token-embedding sequence (shape `(s, d=4096)`).
5. **Step 5: First CUDA kernel — matrix multiplication (GEMM).** Tiling, shared-memory reuse, coalesced global-memory access mandatory.

**Suggested guideline** (not mandatory):

- Tokenizer API in `include/tokenizer.h`; partial impl in `src/`. ~10 LOC missing.
- C++ loader may use `mmap`. Suggested format: small header (name, shape, dtype) + raw bytes. Per-operator or per-layer dump fine; embedding table dumped separately.
- Per-operator or per-layer dump format is your call.
- Weights stored as BF16 on disk; you may load as FP32 for simplicity. BF16 throughout is encouraged.
- Verify correctness incrementally; compare against PyTorch where possible.

**M1 Bonus thought experiments:**

- **B1.1.** What are special tokens (`<|start_header_id|>`, `<|begin_of_text|>`)? Why are they necessary? How to benefit from each?
- **B1.2.** Tradeoffs of fine-grained subword tokenization vs. one-token-per-word.
- **B1.3.** Loading all model weights is slow — applications bottlenecked by this; ways to speed it up.
- **B1.4.** Where is best to store the embedding table — CPU or GPU? Tradeoffs.
- **B1.5.** Disk → CPU → GPU is suboptimal — tricks/hacks to speed it up.

---

## Milestone 2 — RMSNorm + Q/K/V Projections

**Architecture constants** (Llama-3-8B): `d=4096`, `h=32` query heads, `h_k=8` KV heads (GQA), `h_d=128` per-head dim, `d_ff=14336`, `V=128256`, `eps=1e-5`, RoPE base `500000`.

**Required functionality:**

1. **Step 1: RMSNorm kernel.** Row-wise: `RMSNorm(x) = x · γ / RMS(x)` where `RMS(x) = sqrt(mean(x²) + eps)`. `γ` is the learned `input_layernorm.weight` (per layer). `eps` is *inside* the sqrt.
2. **Step 2: Q projection.** `Q = X_norm · W_Q^T`, shape `(s, h·h_d)`, weight `q_proj.weight` of shape `(h·h_d, d)`.
3. **Step 3: K projection.** `K = X_norm · W_K^T`, shape `(s, h_k·h_d)`.
4. **Step 4: V projection.** Analogous: `V = X_norm · W_V^T`, shape `(s, h_k·h_d)`.

**Notes:** Q has more columns than K and V because of GQA (more query heads than KV heads). All three Q/K/V projections reuse the matmul kernel; don't write a new one.

**Suggested guideline:**

- RMSNorm: one block per row; PMPP-style shared-memory tree reduction for the sum-of-squares; thread 0 broadcasts the rsqrt.
- ε goes inside the sqrt to avoid division-by-zero on near-zero rows.
- Verify operators independently before composing.

**M2 Bonus thought experiments:**

- **B2.1.** Predict whether RMSNorm is memory-bandwidth-bound or compute-bound. Estimate arithmetic intensity (FLOPs/byte) assuming all reads from HBM. Profile with `ncu`. Same for matmul. Were predictions correct?
- **B2.2.** RMSNorm reads each row twice from HBM (once for sum-of-squares, once for the scale). What's the total HBM traffic per row? If you kept the row in shared memory across both passes, how much traffic would you save? What prevents naïve implementations from doing this for very long rows?
- **B2.3.** Q, K, V projections all read `X_norm` from HBM in three separate kernels. Total HBM read traffic across all three? If `X_norm` fit in L2 after the first read, how would effective traffic change? Does it actually fit on this GPU?

---

## Milestone 3 — RoPE, GQA, FFN, Decoder Block, End-to-End Inference

**Required functionality:**

1. **Reshape Q, K, V.** Logical reshape `(s, h·h_d) → (h, s, h_d)` (and analogous for K/V with `h_k`). No data movement.
2. **Step 1: RoPE.** Pair dim `i` with dim `i + h_d/2` (HuggingFace `rotate_half` convention, NOT `(q0, q1), (q2, q3)…`). Angle `θ_i = 1 / 500000^(2i/h_d)` (Llama 3 base is 500000, not 10000). Pre-compute `cos(p·θ_i)` and `sin(p·θ_i)` tables on the GPU. Applied independently to every K head.
3. **Step 2: GQA with causal masking + numerically stable softmax.** For query head `i`, KV head index `g = i / (h/h_k)`. Compute `S_i = Q_i K_g^T / sqrt(h_d)`, apply causal mask (`q > p` ⇒ `-inf` or large negative), find row max, subtract, exponentiate, normalize. Numerically stable softmax is **required** — subtract row max before exp.
4. **Step 3: Output projection.** `attn_out = O · W_O^T`. Matmul kernel.
5. **Step 4: Residual add.** Elementwise. Reused for both decoder residuals.
6. **Step 5: SwiGLU FFN, decoder block, 32-layer loop.** FFN decomposes into:
   - `gate = X_norm · W_gate^T` (matmul)
   - `up = X_norm · W_up^T` (matmul)
   - `H = SiLU(gate) ⊙ up` (new elementwise kernel)
   - `ffn_out = H · W_down^T` (matmul)
   
   Decoder block order: RMSNorm → QKV → RoPE → Attention → output proj → residual → RMSNorm → SwiGLU FFN → residual. Loop 32 times.
7. **Step 6: Output layer + token generation.** After 32 layers: final RMSNorm (`model.norm.weight`), extract last token's vector, project via `W_lm` (= `lm_head.weight`, **shares storage with `embed_tokens.weight`** in Llama 3). `logits = x_last · W_lm^T`. Argmax. Decode token ID via the tokenizer. Verify against `reference.py`.

**Suggested guideline:**

- RoPE: precompute `cos`/`sin` for all `(p, i)` pairs and store on GPU before the forward pass.
- GQA: a workable first approach is a host-side loop over `h=32` heads, launching kernels per head for scores, mask, softmax, weighted sum. Use `g = i / (h/h_k)` integer division.
- Causal mask: add a large negative (e.g. `-1e6`) to all positions where `q > p` to avoid branching.
- Softmax order: row max → subtract → exp → divide by sum. Required.
- SwiGLU: implement only the elementwise `SiLU(gate) ⊙ up` step as a CUDA kernel; reuse matmul for the three projections.
- Output layer: only the **last** token's vector goes through `lm_head` — projecting all `s` rows is wasteful at `V=128256`.

**Common pitfalls (per part2.pdf §4):**

- Softmax must subtract row max first — direct `exp(S)/sum(exp(S))` overflows.
- RoPE pairs first half with second half, NOT `(q0, q1)`. Wrong pairing produces silently wrong outputs.
- RoPE base is **500000**, not 10000.
- Each layer has TWO RMSNorm weight vectors: `input_layernorm.weight` and `post_attention_layernorm.weight`.
- Don't forget the `γ` scale step in RMSNorm.
- Weights stored transposed (`(out_dim, in_dim)`) — matmul always computes `X · W^T`.
- `lm_head.weight` shares storage with `model.embed_tokens.weight` — load once.
- Project only the last token before `lm_head` (saves a `s × V` matmul over all rows).
- Causal mask applies to every row: `q > p` for every `p`, not just the last row.

**M3 Bonus thought experiments:**

- **B3.1.** Profile a forward pass with `nsys`. Which operator dominates wall time? Double `s` and re-profile. Which grows fastest with `s`, and why? What does that imply about the practical bottleneck of inference without KV caching?
- **B3.2.** Score matrix `S_i ∈ R^(s × s)` is materialized for all `h=32` heads. Total memory at `s=512` in FP32? How does it scale with `s`? At what `s` does score-matrix memory exceed total model weights?
- **B3.3.** Without KV caching, generating `T` new tokens from prompt length `s_0` requires `T` forward passes on lengths `s_0+1, …, s_0+T`. Total FLOPs vs a system with KV cache (one-token-per-step)? Implication for latency at large `T`.

---

## Test API surface (`tests/test_api.h`, **DO NOT CHANGE**)

The test driver hits exactly these methods:

```cpp
class TestAPI {
  public:
    vector<int>   tokenize(string input);
    string        detokenize(vector<int> token_ids);
    vector<float> get_embeddings(vector<int> token_ids);
    vector<float> matmul(const vector<float> &A, const vector<float> &B, int M, int K, int N);
    vector<float> rmsnorm(const vector<float> &x, const vector<float> &gamma, int s, int d);
    vector<float> rope(const vector<float> &qk, int n_heads, int s, int h_d);
    vector<float> gqa_attention(const vector<float> &Q, const vector<float> &K, const vector<float> &V, int s);
    vector<float> residual_add(const vector<float> &a, const vector<float> &b);
    vector<float> silu_mul(const vector<float> &gate, const vector<float> &up);
    vector<float> swiglu_ffn(const vector<float> &x_norm, int layer_idx, int s);
    vector<float> decoder_block(const vector<float> &x, int layer_idx, int s);
    int           forward_one_step(const vector<int> &token_ids);
    vector<int>   generate(const vector<int> &token_ids, int n_new);
};
```

The 18 numbered test fixtures live in `tests/data/test<N>_*.bin` and were generated from `reference.py` (the PyTorch oracle).
