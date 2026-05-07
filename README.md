# CS265 LLM Inference Project

A from-scratch Llama-3-8B-Instruct inference engine: C++ controller + hand-written CUDA kernels, all BF16 on the GPU.

## One-time setup

```bash
module load cuda/12.2.0-fasrc01

# Download the HF snapshot to ./persisted_assets/llama3/  (needs $HF_TOKEN)
uv run python tools/llama3_downloader.py --out ./persisted_assets/llama3/

# Dump per-tensor blobs (used by the C++ loader) into the same directory
uv run python tools/dumper.py

# Mirror persisted_assets/ to a fast netscratch path and symlink ./assets -> there
tools/sync_to_netscratch.sh /n/netscratch/<lab>/Lab/$USER/cs265-assets
```

`module load cuda/12.2.0-fasrc01` is required in every fresh shell.

## Build

```bash
make                 # bin/llm     (interactive demo)
make tests           # bin/tests   (the 18-test harness)
make my_tests        # bin/my_tests
make clean           # rm -rf build/ bin/
```

## Run

```bash
./bin/tests <N>                  # run test N (1..18)
./bin/llm "Once upon a time" 5   # prompt + n_new tokens, streamed to stdout
```

## Reference (for comparing against the canonical Python path)

```bash
uv run python tools/reference_generate.py "Once upon a time" 5
```

Same prompt, same model, greedy decoding via HuggingFace `transformers`. Use this to sanity-check that our C++/CUDA stack produces the same continuation as the reference.

## Where things live

- `kernel/*.cu` — CUDA kernels (BF16 in/out, FP32 register accumulators).
- `include/model.h` + `src/model.cpp` — `Model` class, eager-loads all weights at construction.
- `tests/test_api.cpp` — thin wrappers around `Model{}.foo()` and direct kernel launches; converts `vector<float>` ↔ BF16 at the boundary.
- `tools/` — Python scripts (download, dump, reference generation) and the netscratch sync.
- `persisted_assets/` (HOME) — source-of-truth model files; never modified after download/dump.
- `assets/` — symlink to a netscratch copy of `persisted_assets/`. Created by `tools/sync_to_netscratch.sh`.
