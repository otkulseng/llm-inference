"""
Reference greedy-decoding via HuggingFace transformers. Mirrors `bin/llm` so
you can diff our C++/CUDA output against the canonical PyTorch path.

Usage:
  uv run python tools/reference_generate.py                          # default prompt, 10 tokens
  uv run python tools/reference_generate.py "Your prompt"            # custom prompt, 10 tokens
  uv run python tools/reference_generate.py "Your prompt" 25         # custom prompt, 25 tokens

Reuses ./assets/llama3/ (the same snapshot tools/llama3_downloader.py and
tools/dumper.py operate on) — no network access required at runtime.
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "./assets/llama3"   # matches tools/llama3_downloader.py


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"
    n_new  = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    # bfloat16 = HF default for Llama-3 inference. Our C++ does FP32 compute
    # internally; results should match to within a few ULPs but may differ on
    # tie-breaks. Switch to torch.float32 if you want bit-equivalent comparison
    # (needs ~32 GB GPU vs ~16 GB for bfloat16).
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16
    ).to("cuda")
    model.eval()

    inputs = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=n_new,
            do_sample=False,                     # greedy (matches our argmax)
            pad_token_id=tok.eos_token_id,       # silences pad-id warning
        )
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
