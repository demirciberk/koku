"""Merge the student's LoRA adapter into Qwen3-0.6B and export GGUF files for llama.cpp.

Output: models/qwen3-0.6b-koku-merged/ (HF, bf16) and models/gguf/koku-0.6b-{f16,q8_0,q4_k_m}.gguf.
Needs the llama.cpp converter checked out in tools/llama-src (see README) and the
prebuilt tools/llama.cpp/llama-quantize.exe.

Run: uv run python distill/export.py
"""
import os
import subprocess
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "distill")
from sft import BASE, OUT  # noqa: E402

MERGED = "models/qwen3-0.6b-koku-merged"
GGUF = "models/gguf"
QUANTS = ["q8_0", "q4_k_m"]

if __name__ == "__main__":
    if not os.path.exists(f"{MERGED}/config.json"):
        model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, OUT).merge_and_unload()
        model.save_pretrained(MERGED)
        AutoTokenizer.from_pretrained(BASE).save_pretrained(MERGED)
        print("merged", MERGED)
    os.makedirs(GGUF, exist_ok=True)
    f16 = f"{GGUF}/koku-0.6b-f16.gguf"
    if not os.path.exists(f16):
        subprocess.run([sys.executable, "tools/llama-src/convert_hf_to_gguf.py", MERGED, "--outtype", "f16", "--outfile", f16], check=True)
    for q in QUANTS:
        out = f"{GGUF}/koku-0.6b-{q}.gguf"
        if not os.path.exists(out):
            subprocess.run(["tools/llama.cpp/llama-quantize.exe", f16, out, q.upper()], check=True)
    for f in sorted(os.listdir(GGUF)):
        print(f, round(os.path.getsize(f"{GGUF}/{f}") / 1e6), "MB")
