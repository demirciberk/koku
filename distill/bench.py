"""Quality and speed of the exported student GGUFs under llama.cpp.

For each GGUF: start llama-server, generate for the same held-out perfumes as
`distill/sft.py --eval`, report valid JSON, foreign-note rate, agreement with the
teacher's description (token F1), and decode speed. Prompt is the exact chat
template used in training (thinking disabled), sent to /completion.

Run: uv run python distill/bench.py [--n 200]
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer

sys.path.insert(0, "distill")
sys.path.insert(0, "eval")
from generate import parse  # noqa: E402
from sft import BASE, chat, split, unfaithful  # noqa: E402

FILES = ["f16", "q8_0", "q4_k_m"]
PORT = 8091


def f1(a: str, b: str) -> float:
    x, y = Counter(a.lower().split()), Counter(b.lower().split())
    common = sum((x & y).values())
    return 0.0 if not common else 2 * common / (sum(x.values()) + sum(y.values()))


def complete(prompt: str) -> dict:
    body = {"prompt": prompt, "n_predict": 320, "temperature": 0, "cache_prompt": True}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def bench(name: str, prompts: list[str], rows: list[dict], by: dict) -> dict:
    srv = subprocess.Popen(["tools/llama.cpp/llama-server.exe", "-m", f"models/gguf/koku-0.6b-{name}.gguf", "-ngl", "99",
                            "-c", "8192", "-np", "4", "--port", str(PORT), "--host", "127.0.0.1"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except OSError:
                time.sleep(1)
        t0 = time.time()
        with ThreadPoolExecutor(4) as pool:
            outs = list(pool.map(complete, prompts))
        wall = time.time() - t0
    finally:
        srv.terminate()
        srv.wait()
    valid = foreign = 0
    f1s, tps, tokens = [], [], 0
    for r, o in zip(rows, outs):
        tokens += o["timings"]["predicted_n"]
        tps.append(o["timings"]["predicted_per_second"])
        d = parse(o["content"])
        if d:
            valid += 1
            foreign += bool(unfaithful(d["description"], by[r["id"]]))
            f1s.append(f1(d["description"], r["description"]))
    return {"model": name, "valid_json": round(valid / len(rows), 3), "desc_with_foreign_note": round(foreign / max(valid, 1), 3),
            "desc_f1_vs_teacher": round(sum(f1s) / max(len(f1s), 1), 3),
            "tokens_per_s_single_stream": round(sorted(tps)[len(tps) // 2], 1),
            "tokens_per_s_4_parallel": round(tokens / wall, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    _, held, by = split()
    rows = held[: args.n]
    tok = AutoTokenizer.from_pretrained(BASE)
    prompts = [chat(tok, by[r["id"]]) for r in rows]
    results = [bench(f, prompts, rows, by) for f in FILES]
    for r in results:
        print(json.dumps(r))
    with open("runs/gguf-bench.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
