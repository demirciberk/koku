"""Dense retrieval with off-the-shelf multilingual embedders (no fine-tuning).

Turkish queries are embedded as-is. Perfumes can be embedded from three texts:
  en    the English fields BM25 sees (name, brand, gender, accords, notes)
  tr    the same fields with notes/accords mapped to Turkish via distill/tr_names.json (no LLM)
  desc  tr + the teacher's Turkish description from data/teacher.clean.jsonl (document expansion)
Cosine similarity, top 100. Document embeddings are cached in runs/cache/.

Run: uv run python retrieve/dense.py bge-m3 --docs desc
     uv run python eval/evaluate.py runs/dense-bge-m3-desc.jsonl
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "eval")
sys.path.insert(0, "retrieve")
from bm25 import doc_text, load_queries  # noqa: E402
from label import corpus  # noqa: E402

# name -> (HF model id, query prefix, document prefix). e5 models are trained with these prefixes.
MODELS = {
    "e5-small": ("intfloat/multilingual-e5-small", "query: ", "passage: "),
    "e5-base": ("intfloat/multilingual-e5-base", "query: ", "passage: "),
    "bge-m3": ("BAAI/bge-m3", "", ""),
}
TOP = 100
GENDER_TR = {"men": "erkek", "women": "kadın", "unisex": "unisex"}


def tr_text(p: dict, tr: dict) -> str:
    n, a = tr["notes"], tr["accords"]
    return " ".join([p["name"], p["brand"], GENDER_TR.get(p["gender"], p["gender"]), *(a.get(x, x) for x in p["accords"]),
                     *(n.get(x, x) for k in ("top", "middle", "base") for x in p[k])])


def doc_builder(kind: str, desc_path: str = "data/teacher.clean.jsonl"):
    if kind == "en":
        return doc_text
    with open("distill/tr_names.json", encoding="utf-8") as f:
        tr = json.load(f)
    if kind == "tr":
        return lambda p: tr_text(p, tr)
    with open(desc_path, encoding="utf-8") as f:
        desc = {r["id"]: r["description"] for r in map(json.loads, f)}
    # Perfumes dropped by distill/check.py have no description and fall back to the Turkish fields alone.
    return lambda p: (tr_text(p, tr) + " " + desc.get(p["id"], "")).strip()


def embed(model: SentenceTransformer, texts: list[str], batch: int) -> np.ndarray:
    return model.encode(texts, batch_size=batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


def load_model(model_id: str) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_id, device=device)
    if device == "cuda":
        model.half()  # 8 GB laptop GPU: fp16 halves memory, retrieval quality is unaffected in practice
    model.max_seq_length = 256
    return model


def run(name: str, docs_kind: str = "en", model_id: str | None = None) -> str:
    """Rank all eval queries. `model_id` overrides the MODELS entry (used for fine-tuned checkpoints)."""
    mid, qp, dp = MODELS[name] if name in MODELS else (None, "", "")
    model = load_model(model_id or mid)
    tag = name if docs_kind == "en" else f"{name}-{docs_kind}"

    docs = corpus()
    os.makedirs("runs/cache", exist_ok=True)
    cache = f"runs/cache/{tag}-docs.npy"
    if os.path.exists(cache):
        D = np.load(cache)
    else:
        text = doc_builder(docs_kind)
        D = embed(model, [dp + text(p) for p in docs], batch=64)
        np.save(cache, D)
    assert D.shape[0] == len(docs), "stale cache: delete runs/cache and rerun"

    queries = load_queries()
    Q = embed(model, [qp + q["query"] for q in queries], batch=64)
    S = Q.astype(np.float32) @ D.astype(np.float32).T
    out = f"runs/dense-{tag}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for q, s in zip(queries, S):
            top = np.argsort(-s)[:TOP]
            f.write(json.dumps({"id": q["id"], "ranking": [docs[i]["id"] for i in top]}) + "\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=list(MODELS))
    ap.add_argument("--docs", choices=["en", "tr", "desc"], default="en")
    args = ap.parse_args()
    for n in args.models:
        print("wrote", run(n, args.docs))
