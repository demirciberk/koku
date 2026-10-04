"""Zero-shot dense baselines: off-the-shelf multilingual embedders, no fine-tuning.

Perfumes are embedded from the same English text BM25 sees; Turkish queries are
embedded as-is. Cosine similarity, top 100. Embeddings are cached in runs/cache/.

Run: uv run python retrieve/dense.py e5-small
     uv run python eval/evaluate.py runs/dense-e5-small.jsonl
"""
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


def embed(model: SentenceTransformer, texts: list[str], batch: int) -> np.ndarray:
    return model.encode(texts, batch_size=batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


def run(name: str) -> str:
    model_id, qp, dp = MODELS[name]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_id, device=device)
    if device == "cuda":
        model.half()  # 8 GB laptop GPU: fp16 halves memory, retrieval quality is unaffected in practice
    model.max_seq_length = 256

    docs = corpus()
    os.makedirs("runs/cache", exist_ok=True)
    cache = f"runs/cache/{name}-docs.npy"
    if os.path.exists(cache):
        D = np.load(cache)
    else:
        D = embed(model, [dp + doc_text(p) for p in docs], batch=64)
        np.save(cache, D)
    assert D.shape[0] == len(docs), "stale cache: delete runs/cache and rerun"

    queries = load_queries()
    Q = embed(model, [qp + q["query"] for q in queries], batch=64)
    S = Q.astype(np.float32) @ D.astype(np.float32).T
    out = f"runs/dense-{name}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for q, s in zip(queries, S):
            top = np.argsort(-s)[:TOP]
            f.write(json.dumps({"id": q["id"], "ranking": [docs[i]["id"] for i in top]}) + "\n")
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or list(MODELS)
    for n in names:
        print("wrote", run(n))
