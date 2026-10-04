"""Score a ranking file against eval/qrels.jsonl.

A run file is JSONL: {"id": "q001", "ranking": [perfume_id, ...]} (best first).
Metrics at k=10: precision, nDCG (binary gains), MRR. Recall is skipped on purpose:
many queries have hundreds of relevant perfumes, so recall@10 is capped near zero.

Run: python eval/evaluate.py runs/popularity.jsonl
     python eval/evaluate.py --baseline     # writes and scores the popularity baseline
"""
import json
import math
import sys

K = 10


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f) if r}


def score(ranking: list[int], relevant: set[int], k: int = K) -> dict:
    top = ranking[:k]
    hits = [1 if d in relevant else 0 for d in top]
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    first = next((i for i, h in enumerate(hits) if h), None)
    return {"p@10": sum(hits) / k, "ndcg@10": dcg / idcg if idcg else 0.0, "mrr@10": 1 / (first + 1) if first is not None else 0.0}


def mean(per: list[dict]) -> dict:
    return {m: round(sum(p[m] for p in per) / len(per), 4) for m in per[0]}


def evaluate(run: dict, qrels: dict, types: dict | None = None) -> dict:
    """Overall metrics, plus one row per query type when `types` (id -> type) is given."""
    missing = set(qrels) - set(run)
    assert not missing, f"run is missing queries: {sorted(missing)[:5]}"
    per = {q: score(run[q]["ranking"], set(qrels[q]["relevant"])) for q in qrels}
    out = {"all": mean(list(per.values()))}
    for t in sorted(set((types or {}).values())):
        out[t] = mean([s for q, s in per.items() if types[q] == t])
    return out


def popularity_baseline(path: str) -> None:
    """Same ranking for every query: most-voted perfumes first. A floor any real retriever must beat."""
    sys.path.insert(0, "eval")
    from label import corpus
    ranking = [p["id"] for p in sorted(corpus(), key=lambda p: -p["votes"])[:100]]
    with open(path, "w", encoding="utf-8") as f:
        for qid in load("eval/qrels.jsonl"):
            f.write(json.dumps({"id": qid, "ranking": ranking}) + "\n")


def selftest() -> None:
    s = score([1, 2, 3], {1})
    assert s["mrr@10"] == 1.0 and s["ndcg@10"] == 1.0 and s["p@10"] == 0.1
    s = score([9, 1], {1, 2})
    assert s["mrr@10"] == 0.5 and 0 < s["ndcg@10"] < 1
    assert score([9], {1}) == {"p@10": 0.0, "ndcg@10": 0.0, "mrr@10": 0.0}


if __name__ == "__main__":
    selftest()
    path = "runs/popularity.jsonl" if sys.argv[1:] == ["--baseline"] else sys.argv[1]
    if sys.argv[1:] == ["--baseline"]:
        import os
        os.makedirs("runs", exist_ok=True)
        popularity_baseline(path)
    types = {q["id"]: q["type"] for q in load("eval/queries.jsonl").values()}
    for name, metrics in evaluate(load(path), load("eval/qrels.jsonl"), types).items():
        print(f"{path} {name:9s}", metrics)
