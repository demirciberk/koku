"""Lexical baseline: BM25 over perfume text, queries used as-is (Turkish).

Documents are English (name, brand, gender, accords, notes), so Turkish queries
only match on shared tokens (brand names, loanwords like "vanilya" vs "vanilla" do NOT match).
That gap is the point of this baseline: it measures what plain keyword search gets.
Ties (including all-zero scores) break by vote count, so a query with no matching
token falls back to the popularity ranking.

Run: python retrieve/bm25.py && python eval/evaluate.py runs/bm25.jsonl
"""
import json
import math
import os
import re
import sys
from collections import Counter

sys.path.insert(0, "eval")
from label import corpus  # noqa: E402

K1, B, TOP = 1.5, 0.75, 100
WORD = re.compile(r"\w+")


def tokens(text: str) -> list[str]:
    # Turkish dotted/dotless I before lower(), otherwise "İ" lowers to "i" + combining dot.
    return WORD.findall(text.replace("İ", "i").replace("I", "ı").lower())


def doc_text(p: dict) -> str:
    return " ".join([p["name"], p["brand"], p["gender"], *p["accords"], *p["top"], *p["middle"], *p["base"]])


class BM25:
    def __init__(self, docs: list[list[str]]):
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / len(docs)
        df = Counter(t for d in self.tf for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, dl in zip(self.tf, self.len):
            s = 0.0
            for t in query:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (K1 + 1) / (f + K1 * (1 - B + B * dl / self.avg))
            out.append(s)
        return out


def load_queries(path: str = "eval/queries.jsonl") -> list[dict]:
    # The retriever may only see the query text; constraints are the answer key.
    with open(path, encoding="utf-8") as f:
        return [{"id": q["id"], "query": q["query"]} for q in map(json.loads, f)]


def selftest() -> None:
    assert tokens("İSTANBUL ılık Işık") == ["istanbul", "ılık", "ışık"]
    bm = BM25([["vanilla", "musk"], ["rose", "rose", "oud"], ["citrus"]])
    s = bm.scores(["rose"])
    assert s[1] > 0 and s[0] == s[2] == 0
    assert set(load_queries()[0]) == {"id", "query"}, "constraints must not reach the retriever"


if __name__ == "__main__":
    selftest()
    docs = corpus()
    bm = BM25([tokens(doc_text(p)) for p in docs])
    os.makedirs("runs", exist_ok=True)
    with open("runs/bm25.jsonl", "w", encoding="utf-8") as f:
        for q in load_queries():
            s = bm.scores(tokens(q["query"]))
            order = sorted(range(len(docs)), key=lambda i: (-s[i], -docs[i]["votes"]))[:TOP]
            f.write(json.dumps({"id": q["id"], "ranking": [docs[i]["id"] for i in order]}) + "\n")
    print("wrote runs/bm25.jsonl")
