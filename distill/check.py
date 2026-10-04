"""Quality report for data/teacher.jsonl. Run after a pilot, before the full run.

Checks: English words leaking into Turkish text, brand/name leaks in queries,
scent words in occasion queries, and diversity (share of distinct leading trigrams).

Run: uv run python distill/check.py [--samples 4]
"""
import argparse
import json
import random
import re
import sys
from collections import Counter

sys.path.insert(0, "eval")
sys.path.insert(0, "distill")
from generate import ANGLES  # noqa: E402
from label import corpus  # noqa: E402

# Scent vocabulary that must not appear in occasion queries (Turkish stems).
SCENT = re.compile(r"vanily|odun|narenciye|misk|gül|çiçek|deri|tütün|baharat|amber|\bud\b|paçuli|pudra|tatlı|meyve|"
                   r"lavanta|yasemin|sandal|bergamot|turunç|limon|kehribar|vetiver|tonka|iris|tütsü|reçine|nota|akor")
# Common English scent words; Turkish text should not contain them (proper note names like yuzu/oud are allowed).
ENGLISH = re.compile(r"\b(leather|smoky|woody|fruity|floral|sweet|fresh|spicy|musk|notes?|amber wood|warm|powdery|citrus)\b")


def trigram(s: str) -> str:
    return " ".join(s.lower().split()[:3])


def report(rows: list[dict], by: dict) -> dict:
    n = len(rows)
    eng = sum(1 for r in rows if ENGLISH.search(" ".join([r["description"], *r["queries"].values()]).lower()))
    leak = sum(1 for r in rows if any(x.lower() in " ".join(r["queries"].values()).lower()
                                      for x in (by[r["id"]]["brand"], by[r["id"]]["name"]) if len(x) > 3))
    occ = sum(1 for r in rows if SCENT.search(r["queries"]["occasion"].lower()))
    div = {k: len({trigram(r["queries"][k]) for r in rows}) / n for k in ("note", "style", "occasion")}
    top = Counter(trigram(r["queries"]["occasion"]) for r in rows).most_common(3)
    examples = [e for _, pool in ANGLES for e in pool]
    copied = sum(1 for r in rows if any(e in r["queries"]["occasion"].lower() for e in examples))
    return {"rows": n, "english_words": eng, "brand_leak": leak, "occasion_scent_words": occ,
            "occasion_copied_example": copied,
            "distinct_openings": {k: round(v, 2) for k, v in div.items()}, "top_occasion_openings": top}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=4)
    args = ap.parse_args()
    by = {p["id"]: p for p in corpus()}
    rows = [json.loads(l) for l in open("data/teacher.jsonl", encoding="utf-8") if l.strip()]
    print(json.dumps(report(rows, by), ensure_ascii=False, indent=1))
    random.seed(1)
    for r in random.sample(rows, min(args.samples, len(rows))):
        p = by[r["id"]]
        print(f"\n## {p['brand']} {p['name']} | {', '.join(p['accords'][:3])}")
        print("D:", r["description"])
        for k, v in r["queries"].items():
            print(f"{k}: {v}")
