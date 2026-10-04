"""Clean the Fragrantica Kaggle dump into data/perfumes.jsonl (one perfume per line).

Run: uv run --with pandas python clean.py
"""
import json
import re

import pandas as pd

SRC = "data/fra_cleaned.csv"
OUT = "data/perfumes.jsonl"

# The file is cp1252, not latin-1: bytes like 0x99/0x92 are the TM sign and a curly apostrophe.
MARKS = re.compile(r"[\u00ae\u2122]")  # (R), TM on trademarked raw-material names


def note_list(cell) -> list[str]:
    if not isinstance(cell, str):
        return []
    out = []
    for n in cell.split(","):
        n = MARKS.sub("", n).replace("\u2019", "'").strip().lower()
        if n and n not in out:
            out.append(n)
    return out


def title(slug: str) -> str:
    return " ".join(w.capitalize() for w in str(slug).replace("-", " ").split())


def clean(df: pd.DataFrame) -> list[dict]:
    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        accords = [a.strip().lower() for a in (r.mainaccord1, r.mainaccord2, r.mainaccord3, r.mainaccord4, r.mainaccord5)
                   if isinstance(a, str) and a.strip()]
        rows.append({
            "id": i,
            "name": title(r.Perfume),
            "brand": title(r.Brand),
            "gender": r.Gender,
            "year": int(r.Year) if pd.notna(r.Year) else None,
            "rating": float(r._5),        # "Rating Value"
            "votes": int(r._6),           # "Rating Count"
            "top": note_list(r.Top),
            "middle": note_list(r.Middle),
            "base": note_list(r.Base),
            "accords": accords,
            "perfumers": [p for p in (r.Perfumer1, r.Perfumer2) if isinstance(p, str) and p.strip() and p != "unknown"],
            "url": r.url,
        })
    return rows


def check(rows: list[dict]) -> None:
    assert len(rows) == 24063, len(rows)
    assert len({r["url"] for r in rows}) == len(rows), "duplicate urls"
    assert all(r["top"] or r["middle"] or r["base"] for r in rows), "perfume without notes"
    assert all(r["accords"] for r in rows), "perfume without accords"
    notes = {n for r in rows for k in ("top", "middle", "base") for n in r[k]}
    bad = [n for n in notes if any(c in n for c in "\x92\x99\u00ae\u2122")]
    assert not bad, bad[:5]
    assert all(0 <= r["rating"] <= 5 for r in rows)


if __name__ == "__main__":
    df = pd.read_csv(SRC, sep=";", encoding="cp1252", decimal=",")
    rows = clean(df)
    check(rows)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    notes = {n for r in rows for k in ("top", "middle", "base") for n in r[k]}
    print(f"wrote {len(rows)} perfumes, {len(notes)} distinct notes -> {OUT}")
    print(json.dumps(rows[0], ensure_ascii=False))
