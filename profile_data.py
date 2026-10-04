"""Profile the Fragrantica Kaggle dump: size, missing fields, note/accord distribution.

Run: uv run --with pandas python profile_data.py
"""
from collections import Counter

import pandas as pd

# fra_cleaned.csv: ';' separated, decimal comma. Try utf-8 first; latin-1 only if it fails.
try:
    c = pd.read_csv("data/fra_cleaned.csv", sep=";", encoding="utf-8", decimal=",")
    enc = "utf-8"
except UnicodeDecodeError:
    c = pd.read_csv("data/fra_cleaned.csv", sep=";", encoding="latin-1", decimal=",")
    enc = "latin-1"
p = pd.read_csv("data/fra_perfumes.csv")

print(f"fra_cleaned: {len(c)} rows, encoding={enc}, unique urls={c.url.nunique()}")
print(f"fra_perfumes: {len(p)} rows, unique urls={p.url.nunique()}")
print("overlap by url:", len(set(c.url) & set(p.url)))

notes = lambda col: Counter(n.strip() for s in c[col].fillna("") for n in str(s).split(",") if n.strip())
top, mid, base = notes("Top"), notes("Middle"), notes("Base")
allnotes = top + mid + base
empty = ((c.Top.fillna("").str.strip() == "") & (c.Middle.fillna("").str.strip() == "") & (c.Base.fillna("").str.strip() == "")).sum()
print(f"\nnotes: {len(allnotes)} distinct, perfumes with no notes at all: {empty}")
per = c[["Top", "Middle", "Base"]].fillna("").apply(lambda r: sum(len([x for x in str(v).split(',') if x.strip()]) for v in r), axis=1)
print("notes per perfume: median", per.median(), "p10", per.quantile(.1), "p90", per.quantile(.9))
print("top 25 notes:", allnotes.most_common(25))
print("singleton notes:", sum(1 for v in allnotes.values() if v == 1))

acc = Counter(a for col in [f"mainaccord{i}" for i in range(1, 6)] for a in c[col].dropna())
print(f"\naccords: {len(acc)} distinct:", acc.most_common(30))
print("\ngender:", c.Gender.value_counts().to_dict())
print("year:", c.Year.describe()[["min", "50%", "max"]].to_dict(), "| >=2020:", int((c.Year >= 2020).sum()))
print("rating count: median", c["Rating Count"].median(), "| >=100 votes:", int((c["Rating Count"] >= 100).sum()))
print("brands:", c.Brand.nunique(), c.Brand.value_counts().head(8).to_dict())

# mojibake check on names/notes (UTF-8 bytes decoded as latin-1)
moj = c.apply(lambda r: any(s in " ".join(map(str, r.values)) for s in ("Ã", "â€")), axis=1).sum()
print("\nrows with mojibake markers:", int(moj))
print("non-ascii note samples:", [n for n in allnotes if not n.isascii()][:15])

print("\nfra_perfumes description length (chars): median", int(p.Description.str.len().median()))
print("sample description:", p.Description.iloc[0][:400])
