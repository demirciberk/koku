"""Build the data file for the fly wardrobe demo (demo/fly.json).

A fly goes to three places (office, date, gym) and before each trip picks one perfume
from a wardrobe that is reshuffled every time. Picks must never look wrong, so a
perfume is a "fit" for a scene only when both hold:
  - the fine-tuned BGE-M3 (bge-m3-koku) ranks it in the top FIT_PCT of the pool for
    the scene's Turkish queries (cosine, averaged over query variants), and
  - its Fragrantica accords pass a hand-written rule for the scene (no oud at the gym).
The browser builds each wardrobe from a few fits plus non-fits and samples among the fits,
so variety comes from the wardrobe, correctness from the model plus the rule.

Pool: men's and unisex perfumes (the fly is one character, so its wardrobe stays consistent)
that already carry a student description in the koku replay (../website/demo/koku/data.json),
so the page can show a Turkish description.

Run: uv run python demo/fly.py
"""
import json
import random
import sys

import numpy as np

sys.path.insert(0, "eval")
sys.path.insert(0, "retrieve")
from dense import embed, load_model  # noqa: E402
from label import corpus  # noqa: E402

REPLAY = "../website/demo/koku/data.json"
DOCS = "runs/cache/bge-m3-koku-desc-docs.npy"  # written by retrieve/dense.py bge-m3-koku --docs desc
OUT = "demo/fly.json"
FIT_PCT = 10  # top 10% of the pool by model score
KEEP_FITS = 40  # strongest fits exported per scene
DISTRACTORS = 120

HEAVY = {"oud", "leather", "tobacco", "animalic", "smoky", "honey", "whiskey", "rum", "cannabis",
         "caramel", "chocolate", "cacao", "coffee", "balsamic", "metallic"}
FRESH = {"citrus", "fresh", "green", "aquatic", "marine", "ozonic", "fresh spicy", "aromatic"}

SCENES = {
    "office": {
        "label": "Office",
        "queries": ["ofiste kimseyi rahatsız etmeyecek, yakından duyulan temiz bir koku",
                    "toplantıda ciddi ve güvenilir görünmek istiyorum",
                    "sabah işe giderken sürülecek sade, temiz ve hafif bir parfüm"],
        # clean and quiet: fresh/woody/powdery lead, nothing heavy, not sweet up front
        "rule": lambda a: bool(set(a[:2]) & (FRESH | {"woody", "powdery", "musky", "iris", "lavender", "white floral"}))
        and not set(a) & HEAVY and not set(a[:2]) & {"sweet", "vanilla", "fruity", "tropical"},
    },
    "date": {
        "label": "Date",
        "queries": ["ilk buluşmada çekici ama abartısız bir koku",
                    "akşam yemeği randevusu için sıcak, tatlı ve baharatlı bir parfüm",
                    "sarılınca karşıdakinin üstünde kalan sıcak, çekici bir koku"],
        # warm and close: sweet/amber/spicy/floral in the top three, not a fresh splash
        "rule": lambda a: bool(set(a[:3]) & {"sweet", "vanilla", "amber", "warm spicy", "soft spicy", "rose",
                                              "musky", "powdery", "tuberose", "almond", "patchouli"})
        and a[0] not in FRESH and not set(a) & {"animalic", "cannabis", "metallic", "marine", "ozonic"},
    },
    "gym": {
        "label": "Gym",
        "queries": ["spor salonu sonrası için ferah, yeşil ve narenciye kokan bir şey",
                    "antrenmana giderken sürülecek hafif, enerjik ve temiz bir koku",
                    "terledikten sonra bile bunaltmayan sulu, ferah bir parfüm"],
        # fresh lead, nothing warm or heavy anywhere
        "rule": lambda a: a[0] in FRESH
        and not set(a) & (HEAVY | {"vanilla", "amber", "sweet", "powdery", "warm spicy", "patchouli"}),
    },
}


def main() -> None:
    with open(REPLAY, encoding="utf-8") as f:
        replay = {int(k): v for k, v in json.load(f)["perfumes"].items()}
    docs = corpus()
    D = np.load(DOCS).astype(np.float32)
    assert D.shape[0] == len(docs), "stale cache: rerun retrieve/dense.py"
    pool = [i for i, p in enumerate(docs) if p["gender"] in ("men", "unisex") and replay.get(p["id"], {}).get("d")]

    model = load_model("models/bge-m3-koku")
    scores, fits = {}, {}
    for key, s in SCENES.items():
        Q = embed(model, s["queries"], batch=8).astype(np.float32)
        sc = (Q @ D[pool].T).mean(0)
        cut = np.percentile(sc, 100 - FIT_PCT)
        scores[key] = dict(zip(pool, sc.tolist()))
        ok = [i for i, v in zip(pool, sc) if v >= cut and s["rule"](docs[i]["accords"])]
        fits[key] = sorted(ok, key=lambda i: -scores[key][i])[:KEEP_FITS]
        print(f"{key}: {len(ok)} fits, top: " + " | ".join(f"{docs[i]['name']} ({docs[i]['brand']})" for i in fits[key][:5]))

    keep = set().union(*fits.values())
    rest = [i for i in pool if i not in keep]
    keep |= set(random.Random(0).sample(rest, DISTRACTORS))

    out = {"scenes": {k: {"label": s["label"], "queries": s["queries"]} for k, s in SCENES.items()}, "perfumes": []}
    for i in sorted(keep):
        p, r = docs[i], replay[docs[i]["id"]]
        out["perfumes"].append({"id": p["id"], "n": r["n"], "b": r["b"], "g": r["g"], "a": r["a"], "d": r["d"],
                                "s": [round(scores[k][i], 4) for k in SCENES],
                                "f": [int(i in fits[k]) for k in SCENES]})

    # every scene needs enough fits for varied wardrobes, and enough non-fits to fill them
    for j, k in enumerate(SCENES):
        n = sum(p["f"][j] for p in out["perfumes"])
        assert n >= 20, f"{k}: only {n} fits"
        assert len(out["perfumes"]) - n >= 60, f"{k}: too few distractors"
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {OUT}: {len(out['perfumes'])} perfumes")


if __name__ == "__main__":
    main()
