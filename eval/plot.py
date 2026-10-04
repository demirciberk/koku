"""Results charts for the README: nDCG@10 per query type for every run, and the
Turkish-English vocabulary gap that motivates a trained retriever.

Run: uv run python eval/plot.py
Writes assets/results.png and assets/vocab-gap.png.
"""
import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, "eval")
sys.path.insert(0, "retrieve")
from evaluate import evaluate, load  # noqa: E402

# Runs shown, in order. Missing run files are skipped so the chart grows with the project.
RUNS = [
    ("runs/popularity.jsonl", "Popularity"),
    ("runs/bm25.jsonl", "BM25"),
    ("runs/dense-e5-small.jsonl", "multilingual-e5-small"),
    ("runs/dense-e5-base.jsonl", "multilingual-e5-base"),
    ("runs/dense-bge-m3.jsonl", "BGE-M3"),
]
TYPES = [("note", "Note"), ("style", "Style"), ("occasion", "Occasion"), ("all", "All queries")]

INK, MUTED, RULE, BG = "#1A1A1A", "#77746E", "#E4E1DA", "#FBFAF7"
# Older/weaker runs fade into the background; the newest run carries the accent.
SHADES = ["#D9D5CC", "#B9B3A7", "#8C8579", "#5E584E", "#9F2F2D"]


def fonts() -> tuple[str, str]:
    have = {f.name for f in font_manager.fontManager.ttflist}
    serif = next((f for f in ("Instrument Serif", "Georgia", "DejaVu Serif") if f in have), "serif")
    sans = next((f for f in ("Geist", "Segoe UI", "Helvetica Neue", "DejaVu Sans") if f in have), "sans-serif")
    return serif, sans


def style(ax) -> None:
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)
    ax.grid(axis="y", color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)


def results_chart(serif: str, sans: str) -> str:
    qrels = load("eval/qrels.jsonl")
    types = {q["id"]: q["type"] for q in load("eval/queries.jsonl").values()}
    runs = [(label, evaluate(load(p), qrels, types)) for p, label in RUNS if os.path.exists(p)]
    shades = SHADES[: len(runs) - 1] + [SHADES[-1]]  # newest run always gets the accent

    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    n, width = len(runs), 0.8 / len(runs)
    for i, (label, res) in enumerate(runs):
        xs = [t + (i - (n - 1) / 2) * width for t in range(len(TYPES))]
        vals = [res[k]["ndcg@10"] for k, _ in TYPES]
        bars = ax.bar(xs, vals, width * 0.88, color=shades[i], label=label, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}".lstrip("0"), ha="center", va="bottom",
                    fontsize=8, color=INK if i == n - 1 else MUTED, family=sans)
    style(ax)
    ax.axvline(len(TYPES) - 1.5, color=RULE, linewidth=1, zorder=1)  # per-type groups | overall
    ax.set_xticks(range(len(TYPES)), [t for _, t in TYPES], family=sans, fontsize=11, color=INK)
    ax.set_ylabel("nDCG@10", family=sans, color=MUTED, fontsize=10)
    ax.set_ylim(0, max(max(r[k]["ndcg@10"] for k, _ in TYPES) for _, r in runs) * 1.18)
    leg = ax.legend(frameon=False, loc="upper left", ncol=min(n, 5), fontsize=9.5, prop={"family": sans},
                    bbox_to_anchor=(0, 1.02), handlelength=1.2, columnspacing=1.6)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.text(0.06, 0.965, "Retrieval quality by query type", family=serif, fontsize=22, color=INK, va="top")
    fig.text(0.06, 0.885, "183 Turkish queries over 13,718 perfumes. Higher is better.", family=sans,
             fontsize=10.5, color=MUTED, va="top")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.76, bottom=0.09)
    os.makedirs("assets", exist_ok=True)
    out = "assets/results.png"
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    return out


def vocab_chart(serif: str, sans: str) -> str:
    from bm25 import doc_text, load_queries, tokens
    from label import corpus

    vocab = {t for p in corpus() for t in tokens(doc_text(p))}
    qtype = {q["id"]: q["type"] for q in load("eval/queries.jsonl").values()}
    counts = {k: [0, 0] for k, _ in TYPES[:3]}
    for q in load_queries():
        counts[qtype[q["id"]]][0 if set(tokens(q["query"])) & vocab else 1] += 1

    fig, ax = plt.subplots(figsize=(11, 3.6), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    labels = [t for _, t in TYPES[:3]]
    totals = [sum(counts[k]) for k, _ in TYPES[:3]]
    share = [counts[k][0] / t * 100 for (k, _), t in zip(TYPES[:3], totals)]
    none = [counts[k][1] / t * 100 for (k, _), t in zip(TYPES[:3], totals)]
    ax.barh(labels, share, color="#5E584E", height=0.56, zorder=3, label="shares at least one token")
    ax.barh(labels, none, left=share, color="#E9D9D6", height=0.56, zorder=3, label="shares no token (BM25 falls back to popularity)")
    for y, (k, _) in enumerate(TYPES[:3]):
        ax.text(101.5, y, f"{counts[k][1]} of {totals[y]}  ({none[y]:.0f}%)", va="center", family=sans, fontsize=10, color="#9F2F2D")
    ax.set_xlim(0, 100)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, length=0, labelsize=11)
    ax.set_xticks([])
    ax.invert_yaxis()
    for t in ax.get_yticklabels():
        t.set_family(sans)
    leg = ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=9.5, prop={"family": sans}, bbox_to_anchor=(0, 1.0))
    for t in leg.get_texts():
        t.set_color(INK)
    total = sum(counts[k][1] for k, _ in TYPES[:3])
    fig.text(0.06, 0.95, "Why keyword search fails here", family=serif, fontsize=22, color=INK, va="top")
    fig.text(0.06, 0.80, f"{total} of {sum(totals)} Turkish queries share no word with the English perfume data.",
             family=sans, fontsize=10.5, color=MUTED, va="top")
    fig.subplots_adjust(left=0.1, right=0.84, top=0.58, bottom=0.05)
    out = "assets/vocab-gap.png"
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    return out


if __name__ == "__main__":
    serif, sans = fonts()
    print("fonts:", serif, "/", sans)
    print("wrote", results_chart(serif, sans))
    print("wrote", vocab_chart(serif, sans))
