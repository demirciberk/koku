"""Explanatory figures for the README and the website write-up, all computed from real run files.

Writes assets/pipeline.png, assets/progression.png, assets/teacher-qa.png,
assets/student.png and assets/quantization.png in the same style as eval/plot.py.
Needs the local data/ and models/ outputs from the pipeline (not in git).

Run: uv run python eval/figures.py
"""
import json
import os
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, "eval")
sys.path.insert(0, "retrieve")
sys.path.insert(0, "distill")
from evaluate import evaluate, load  # noqa: E402
from plot import BG, INK, MUTED, RULE, SHADES, fonts, style  # noqa: E402

ACCENT = SHADES[-1]
SERIF, SANS = fonts()


def jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def ndcg(path: str) -> float:
    return evaluate(load(path), load("eval/qrels.jsonl"))["all"]["ndcg@10"]


def header(fig, title: str, sub: str, top: float = 0.95) -> None:
    fig.text(0.04, top, title, family=SERIF, fontsize=22, color=INK, va="top")
    fig.text(0.04, top - 0.085, sub, family=SANS, fontsize=10.5, color=MUTED, va="top")


def save(fig, name: str) -> str:
    out = f"assets/{name}.png"
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    return out


def pipeline() -> str:
    from label import corpus
    n_all = sum(1 for _ in open("data/perfumes.jsonl", encoding="utf-8"))
    n_corpus = len(corpus())
    n_teacher = len(jsonl("data/teacher.jsonl"))
    n_clean = len(jsonl("data/teacher.clean.jsonl"))
    gen_min = float(re.search(r"in ([\d.]+) min", open("data/generate.log", encoding="utf-8").read()).group(1))
    pairs = int(re.search(r"train pairs=(\d+)", open("data/finetune.log", encoding="utf-8").read()).group(1))
    bench = {r["model"]: r for r in json.load(open("runs/gguf-bench.json", encoding="utf-8"))}
    sft = json.load(open("runs/sft-tuned.json", encoding="utf-8"))
    demo = json.load(open("demo/data.json", encoding="utf-8"))
    q4_mb = os.path.getsize("models/gguf/koku-0.6b-q4_k_m.gguf") / 1e6

    fig = plt.figure(figsize=(12, 5.4), dpi=200)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 0.80])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    W, H = 0.17, 0.30

    def box(x, y, title, lines, accent=False):
        ax.add_patch(FancyBboxPatch((x, y - H / 2), W, H, boxstyle="round,pad=0,rounding_size=0.012",
                                    linewidth=1.2 if accent else 1, edgecolor=ACCENT if accent else RULE,
                                    facecolor="#FFFDF8"))
        ax.text(x + 0.012, y + H / 2 - 0.045, title, family=SERIF, fontsize=14, color=INK, va="top")
        ax.text(x + 0.012, y + H / 2 - 0.125, "\n".join(lines), family=SANS, fontsize=8.6, color=MUTED, va="top", linespacing=1.45)

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, color=MUTED, linewidth=1, shrinkA=2, shrinkB=2))

    xs = [0.02, 0.215, 0.41, 0.605, 0.80]
    mid, top, bot = 0.50, 0.77, 0.23
    box(xs[0], mid, "Data", [f"{n_all:,} Fragrantica perfumes", f"{n_corpus:,} with 100+ votes", "183 Turkish eval queries"])
    box(xs[1], mid, "Teacher", ["Qwen3-8B, Q4_K_M, local", f"{n_teacher:,} perfumes x", "description + 3 queries", f"{gen_min / 60:.1f} h on an RTX 4060"])
    box(xs[2], mid, "QA filter", ["drop English, brand leaks,", "scent words in occasion", f"{n_clean:,} clean rows"])
    box(xs[3], top, "Retriever", ["BGE-M3, InfoNCE", f"{pairs:,} query-perfume pairs", f"nDCG@10 {ndcg('runs/dense-bge-m3.jsonl'):.3f} to {ndcg('runs/dense-bge-m3-koku-desc.jsonl'):.3f}"], accent=True)
    box(xs[3], bot, "Student", ["Qwen3-0.6B + LoRA SFT", f"valid JSON 0% to {sft['valid_json']:.0%}", f"foreign notes {sft['desc_with_foreign_note']:.1%}"], accent=True)
    box(xs[4], bot, "Quantize", ["GGUF via llama.cpp", f"Q4_K_M {q4_mb:.0f} MB", f"{bench['q4_k_m']['tokens_per_s_single_stream']:.0f} tokens/s, 1 stream"])
    box(xs[4], top, "Demo", ["static, no server", f"{len(demo['queries'])} queries replayed", f"{len(demo['perfumes']):,} student descriptions"])
    for i in range(2):
        arrow((xs[i] + W, mid), (xs[i + 1], mid))
    arrow((xs[2] + W, mid + 0.05), (xs[3], top))
    arrow((xs[2] + W, mid - 0.05), (xs[3], bot))
    arrow((xs[3] + W, top), (xs[4], top))
    arrow((xs[3] + W, bot), (xs[4], bot))
    arrow((xs[4] + W / 2, bot + H / 2), (xs[4] + W / 2, top - H / 2))
    header(fig, "How koku is built", "One local teacher writes Turkish training data; two small models learn from it. Every number comes from a run file.")
    return save(fig, "pipeline")


def progression() -> str:
    steps = [("Popularity", "runs/popularity.jsonl"), ("BM25", "runs/bm25.jsonl"),
             ("BGE-M3, English docs", "runs/dense-bge-m3.jsonl"), ("+ Turkish note names", "runs/dense-bge-m3-tr.jsonl"),
             ("+ teacher description", "runs/dense-bge-m3-desc.jsonl"), ("+ fine-tuning", "runs/dense-bge-m3-koku-desc.jsonl")]
    vals = [ndcg(p) for _, p in steps]
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    colors = SHADES[:4] + [SHADES[3], ACCENT]
    bars = ax.bar(range(len(steps)), vals, 0.62, color=colors, zorder=3)
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}".lstrip("0"), ha="center", va="bottom",
                family=SANS, fontsize=10, color=INK if i == len(vals) - 1 else MUTED)
        if i >= 3:
            ax.text(b.get_x() + b.get_width() / 2, v / 2, f"+{(v / vals[i - 1] - 1):.0%}", ha="center", va="center",
                    family=SANS, fontsize=9, color="#FFFDF8")
    style(ax)
    ax.set_xticks(range(len(steps)), [s for s, _ in steps], family=SANS, fontsize=10, color=INK)
    ax.set_ylabel("nDCG@10, all queries", family=SANS, color=MUTED, fontsize=10)
    ax.set_ylim(0, max(vals) * 1.18)
    header(fig, "Where the gains came from", "183 Turkish queries over 13,718 perfumes. Translating note names was the biggest single step. Higher is better.")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.1)
    return save(fig, "progression")


def teacher_qa() -> str:
    from check import ENGLISH, SCENT
    from label import corpus
    by = {p["id"]: p for p in corpus()}
    rows = jsonl("data/teacher.jsonl")
    reasons = {"English words left": 0, "brand or name in a query": 0, "scent words in an occasion query": 0}
    for r in rows:  # first failing reason only, so the bars add up to the drop
        p, qs = by[r["id"]], " ".join(r["queries"].values()).lower()
        if ENGLISH.search(r["description"].lower() + " " + qs):
            reasons["English words left"] += 1
        elif any(len(x) > 3 and x.lower() in qs for x in (p["brand"], p["name"])):
            reasons["brand or name in a query"] += 1
        elif SCENT.search(r["queries"]["occasion"].lower()):
            reasons["scent words in an occasion query"] += 1
    kept = len(rows) - sum(reasons.values())
    assert kept == len(jsonl("data/teacher.clean.jsonl")), "filter logic drifted from distill/check.py"
    labels = ["generated"] + [f"dropped: {k}" for k in reasons] + ["kept for training"]
    vals = [len(rows), *reasons.values(), kept]
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    colors = [SHADES[2], "#E9D9D6", "#E9D9D6", "#E9D9D6", ACCENT]
    ax.barh(range(len(vals)), vals, 0.6, color=colors, zorder=3)
    for y, v in enumerate(vals):
        ax.text(v + 120, y, f"{v:,}", va="center", family=SANS, fontsize=10, color=INK)
    ax.set_yticks(range(len(vals)), labels, family=SANS, fontsize=10.5, color=INK)
    ax.invert_yaxis()
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(RULE)
    ax.tick_params(length=0)
    ax.set_xticks([])
    ax.set_xlim(0, len(rows) * 1.12)
    header(fig, "Teacher data, before and after QA", f"Qwen3-8B wrote {len(rows):,} rows with 0 malformed JSON; {len(rows) - kept} rows were dropped by rule-based checks.", top=0.96)
    fig.subplots_adjust(left=0.26, right=0.97, top=0.72, bottom=0.04)
    return save(fig, "teacher-qa")


def student() -> str:
    from sft import split, unfaithful
    base = json.load(open("runs/sft-base.json", encoding="utf-8"))
    tuned = json.load(open("runs/sft-tuned.json", encoding="utf-8"))
    _, held, by = split()
    rows = held[: tuned["n"]]
    teacher_foreign = sum(bool(unfaithful(r["description"], by[r["id"]])) for r in rows) / len(rows)
    names = ["Qwen3-0.6B\nuntuned", "Qwen3-0.6B\n+ LoRA SFT", "Qwen3-8B\nteacher"]
    panels = [("Valid JSON", [base["valid_json"], tuned["valid_json"], 1.0], "higher is better"),
              ("Description names a note\nthe perfume does not have", [None, tuned["desc_with_foreign_note"], teacher_foreign], "lower is better")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=200)
    fig.patch.set_facecolor(BG)
    for ax, (title, vals, note) in zip(axes, panels):
        ax.set_facecolor(BG)
        shown = [v if v is not None else 0 for v in vals]
        ax.bar(range(3), shown, 0.56, color=[SHADES[1], ACCENT, SHADES[3]], zorder=3)
        for i, v in enumerate(vals):
            ax.text(i, (v or 0) + max(shown) * 0.03, "n/a" if v is None else f"{v:.1%}".replace(".0%", "%"),
                    ha="center", va="bottom", family=SANS, fontsize=10, color=INK if i == 1 else MUTED)
        style(ax)
        ax.set_yticks([])
        ax.set_xticks(range(3), names, family=SANS, fontsize=9.5, color=INK)
        ax.set_ylim(0, max(shown) * 1.25)
        ax.set_title(f"{title}  ({note})", family=SANS, fontsize=10.5, color=INK, loc="left", pad=10)
    header(fig, "Distilling 8B into 0.6B", f"{tuned['n']} held-out perfumes the student never trained on, greedy decoding. The untuned model writes no usable JSON.")
    fig.subplots_adjust(left=0.04, right=0.98, top=0.68, bottom=0.14, wspace=0.18)
    return save(fig, "student")


def quantization() -> str:
    bench = json.load(open("runs/gguf-bench.json", encoding="utf-8"))
    names = [r["model"].upper().replace("_K_M", "_K_M") for r in bench]
    size = [os.path.getsize(f"models/gguf/koku-0.6b-{r['model']}.gguf") / 1e6 for r in bench]
    panels = [("Size (MB)", size, "{:.0f}"), ("Description F1 vs teacher", [r["desc_f1_vs_teacher"] for r in bench], "{:.3f}"),
              ("Tokens/s, one stream", [r["tokens_per_s_single_stream"] for r in bench], "{:.0f}")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), dpi=200)
    fig.patch.set_facecolor(BG)
    for ax, (title, vals, fmt) in zip(axes, panels):
        ax.set_facecolor(BG)
        ax.bar(range(len(vals)), vals, 0.56, color=[SHADES[1], SHADES[3], ACCENT], zorder=3)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.03, fmt.format(v), ha="center", va="bottom", family=SANS, fontsize=10,
                    color=INK if i == len(vals) - 1 else MUTED)
        style(ax)
        ax.set_yticks([])
        ax.set_xticks(range(len(vals)), names, family=SANS, fontsize=10, color=INK)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.set_title(title, family=SANS, fontsize=10.5, color=INK, loc="left", pad=10)
    header(fig, "Quantizing the student", f"llama.cpp, {bench[0].get('n', 200)} held-out perfumes. Q8_0 is lossless; Q4_K_M is a third of the size for ~4% F1.")
    fig.subplots_adjust(left=0.03, right=0.98, top=0.68, bottom=0.1, wspace=0.12)
    return save(fig, "quantization")


if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    for f in (pipeline, progression, teacher_qa, student, quantization):
        print("wrote", f())
