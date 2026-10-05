"""Fine-tune BGE-M3 on teacher-generated (Turkish query, perfume) pairs.

Training pairs: every perfume in data/teacher.clean.jsonl has three teacher queries
(note, style, occasion). The document side is the same 'desc' text used at retrieval
time (Turkish fields + teacher description), so train and test see one format.
The eval set (eval/queries.jsonl) is never read here.

Loss: InfoNCE with in-batch negatives (each batch holds distinct perfumes), temperature 0.05.
Memory (8 GB GPU): the 250k-token embedding matrix and the lower half of the encoder are
frozen, bf16 autocast, gradient checkpointing.

Run: uv run python retrieve/finetune.py
     uv run python retrieve/dense.py ... (see __main__: scores the checkpoint on the eval set)
"""
import argparse
import json
import random
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, "eval")
sys.path.insert(0, "retrieve")
from dense import doc_builder  # noqa: E402
from label import corpus  # noqa: E402

BASE = "BAAI/bge-m3"
OUT = "models/bge-m3-koku"
KEYS = ("note", "style", "occasion")


def pairs(seed: int = 0) -> tuple[list[tuple[str, int]], list[tuple[str, int]], dict[int, str]]:
    """(query, perfume id) pairs split by perfume (5% held out), plus id -> document text."""
    by = {p["id"]: p for p in corpus()}
    text = doc_builder("desc")
    with open("data/teacher.clean.jsonl", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    ids = sorted(r["id"] for r in rows)
    random.Random(seed).shuffle(ids)
    val_ids = set(ids[: len(ids) // 20])
    train, val = [], []
    for r in rows:
        for k in KEYS:
            (val if r["id"] in val_ids else train).append((r["queries"][k], r["id"]))
    return train, val, {i: text(p) for i, p in by.items()}


def batches(train: list[tuple[str, int]], size: int, rng: random.Random):
    """Shuffle, then greedily fill batches with distinct perfumes so no positive is also a negative."""
    pool = train[:]
    rng.shuffle(pool)
    while pool:
        seen, batch, rest = set(), [], []
        for q, i in pool:
            if len(batch) < size and i not in seen:
                batch.append((q, i))
                seen.add(i)
            else:
                rest.append((q, i))
        pool = rest
        if len(batch) == size:  # ponytail: drops the last partial batch per epoch (< size pairs)
            yield batch


def encode(model, tok, texts: list[str], max_len: int) -> torch.Tensor:
    x = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(model.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model(**x).last_hidden_state[:, 0]  # BGE-M3 dense embedding = CLS token
    return F.normalize(h.float(), dim=-1)


@torch.no_grad()
def recall_at(model, tok, val, docs: dict[int, str], k: int = 10) -> float:
    """Held-out recall@k: rank all corpus perfumes for each held-out teacher query."""
    model.eval()
    ids = list(docs)
    D = torch.cat([encode(model, tok, [docs[i] for i in ids[s : s + 128]], 256) for s in range(0, len(ids), 128)])
    pos = {i: n for n, i in enumerate(ids)}
    hits = 0
    for s in range(0, len(val), 128):
        chunk = val[s : s + 128]
        Q = encode(model, tok, [q for q, _ in chunk], 64)
        top = (Q @ D.T).topk(k, dim=1).indices
        hits += sum(pos[i] in t.tolist() for (_, i), t in zip(chunk, top))
    model.train()
    return hits / len(val)


def freeze_lower(model, keep_top: int) -> None:
    for p in model.embeddings.parameters():
        p.requires_grad = False
    layers = model.encoder.layer
    for layer in layers[: len(layers) - keep_top]:
        for p in layer.parameters():
            p.requires_grad = False


def selftest() -> None:
    data = [("a", 1), ("b", 1), ("c", 2), ("d", 3), ("e", 2), ("f", 4)]
    for b in batches(data, 2, random.Random(0)):
        assert len({i for _, i in b}) == len(b)


if __name__ == "__main__":
    selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--keep-top", type=int, default=12, help="trainable top encoder layers (of 24)")
    ap.add_argument("--temp", type=float, default=0.05)
    args = ap.parse_args()

    torch.manual_seed(0)
    train, val, docs = pairs()
    print(f"train pairs={len(train)} val pairs={len(val)} docs={len(docs)}")
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModel.from_pretrained(BASE).cuda()
    # Non-reentrant: with frozen lower layers the checkpointed inputs carry no grad, and the
    # reentrant variant would then silently skip gradients for the trainable layers.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    freeze_lower(model, args.keep_top)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.0f}M")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(train) // args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, 1 - s / steps))

    print(f"val recall@10 before: {recall_at(model, tok, val, docs):.3f}", flush=True)
    rng, step, t0 = random.Random(0), 0, time.time()
    model.train()
    for epoch in range(args.epochs):
        for b in batches(train, args.batch, rng):
            q = encode(model, tok, [x for x, _ in b], 64)
            d = encode(model, tok, [docs[i] for _, i in b], 256)
            logits = q @ d.T / args.temp
            labels = torch.arange(len(b), device=logits.device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{steps} loss {loss.item():.3f} {(time.time() - t0) / step:.2f}s/step", flush=True)
        print(f"val recall@10 after epoch {epoch}: {recall_at(model, tok, val, docs):.3f}", flush=True)

    # Save as a SentenceTransformer so retrieve/dense.py loads it with BGE-M3's CLS pooling
    # (a bare HF checkpoint would silently get mean pooling).
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(BASE, device="cpu")
    st[0].auto_model.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    st.save(OUT)
    print("saved", OUT)
