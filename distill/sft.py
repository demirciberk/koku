"""Sequence-level distillation: fine-tune a small student (Qwen3-0.6B, LoRA) on the teacher's outputs.

Input  = the perfume's Turkish fields (same compact line the teacher saw, minus the long instructions).
Target = the teacher's JSON (description + three queries) from data/teacher.clean.jsonl.
0.6B is chosen over 1.7B because the end goal is an in-browser demo.

Held-out perfumes (5%, same seeded split as retrieve/finetune.py) are never trained on.
`--eval` generates for held-out perfumes with the base or tuned student and reports
JSON validity, note faithfulness (Turkish note names in the description that are not in the input)
and generation speed.

Run: uv run python distill/sft.py                 # train, saves models/qwen3-0.6b-koku (LoRA adapter)
     uv run python distill/sft.py --eval base     # untuned student on held-out perfumes
     uv run python distill/sft.py --eval tuned
"""
import argparse
import json
import random
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "eval")
sys.path.insert(0, "distill")
from generate import GENDER_TR, TR, parse  # noqa: E402
from label import corpus  # noqa: E402

BASE = "Qwen/Qwen3-0.6B"
OUT = "models/qwen3-0.6b-koku"
INSTR = "Bu parfüm için Türkçe bir tanım ve üç arama sorgusu (note, style, occasion) üret. Yalnızca JSON yaz."


def user_text(p: dict) -> str:
    j = lambda xs, t: ", ".join(t.get(x, x) for x in xs) or "-"
    n = TR["notes"]
    return (f"{INSTR}\nParfüm: {GENDER_TR.get(p['gender'], p['gender'])} | akorlar: {j(p['accords'], TR['accords'])} | "
            f"üst: {j(p['top'], n)} | orta: {j(p['middle'], n)} | alt: {j(p['base'], n)}")


def split(seed: int = 0) -> tuple[list[dict], list[dict], dict[int, dict]]:
    """Train/held-out rows by perfume id; same 5% held-out set as retrieve/finetune.py."""
    by = {p["id"]: p for p in corpus()}
    with open("data/teacher.clean.jsonl", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    ids = sorted(r["id"] for r in rows)
    random.Random(seed).shuffle(ids)
    held = set(ids[: len(ids) // 20])
    return [r for r in rows if r["id"] not in held], [r for r in rows if r["id"] in held], by


def chat(tok, p: dict, answer: str | None = None) -> str:
    msgs = [{"role": "user", "content": user_text(p)}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return prompt if answer is None else prompt + answer + tok.eos_token


def target(r: dict) -> str:
    return json.dumps({"description": r["description"], "queries": r["queries"]}, ensure_ascii=False)


def encode(tok, p: dict, r: dict, max_len: int = 512) -> tuple[list[int], list[int]]:
    """Token ids and labels; prompt tokens are masked (-100) so loss is on the answer only."""
    prompt_ids = tok(chat(tok, p), add_special_tokens=False)["input_ids"]
    full = tok(chat(tok, p, target(r)), add_special_tokens=False)["input_ids"][:max_len]
    labels = [-100] * len(prompt_ids) + full[len(prompt_ids):]
    return full, labels[: len(full)]


# Turkish note names from the translation table, longest first so "sandal ağacı" wins over "sandal".
NOTE_NAMES = sorted({v for v in TR["notes"].values() if len(v) > 3}, key=len, reverse=True)


def unfaithful(desc: str, p: dict) -> list[str]:
    """Note names mentioned in the description that are not among the perfume's own notes/accords."""
    have = " ".join(TR["notes"].get(x, x) for k in ("top", "middle", "base") for x in p[k])
    have += " " + " ".join(TR["accords"].get(a, a) for a in p["accords"])
    text, out = desc.lower(), []
    for name in NOTE_NAMES:
        if re.search(rf"(?<!\w){re.escape(name)}", text):
            text = text.replace(name, " ")
            if name not in have:
                out.append(name)
    return out


@torch.no_grad()
def evaluate(model, tok, held: list[dict], by: dict, n: int, batch: int = 16) -> dict:
    model.eval()
    tok.padding_side = "left"
    rows, valid, bad_notes, tokens, t0 = held[:n], 0, 0, 0, time.time()
    samples = []
    for s in range(0, len(rows), batch):
        chunk = rows[s : s + batch]
        x = tok([chat(tok, by[r["id"]]) for r in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        y = model.generate(**x, max_new_tokens=320, do_sample=False, pad_token_id=tok.pad_token_id)
        for r, out in zip(chunk, y[:, x["input_ids"].shape[1]:]):
            tokens += int((out != tok.pad_token_id).sum())
            text = tok.decode(out, skip_special_tokens=True)
            d = parse(text)
            if d:
                valid += 1
                bad_notes += bool(unfaithful(d["description"], by[r["id"]]))
                if len(samples) < 3:
                    samples.append({"id": r["id"], **d})
    dt = time.time() - t0
    return {"n": len(rows), "valid_json": round(valid / len(rows), 3),
            "desc_with_foreign_note": round(bad_notes / max(valid, 1), 3),
            "tokens_per_s": round(tokens / dt, 1), "samples": samples}


def selftest() -> None:
    p = {"top": ["bergamot"], "middle": ["rose"], "base": ["musk"], "accords": ["woody"]}
    assert unfaithful("Bergamot ve gül ile açılır, misk ile biter.", p) == []
    assert unfaithful("Vanilya ve gül ile açılır.", p) == ["vanilya"]


if __name__ == "__main__":
    selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", choices=["base", "tuned"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    args = ap.parse_args()

    torch.manual_seed(0)
    train, held, by = split()
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda()

    if args.eval:
        if args.eval == "tuned":
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, OUT).merge_and_unload()
        res = evaluate(model, tok, held, by, args.n)
        print(json.dumps(res, ensure_ascii=False, indent=1))
        with open(f"runs/sft-{args.eval}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        sys.exit()

    from peft import LoraConfig, get_peft_model
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    data = [encode(tok, by[r["id"]], r) for r in train]
    print(f"train examples={len(data)} held-out={len(held)} max_len={max(len(x) for x, _ in data)}")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = args.epochs * len(data) // (args.batch * args.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.03 * steps)) * max(0.0, 1 - s / steps))
    pad = tok.pad_token_id
    rng, step, t0 = random.Random(0), 0, time.time()
    model.train()
    for epoch in range(args.epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        for b, s in enumerate(range(0, len(order) - args.batch + 1, args.batch)):
            chunk = [data[i] for i in order[s : s + args.batch]]
            L = max(len(x) for x, _ in chunk)
            ids = torch.tensor([x + [pad] * (L - len(x)) for x, _ in chunk], device="cuda")
            lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in chunk], device="cuda")
            att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x, _ in chunk], device="cuda")
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / args.accum
            loss.backward()
            if (b + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    print(f"epoch {epoch} step {step}/{steps} loss {loss.item() * args.accum:.3f} "
                          f"{(time.time() - t0) / step:.2f}s/step", flush=True)
    model.save_pretrained(OUT)
    print("saved", OUT)
