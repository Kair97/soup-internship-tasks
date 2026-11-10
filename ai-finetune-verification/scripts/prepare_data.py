"""Build the train / held-out split from databricks-dolly-15k.

Choices (justified in REPORT.md):
  * human-written (not distilled from another model), CC BY-SA 3.0;
  * only GROUNDED categories, where the answer can be checked against the given
    context by a human: closed_qa, information_extraction, summarization, classification;
  * English, which the author can evaluate;
  * 2,000 examples: 1,700 train / 300 held-out, length-filtered so that
    prompt + answer fit in --max-tokens with the model's chat template;
  * de-duplicated on a normalised instruction AND context, and the split is checked
    for overlap, so the held-out loss cannot be flattered by leaked rows.

Output: data/train.jsonl, data/eval.jsonl (alpaca format: instruction / input / output)
and data/manifest.json with counts and hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

KEEP = {"closed_qa", "information_extraction", "summarization", "classification"}


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--source", default="data/raw/databricks-dolly-15k.jsonl",
                    help="raw jsonl from huggingface.co/datasets/databricks/databricks-dolly-15k")
    ap.add_argument("--out", default="data")
    ap.add_argument("--train", type=int, default=1700)
    ap.add_argument("--eval", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    ds = [json.loads(l) for l in Path(a.source).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows, seen, dropped = [], set(), Counter()
    for r in ds:
        if r["category"] not in KEEP:
            dropped["category"] += 1
            continue
        instr, ctx, out = r["instruction"].strip(), r["context"].strip(), r["response"].strip()
        if not instr or not out:
            dropped["empty"] += 1
            continue
        key = hashlib.sha1((norm(instr) + "|" + norm(ctx)).encode()).hexdigest()
        if key in seen:
            dropped["duplicate"] += 1
            continue
        user = f"{instr}\n\n{ctx}" if ctx else instr
        n = len(tok.apply_chat_template(
            [{"role": "user", "content": user}, {"role": "assistant", "content": out}],
            tokenize=True, return_dict=False))  # transformers 5 returns a dict otherwise
        if n > a.max_tokens:
            dropped["too_long"] += 1
            continue
        seen.add(key)
        rows.append({"instruction": instr, "input": ctx, "output": out,
                     "category": r["category"], "tokens": n, "key": key})

    random.Random(a.seed).shuffle(rows)
    need = a.train + a.eval
    assert len(rows) >= need, f"only {len(rows)} usable rows"
    # stratify the held-out set by category so every category is evaluated
    by_cat: dict[str, list] = {}
    for r in rows[:need]:
        by_cat.setdefault(r["category"], []).append(r)
    eval_rows, train_rows = [], []
    for cat, items in by_cat.items():
        k = round(a.eval * len(items) / need)
        eval_rows += items[:k]
        train_rows += items[k:]
    while len(eval_rows) > a.eval:
        train_rows.append(eval_rows.pop())
    while len(eval_rows) < a.eval:
        eval_rows.append(train_rows.pop())
    train_rows, eval_rows = train_rows[: a.train], eval_rows[: a.eval]

    overlap = {r["key"] for r in train_rows} & {r["key"] for r in eval_rows}
    assert not overlap, f"{len(overlap)} rows in both splits"

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train_rows), ("eval", eval_rows)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps({k: r[k] for k in ("instruction", "input", "output", "category")},
                                   ensure_ascii=False) + "\n")
    manifest = {
        "source": "databricks/databricks-dolly-15k (CC BY-SA 3.0)",
        "categories": sorted(KEEP), "seed": a.seed, "max_tokens": a.max_tokens,
        "usable_rows": len(rows), "dropped": dict(dropped),
        "train": len(train_rows), "eval": len(eval_rows), "train_eval_overlap": 0,
        "train_categories": dict(Counter(r["category"] for r in train_rows)),
        "eval_categories": dict(Counter(r["category"] for r in eval_rows)),
        "train_tokens_mean": round(sum(r["tokens"] for r in train_rows) / len(train_rows), 1),
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (out / "train.jsonl", out / "eval.jsonl")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
