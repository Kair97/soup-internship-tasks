"""Part 3: break one link of the chain at a time, without any error, and measure it.

Every variant is scored with the same held-out, answer-only NLL as verify_artifacts.py,
so the numbers are directly comparable. Nothing here raises an exception when the
link is broken: that is the point.

  S1  merge dtype      soup merge's DEFAULT --dtype is float16; the base is bf16.
                       Compare a float16 merge with a bfloat16 merge (+ inf/nan scan).
  S2  QLoRA mismatch   the adapter was trained on an NF4 base. Merging it into the
                       bf16 base gives a model that was never trained as such.
                       Compare: adapter on NF4 base (what training saw) vs merged bf16
                       vs merged->NF4 (what ships).
  S3  chat template    serving the merged model with a raw prompt instead of the chat
                       template (a common serving-config mistake).
  S4  stale tokenizer  the merged folder's tokenizer/template vs the base's.
  S5  train vs held-out  answer NLL on 200 TRAIN rows vs 200 held-out rows, base and
                       merged: how much of the "improvement" is memorisation.

Usage: python scripts/silent_checks.py --adapter runs/main --merged-bf16 runs/main/merged \
          --merged-fp16 runs/main/merged_fp16 --eval data/eval.jsonl --train data/train.jsonl
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEV = "cuda"
NF4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)


def log(m: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {m}", flush=True)


def rows_of(path: Path, n: int) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()][:n]


def user_text(r: dict) -> str:
    return f"{r['instruction']}\n\n{r['input']}" if r["input"] else r["instruction"]


def encode(tok, r: dict, template: bool = True) -> tuple[list[int], int]:
    if template:
        p = tok.apply_chat_template([{"role": "user", "content": user_text(r)}], tokenize=True,
                                    add_generation_prompt=True, return_dict=False)
        full = tok.apply_chat_template([{"role": "user", "content": user_text(r)},
                                        {"role": "assistant", "content": r["output"]}],
                                       tokenize=True, return_dict=False)
        return full, len(p)
    # S3: raw "prompt\n\nanswer" with EOS, no role markers
    p = tok(user_text(r) + "\n\n", add_special_tokens=False)["input_ids"]
    a = tok(r["output"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return p + a, len(p)


@torch.no_grad()
def nll(model, tok, rows, template=True) -> dict:
    vals, nonfinite = [], 0
    for r in rows:
        ids, p = encode(tok, r, template)
        x = torch.tensor([ids], device=DEV)
        logits = model(x).logits[0, p - 1:-1].float()
        if not torch.isfinite(logits).all():
            nonfinite += 1
        lp = torch.log_softmax(logits, -1)
        vals.append(-lp.gather(1, x[0, p:, None]).mean().item())
    finite = [v for v in vals if math.isfinite(v)]
    return {"mean_nll": sum(finite) / max(1, len(finite)), "n": len(vals),
            "nonfinite_examples": nonfinite, "per_example": vals}


def load(path, *, nf4=False, dtype=torch.bfloat16):
    kw = {"device_map": {"": 0}, "torch_dtype": dtype}
    if nf4:
        kw["quantization_config"] = NF4
    return AutoModelForCausalLM.from_pretrained(str(path), **kw).eval()


def free(*ms):
    for m in ms:
        del m
    gc.collect()
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="../models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--merged-bf16", type=Path, required=True)
    ap.add_argument("--merged-fp16", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("results/silent_checks.json"))
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.base)
    ev, tr = rows_of(a.eval, a.n), rows_of(a.train, a.n)
    res: dict = {}

    def run(name, model_fn, rows=ev, template=True):
        m = model_fn()
        torch.cuda.reset_peak_memory_stats()
        r = nll(m, tok, rows, template)
        r["peak_alloc_mib"] = torch.cuda.max_memory_allocated() / 2**20
        free(m)
        log(f"{name:<44} NLL {r['mean_nll']:.4f}  non-finite {r['nonfinite_examples']}  "
            f"peak {r['peak_alloc_mib']:.0f} MiB")
        res[name] = r

    run("base bf16 | held-out", lambda: load(a.base))
    run("merged bf16 | held-out", lambda: load(a.merged_bf16))
    # S1
    run("S1 merged fp16 (soup merge default dtype)", lambda: load(a.merged_fp16, dtype=torch.float16))
    # S2
    run("S2 adapter on NF4 base (as trained)",
        lambda: PeftModel.from_pretrained(load(a.base, nf4=True), str(a.adapter)).eval())
    run("S2 adapter on bf16 base (unmerged)",
        lambda: PeftModel.from_pretrained(load(a.base), str(a.adapter)).eval())
    run("S2 merged bf16 -> NF4 (what ships)", lambda: load(a.merged_bf16, nf4=True))
    # S3
    run("S3 merged bf16, NO chat template", lambda: load(a.merged_bf16), template=False)
    run("S3 base bf16, NO chat template", lambda: load(a.base), template=False)
    # S5
    run("S5 base bf16 | TRAIN rows", lambda: load(a.base), rows=tr)
    run("S5 merged bf16 | TRAIN rows", lambda: load(a.merged_bf16), rows=tr)

    # S4
    mt = AutoTokenizer.from_pretrained(a.merged_bf16)
    res["S4 tokenizer"] = {"vocab_identical": mt.get_vocab() == tok.get_vocab(),
                           "chat_template_identical": mt.chat_template == tok.chat_template,
                           "eos_identical": mt.eos_token_id == tok.eos_token_id,
                           "pad_token": mt.pad_token, "base_pad_token": tok.pad_token}
    log(f"S4 tokenizer {res['S4 tokenizer']}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
