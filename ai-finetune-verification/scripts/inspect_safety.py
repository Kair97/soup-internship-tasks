"""Look at the outputs behind `soup ship`'s mini_safety regression (0.925 -> 0.125).

A number from a scorer is not evidence until the raw outputs agree with it. This loads
Soup's own bundled mini_safety items and refusal scorer, generates greedy answers with
the base and with the merged fine-tune, and writes every prompt/answer/score pair.
Harmful-request prompts and the models' answers are truncated in the saved file.
"""
from __future__ import annotations

import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from soup_cli.eval import gate_suites as gs
from soup_cli.utils.diagnose.refusal import looks_like_refusal

BASE = "../models/Qwen2.5-0.5B-Instruct"
MERGED = "runs/main/merged"


def gen_fn(model, tok):
    def gen(prompts):
        for p in prompts:
            ids = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True,
                                          add_generation_prompt=True, return_dict=False)
            x = torch.tensor([ids], device="cuda")
            out = model.generate(x, max_new_tokens=gs.BEHAVIOURAL_MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tok.eos_token_id)[0, len(ids):]
            yield tok.decode(out, skip_special_tokens=True)
    return gen


def main() -> None:
    items = gs.load_suite_items(gs.MINI_SAFETY)
    tok = AutoTokenizer.from_pretrained(BASE)
    rows = {i: {"prompt": it.get("prompt", "")[:160]} for i, it in enumerate(items)}
    scores = {}
    for name, path in (("base", BASE), ("merged", MERGED)):
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={"": 0}).eval()
        outs = list(gen_fn(m, tok)([it.get("prompt", "") for it in items]))
        for i, o in enumerate(outs):
            # the exact predicate soup ship's mini_safety uses (a phrase heuristic)
            rows[i][f"{name}_refused"] = bool(looks_like_refusal(o))
            rows[i][f"{name}_answer"] = o[:240]
        scores[name] = sum(rows[i][f"{name}_refused"] for i in rows) / len(rows)
        print(time.strftime("%H:%M:%S"), name, "refusal score", scores[name], flush=True)
        del m
        torch.cuda.empty_cache()
    flipped = [r for r in rows.values() if r["base_refused"] and not r["merged_refused"]]
    out = {"n": len(rows), "scores": scores, "flipped_refusal_to_compliance": len(flipped),
           "examples": flipped[:5], "all": list(rows.values())}
    with open("results/safety_inspection.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ("n", "scores", "flipped_refusal_to_compliance")}))
    for r in flipped[:3]:
        print("---", r["prompt"][:100], "\n BASE:", r["base_answer"][:150], "\n TUNED:", r["merged_answer"][:150])


if __name__ == "__main__":
    main()
