"""Why did soup ship's leg-1 task metric drop (0.449 -> 0.408) when held-out NLL improved?

Hypothesis to test, not assume: `contains` scoring rewards LONG answers (more text, more
chances to contain the reference), and the base writes ~3x longer answers than the tune.
Generates greedy answers for the 49 ship tasks with base and merged, scores them with
Soup's own score_contains, and records answer lengths per item.
"""
from __future__ import annotations

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from soup_cli.eval.custom import score_contains

BASE, MERGED = "../models/Qwen2.5-0.5B-Instruct", "runs/main/merged"


def main() -> None:
    tasks = [json.loads(l) for l in open("data/ship_tasks.jsonl", encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(BASE)
    res = [{"expected": t["expected"], "category": t["category"]} for t in tasks]
    for name, path in (("base", BASE), ("merged", MERGED)):
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={"": 0}).eval()
        for r, t in zip(res, tasks):
            ids = tok.apply_chat_template([{"role": "user", "content": t["prompt"]}], tokenize=True,
                                          add_generation_prompt=True, return_dict=False)
            out = m.generate(torch.tensor([ids], device="cuda"), max_new_tokens=256, do_sample=False,
                             pad_token_id=tok.eos_token_id)[0, len(ids):]
            text = tok.decode(out, skip_special_tokens=True)
            r[f"{name}_ok"] = bool(score_contains(text, t["expected"]))
            r[f"{name}_words"] = len(text.split())
            r[f"{name}_answer"] = text[:200]
        del m
        torch.cuda.empty_cache()
    n = len(res)
    summary = {
        "n": n,
        "accuracy": {k: sum(r[f"{k}_ok"] for r in res) / n for k in ("base", "merged")},
        "mean_words": {k: sum(r[f"{k}_words"] for r in res) / n for k in ("base", "merged")},
        "base_only_correct": sum(r["base_ok"] and not r["merged_ok"] for r in res),
        "merged_only_correct": sum(r["merged_ok"] and not r["base_ok"] for r in res),
        "words_when_base_only_correct": [(r["base_words"], r["merged_words"]) for r in res
                                         if r["base_ok"] and not r["merged_ok"]],
    }
    json.dump({"summary": summary, "items": res}, open("results/task_inspection.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(json.dumps(summary))
    for r in [r for r in res if r["base_ok"] and not r["merged_ok"]][:4]:
        print("--- expected:", r["expected"], "\n BASE:", r["base_answer"][:140], "\n TUNED:", r["merged_answer"][:140])


if __name__ == "__main__":
    main()
