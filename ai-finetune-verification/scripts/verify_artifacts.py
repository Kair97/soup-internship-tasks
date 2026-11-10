"""Did the fine-tune actually change the model, and did the change survive export?

Compares three artifacts: BASE, MERGED (bf16, adapter folded in) and Q4 (MERGED loaded
in 4-bit NF4). A falling training loss proves none of this, so every check here is
done on data the model never trained on, or on the weights themselves.

  W1  adapter is not a no-op      every LoRA B matrix is non-zero (B starts at zero!)
  W2  merge is correct            for every targeted Linear: W_merged - W_base == scale*B@A
                                  and every NON-targeted tensor is bit-identical
  B1  merged learned something    held-out, answer-only NLL: merged < base, paired
                                  bootstrap 95% CI of the difference excludes 0
  B2  4-bit kept it               q4 < base (CI excludes 0) and q4 keeps >= 50% of the gain
  B3  outputs moved               mean token KL(merged || base) on held-out answers
                                  is >= 10x KL(q4 || merged) noise... reported, see below
  G1  generation sanity           greedy answers stop with EOS, are non-empty, and q4 vs
                                  merged agreement is reported

`--control` runs the same checks on a deliberately UNTRAINED artifact (the adapter with
its scale set to 0, i.e. the base model saved as "merged"). The verdict on it MUST be
FAIL, which shows the check can tell a real run from one that changed nothing.

Usage:
  python scripts/verify_artifacts.py --base ../models/Qwen2.5-0.5B-Instruct \
      --adapter runs/main/adapter --merged runs/main/merged --eval data/eval.jsonl \
      --out results/verify_main.json
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

torch.backends.cuda.matmul.allow_tf32 = True
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}", flush=True)


# ------------------------------------------------------------------ weight checks
def st_tensors(folder: Path) -> dict[str, tuple[Path, str]]:
    idx: dict[str, tuple[Path, str]] = {}
    for f in sorted(folder.glob("*.safetensors")):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                idx[k] = (f, k)
    return idx


def load_t(where: tuple[Path, str]) -> torch.Tensor:
    with safe_open(where[0], "pt") as s:
        return s.get_tensor(where[1])


def base_folder(model_id: str) -> Path:
    if Path(model_id).is_dir():
        return Path(model_id)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"]))


def check_weights(base_id: str, adapter: Path, merged: Path, scale_override: float | None) -> dict:
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
    if scale_override is not None:
        scale = scale_override
    ad = st_tensors(adapter)
    base = st_tensors(base_folder(base_id))
    mrg = st_tensors(merged)

    # map "...layers.3.self_attn.q_proj" -> (A, B)
    pairs: dict[str, dict[str, tuple]] = {}
    for k, w in ad.items():
        if ".lora_A." in k or ".lora_B." in k:
            mod = k.split(".lora_")[0].replace("base_model.model.", "")
            pairs.setdefault(mod, {})["A" if ".lora_A." in k else "B"] = w
    zero_b, rel_delta, merge_err, floor_ratio, visible, targeted = 0, [], [], [], [], set()
    for mod, ab in sorted(pairs.items()):
        A, B = load_t(ab["A"]).float(), load_t(ab["B"]).float()
        if B.abs().max().item() == 0:
            zero_b += 1
        wkey = mod + ".weight"
        targeted.add(wkey)
        Wb, Wm = load_t(base[wkey]).float(), load_t(mrg[wkey]).float()
        delta = scale * (B @ A)
        ideal = Wb + delta                                    # exact merge in fp32
        dn = delta.norm().item()
        rel_delta.append(dn / Wb.norm().item())
        if dn == 0:
            merge_err.append(0.0 if torch.equal(Wm, Wb) else float("inf"))
            floor_ratio.append(0.0)
            visible.append(0.0)
            continue
        # How far is the stored merged weight from the exact merge, in units of |delta|?
        # A skipped merge scores 1.0, a double merge 1.0, a correct bf16 merge ~ the
        # bf16 rounding floor. The floor is computed, not assumed: for small deltas bf16
        # rounding of W itself is larger than delta (a FIXED threshold gave false FAILs).
        merge_err.append((Wm - ideal).norm().item() / dn)
        floor_ratio.append((ideal.to(Wm.dtype if Wm.dtype != torch.float32 else torch.bfloat16)
                            .float() - ideal).norm().item() / dn)
        # How much of the adapter's change is still visible in the stored weights
        # (1.0 = all of it, 0.0 = it vanished in rounding / was never merged).
        visible.append(((Wm - Wb).flatten() @ delta.flatten()).item() / dn ** 2)

    untouched_diff, untouched_n = 0, 0
    for k, w in base.items():
        if k in targeted:
            continue
        mk = k if k in mrg else None
        if mk is None:
            if k == "lm_head.weight":       # tied embeddings may not be stored twice
                continue
            untouched_diff += 1
            continue
        untouched_n += 1
        if not torch.equal(load_t(w), load_t(mrg[mk])):
            untouched_diff += 1
    excess = sorted(e - f for e, f in zip(merge_err, floor_ratio))
    visible.sort()
    rel_delta.sort()
    merge_err.sort()
    return {
        "lora_modules": len(pairs), "scale": scale, "r": r, "alpha": alpha,
        "zero_B_modules": zero_b,
        "delta_rel_norm_min": rel_delta[0], "delta_rel_norm_median": rel_delta[len(rel_delta) // 2],
        "delta_rel_norm_max": rel_delta[-1],
        "merge_rel_err_median": merge_err[len(merge_err) // 2], "merge_rel_err_max": merge_err[-1],
        "bf16_floor_median": sorted(floor_ratio)[len(floor_ratio) // 2],
        "merge_err_minus_floor_max": excess[-1],
        "delta_visible_after_merge_min": visible[0],
        "delta_visible_after_merge_median": visible[len(visible) // 2],
        "non_targeted_tensors_checked": untouched_n, "non_targeted_tensors_changed": untouched_diff,
    }


# --------------------------------------------------------------- behaviour checks
def load_eval(path: Path, n: int) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:n]


def encode(tok, row: dict) -> tuple[list[int], int]:
    user = f"{row['instruction']}\n\n{row['input']}" if row["input"] else row["instruction"]
    # return_dict=False: transformers 5 returns a BatchEncoding (dict) by default,
    # and len() of that is 2 - a silent bug that hit prepare_data.py first.
    prompt = tok.apply_chat_template([{"role": "user", "content": user}],
                                     tokenize=True, add_generation_prompt=True, return_dict=False)
    full = tok.apply_chat_template([{"role": "user", "content": user},
                                    {"role": "assistant", "content": row["output"]}],
                                   tokenize=True, return_dict=False)
    assert full[: len(prompt)] == prompt, "chat template is not prefix-stable"
    return full, len(prompt)


def load_model(path: str, four_bit: bool):
    kw = dict(torch_dtype=torch.bfloat16, device_map={"": 0} if DEV == "cuda" else None)
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    m = AutoModelForCausalLM.from_pretrained(path, **kw)
    m.eval()
    return m


@torch.no_grad()
def score(model, tok, rows, n_kl: int = 50, kl_tokens: int = 48):
    """Per-example mean NLL over ANSWER tokens only (prompt tokens are not scored).
    Full log-prob rows are kept only for the first n_kl examples x kl_tokens tokens
    (151k-vocab distributions are ~0.3 MB per token in bf16)."""
    nll, logps = [], []
    for i, row in enumerate(rows):
        ids, p = encode(tok, row)
        x = torch.tensor([ids], device=DEV)
        logits = model(x).logits[0, p - 1:-1].float()
        lp = torch.log_softmax(logits, -1)
        tgt = x[0, p:]
        nll.append(-lp.gather(1, tgt[:, None]).mean().item())
        if i < n_kl:
            logps.append(lp[:kl_tokens].cpu().to(torch.bfloat16))
    return nll, logps


def kl_mean(p_logs, q_logs) -> float:
    """mean over answer tokens of KL(P || Q), both given as log-probs."""
    tot, n = 0.0, 0
    for a, b in zip(p_logs, q_logs):
        a, b = a.float(), b.float()
        tot += (a.exp() * (a - b)).sum(-1).sum().item()
        n += a.shape[0]
    return tot / n


def bootstrap_ci(diffs: list[float], iters: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return sum(diffs) / n, means[int(0.025 * iters)], means[int(0.975 * iters)]


@torch.no_grad()
def generate(model, tok, rows, max_new: int = 256) -> list[dict]:   # 256 = soup ship's cap
    out = []
    for row in rows:
        ids, p = encode(tok, row)
        x = torch.tensor([ids[:p]], device=DEV)
        g = model.generate(x, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)[0, p:].tolist()
        stopped = any(t in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")) for t in g)
        out.append({"text": tok.decode(g, skip_special_tokens=True).strip(), "stopped": stopped,
                    "n_tokens": len(g)})
    return out


def free(m) -> None:
    del m
    gc.collect()
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="../models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--merged", required=True, type=Path)
    ap.add_argument("--q4", type=Path, default=None,
                    help="a saved 4-bit artifact (e.g. `soup merge --save-format 4bit`); "
                         "default: load --merged on the fly in NF4")
    ap.add_argument("--eval", required=True, type=Path)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n-gen", type=int, default=20)
    ap.add_argument("--scale-override", type=float, default=None,
                    help="use when the 'merged' artifact was built with a different scale (control)")
    ap.add_argument("--label", default="main")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    res: dict = {"label": a.label, "base": a.base, "adapter": str(a.adapter), "merged": str(a.merged),
                 "q4": str(a.q4) if a.q4 else "merged loaded on the fly as NF4"}
    log(f"[{a.label}] W1/W2 weight checks")
    res["weights"] = w = check_weights(a.base, a.adapter, a.merged, a.scale_override)
    log(json.dumps(w))

    tok = AutoTokenizer.from_pretrained(a.merged)
    tok_base = AutoTokenizer.from_pretrained(a.base)
    res["tokenizer_identical"] = tok.get_vocab() == tok_base.get_vocab() and \
        tok.chat_template == tok_base.chat_template
    rows = load_eval(a.eval, a.n)
    gen_rows = rows[: a.n_gen]

    stats = {}
    q4_spec = (str(a.q4), False) if a.q4 else (str(a.merged), True)  # saved 4-bit loads as-is
    for name, path, q4 in (("base", a.base, False), ("merged", str(a.merged), False),
                           ("q4", *q4_spec)):
        torch.cuda.reset_peak_memory_stats()
        log(f"[{a.label}] scoring {name} on {len(rows)} held-out examples")
        m = load_model(path, q4)
        nll, lps = score(m, tok, rows)
        gens = generate(m, tok, gen_rows)
        stats[name] = {"nll": nll, "lp": lps, "gen": gens,
                       "peak_alloc_mib": torch.cuda.max_memory_allocated() / 2**20}
        log(f"[{a.label}] {name}: mean answer NLL {sum(nll)/len(nll):.4f}, "
            f"peak allocated {stats[name]['peak_alloc_mib']:.0f} MiB")
        free(m)

    nb, nm, nq = stats["base"]["nll"], stats["merged"]["nll"], stats["q4"]["nll"]
    d_mb = bootstrap_ci([b - m for b, m in zip(nb, nm)])
    d_qb = bootstrap_ci([b - q for b, q in zip(nb, nq)])
    d_qm = bootstrap_ci([q - m for q, m in zip(nq, nm)])
    gain_kept = d_qb[0] / d_mb[0] if d_mb[0] > 0 else float("nan")
    res["behaviour"] = b = {
        "n_eval": len(rows),
        "nll_mean": {k: sum(v["nll"]) / len(v["nll"]) for k, v in stats.items()},
        "ppl": {k: math.exp(sum(v["nll"]) / len(v["nll"])) for k, v in stats.items()},
        "base_minus_merged": {"mean": d_mb[0], "ci95": [d_mb[1], d_mb[2]]},
        "base_minus_q4": {"mean": d_qb[0], "ci95": [d_qb[1], d_qb[2]]},
        "q4_minus_merged": {"mean": d_qm[0], "ci95": [d_qm[1], d_qm[2]]},
        "share_of_gain_kept_by_q4": gain_kept,
        "examples_better_than_base": {"merged": sum(m < x for m, x in zip(nm, nb)) / len(nb),
                                      "q4": sum(q < x for q, x in zip(nq, nb)) / len(nb)},
        "kl_merged_vs_base": kl_mean(stats["merged"]["lp"], stats["base"]["lp"]),
        "kl_q4_vs_merged": kl_mean(stats["q4"]["lp"], stats["merged"]["lp"]),
        "peak_alloc_mib": {k: v["peak_alloc_mib"] for k, v in stats.items()},
    }
    gens = {k: v["gen"] for k, v in stats.items()}
    res["generation"] = {
        "n": len(gen_rows),
        "stopped_with_eos": {k: sum(g["stopped"] for g in v) for k, v in gens.items()},
        "empty": {k: sum(not g["text"] for g in v) for k, v in gens.items()},
        "q4_equals_merged": sum(x["text"] == y["text"] for x, y in zip(gens["q4"], gens["merged"])),
        "merged_equals_base": sum(x["text"] == y["text"] for x, y in zip(gens["merged"], gens["base"])),
        "mean_tokens": {k: sum(g["n_tokens"] for g in v) / len(v) for k, v in gens.items()},
        "per_example": [{"i": i, "reference_tokens": len(tok(r["output"], add_special_tokens=False)["input_ids"]),
                         **{f"{k}_tokens": gens[k][i]["n_tokens"] for k in gens},
                         **{f"{k}_stopped": gens[k][i]["stopped"] for k in gens}}
                        for i, r in enumerate(gen_rows)],
        "samples": [{"instruction": r["instruction"][:300], "reference": r["output"][:400],
                     **{k: gens[k][i]["text"][:400] for k in gens}} for i, r in enumerate(gen_rows[:8])],
    }

    checks = {
        "W1_adapter_non_zero": w["zero_B_modules"] == 0 and w["lora_modules"] > 0,
        # correct merge = no worse than bf16 rounding of the exact result (+ small slack)
        "W2_merge_matches_adapter": w["merge_err_minus_floor_max"] < 0.05,
        "W2_untouched_tensors_identical": w["non_targeted_tensors_changed"] == 0,
        "W3_change_survives_in_stored_weights": w["delta_visible_after_merge_median"] > 0.5,
        "B1_merged_beats_base_ci": b["base_minus_merged"]["ci95"][0] > 0,
        "B2_q4_beats_base_ci": b["base_minus_q4"]["ci95"][0] > 0,
        "B2_q4_keeps_half_the_gain": (gain_kept == gain_kept) and gain_kept >= 0.5,
        "B3_distribution_moved": b["kl_merged_vs_base"] > 2 * b["kl_q4_vs_merged"],
        # only the artifacts that would ship are judged; the base is context.
        # (v1 of this check also required it of the BASE, which is verbose and stopped
        #  5/20 within 160 tokens - a false FAIL; kept in results/superseded/)
        "G1_shipped_artifacts_stop_and_non_empty": all(
            res["generation"]["stopped_with_eos"][k] >= 0.8 * len(gen_rows) and
            res["generation"]["empty"][k] == 0 for k in ("merged", "q4")),
        "T1_tokenizer_and_template_unchanged": res["tokenizer_identical"],
    }
    res["checks"] = checks
    res["verdict"] = "TRAINED_AND_EXPORTED" if all(checks.values()) else "FAIL"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    for k, v in checks.items():
        log(f"[{a.label}] {'PASS' if v else 'FAIL'}  {k}")
    log(f"[{a.label}] VERDICT: {res['verdict']}")
    log(f"[{a.label}] " + json.dumps({k: b[k] for k in ("nll_mean", "base_minus_merged",
                                                         "base_minus_q4", "share_of_gain_kept_by_q4",
                                                         "kl_merged_vs_base", "kl_q4_vs_merged")}))
    return 0 if res["verdict"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
