"""VRAM budget, computed BEFORE training from the model config alone.

Every term is written out so it can be checked by hand. The actual peaks measured
during the runs are compared to these numbers in REPORT.md.

Usage: python scripts/memory_budget.py --model Qwen/Qwen2.5-1.5B-Instruct \
           --seq 512 --batch 1 --rank 16 [--json out.json]
"""
from __future__ import annotations

import argparse
import json

from transformers import AutoConfig

MiB = 1024 ** 2
GiB = 1024 ** 3


def linear_shapes(c) -> dict[str, tuple[int, int]]:
    """(in, out) of every LoRA-targetable linear layer in one decoder block."""
    h, i = c.hidden_size, c.intermediate_size
    hd = getattr(c, "head_dim", None) or h // c.num_attention_heads
    q_out = c.num_attention_heads * hd
    kv_out = c.num_key_value_heads * hd
    return {"q_proj": (h, q_out), "k_proj": (h, kv_out), "v_proj": (h, kv_out),
            "o_proj": (q_out, h), "gate_proj": (h, i), "up_proj": (h, i), "down_proj": (i, h)}


def budget(model: str, seq: int, batch: int, rank: int, targets: list[str],
           grad_ckpt: bool = True) -> dict:
    c = AutoConfig.from_pretrained(model)
    L, h, V = c.num_hidden_layers, c.hidden_size, c.vocab_size
    shapes = linear_shapes(c)
    linear_params = L * sum(a * b for a, b in shapes.values())
    embed_params = V * h
    tied = bool(getattr(c, "tie_word_embeddings", False))
    lm_head_params = 0 if tied else V * h
    other = L * 2 * h + h + L * (shapes["q_proj"][1] + 2 * shapes["k_proj"][1])  # norms + qkv bias
    total = linear_params + embed_params + lm_head_params + other

    lora_params = L * sum(rank * (a + b) for n, (a, b) in shapes.items() if n in targets)
    tokens = seq * batch
    r = {}

    # ---------------- training (QLoRA: frozen NF4 base, bf16 compute, LoRA in fp32)
    # NF4 = 4 bits/param + double-quantised absmax (~0.127 bits/param with blocksize 64).
    nf4_bytes = linear_params * (4 + 0.127) / 8
    # bitsandbytes leaves embeddings, lm_head and norms un-quantized, and peft's
    # prepare_model_for_kbit_training casts them to fp32 (v1 of this script assumed bf16).
    fp32_rest = (embed_params + lm_head_params + other) * 4
    bf16_rest = (embed_params + lm_head_params + other) * 2   # for the inference artifact
    r["train.base_weights (nf4 linears + fp32 embeddings/norms)"] = nf4_bytes + fp32_rest
    r["train.lora_weights_fp32"] = lora_params * 4
    r["train.lora_grads_fp32"] = lora_params * 4
    r["train.adamw_states_fp32 (m+v)"] = lora_params * 8
    attn_heads = c.num_attention_heads
    # what autograd keeps per layer (bf16): norms, q/k/v/o, MLP gate/up/act, plus the
    # inputs bitsandbytes 4-bit linears keep for backward, plus LoRA dropout outputs
    one_layer = tokens * (
        10 * h                                   # norms, q,k,v,o in/out, residuals
        + 3 * c.intermediate_size                # gate, up, act(gate)*up
        + 2 * c.intermediate_size                # nf4 linears keep their inputs (bnb)
    ) * 2 + batch * attn_heads * seq * seq * 2   # attention probs if not fused SDPA
    if grad_ckpt:
        r["train.checkpointed_hidden (L x tokens x h x bf16)"] = L * tokens * h * 2
        r["train.recompute_one_layer (bf16)"] = one_layer
    else:
        r["train.activations_all_layers (no checkpointing)"] = L * one_layer
    # Logits: under 4-bit the (fp32) head emits fp32 logits; the loss keeps a fp32
    # log-softmax and backward a fp32 gradient: ~3 fp32 copies of tokens x vocab.
    r["train.logits fp32 + log_softmax + grad"] = tokens * V * (4 + 4 + 4)
    r["train.cuda_context (not seen by torch allocator)"] = 400 * MiB
    train_total = sum(v for k, v in r.items() if k.startswith("train."))
    r["train.TOTAL_ESTIMATE"] = train_total

    # ---------------- export
    bf16_full = total * 2
    largest_linear = max(a * b for a, b in shapes.values())
    r["merge.full_bf16_model"] = bf16_full
    r["merge.peak (model + one delta B@A in fp32)"] = bf16_full + largest_linear * 4
    # On-the-fly NF4 quantisation of the merged model when loading with load_in_4bit:
    # quantised layers accumulate, plus one bf16 layer being converted at a time.
    r["quantize.final_nf4_model"] = nf4_bytes + bf16_rest
    r["quantize.peak (nf4 so far + one bf16 linear + context)"] = (
        nf4_bytes + bf16_rest + largest_linear * 2 + 400 * MiB)

    return {"model": model, "params_total": total, "params_linear": linear_params,
            "params_embedding": embed_params, "tied_embeddings": tied,
            "lora_params": lora_params, "seq": seq, "batch": batch, "rank": rank,
            "bytes": r}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--targets", nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    b = budget(a.model, a.seq, a.batch, a.rank, a.targets, grad_ckpt=not a.no_grad_ckpt)
    print(f"model {b['model']}: {b['params_total']/1e9:.3f} B params "
          f"(linear {b['params_linear']/1e9:.3f} B, embedding {b['params_embedding']/1e6:.0f} M, "
          f"tied={b['tied_embeddings']}); LoRA r={a.rank}: {b['lora_params']/1e6:.2f} M params")
    print(f"seq={a.seq} batch={a.batch} gradient_checkpointing={not a.no_grad_ckpt}")
    for k, v in b["bytes"].items():
        print(f"  {k:<58} {v / MiB:9.0f} MiB")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(b, f, indent=2)


if __name__ == "__main__":
    main()
