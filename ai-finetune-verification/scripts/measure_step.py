"""Measure what ONE training step really allocates, to compare with the budget.

nvidia-smi only shows the whole card (other processes + CUDA context + PyTorch's cache).
This replays the main run's step outside Soup with the same pieces (NF4 base,
prepare_model_for_kbit_training, LoRA r16 on all linears, gradient checkpointing, bf16
autocast, AdamW) on the WORST-CASE batch, 1 x 512 tokens (batch 2 OOMed twice, see REPORT), and reports
torch.cuda.max_memory_allocated / max_memory_reserved after each phase.

It is a replay, not Soup itself: the gap to Soup's run is part of the analysis.
"""
from __future__ import annotations

import json
import time

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

MiB = 2 ** 20


def snap(tag, out):
    torch.cuda.synchronize()
    out[tag] = {"allocated_mib": round(torch.cuda.memory_allocated() / MiB),
                "max_allocated_mib": round(torch.cuda.max_memory_allocated() / MiB),
                "reserved_mib": round(torch.cuda.memory_reserved() / MiB),
                "max_reserved_mib": round(torch.cuda.max_memory_reserved() / MiB)}
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), tag, out[tag], flush=True)


def main() -> None:
    out: dict = {}
    torch.cuda.reset_peak_memory_stats()
    m = AutoModelForCausalLM.from_pretrained(
        "../models/Qwen2.5-0.5B-Instruct", device_map={"": 0}, torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_use_double_quant=True,
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    snap("1_base_loaded_nf4", out)
    m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=True)
    snap("2_after_prepare_kbit (fp32 upcast of non-quantized params)", out)
    m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                     "gate_proj", "up_proj", "down_proj"]))
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-4)
    # v1 of this probe forgot m.train(): from_pretrained returns eval mode, where
    # gradient checkpointing is NOT applied -> it measured the no-checkpointing peak
    # (3,621 MiB). Kept as results/measure_step_v1_eval_mode_no_checkpointing.json.
    m.train()
    snap("3_lora_attached", out)
    x = torch.randint(0, 150000, (1, 512), device="cuda")   # the main run: batch 1, max_length 512
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = m(input_ids=x, labels=x).loss
    snap("4_forward (peak = activations + logits)", out)
    loss.backward()
    snap("5_backward", out)
    opt.step()
    opt.zero_grad(set_to_none=True)
    snap("6_optimizer_step (Adam states now exist)", out)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = m(input_ids=x, labels=x).loss
    loss.backward()
    opt.step()
    snap("7_second_step (steady state)", out)
    print(json.dumps(out))
    with open("results/measure_step.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
