"""Build the negative controls for verify_artifacts.py from a real adapter.

  zero_B   every lora_B set to 0. Exactly what a run that never updated the adapter
           leaves behind (LoRA initialises B to zero). Soup's `adapter_noop` rule
           targets this case.
  tiny     every lora_B multiplied by 1e-3. Non-zero, so a "is B all zeros?" check
           passes, but the change to the model is negligible: a run that finished,
           whose loss went down a bit, and that changed nothing that matters.

The check is only trustworthy if it says FAIL on both.
Usage: python scripts/make_controls.py --adapter runs/main --out runs/controls
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    src = a.adapter / "adapter_model.safetensors"
    tensors = load_file(src)
    for name, factor in (("zero_B", 0.0), ("tiny", 1e-3)):
        dst = a.out / name
        dst.mkdir(parents=True, exist_ok=True)
        for f in a.adapter.iterdir():
            if f.is_file() and f.suffix in (".json", ".txt", ".jinja", ".model") or f.name.startswith("tokenizer"):
                shutil.copy2(f, dst / f.name)
        out = {k: (v * factor if ".lora_B." in k else v).contiguous() for k, v in tensors.items()}
        save_file(out, dst / "adapter_model.safetensors", metadata={"format": "pt"})
        print(f"{name}: {sum('.lora_B.' in k for k in out)} lora_B tensors x {factor} -> {dst}")


if __name__ == "__main__":
    main()
