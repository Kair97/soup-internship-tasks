# LoRA → merge → 4-bit: ship this artifact or not?

Fine-tune **Qwen2.5-0.5B-Instruct** with **Soup** (QLoRA) on 1,700 grounded instruction examples, merge the adapter,
quantize to 4-bit. Then decide, with evidence rather than a loss curve, whether the result should go to users.

| Document | What is in it |
|---|---|
| [REPORT.md](REPORT.md) | the ≤2-page report: memory budget vs. actual, proof of training, silent failures, verdict, surprises |
| [scripts/verify_artifacts.py](scripts/verify_artifacts.py) | **the part-2 check**: base vs. merged vs. 4-bit, with a no-op control that must fail |
| [logs/](logs/) | raw timestamped logs of every stage + raw `nvidia-smi` samples (500 ms) and snapshots |
| [results/](results/) | JSON output of every check, memory budget, silent-failure experiments |

## Hardware actually used
A laptop **RTX 3050, 4 GB VRAM** (Windows 11, driver 610.62, CUDA 12.8 wheels). It is smaller than the
brief's Colab T4 (16 GB), which makes the memory budget a real constraint. Everything was run here and
nothing was copied from elsewhere.

## Reproduce

```bash
# 1. environment (Python 3.12)
python -m venv .venv && .venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/pip install "soup-cli[train]==0.77.2"   # transformers 5.22, peft 0.21, trl 0.29, bitsandbytes 0.50

# 2. model (pinned revision, see ../models/Qwen2.5-0.5B-Instruct/REVISION) and data
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ../models/Qwen2.5-0.5B-Instruct
curl -L -o data/raw/databricks-dolly-15k.jsonl \
  https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl
python scripts/prepare_data.py                 # -> data/train.jsonl, data/eval.jsonl, manifest.json

# 3. budget BEFORE training
python scripts/memory_budget.py --model ../models/Qwen2.5-0.5B-Instruct --seq 512 --batch 1 --rank 16   # first estimates used batch 4 / 2, see REPORT §1

# 4. pipeline, every stage wrapped for raw logs + nvidia-smi (exactly what was run)
python scripts/run_stage.py data_doctor -- soup data doctor data/train.jsonl --model ../models/Qwen2.5-0.5B-Instruct --max-length 512 --show-mask 1
python scripts/run_stage.py train_main  -- soup train --config configs/main.yaml --name main --yes
bash scripts/post_train.sh main      # merge bf16 + fp16, 4-bit, verify, controls, silent checks, memory probe
python scripts/run_stage.py train_template_defaults -- soup train --config configs/template_defaults.yaml --name template_defaults --yes
bash scripts/post_train.sh template_defaults --no-controls
bash scripts/ship_runs.sh            # soup ship on adapter, on the 4-bit artifact, on the tiny control
python scripts/summarize.py          # -> results/SUMMARY.md (every number in REPORT.md)
```

The exact command of every stage is the first line of its file in `logs/`; `logs/stages.jsonl` has the exit code,
wall time and nvidia-smi peak of every stage, including the failed ones (`logs/failed/`).

| folder | content |
|---|---|
| `configs/` | `main.yaml` (the run), `template_defaults.yaml` (Soup's chat-template settings) |
| `scripts/` | data prep, memory budget, stage runner, verification, controls, silent checks, memory probe, summary |
| `data/` | train/eval splits + manifest (hashes, counts) + `ship_tasks.jsonl` |
| `logs/` | raw timestamped stage logs + nvidia-smi CSV/snapshots; `failed/` = the two crashed training attempts |
| `results/` | every check's JSON, `SUMMARY.md`; `superseded/` = earlier, wrong versions of a check |
| `runs/` | adapters + run metadata (merged/4-bit weights are not committed, ~1.4 GB; rebuild with `post_train.sh`) |
