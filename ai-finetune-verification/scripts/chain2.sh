#!/usr/bin/env bash
# resume after the G1 fix: re-verify main, controls, silent checks, probe, template run
cd "$(dirname "$0")/.."
PY=../.venv-ai/Scripts/python.exe; SOUP=../.venv-ai/Scripts/soup.exe; BASE=../models/Qwen2.5-0.5B-Instruct
ST="$PY scripts/run_stage.py"; export COLUMNS=160
$ST verify_main -- $PY scripts/verify_artifacts.py --label main --base $BASE --adapter runs/main \
    --merged runs/main/merged --q4 runs/main/q4 --eval data/eval.jsonl --out results/verify_main.json
for C in zero_B tiny; do
  [ -d runs/controls/$C/merged ] || $ST merge_control_$C -- $SOUP merge --adapter runs/controls/$C --base $BASE --output runs/controls/$C/merged --dtype bfloat16
  $ST verify_control_$C -- $PY scripts/verify_artifacts.py --label control_$C --base $BASE \
      --adapter runs/controls/$C --merged runs/controls/$C/merged --eval data/eval.jsonl --out results/verify_control_$C.json
done
$ST silent_checks -- $PY scripts/silent_checks.py --base $BASE --adapter runs/main --merged-bf16 runs/main/merged \
    --merged-fp16 runs/main/merged_fp16 --eval data/eval.jsonl --train data/train.jsonl
$ST measure_step -- $PY scripts/measure_step.py
$ST train_template_defaults -- $SOUP train --config configs/template_defaults.yaml --name template_defaults --yes > /dev/null
bash scripts/post_train.sh template_defaults --no-controls
$PY scripts/summarize.py > /dev/null
echo "CHAIN DONE"
