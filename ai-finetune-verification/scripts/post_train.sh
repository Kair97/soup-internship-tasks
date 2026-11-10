#!/usr/bin/env bash
# Everything after `soup train` for one run, each stage logged by run_stage.py.
# Usage: bash scripts/post_train.sh main            (run dir = runs/main)
#        bash scripts/post_train.sh template_defaults --no-controls
set -u
RUN="$1"; shift
CONTROLS=1; [ "${1:-}" = "--no-controls" ] && CONTROLS=0
cd "$(dirname "$0")/.."
PY=../.venv-ai/Scripts/python.exe
SOUP=../.venv-ai/Scripts/soup.exe
BASE=../models/Qwen2.5-0.5B-Instruct
ST="$PY scripts/run_stage.py"
export COLUMNS=160

# adapter dir: Soup writes the final adapter into the run's output dir
AD=runs/$RUN
$ST merge_${RUN}_bf16 -- $SOUP merge --adapter $AD --base $BASE --output runs/$RUN/merged --dtype bfloat16
$ST merge_${RUN}_fp16 -- $SOUP merge --adapter $AD --base $BASE --output runs/$RUN/merged_fp16   # Soup default dtype
$ST q4_${RUN} -- $SOUP merge --adapter $AD --base $BASE --output runs/$RUN/q4 --save-format 4bit --dtype bfloat16
$ST verify_${RUN} -- $PY scripts/verify_artifacts.py --label $RUN --base $BASE --adapter $AD \
    --merged runs/$RUN/merged --q4 runs/$RUN/q4 --eval data/eval.jsonl --out results/verify_${RUN}.json

if [ "$CONTROLS" = 1 ]; then
  $ST controls_build -- $PY scripts/make_controls.py --adapter $AD --out runs/controls
  for C in zero_B tiny; do
    $ST merge_control_$C -- $SOUP merge --adapter runs/controls/$C --base $BASE --output runs/controls/$C/merged --dtype bfloat16
    $ST verify_control_$C -- $PY scripts/verify_artifacts.py --label control_$C --base $BASE \
        --adapter runs/controls/$C --merged runs/controls/$C/merged --eval data/eval.jsonl \
        --out results/verify_control_$C.json
  done
  $ST silent_checks -- $PY scripts/silent_checks.py --base $BASE --adapter $AD --merged-bf16 runs/$RUN/merged \
      --merged-fp16 runs/$RUN/merged_fp16 --eval data/eval.jsonl --train data/train.jsonl
  $ST measure_step -- $PY scripts/measure_step.py
fi
echo "post_train $RUN done"
