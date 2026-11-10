#!/usr/bin/env bash
# main training -> post-processing -> template-defaults training -> its verification
cd "$(dirname "$0")/.."
L=logs/train_main.log
until grep -q "\[run_stage\] exit=" "$L"; do sleep 10; done
grep -q "\[run_stage\] exit=0" "$L" || { echo "MAIN TRAINING FAILED"; exit 1; }
ls runs/main
bash scripts/post_train.sh main
COLUMNS=160 ../.venv-ai/Scripts/python.exe scripts/run_stage.py train_template_defaults -- \
    ../.venv-ai/Scripts/soup.exe train --config configs/template_defaults.yaml --name template_defaults --yes > /dev/null
bash scripts/post_train.sh template_defaults --no-controls
../.venv-ai/Scripts/python.exe scripts/summarize.py > /dev/null
echo "CHAIN DONE"
