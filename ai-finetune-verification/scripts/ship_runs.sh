#!/usr/bin/env bash
# Soup's own verdict on: main adapter, the shipped 4-bit artifact (--tuned), and the tiny control.
cd "$(dirname "$0")/.."
PY=../.venv-ai/Scripts/python.exe; SOUP=../.venv-ai/Scripts/soup.exe; BASE=../models/Qwen2.5-0.5B-Instruct
ST="$PY scripts/run_stage.py"; export COLUMNS=160
$ST measure_step -- $PY scripts/measure_step.py
$ST ship_main_adapter -- $SOUP ship --base $BASE --adapter runs/main --task-eval data/ship_tasks.jsonl \
    --task-mode metric --device cuda --output results/ship_main_adapter.json
$ST ship_main_q4 -- $SOUP ship --base $BASE --tuned runs/main/q4 --task-eval data/ship_tasks.jsonl \
    --task-mode metric --device cuda --output results/ship_main_q4.json
$ST ship_control_tiny -- $SOUP ship --base $BASE --adapter runs/controls/tiny --task-eval data/ship_tasks.jsonl \
    --task-mode metric --device cuda --output results/ship_control_tiny.json
$PY scripts/summarize.py > /dev/null
echo "SHIP DONE"
