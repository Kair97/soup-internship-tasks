#!/usr/bin/env bash
cd "$(dirname "$0")/.."
until grep -q "CHAIN DONE" logs/chain2.out; do sleep 10; done
bash scripts/ship_runs.sh
