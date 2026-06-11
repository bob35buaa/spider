#!/usr/bin/env bash
# E059 one-shot: train baseline + warm on box023, then eval + keyframes.
#
# Usage:
#   bash workspace/core4d/scripts/run_E059.sh [parallel|serial] [GPU_BASE] [GPU_WARM]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-parallel}"
GPU_BASE="${2:-0}"
GPU_WARM="${3:-1}"

echo "=== E059 step 1/3: train (baseline + warm) ==="
bash workspace/core4d/scripts/train/train_E059.sh "$MODE" "$GPU_BASE" "$GPU_WARM"

echo
echo "=== E059 step 2/3: eval ==="
.venv/bin/python workspace/core4d/scripts/eval/eval_E059.py

echo
echo "=== E059 step 3/3: keyframes ==="
bash workspace/core4d/scripts/eval/extract_E059_keyframes.sh

echo
echo "=== E059 done. Inspect: ==="
ls -lh workspace/core4d/results/E059/
