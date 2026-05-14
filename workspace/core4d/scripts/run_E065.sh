#!/usr/bin/env bash
# E065 one-button: train (parallel) → eval → keyframes.
# Usage: bash workspace/core4d/scripts/run_E065.sh [parallel|serial] [GPU_A] [GPU_D]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-parallel}"
GPU_A="${2:-0}"
GPU_D="${3:-1}"

echo "=== [1/3] train (mode=$MODE, GPU A=$GPU_A, D=$GPU_D) ==="
bash workspace/core4d/scripts/train/train_E065.sh "$MODE" "$GPU_A" "$GPU_D"

echo
echo "=== [2/3] eval ==="
.venv/bin/python workspace/core4d/scripts/eval/eval_E065.py

echo
echo "=== [3/3] extract keyframes ==="
bash workspace/core4d/scripts/eval/extract_E065_keyframes.sh

echo
echo "=== E065 done ==="
ls -lh workspace/core4d/results/E065/
