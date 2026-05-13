#!/usr/bin/env bash
# E063 one-button: train (parallel) → eval → keyframes.
# Usage: bash workspace/core4d/scripts/run_E063.sh [parallel|serial] [GPU_BOX023] [GPU_BOX025]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-parallel}"
GPU_BOX023="${2:-0}"
GPU_BOX025="${3:-1}"

echo "=== [1/3] train (mode=$MODE, GPU box023=$GPU_BOX023, box025=$GPU_BOX025) ==="
bash workspace/core4d/scripts/train/train_E063.sh "$MODE" "$GPU_BOX023" "$GPU_BOX025"

echo
echo "=== [2/3] eval ==="
.venv/bin/python workspace/core4d/scripts/eval/eval_E063.py

echo
echo "=== [3/3] extract keyframes ==="
bash workspace/core4d/scripts/eval/extract_E063_keyframes.sh

echo
echo "=== E063 done ==="
ls -lh workspace/core4d/results/E063/
