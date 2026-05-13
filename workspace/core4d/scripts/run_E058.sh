#!/usr/bin/env bash
# E058 one-shot: train baseline + warm, then eval + keyframes.
#
# Usage:
#   bash workspace/core4d/scripts/run_E058.sh [GPU_ID]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

GPU="${1:-0}"

echo "=== E058 step 1/3: train (baseline + warm) on GPU $GPU ==="
bash workspace/core4d/scripts/train/train_E058.sh "$GPU"

echo
echo "=== E058 step 2/3: eval ==="
.venv/bin/python workspace/core4d/scripts/eval/eval_E058.py

echo
echo "=== E058 step 3/3: keyframes ==="
bash workspace/core4d/scripts/eval/extract_E058_keyframes.sh

echo
echo "=== E058 done. Inspect: ==="
ls -lh workspace/core4d/results/E058/
