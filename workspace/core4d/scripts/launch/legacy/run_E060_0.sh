#!/usr/bin/env bash
# E060.0 one-shot: train baseline (E041c) on box023 + bucket005_s2 with
# data-layer-fixed scenes, then eval + keyframes.
#
# Usage:
#   bash workspace/core4d/scripts/run_E060_0.sh [parallel|serial] [GPU_BOX] [GPU_BUCKET]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-parallel}"
GPU_BOX="${2:-0}"
GPU_BUCKET="${3:-1}"

echo "=== E060.0 step 1/3: train (parallel baseline on 2 cases) ==="
bash workspace/core4d/scripts/train/train_E060_0.sh "$MODE" "$GPU_BOX" "$GPU_BUCKET"

echo
echo "=== E060.0 step 2/3: eval ==="
.venv/bin/python workspace/core4d/scripts/eval/eval_E060_0.py

echo
echo "=== E060.0 step 3/3: keyframes ==="
bash workspace/core4d/scripts/eval/extract_E060_0_keyframes.sh

echo
echo "=== E060.0 done. Inspect: ==="
ls -lh workspace/core4d/results/E060/E060_0_*.{npz,mp4,jpg} 2>/dev/null | head
echo
cat workspace/core4d/results/E060/eval_summary.csv
