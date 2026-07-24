#!/usr/bin/env bash
# Render all 16 E179 trajectories and paired E173/E179 videos.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "$PYTHON_BIN" \
  workspace/core4d/scripts/experiments/E179/render_paired_results.py \
  --require-all "$@"
