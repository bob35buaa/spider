#!/usr/bin/env bash
# E070: MJWarp vs MuJoCo ref-control commit parity diagnosis.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

GPU="${1:-0}"
RESULTS=workspace/core4d/results/E070
LOGS=logs/E070
mkdir -p "$RESULTS" "$LOGS"

echo "[$(date '+%H:%M:%S')] === E070 ref-control parity (GPU $GPU) ==="
CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u \
  workspace/core4d/scripts/debug/diagnose_E070_ref_control_parity.py \
  --override core4d_e069w02_box023 \
  --task box023_person1 \
  --device cuda:0 \
  --num-samples 4 \
  --steps 12 \
  --out-dir "$RESULTS" \
  --e069-npz workspace/core4d/results/E069/E069W02_box023.npz \
  2>&1 | tee "$LOGS/ref_control_parity.log"

echo "[$(date '+%H:%M:%S')] E070 done."
