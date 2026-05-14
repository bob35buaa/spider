#!/usr/bin/env bash
# E068: MJWP init drift diagnosis and optional verification.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-diagnose}"
GPU_BOX023="${2:-0}"
GPU_BOX025="${3:-1}"
DEVICE="${E068_DEVICE:-cpu}"
RESULTS=workspace/core4d/results/E068
LOGS=logs/E068
mkdir -p "$RESULTS" "$LOGS"

if [[ "$MODE" == "diagnose" ]]; then
  echo "[$(date '+%H:%M:%S')] === E068 diagnose init drift (device=$DEVICE) ==="
  MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u \
      workspace/core4d/scripts/debug/diagnose_E068_init_drift.py \
      --override core4d_e062_box023 \
      --task box023_person1 \
      --device "$DEVICE" \
      --num-samples 4 \
      --out-dir "$RESULTS" \
      2>&1 | tee "$LOGS/diagnose_init_drift.log"
  echo "[$(date '+%H:%M:%S')] diagnose done."
  exit 0
fi

if [[ "$MODE" == "verify" ]]; then
  echo "E068 verify is intentionally left disabled until diagnose confirms the init fix."
  echo "Expected follow-up: add core4d_e068_box023/core4d_e068_box025 and mjwp_init_mode=forward."
  exit 2
fi

echo "Unknown MODE=$MODE (expected: diagnose | verify)" >&2
exit 1
