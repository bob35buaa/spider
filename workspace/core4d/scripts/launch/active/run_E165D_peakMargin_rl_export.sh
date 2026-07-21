#!/usr/bin/env bash
# E165D: export peakMargin025 three-case rows for downstream RL diagnostics.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PARTNER_PYTHON_BIN="${PARTNER_PYTHON_BIN:-.venv/bin/python}"
CORE4D_RAW_ROOT="${CORE4D_RAW_ROOT:-/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real}"
SMPLX_MODEL_DIR="${SMPLX_MODEL_DIR:-/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/smplx}"
HOLOSOMA_REPO="${HOLOSOMA_REPO:-/home/ubuntu/Workspace/holosoma}"

exec "$PYTHON_BIN" workspace/core4d/scripts/experiments/E165/export_peakMargin_rl_handoff.py \
  --execute-partner \
  --allow-partner-failure \
  --core4d-raw-root "$CORE4D_RAW_ROOT" \
  --smplx-model-dir "$SMPLX_MODEL_DIR" \
  --holosoma-repo "$HOLOSOMA_REPO" \
  --python-bin "$PARTNER_PYTHON_BIN" \
  "$@"
