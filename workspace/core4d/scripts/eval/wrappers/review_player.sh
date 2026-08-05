#!/usr/bin/env bash
# Launch the PRG retargeting review player (viser, E170-E188).
#
# Usage:
#   bash workspace/core4d/scripts/eval/wrappers/review_player.sh [E188] [--port 8080] [--check] [...]
#
# --check runs a headless index audit (case counts vs summary.json) and exits.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
APP=workspace/core4d/scripts/eval/review/viser_review_player.py

# Tracking-error red lines, aligned with the E178 numeric gates.
export CORE4D_REVIEW_ROOT_POS_MAX_CM="${CORE4D_REVIEW_ROOT_POS_MAX_CM:-20}"
export CORE4D_REVIEW_ROOT_ORI_MAX_DEG="${CORE4D_REVIEW_ROOT_ORI_MAX_DEG:-20}"
export CORE4D_REVIEW_HAND_POS_MAX_CM="${CORE4D_REVIEW_HAND_POS_MAX_CM:-20}"
export CORE4D_REVIEW_HAND_ORI_MAX_DEG="${CORE4D_REVIEW_HAND_ORI_MAX_DEG:-20}"
export CORE4D_REVIEW_OBJECT_POS_MAX_CM="${CORE4D_REVIEW_OBJECT_POS_MAX_CM:-20}"
export CORE4D_REVIEW_OBJECT_ORI_MAX_DEG="${CORE4D_REVIEW_OBJECT_ORI_MAX_DEG:-10}"

# viser (and its trimesh dep) are listed in pyproject but not always installed.
if ! "$PYTHON_BIN" -c "import viser, trimesh" 2>/dev/null; then
  echo "[review] installing viser (+ trimesh) ..." >&2
  uv pip install viser trimesh || "$PYTHON_BIN" -m pip install viser trimesh
fi

if [[ "${1:-}" =~ ^E[0-9]+$ ]]; then
  exp_id="$1"
  shift
  exec "$PYTHON_BIN" "$APP" --exps "$exp_id" "$@"
fi

exec "$PYTHON_BIN" "$APP" "$@"
