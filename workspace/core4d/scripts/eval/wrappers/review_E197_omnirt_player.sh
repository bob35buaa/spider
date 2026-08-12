#!/usr/bin/env bash
# Read-only Viser review for E197 OmniRetarget converted trajectories.
# Usage: bash workspace/core4d/scripts/eval/wrappers/review_E197_omnirt_player.sh [--port 8097] [--check]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
APP=workspace/core4d/scripts/eval/review/viser_e197_omnirt_player.py

if ! "$PYTHON_BIN" -c "import mujoco, viser, trimesh" 2>/dev/null; then
  echo "[E197 Omni] missing viewer dependencies; install viser and trimesh into $PYTHON_BIN" >&2
  exit 1
fi
exec "$PYTHON_BIN" "$APP" "$@"
