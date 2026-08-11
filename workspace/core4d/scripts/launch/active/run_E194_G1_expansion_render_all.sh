#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
"$PYTHON_BIN" workspace/core4d/scripts/experiments/E194/render_g1_expansion.py "$@"
