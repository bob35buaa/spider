#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
MUJOCO_GL="${MUJOCO_GL:-egl}" .venv/bin/python \
  workspace/core4d/scripts/eval/runners/eval_E168_e167a_metrics.py "$@"
