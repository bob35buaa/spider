#!/bin/bash
# E215 eval: score rot-aug rollouts, pair vs each case's SAME-arm orig rollout,
# through the public eval.core.core_metrics. Run after run_e215_cem.py finishes.
#
# Usage: bash workspace/core4d/scripts/eval/wrappers/eval_E215_rot_augmentation.sh [--limit N]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO"
MUJOCO_GL="${MUJOCO_GL:-disable}" "${PY:-$REPO/.venv/bin/python}" \
  workspace/core4d/scripts/eval/runners/eval_E215_rot_augmentation.py "$@"
