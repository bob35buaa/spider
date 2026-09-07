#!/bin/bash
# E213 Phase C eval: score selected-arm aug rollouts, pair vs each case's
# selected-arm orig rollout. Run after the 4 CEM shards finish + merge_shards.py.
#
# Usage: bash workspace/core4d/scripts/eval/wrappers/eval_E213_aug.sh [--limit N]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO"
MUJOCO_GL="${MUJOCO_GL:-disable}" "${PY:-$REPO/.venv/bin/python}" \
  workspace/core4d/scripts/eval/runners/eval_E213_aug.py "$@"
