#!/usr/bin/env bash
# E214 eval wrapper: compute the 15-metric set for all cem_ok ablation rollouts
# (+ box023 E173 baseline), then generate the ablation comparison table.
#
# Usage:
#   bash workspace/core4d/scripts/eval/wrappers/eval_E214_ablation.sh [--fresh]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=.venv/bin/python
export MUJOCO_GL="${MUJOCO_GL:-egl}"

echo "=== [E214] eval: per-case metrics ==="
$PY workspace/core4d/scripts/eval/runners/eval_E214_ablation.py "$@"

echo "=== [E214] eval: single-variable audit ==="
$PY workspace/core4d/scripts/experiments/E214/audit_single_variable.py || true

echo "=== [E214] report: ablation table ==="
$PY workspace/core4d/scripts/eval/reports/gen_E214_ablation_table.py
