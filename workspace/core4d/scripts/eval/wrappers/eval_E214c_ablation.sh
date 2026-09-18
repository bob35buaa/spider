#!/usr/bin/env bash
# E214c eval wrapper: compute the per-part-penetration metric set for full + A5 + A6,
# run the single-variable audit, then generate the part-wise ablation table(s).
#
# Usage:
#   bash workspace/core4d/scripts/eval/wrappers/eval_E214c_ablation.sh [--fresh] [--series full,A5_uniform_hand]
#   (args are forwarded to the eval runner only; for report options like --mean-only,
#    run gen_E214c_ablation_table.py directly.)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=.venv/bin/python
export MUJOCO_GL="${MUJOCO_GL:-egl}"

echo "=== [E214c] eval: per-case metrics (full + A5 + A6) ==="
$PY workspace/core4d/scripts/eval/runners/eval_E214c_ablation.py "$@"

echo "=== [E214c] eval: single-variable audit ==="
$PY workspace/core4d/scripts/experiments/E214c/audit_single_variable.py || true

echo "=== [E214c] report: part-wise table (+ 6-arm) ==="
$PY workspace/core4d/scripts/eval/reports/gen_E214c_ablation_table.py
