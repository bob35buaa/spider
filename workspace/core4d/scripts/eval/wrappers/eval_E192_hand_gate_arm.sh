#!/usr/bin/env bash
# Canonical E192 evaluator: baseline_sentinel|canary|full.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
STAGE="${1:-full}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
# Do not force a GL backend: this host's OSMesa import can fail before MuJoCo
# loads (E194 environment evidence).  Render/video is a separate post-step.
unset MUJOCO_GL
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
"$PYTHON_BIN" workspace/core4d/scripts/eval/runners/eval_E192_hand_gate_arm.py "$STAGE"
if [ "$STAGE" != "baseline_sentinel" ]; then
  "$PYTHON_BIN" workspace/core4d/scripts/eval/reports/gen_E192_arm_comparison.py "$STAGE"
fi
