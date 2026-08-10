#!/usr/bin/env bash
# Canonical E195 paired evaluator and report entry point.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
unset MUJOCO_GL
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
"$PYTHON_BIN" workspace/core4d/scripts/eval/runners/eval_E195_stricter_hand_gate.py
"$PYTHON_BIN" workspace/core4d/scripts/eval/reports/gen_E195_comparison.py

