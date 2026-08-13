#!/usr/bin/env bash
# E198 2x2 factorial eval — MuJoCo headless + thread caps (avoid 192-core BLAS thrash).
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
# eval is physics/SDF only (no rendering) -> do not init any GL backend
unset MUJOCO_GL
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export PYTHONPATH="workspace/core4d/scripts:${PYTHONPATH:-}"
exec .venv/bin/python workspace/core4d/scripts/eval/runners/eval_E198_factorial.py "$@"
