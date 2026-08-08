#!/usr/bin/env bash
# E194 four-cell paired eval entrypoint.
#   bash workspace/core4d/scripts/eval/wrappers/eval_E194_gravcomp_arms.sh [canary|full]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
STAGE="${1:-full}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
# osmesa avoids the GL-backend probe that deadlocks `import mujoco` on this host
# when MUJOCO_GL is unset (verified 2026-08-08). Thread caps avoid the 127-thread
# BLAS thrash on the 192-core shared box.
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
"$PYTHON_BIN" workspace/core4d/scripts/eval/runners/eval_E194_gravcomp_arms.py "$STAGE"
