#!/usr/bin/env bash
# E183 CPU-only Full27 static-P coverage audit. No CUDA/GPU process is launched.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.."

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TBB_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
WORKERS="${E183_CPU_WORKERS:-4}"
RUNNER="workspace/core4d/scripts/experiments/E183/audit_full27_static_p.py"

"${PYTHON_BIN}" "${RUNNER}" protocol --workers "${WORKERS}"
"${PYTHON_BIN}" "${RUNNER}" query --workers "${WORKERS}"
"${PYTHON_BIN}" "${RUNNER}" score --workers "${WORKERS}"
"${PYTHON_BIN}" "${RUNNER}" aggregate --workers "${WORKERS}"
"${PYTHON_BIN}" "${RUNNER}" visual --workers "${WORKERS}"
"${PYTHON_BIN}" "${RUNNER}" validate --workers "${WORKERS}"
