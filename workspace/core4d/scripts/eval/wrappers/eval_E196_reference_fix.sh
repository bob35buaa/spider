#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SCOPE="${1:-full}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
"$PYTHON_BIN" workspace/core4d/scripts/eval/runners/eval_E196_reference_fix.py "$SCOPE" --require-all
if [ "$SCOPE" = "full" ]; then
  "$PYTHON_BIN" workspace/core4d/scripts/eval/reports/gen_E196_reference_fix_report.py --require-all
  RECALC="${XLSX_RECALC:-/home/ubuntu/.codex/skills/xlsx/scripts/recalc.py}"
  "$PYTHON_BIN" "$RECALC" workspace/core4d/results/E196/s6_downstream/eval/full_reference_fix/E196_reference_fix_comparison.xlsx 120
  "$PYTHON_BIN" workspace/core4d/scripts/eval/reports/sync_E196_reference_fix_provenance.py --require-all --replace
fi
