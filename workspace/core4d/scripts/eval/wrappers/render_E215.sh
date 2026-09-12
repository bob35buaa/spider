#!/usr/bin/env bash
# Render E215 rot rollouts as sim-vs-aug-ref MP4s (osmesa CPU, GPUs untouched).
#
# Usage:
#   bash workspace/core4d/scripts/eval/wrappers/render_E215.sh            # all 68 rot rows
#   bash workspace/core4d/scripts/eval/wrappers/render_E215.sh --cases box021_20231011_034_p1 --variants rot0
#   bash workspace/core4d/scripts/eval/wrappers/render_E215.sh --gate     # the 9-case C6 gate sample only
#
# Rendering is CPU work (MUJOCO_GL=osmesa) and does NOT need the GPUs, so the
# keepalive occupancy script /mnt/ali-sh-1/usr/xiayibo/.cache/run.py can stay
# attached. Videos land in results/E215/s6_downstream/render/c6/ (gitignored).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
APP=workspace/core4d/scripts/experiments/E215/render_c6.py

# --gate = keep render_c6's default 9-case gate sample; anything else defaults to
# the full 68-rot sweep. Explicit --cases/--variants pass straight through.
if [[ "${1:-}" == "--gate" ]]; then
  shift
  exec "$PYTHON_BIN" "$APP" "$@"
fi
if [[ "$*" == *"--cases"* || "$*" == *"--variants"* || "$*" == *"--all"* ]]; then
  exec "$PYTHON_BIN" "$APP" "$@"
fi
exec "$PYTHON_BIN" "$APP" --all "$@"
