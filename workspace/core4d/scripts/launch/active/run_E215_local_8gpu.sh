#!/usr/bin/env bash
# E215 rot object-augmentation -- full CEM 8-GPU priority queue (76 rot variants).
# Tiers: P0 (first rot0 per object) -> P1 (rest rot0) -> P2 (rot1). One arm per case
# (bucket003 PRG / bucket007 PRG+G1 / box021 PRG / box023 noPRG / box001/004/024 G1A2),
# all encoded in the frozen manifest.
#
# Coexists with other GPU jobs: only dispatches when free mem >= PER_GPU_MEM_MIB;
# never kills/preempts foreign processes. Resume-safe. Run ONE instance (flock).
#
# Prereq: bash workspace/core4d/scripts/train/train_E215.sh   (builds tasks + freezes manifest)
#
# Usage:
#   bash workspace/core4d/scripts/launch/active/run_E215_local_8gpu.sh
#   DRY_RUN=1 bash .../run_E215_local_8gpu.sh              # preview commands
#   LIMIT=1  bash .../run_E215_local_8gpu.sh               # smoke: one variant then exit
#   CASES=box021_20231011_034_p1 VARIANTS=rot0 bash .../run_E215_local_8gpu.sh
# Env: GPUS, PER_GPU_MEM_MIB, MAX_PER_GPU, POLL_INTERVAL, PER_TASK_TIMEOUT_MIN, DRY_RUN, LIMIT, CASES, VARIANTS
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PER_GPU_MEM_MIB="${PER_GPU_MEM_MIB:-5000}"
MAX_PER_GPU="${MAX_PER_GPU:-1}"
POLL_INTERVAL="${POLL_INTERVAL:-15}"
PER_TASK_TIMEOUT_MIN="${PER_TASK_TIMEOUT_MIN:-180}"
DRY_RUN="${DRY_RUN:-0}"
LIMIT="${LIMIT:-0}"
CASES="${CASES:-}"
VARIANTS="${VARIANTS:-}"
PY=".venv/bin/python"
Q="workspace/core4d/scripts/experiments/E215/run_e215_cem.py"

EXTRA=""
[[ "$DRY_RUN" == "1" ]] && EXTRA="$EXTRA --dry-run"
[[ "$LIMIT" != "0" ]] && EXTRA="$EXTRA --limit $LIMIT"
[[ -n "$CASES" ]] && EXTRA="$EXTRA --cases $CASES"
[[ -n "$VARIANTS" ]] && EXTRA="$EXTRA --variants $VARIANTS"

echo "[run_E215] GPUS=$GPUS mem>=$PER_GPU_MEM_MIB timeout=${PER_TASK_TIMEOUT_MIN}min LIMIT=$LIMIT $EXTRA"
exec "$PY" "$Q" --gpus "$GPUS" --per-gpu-mem-mib "$PER_GPU_MEM_MIB" \
  --max-per-gpu "$MAX_PER_GPU" --poll-interval "$POLL_INTERVAL" \
  --per-task-timeout-min "$PER_TASK_TIMEOUT_MIN" $EXTRA
