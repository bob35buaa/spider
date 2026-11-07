#!/usr/bin/env bash
# E216 trans 0.1m object-augmentation -- full CEM 8-GPU priority queue.
# One arm per object (box021 PRG / box023 noPRG / box001/004/024 G1A2 /
# bucket003/007 PRG), all encoded in the frozen manifest.  Tiers: P0 (first
# trans0 per object) -> P1 (rest trans0) -> P2 (trans1/trans2).
#
# Coexists with other GPU jobs: only dispatches when free mem >= PER_GPU_MEM_MIB;
# never kills/preempts foreign processes.  Resume-safe.  Run ONE instance (flock).
#
# Prereq: bash workspace/core4d/scripts/train/train_E216.sh FREEZE=1
#
# Usage:
#   bash workspace/core4d/scripts/launch/active/run_E216_local_8gpu.sh
#   DRY_RUN=1 bash .../run_E216_local_8gpu.sh
#   LIMIT=1   bash .../run_E216_local_8gpu.sh
#   CASES=box021_20231011_034_p1 VARIANTS=trans0 bash .../run_E216_local_8gpu.sh
# Env: GPUS, PER_GPU_MEM_MIB, MAX_PER_GPU, POLL_INTERVAL, PER_TASK_TIMEOUT_MIN,
#      DRY_RUN, LIMIT, CASES, VARIANTS, MANIFEST
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
MANIFEST="${MANIFEST:-}"
PY=".venv/bin/python"
Q="workspace/core4d/scripts/experiments/E216/run_E216_cem.py"

EXTRA=""
[[ "$DRY_RUN" == "1" ]] && EXTRA="$EXTRA --dry-run"
[[ "$LIMIT" != "0" ]] && EXTRA="$EXTRA --limit $LIMIT"
[[ -n "$CASES" ]] && EXTRA="$EXTRA --cases $CASES"
[[ -n "$VARIANTS" ]] && EXTRA="$EXTRA --variants $VARIANTS"
[[ -n "$MANIFEST" ]] && EXTRA="$EXTRA --manifest $MANIFEST"

echo "[run_E216] GPUS=$GPUS mem>=$PER_GPU_MEM_MIB timeout=${PER_TASK_TIMEOUT_MIN}min LIMIT=$LIMIT $EXTRA"
exec "$PY" "$Q" --gpus "$GPUS" --per-gpu-mem-mib "$PER_GPU_MEM_MIB" \
  --max-per-gpu "$MAX_PER_GPU" --poll-interval "$POLL_INTERVAL" \
  --per-task-timeout-min "$PER_TASK_TIMEOUT_MIN" $EXTRA
