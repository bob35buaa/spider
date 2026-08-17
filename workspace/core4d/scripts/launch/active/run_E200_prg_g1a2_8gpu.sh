#!/usr/bin/env bash
# E200 arm PRG+G1+A2 -- full CEM 8-GPU priority queue (all feasible E199 trans variants).
# Reuses E199 aug trajectories; scene = gravcomp (G1) sidecar; override = E199 PRG
# override + CLI scene_name(gravcomp) + A2 hand-gate pack. orig baseline (eval) =
# E198 G1A2 arm (all 87 box cases).
#
# Coexists with other GPU jobs: only dispatches when free mem >= PER_GPU_MEM_MIB;
# never kills/preempts foreign processes. Resume-safe. Run ONE instance for this arm.
#
# Prereq: bash workspace/core4d/scripts/train/train_E200.sh   (builds scenes + manifest)
#
# Usage:
#   bash workspace/core4d/scripts/launch/active/run_E200_prg_g1a2_8gpu.sh
#   DRY_RUN=1 bash .../run_E200_prg_g1a2_8gpu.sh          # preview commands
#   LIMIT=1  bash .../run_E200_prg_g1a2_8gpu.sh           # smoke: one case then exit
# Env overrides: GPUS, PER_GPU_MEM_MIB, MAX_PER_GPU, POLL_INTERVAL, DRY_RUN, LIMIT
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

ARM="prg_g1a2"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PER_GPU_MEM_MIB="${PER_GPU_MEM_MIB:-5000}"
MAX_PER_GPU="${MAX_PER_GPU:-1}"
POLL_INTERVAL="${POLL_INTERVAL:-15}"
DRY_RUN="${DRY_RUN:-0}"
LIMIT="${LIMIT:-0}"
PY=".venv/bin/python"
Q="workspace/core4d/scripts/experiments/E200/run_local_priority_queue.py"

EXTRA=""
[[ "$DRY_RUN" == "1" ]] && EXTRA="$EXTRA --dry-run"
[[ "$LIMIT" != "0" ]] && EXTRA="$EXTRA --limit $LIMIT"

echo "[run_E200:$ARM] GPUS=$GPUS PER_GPU_MEM_MIB=$PER_GPU_MEM_MIB MAX_PER_GPU=$MAX_PER_GPU LIMIT=$LIMIT $EXTRA"
exec "$PY" "$Q" --arm "$ARM" --gpus "$GPUS" \
  --per-gpu-mem-mib "$PER_GPU_MEM_MIB" --max-per-gpu "$MAX_PER_GPU" \
  --poll-interval "$POLL_INTERVAL" $EXTRA
