#!/usr/bin/env bash
# E202 bucket object augmentation -- full CEM local 8-GPU priority queue.
# Reuses the E199 manifest-driven priority queue (all E202 run params live in the
# manifest rows). Coexists with other GPU jobs: only dispatches when free mem >=
# PER_GPU_MEM_MIB; never kills/preempts foreign processes. Resume-safe. All rows
# are P1 (translation); the 27 orig baselines are reused from E178 (not queued).
#
# Usage:
#   bash workspace/core4d/scripts/launch/active/run_E202_local_8gpu.sh
#   DRY_RUN=1 bash workspace/core4d/scripts/launch/active/run_E202_local_8gpu.sh
# Env overrides: GPUS, PER_GPU_MEM_MIB, MAX_PER_GPU, POLL_INTERVAL, DRY_RUN
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PER_GPU_MEM_MIB="${PER_GPU_MEM_MIB:-5000}"
MAX_PER_GPU="${MAX_PER_GPU:-1}"
POLL_INTERVAL="${POLL_INTERVAL:-15}"
DRY_RUN="${DRY_RUN:-0}"
PY=".venv/bin/python"
Q="workspace/core4d/scripts/experiments/E199/run_local_priority_queue.py"
MANIFEST="workspace/core4d/results/E202/s6_downstream/manifests/e202_bucket_priority_manifest.tsv"

EXTRA=""
[[ "$DRY_RUN" == "1" ]] && EXTRA="--dry-run"

echo "[run_E202] GPUS=$GPUS PER_GPU_MEM_MIB=$PER_GPU_MEM_MIB MAX_PER_GPU=$MAX_PER_GPU manifest=$MANIFEST $EXTRA"
exec "$PY" "$Q" --manifest "$MANIFEST" --gpus "$GPUS" \
  --per-gpu-mem-mib "$PER_GPU_MEM_MIB" --max-per-gpu "$MAX_PER_GPU" \
  --poll-interval "$POLL_INTERVAL" $EXTRA
