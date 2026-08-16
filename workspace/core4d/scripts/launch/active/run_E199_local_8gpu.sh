#!/usr/bin/env bash
# E199 OmniRetarget object augmentation -- full CEM local 8-GPU priority queue.
# Coexists with other GPU jobs: only dispatches when free mem >= PER_GPU_MEM_MIB;
# never kills/preempts foreign processes. Resume-safe. Tiers P0(orig)->P1(trans)->P2(rot).
#
# Usage:
#   bash workspace/core4d/scripts/launch/active/run_E199_local_8gpu.sh                       # pilot manifest
#   SCOPE=box_fullscale bash workspace/core4d/scripts/launch/active/run_E199_local_8gpu.sh   # plan229 full-scale manifest
# Env overrides: SCOPE(pilot|box_fullscale), GPUS, PER_GPU_MEM_MIB, MAX_PER_GPU, POLL_INTERVAL, DRY_RUN
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

SCOPE="${SCOPE:-pilot}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PER_GPU_MEM_MIB="${PER_GPU_MEM_MIB:-5000}"
MAX_PER_GPU="${MAX_PER_GPU:-1}"
POLL_INTERVAL="${POLL_INTERVAL:-15}"
DRY_RUN="${DRY_RUN:-0}"
PY=".venv/bin/python"
Q="workspace/core4d/scripts/experiments/E199/run_local_priority_queue.py"
if [[ "$SCOPE" == "box_fullscale" ]]; then
  MANIFEST="workspace/core4d/results/E199/s6_downstream/manifests/e199_fullscale_priority_manifest.tsv"
else
  MANIFEST="workspace/core4d/results/E199/s6_downstream/manifests/e199_priority_full_manifest.tsv"
fi

EXTRA=""
[[ "$DRY_RUN" == "1" ]] && EXTRA="--dry-run"

echo "[run_E199] scope=$SCOPE GPUS=$GPUS PER_GPU_MEM_MIB=$PER_GPU_MEM_MIB MAX_PER_GPU=$MAX_PER_GPU manifest=$MANIFEST $EXTRA"
exec "$PY" "$Q" --manifest "$MANIFEST" --gpus "$GPUS" \
  --per-gpu-mem-mib "$PER_GPU_MEM_MIB" --max-per-gpu "$MAX_PER_GPU" \
  --poll-interval "$POLL_INTERVAL" $EXTRA
