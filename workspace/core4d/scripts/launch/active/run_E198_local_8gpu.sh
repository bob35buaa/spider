#!/usr/bin/env bash
# E198 G1+A2 + E192-ext A2 — local 8-GPU priority queue entry.
# Coexists with other GPU jobs: only dispatches when free mem >= PER_GPU_MEM_MIB;
# never kills/preempts foreign processes. Resume-safe.
#
# Usage:
#   MODE=canary                bash run_E198_local_8gpu.sh
#   MODE=full SENTINEL_ONLY=1  bash run_E198_local_8gpu.sh
#   MODE=full                  bash run_E198_local_8gpu.sh
# Env overrides: GPUS, PER_GPU_MEM_MIB, MAX_PER_GPU, POLL_INTERVAL
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

MODE="${MODE:-full}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PER_GPU_MEM_MIB="${PER_GPU_MEM_MIB:-5000}"
MAX_PER_GPU="${MAX_PER_GPU:-1}"
POLL_INTERVAL="${POLL_INTERVAL:-15}"
SENTINEL_ONLY="${SENTINEL_ONLY:-0}"
PY=".venv/bin/python"
Q="workspace/core4d/scripts/experiments/E198/run_local_priority_queue.py"
MAN_DIR="workspace/core4d/results/E198/s6_downstream/manifests"

case "$MODE" in
  canary)   MANIFEST="$MAN_DIR/e198_priority_canary_manifest.tsv"; EXTRA="" ;;
  full)     MANIFEST="$MAN_DIR/e198_priority_full_manifest.tsv"
            if [[ "$SENTINEL_ONLY" == "1" ]]; then EXTRA="--sentinel-only"; else EXTRA=""; fi ;;
  *) echo "unknown MODE=$MODE" >&2; exit 2 ;;
esac

echo "[run_E198] MODE=$MODE GPUS=$GPUS PER_GPU_MEM_MIB=$PER_GPU_MEM_MIB MAX_PER_GPU=$MAX_PER_GPU manifest=$MANIFEST $EXTRA"
exec "$PY" "$Q" --manifest "$MANIFEST" --gpus "$GPUS" \
  --per-gpu-mem-mib "$PER_GPU_MEM_MIB" --max-per-gpu "$MAX_PER_GPU" \
  --poll-interval "$POLL_INTERVAL" $EXTRA
