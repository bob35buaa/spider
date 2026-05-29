#!/usr/bin/env bash
# E096 remote runner: GPU0/GPU1 split. Does not stop other processes.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
LOG_ROOT="logs/E096/remote"
mkdir -p "$LOG_ROOT"

echo "[$(date '+%H:%M:%S')] === E096 remote ${STAGE} launch ==="
( bash workspace/core4d/scripts/train/train_E096_box004_cem.sh remote-gpu0 "$STAGE" 0 > "$LOG_ROOT/remote_gpu0_${STAGE}.log" 2>&1 ) &
PID0=$!
( bash workspace/core4d/scripts/train/train_E096_box004_cem.sh remote-gpu1 "$STAGE" 1 > "$LOG_ROOT/remote_gpu1_${STAGE}.log" 2>&1 ) &
PID1=$!

echo "[$(date '+%H:%M:%S')] launched GPU0 PID=$PID0 GPU1 PID=$PID1"
wait "$PID0"
echo "[$(date '+%H:%M:%S')] GPU0 done"
wait "$PID1"
echo "[$(date '+%H:%M:%S')] GPU1 done"
echo "[$(date '+%H:%M:%S')] === E096 remote ${STAGE} complete ==="
