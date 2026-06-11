#!/usr/bin/env bash
# E105 remote runner. Run on spider-remote after code/results gate sync.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

WAVE="${1:-wave1}"
STAGE="${STAGE:-full}"
LOG_ROOT="logs/E105/remote"
mkdir -p "$LOG_ROOT"

echo "[$(date '+%H:%M:%S')] === E105 remote ${WAVE} ${STAGE} launch ==="
( E105_SKIP_EVAL=1 bash workspace/core4d/scripts/train/train_E105_box026_clean_full.sh remote-gpu0 "$WAVE" 0 > "$LOG_ROOT/remote_gpu0_${WAVE}_${STAGE}.log" 2>&1 ) &
PID0=$!
( E105_SKIP_EVAL=1 bash workspace/core4d/scripts/train/train_E105_box026_clean_full.sh remote-gpu1 "$WAVE" 1 > "$LOG_ROOT/remote_gpu1_${WAVE}_${STAGE}.log" 2>&1 ) &
PID1=$!

echo "[$(date '+%H:%M:%S')] launched GPU0 PID=$PID0 GPU1 PID=$PID1"
wait "$PID0"
echo "[$(date '+%H:%M:%S')] GPU0 done"
wait "$PID1"
echo "[$(date '+%H:%M:%S')] GPU1 done"
echo "[$(date '+%H:%M:%S')] === E105 remote ${WAVE} ${STAGE} complete ==="
