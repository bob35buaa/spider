#!/usr/bin/env bash
# E096b remote runner: GPU0 only. Does not stop other processes.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
LOG_ROOT="logs/E096b/remote"
mkdir -p "$LOG_ROOT"

echo "[$(date '+%H:%M:%S')] === E096b remote ${STAGE} launch ==="
bash workspace/core4d/scripts/train/train_E096b_mask_cem.sh remote-gpu0 "$STAGE" 0 \
  > "$LOG_ROOT/remote_gpu0_${STAGE}.log" 2>&1
echo "[$(date '+%H:%M:%S')] === E096b remote ${STAGE} complete ==="
