#!/usr/bin/env bash
# E024 remote runner: GPU0 runs remote_gpu0, GPU1 runs remote_gpu1.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

LOGS="logs/core4d_collab_retarget/E024"
mkdir -p "$LOGS"

echo "[$(date '+%H:%M:%S')] E024 remote start: GPU0=remote_gpu0 GPU1=remote_gpu1"

(
  SKIP_EVAL=1 bash workspace/core4d_collab_retarget/scripts/train/train_E024.sh remote_gpu0 0
) > "$LOGS/remote_gpu0.controller.log" 2>&1 &
PID0=$!

(
  SKIP_EVAL=1 bash workspace/core4d_collab_retarget/scripts/train/train_E024.sh remote_gpu1 1
) > "$LOGS/remote_gpu1.controller.log" 2>&1 &
PID1=$!

echo "[$(date '+%H:%M:%S')] launched remote queues: PID0=$PID0 PID1=$PID1"
set +e
wait "$PID0"
STATUS0=$?
echo "[$(date '+%H:%M:%S')] remote_gpu0 status=$STATUS0"
wait "$PID1"
STATUS1=$?
echo "[$(date '+%H:%M:%S')] remote_gpu1 status=$STATUS1"
set -e

if [ "$STATUS0" -ne 0 ] || [ "$STATUS1" -ne 0 ]; then
  exit 1
fi

echo "[$(date '+%H:%M:%S')] E024 remote done"
