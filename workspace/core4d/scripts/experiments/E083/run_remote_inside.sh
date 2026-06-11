#!/usr/bin/env bash
# E083 remote-side worker: run GPU0 and GPU1 splits in parallel.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

mkdir -p logs/E083 workspace/core4d/results/E083

echo "[$(date '+%H:%M:%S')] === E083 remote two-GPU run start ==="
bash workspace/core4d/scripts/train/train_E083.sh remote-gpu0 0 > logs/E083/remote_gpu0.log 2>&1 &
PID0=$!
bash workspace/core4d/scripts/train/train_E083.sh remote-gpu1 1 > logs/E083/remote_gpu1.log 2>&1 &
PID1=$!

wait "$PID0"
STATUS0=$?
wait "$PID1"
STATUS1=$?

echo "[$(date '+%H:%M:%S')] === E083 remote done: gpu0=${STATUS0} gpu1=${STATUS1} ==="
if [ "$STATUS0" -ne 0 ] || [ "$STATUS1" -ne 0 ]; then
  exit 1
fi
