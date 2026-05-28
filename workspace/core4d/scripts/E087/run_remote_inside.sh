#!/usr/bin/env bash
# E087 remote-side worker: run GPU0/GPU1 splits in parallel.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

mkdir -p logs/E087 workspace/core4d/results/E087

echo "[$(date '+%H:%M:%S')] === E087 remote run start ==="
bash workspace/core4d/scripts/train/train_E087.sh remote-gpu0 0 > logs/E087/remote_gpu0.log 2>&1 &
PID0=$!
bash workspace/core4d/scripts/train/train_E087.sh remote-gpu1 1 > logs/E087/remote_gpu1.log 2>&1 &
PID1=$!

wait "$PID0"
STATUS0=$?
wait "$PID1"
STATUS1=$?

echo "[$(date '+%H:%M:%S')] === E087 remote done: gpu0=${STATUS0} gpu1=${STATUS1} ==="
if [ "$STATUS0" -ne 0 ] || [ "$STATUS1" -ne 0 ]; then
  exit 1
fi

