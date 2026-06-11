#!/usr/bin/env bash
# E085 remote-side worker: run remote split(s) in parallel by GPU.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

mkdir -p logs/E085 workspace/core4d/results/E085

echo "[$(date '+%H:%M:%S')] === E085 remote run start ==="
bash workspace/core4d/scripts/train/train_E085.sh remote-gpu0 0 > logs/E085/remote_gpu0.log 2>&1 &
PID0=$!

wait "$PID0"
STATUS0=$?

echo "[$(date '+%H:%M:%S')] === E085 remote done: gpu0=${STATUS0} ==="
if [ "$STATUS0" -ne 0 ]; then
  exit 1
fi
