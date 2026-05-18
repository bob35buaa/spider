#!/usr/bin/env bash
# Run E012 remote GPU queues inside an already-launched remote tmux session.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

LOGS="${LOGS:-logs/core4d_collab_retarget/E012}"
mkdir -p "$LOGS"

echo "[$(date '+%H:%M:%S')] === E012 remote tmux worker start ==="

SKIP_EVAL=1 bash workspace/core4d_collab_retarget/scripts/train/train_E012.sh remote_gpu0 0 \
  > "$LOGS/remote_gpu0.log" 2>&1 &
PID0=$!

SKIP_EVAL=1 bash workspace/core4d_collab_retarget/scripts/train/train_E012.sh remote_gpu1 1 \
  > "$LOGS/remote_gpu1.log" 2>&1 &
PID1=$!

set +e
wait "$PID0"
STATUS0=$?
wait "$PID1"
STATUS1=$?
set -e

echo "[$(date '+%H:%M:%S')] gpu0 status=${STATUS0} gpu1 status=${STATUS1}"

if [ "$STATUS0" -eq 0 ] && [ "$STATUS1" -eq 0 ]; then
  .venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E012.py \
    | tee "$LOGS/eval_remote.log"
fi

if [ "$STATUS0" -ne 0 ] || [ "$STATUS1" -ne 0 ]; then
  exit 1
fi

echo "[$(date '+%H:%M:%S')] === E012 remote tmux worker done ==="
