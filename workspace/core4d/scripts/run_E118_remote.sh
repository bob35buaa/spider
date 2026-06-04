#!/usr/bin/env bash
# E118 remote runner. Run on spider-remote after code/data sync.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-smoke}"
LOG_ROOT="logs/E118/remote"
mkdir -p "$LOG_ROOT"

echo "[$(date '+%H:%M:%S')] === E118 remote ${STAGE} launch ==="
mapfile -t SNAPSHOT_TASKS < <(
  awk -F '\t' 'NF && $1 !~ /^#/ && ($6 == "remote-gpu0" || $6 == "remote-gpu1") {print $4}' \
    workspace/core4d/scripts/E118/variants.tsv | sort -u
)
if [ "${#SNAPSHOT_TASKS[@]}" -gt 0 ]; then
  bash workspace/core4d/scripts/convert/snapshot_scenes.sh E118 "${SNAPSHOT_TASKS[@]}" \
    > "$LOG_ROOT/remote_scene_snapshot_${STAGE}.log" 2>&1
fi

( E118_SKIP_SCENE_SNAPSHOT=1 bash workspace/core4d/scripts/train/train_E118_carry_corridor.sh remote-gpu0 "$STAGE" 0 > "$LOG_ROOT/remote_gpu0_${STAGE}.log" 2>&1 ) &
PID0=$!
( E118_SKIP_SCENE_SNAPSHOT=1 bash workspace/core4d/scripts/train/train_E118_carry_corridor.sh remote-gpu1 "$STAGE" 1 > "$LOG_ROOT/remote_gpu1_${STAGE}.log" 2>&1 ) &
PID1=$!

echo "[$(date '+%H:%M:%S')] launched GPU0 PID=$PID0 GPU1 PID=$PID1"
wait "$PID0"
echo "[$(date '+%H:%M:%S')] GPU0 done"
wait "$PID1"
echo "[$(date '+%H:%M:%S')] GPU1 done"
echo "[$(date '+%H:%M:%S')] === E118 remote ${STAGE} complete ==="
