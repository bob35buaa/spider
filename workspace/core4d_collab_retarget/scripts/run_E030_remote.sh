#!/usr/bin/env bash
# E030 remote launcher.
# GPU0: E029_d003_box021_20231011_035_p2_d6_locked
# GPU1: E029_d003_box021_20231020_019_p1_d6_locked
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

export VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d_collab_retarget/results/E029/d6/manifest.tsv}"
export RESULTS="${RESULTS:-workspace/core4d_collab_retarget/results/E030/d6_locked_cem}"
export LOGS="${LOGS:-logs/core4d_collab_retarget/E030}"

mkdir -p "$RESULTS" "$LOGS"

echo "=== E030 remote start $(date '+%F %T') ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true

( bash workspace/core4d_collab_retarget/scripts/train/train_E030.sh remote-gpu0 0 ) &
PID0=$!
( bash workspace/core4d_collab_retarget/scripts/train/train_E030.sh remote-gpu1 1 ) &
PID1=$!

echo "=== Launched E030 remote: GPU0 PID=$PID0 | GPU1 PID=$PID1 ==="
wait "$PID0"
echo "=== E030 GPU0 done $(date '+%F %T') ==="
wait "$PID1"
echo "=== E030 GPU1 done $(date '+%F %T') ==="

find "$RESULTS" -maxdepth 3 -type f \( -name '*.npz' -o -name '*.mp4' \) | sort
echo "=== E030 remote complete $(date '+%F %T') ==="

