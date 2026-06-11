#!/usr/bin/env bash
# Run E088 remote GPU splits inside a tmux session.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

mkdir -p logs/E088 workspace/core4d/results/E088

(
  echo "[GPU0] $(date '+%F %T') start"
  bash workspace/core4d/scripts/train/train_E088.sh remote-gpu0 0
  echo "[GPU0] $(date '+%F %T') done"
) > logs/E088/remote_gpu0_driver.log 2>&1 &
pid0=$!

(
  echo "[GPU1] $(date '+%F %T') start"
  bash workspace/core4d/scripts/train/train_E088.sh remote-gpu1 1
  echo "[GPU1] $(date '+%F %T') done"
) > logs/E088/remote_gpu1_driver.log 2>&1 &
pid1=$!

wait "$pid0"
wait "$pid1"

RESULTS=workspace/core4d/results/E088 \
VARIANTS_FILE=workspace/core4d/scripts/E088/variants.tsv \
.venv/bin/python workspace/core4d/scripts/eval/eval_E088.py | tee logs/E088/eval_E088_remote_merged.log

echo "E088 remote inside done at $(date '+%F %T')"
