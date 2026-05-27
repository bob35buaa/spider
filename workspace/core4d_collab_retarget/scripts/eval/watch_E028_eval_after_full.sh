#!/usr/bin/env bash
# Wait for the E028 full local tmux run, then evaluate all manifest variants.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

LOGS="logs/core4d_collab_retarget/E028"
mkdir -p "$LOGS"

{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] waiting for e028_full_local"
  while tmux has-session -t e028_full_local 2>/dev/null; do
    sleep "${E028_WATCH_INTERVAL_SECONDS:-300}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] still waiting for e028_full_local"
  done
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] e028_full_local ended; starting eval"
  .venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E028.py --all
  .venv/bin/python workspace/core4d_collab_retarget/scripts/eval/index_E028_online_videos.py --force
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] eval/index done"
} 2>&1 | tee "$LOGS/eval_after_full_driver.log"
