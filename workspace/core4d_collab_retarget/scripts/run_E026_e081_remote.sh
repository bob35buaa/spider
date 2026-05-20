#!/usr/bin/env bash
# E026: launch E081 full rerun remote splits without requiring a git commit.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REMOTE_HOST="${REMOTE_HOST:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"
REMOTE_SESSION="${REMOTE_SESSION:-E026_E081}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-7200}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=10 -o ServerAliveCountMax=3)

FILES=(
  workspace/core4d/scripts/eval/eval_E081.py
  workspace/core4d_collab_retarget/scripts/E026/e081_full_variants.tsv
  workspace/core4d_collab_retarget/scripts/run_E026_e081_preprocess.sh
  workspace/core4d_collab_retarget/scripts/train/train_E026_e081_full.sh
)

echo "[$(date '+%H:%M:%S')] syncing E026 E081 scripts to ${REMOTE_HOST}:${REMOTE_REPO}"
for f in "${FILES[@]}"; do
  ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "mkdir -p '$REMOTE_REPO/$(dirname "$f")'"
  rsync -az -e "ssh ${SSH_OPTS[*]}" "$f" "$REMOTE_HOST:$REMOTE_REPO/$f"
done

echo "[$(date '+%H:%M:%S')] remote preprocess"
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_REPO' && bash workspace/core4d_collab_retarget/scripts/run_E026_e081_preprocess.sh"

echo "[$(date '+%H:%M:%S')] launching remote tmux ${REMOTE_SESSION}"
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_REPO' && (tmux kill-session -t '$REMOTE_SESSION' 2>/dev/null || true)"
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_REPO' && tmux new-session -d -s '$REMOTE_SESSION' \
  \"RUN_TIMEOUT_SECONDS='$RUN_TIMEOUT_SECONDS' bash workspace/core4d_collab_retarget/scripts/train/train_E026_e081_full.sh remote_gpu0 0 > logs/core4d_collab_retarget/E026_E081_full/remote_gpu0.controller.log 2>&1 & \
   RUN_TIMEOUT_SECONDS='$RUN_TIMEOUT_SECONDS' bash workspace/core4d_collab_retarget/scripts/train/train_E026_e081_full.sh remote_gpu1 1 > logs/core4d_collab_retarget/E026_E081_full/remote_gpu1.controller.log 2>&1 & \
   wait\""

echo "Monitor:"
echo "  ssh ${REMOTE_HOST} \"tmux capture-pane -t ${REMOTE_SESSION} -p | tail -40\""
echo "Pull:"
echo "  bash workspace/core4d_collab_retarget/scripts/pull_E026_e081_remote_results.sh"
