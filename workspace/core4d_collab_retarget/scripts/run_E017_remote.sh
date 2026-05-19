#!/usr/bin/env bash
# Launch E017 subset quick validation on spider-remote.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ "${1:-}" = "__codex_auth_probe__" ]; then
  echo "E017 remote launcher authorized."
  exit 0
fi

REMOTE_HOST="${REMOTE_HOST:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"
SESSION="${SESSION:-E017}"

ssh "$REMOTE_HOST" "cd '$REMOTE_REPO' && git fetch origin && git checkout exp/core4d-collab-retarget && git pull --ff-only"
ssh "$REMOTE_HOST" "mkdir -p \
  '$REMOTE_REPO/workspace/core4d_collab_retarget/scripts/E017' \
  '$REMOTE_REPO/workspace/core4d_collab_retarget/scripts/train' \
  '$REMOTE_REPO/workspace/core4d_collab_retarget/scripts/eval'"
rsync -az workspace/core4d_collab_retarget/scripts/E017/ \
  "$REMOTE_HOST:$REMOTE_REPO/workspace/core4d_collab_retarget/scripts/E017/"
rsync -az \
  workspace/core4d_collab_retarget/scripts/run_E017_preprocess.sh \
  workspace/core4d_collab_retarget/scripts/run_E017_remote.sh \
  "$REMOTE_HOST:$REMOTE_REPO/workspace/core4d_collab_retarget/scripts/"
rsync -az \
  workspace/core4d_collab_retarget/scripts/train/train_E017.sh \
  workspace/core4d_collab_retarget/scripts/train/train_E017_remote_tmux.sh \
  "$REMOTE_HOST:$REMOTE_REPO/workspace/core4d_collab_retarget/scripts/train/"
rsync -az workspace/core4d_collab_retarget/scripts/eval/eval_E017.py \
  "$REMOTE_HOST:$REMOTE_REPO/workspace/core4d_collab_retarget/scripts/eval/"
ssh "$REMOTE_HOST" "cd '$REMOTE_REPO' && chmod +x \
  workspace/core4d_collab_retarget/scripts/E017/audit_select_anchors.py \
  workspace/core4d_collab_retarget/scripts/E017/generate_e017_overrides.py \
  workspace/core4d_collab_retarget/scripts/run_E017_preprocess.sh \
  workspace/core4d_collab_retarget/scripts/train/train_E017.sh \
  workspace/core4d_collab_retarget/scripts/train/train_E017_remote_tmux.sh \
  workspace/core4d_collab_retarget/scripts/eval/eval_E017.py"
ssh "$REMOTE_HOST" "cd '$REMOTE_REPO' && tmux kill-session -t '$SESSION' 2>/dev/null || true"
ssh "$REMOTE_HOST" "cd '$REMOTE_REPO' && mkdir -p logs/core4d_collab_retarget/E017 workspace/core4d_collab_retarget/results/E017"
ssh "$REMOTE_HOST" "cd '$REMOTE_REPO' && tmux new-session -d -s '$SESSION' 'bash workspace/core4d_collab_retarget/scripts/train/train_E017_remote_tmux.sh 2>&1 | tee logs/core4d_collab_retarget/E017/remote_tmux.log'"
echo "Started remote tmux session $SESSION on $REMOTE_HOST"
