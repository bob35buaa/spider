#!/usr/bin/env bash
# Launch E074A/E074C on a 2-GPU remote machine after git sync.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REMOTE_HOST="${REMOTE_HOST:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"
REMOTE_SESSION="${REMOTE_SESSION:-E074}"
GPU_A="${GPU_A:-0}"
GPU_C="${GPU_C:-1}"

if [ -n "$(git status --porcelain)" ]; then
  echo "Working tree has uncommitted changes. Commit them before remote launch so git sync is exact." >&2
  git status --short >&2
  exit 1
fi

echo "[$(date '+%H:%M:%S')] pushing local branch to origin"
git push

echo "[$(date '+%H:%M:%S')] launching remote session ${REMOTE_SESSION} on ${REMOTE_HOST}:${REMOTE_REPO}"
ssh "$REMOTE_HOST" "cd '$REMOTE_REPO' && git pull --ff-only && mkdir -p logs/E074 workspace/core4d/results/E074 && (tmux kill-session -t '$REMOTE_SESSION' 2>/dev/null || true)"
ssh "$REMOTE_HOST" "cd '$REMOTE_REPO' && tmux new-session -d -s '$REMOTE_SESSION' \"bash workspace/core4d/scripts/train/train_E074.sh parallel '$GPU_A' '$GPU_C' 2>&1 | tee logs/E074/remote_parallel.log\""

echo "Remote launched."
echo "Monitor:"
echo "  ssh ${REMOTE_HOST} \"tmux capture-pane -t ${REMOTE_SESSION} -p | tail -40\""
echo "  ssh ${REMOTE_HOST} \"tmux attach -t ${REMOTE_SESSION}\""
echo "Check results:"
echo "  ssh ${REMOTE_HOST} \"ls ${REMOTE_REPO}/workspace/core4d/results/E074/*.npz 2>/dev/null | wc -l\""
echo "Pull results when done:"
echo "  REMOTE_HOST=${REMOTE_HOST} REMOTE_REPO=${REMOTE_REPO} bash workspace/core4d/scripts/pull_E074_remote_results.sh"
