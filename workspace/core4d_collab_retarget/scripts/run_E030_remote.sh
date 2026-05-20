#!/usr/bin/env bash
# Launch E030 remote queues on spider-remote.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ "${1:-}" = "__codex_auth_probe__" ]; then
  echo "E030 remote launcher authorized."
  exit 0
fi

REMOTE_HOST="${REMOTE_HOST:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"
REMOTE_WORKTREE="${REMOTE_WORKTREE:-/home/xiayb/pHRI_workspace/spider_e030_worktree}"
REMOTE_BRANCH="${REMOTE_BRANCH:-exp/core4d-collab-retarget-e030-geometry-surface-control}"
SESSION="${SESSION:-E030}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-2400}"
RUN_STALL_TIMEOUT_SECONDS="${RUN_STALL_TIMEOUT_SECONDS:-300}"

ssh "$REMOTE_HOST" "set -euo pipefail
  git -C '$REMOTE_REPO' fetch origin
  if [ ! -e '$REMOTE_WORKTREE/.git' ]; then
    git -C '$REMOTE_REPO' worktree add -B '$REMOTE_BRANCH' '$REMOTE_WORKTREE' 'origin/$REMOTE_BRANCH'
  else
    git -C '$REMOTE_WORKTREE' fetch origin
    git -C '$REMOTE_WORKTREE' switch '$REMOTE_BRANCH'
    git -C '$REMOTE_WORKTREE' pull --ff-only
  fi
  if [ ! -e '$REMOTE_WORKTREE/.venv' ] && [ -e '$REMOTE_REPO/.venv' ]; then
    ln -s '$REMOTE_REPO/.venv' '$REMOTE_WORKTREE/.venv'
  fi
  if [ ! -e '$REMOTE_WORKTREE/example_datasets' ] && [ -e '$REMOTE_REPO/example_datasets' ]; then
    ln -s '$REMOTE_REPO/example_datasets' '$REMOTE_WORKTREE/example_datasets'
  fi"
ssh "$REMOTE_HOST" "set -euo pipefail
  base_data='$REMOTE_REPO/example_datasets/processed/core4d/unitree_g1/humanoid_object'
  work_data='$REMOTE_WORKTREE/example_datasets/processed/core4d/unitree_g1/humanoid_object'
  if [ -d \"\$base_data\" ]; then
    mkdir -p \"\$work_data\"
    for pat in '*_e018b' '*_e022_*' '*_e023_*' '*_e024_*' '*_e025_*' '*_e029_*'; do
      for d in \"\$base_data\"/\$pat; do
        [ -e \"\$d\" ] || continue
        name=\"\$(basename \"\$d\")\"
        if [ ! -e \"\$work_data/\$name\" ]; then
          ln -s \"\$d\" \"\$work_data/\$name\"
        fi
      done
    done
  fi"
ssh "$REMOTE_HOST" "set -euo pipefail
  mkdir -p '$REMOTE_WORKTREE/workspace/core4d_collab_retarget' '$REMOTE_WORKTREE/logs'
  if [ ! -e '$REMOTE_WORKTREE/workspace/core4d_collab_retarget/results' ] && [ -e '$REMOTE_REPO/workspace/core4d_collab_retarget/results' ]; then
    ln -s '$REMOTE_REPO/workspace/core4d_collab_retarget/results' '$REMOTE_WORKTREE/workspace/core4d_collab_retarget/results'
  else
    mkdir -p '$REMOTE_WORKTREE/workspace/core4d_collab_retarget/results'
    for d in E018b E022 E023 E024 E025 E027 E029 holosoma_v2_kinematic E026_full_eval; do
      if [ ! -e '$REMOTE_WORKTREE/workspace/core4d_collab_retarget/results/'\"\$d\" ] && [ -e '$REMOTE_REPO/workspace/core4d_collab_retarget/results/'\"\$d\" ]; then
        ln -s '$REMOTE_REPO/workspace/core4d_collab_retarget/results/'\"\$d\" '$REMOTE_WORKTREE/workspace/core4d_collab_retarget/results/'\"\$d\"
      fi
    done
  fi
  if [ ! -e '$REMOTE_WORKTREE/logs/core4d_collab_retarget' ] && [ -e '$REMOTE_REPO/logs/core4d_collab_retarget' ]; then
    ln -s '$REMOTE_REPO/logs/core4d_collab_retarget' '$REMOTE_WORKTREE/logs/core4d_collab_retarget'
  fi"
ssh "$REMOTE_HOST" "cd '$REMOTE_WORKTREE' && bash workspace/core4d_collab_retarget/scripts/run_E030_preprocess.sh"
ssh "$REMOTE_HOST" "cd '$REMOTE_WORKTREE' && tmux kill-session -t '$SESSION' 2>/dev/null || true"
ssh "$REMOTE_HOST" "cd '$REMOTE_WORKTREE' && mkdir -p logs/core4d_collab_retarget/E030 workspace/core4d_collab_retarget/results/E030"
ssh "$REMOTE_HOST" "cd '$REMOTE_WORKTREE' && tmux new-session -d -s '$SESSION' 'RUN_TIMEOUT_SECONDS=$RUN_TIMEOUT_SECONDS RUN_STALL_TIMEOUT_SECONDS=$RUN_STALL_TIMEOUT_SECONDS bash workspace/core4d_collab_retarget/scripts/train/train_E030_remote_tmux.sh 2>&1 | tee logs/core4d_collab_retarget/E030/remote_tmux.log'"
echo "Started remote tmux session $SESSION on $REMOTE_HOST in $REMOTE_WORKTREE"
