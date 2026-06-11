#!/usr/bin/env bash
# Launch E083 remote GPU0/GPU1 splits without requiring a clean local git tree.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REMOTE_HOST="${REMOTE_HOST:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"
REMOTE_SESSION="${REMOTE_SESSION:-E083_remote_upperobj}"
SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o ServerAliveInterval=10
  -o ServerAliveCountMax=3
)

echo "[$(date '+%H:%M:%S')] checking remote session ${REMOTE_SESSION}"
if ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "tmux has-session -t '$REMOTE_SESSION' 2>/dev/null"; then
  echo "Remote tmux session already exists: ${REMOTE_SESSION}. Not killing it." >&2
  exit 1
fi

echo "[$(date '+%H:%M:%S')] rsync E083 code/data to ${REMOTE_HOST}:${REMOTE_REPO}"
rsync -avR \
  workspace/core4d/scripts/E083/ \
  workspace/core4d/scripts/train/train_E083.sh \
  workspace/core4d/scripts/run_E083_preprocess.sh \
  workspace/core4d/scripts/run_E083_remote.sh \
  workspace/core4d/scripts/pull_E083_remote_results.sh \
  workspace/core4d/scripts/eval/eval_E083.py \
  workspace/core4d/scripts/eval/eval_E081.py \
  workspace/core4d/scripts/eval/diagnose_E082_body_fall.py \
  workspace/core4d/results/E083/contact_masks/ \
  examples/config/override/core4d_E083_d003_box021_20231011_035_p2_upperobj.yaml \
  examples/config/override/core4d_E083_d003_box021_20231020_019_p1_upperobj.yaml \
  examples/config/override/core4d_E083_box023_p2_upperobj_guard.yaml \
  example_datasets/processed/core4d/assets/objects/box021/box021_m.obj \
  example_datasets/processed/core4d/assets/objects/box023/box023_m.obj \
  example_datasets/processed/core4d/unitree_g1/humanoid_object/d003_box021_20231011_035_p2_upperobj_e083/ \
  example_datasets/processed/core4d/unitree_g1/humanoid_object/d003_box021_20231020_019_p1_upperobj_e083/ \
  example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person2_upperobj_e083/ \
  "${REMOTE_HOST}:${REMOTE_REPO}/"

echo "[$(date '+%H:%M:%S')] launching remote E083 two-GPU session"
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_REPO' && mkdir -p logs/E083 workspace/core4d/results/E083 && tmux new-session -d -s '$REMOTE_SESSION' 'bash workspace/core4d/scripts/E083/run_remote_inside.sh'"

echo "Remote launched."
echo "Monitor:"
echo "  ssh ${REMOTE_HOST} \"tmux capture-pane -t ${REMOTE_SESSION} -p | tail -40\""
echo "Pull results when done:"
echo "  REMOTE_HOST=${REMOTE_HOST} REMOTE_REPO=${REMOTE_REPO} bash workspace/core4d/scripts/pull_E083_remote_results.sh"
