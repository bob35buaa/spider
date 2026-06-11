#!/usr/bin/env bash
# Launch E084 remote GPU0/GPU1 main gate splits without requiring a clean git tree.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REMOTE_HOST="${REMOTE_HOST:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"
REMOTE_SESSION="${REMOTE_SESSION:-E084_remote_main_gate}"
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

echo "[$(date '+%H:%M:%S')] rsync E084 code/data to ${REMOTE_HOST}:${REMOTE_REPO}"
rsync -avR \
  spider/config.py \
  spider/simulators/mjwp.py \
  workspace/core4d/scripts/E084/ \
  workspace/core4d/scripts/train/train_E084.sh \
  workspace/core4d/scripts/run_E084_preprocess.sh \
  workspace/core4d/scripts/run_E084_remote.sh \
  workspace/core4d/scripts/pull_E084_remote_results.sh \
  workspace/core4d/scripts/eval/eval_E084.py \
  workspace/core4d/scripts/eval/eval_E083.py \
  workspace/core4d/scripts/eval/eval_E081.py \
  workspace/core4d/scripts/eval/eval_E079.py \
  workspace/core4d/scripts/eval/eval_E078.py \
  workspace/core4d/scripts/eval/eval_E072.py \
  workspace/core4d/scripts/eval/diagnose_E082_body_fall.py \
  workspace/core4d_collab_retarget/scripts/eval/paper_metrics.py \
  workspace/core4d/results/E084/contact_masks/ \
  examples/config/override/core4d_E084A_d003_box021_20231018_029_p2_safety.yaml \
  examples/config/override/core4d_E084B_d003_box021_20231018_029_p2_upright.yaml \
  examples/config/override/core4d_E084C_d003_box021_20231018_029_p2_semantic.yaml \
  examples/config/override/core4d_E084A_box023_p2_safety_guard.yaml \
  examples/config/override/core4d_E084B_box023_p2_upright_guard.yaml \
  examples/config/override/core4d_E084C_box023_p2_semantic_guard.yaml \
  example_datasets/processed/core4d/assets/objects/box021/box021_m.obj \
  example_datasets/processed/core4d/assets/objects/box023/box023_m.obj \
  example_datasets/processed/core4d/unitree_g1/humanoid_object/d003_box021_20231018_029_p2_upperobj_e083/ \
  example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person2_upperobj_e083/ \
  "${REMOTE_HOST}:${REMOTE_REPO}/"

echo "[$(date '+%H:%M:%S')] launching remote E084 main gate session"
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_REPO' && mkdir -p logs/E084 workspace/core4d/results/E084 && tmux new-session -d -s '$REMOTE_SESSION' 'bash workspace/core4d/scripts/E084/run_remote_inside.sh'"

echo "Remote launched."
echo "Monitor:"
echo "  ssh ${REMOTE_HOST} \"tmux capture-pane -t ${REMOTE_SESSION} -p | tail -40\""
echo "Pull results when done:"
echo "  REMOTE_HOST=${REMOTE_HOST} REMOTE_REPO=${REMOTE_REPO} bash workspace/core4d/scripts/pull_E084_remote_results.sh"
