#!/usr/bin/env bash
# Launch E087 remote GPU0/GPU1 splits without requiring a clean git tree.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REMOTE_HOST="${REMOTE_HOST:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"
REMOTE_SESSION="${REMOTE_SESSION:-E087_mass_reward}"
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

echo "[$(date '+%H:%M:%S')] rsync E087 code/data to ${REMOTE_HOST}:${REMOTE_REPO}"
rsync -avR \
  spider/config.py \
  spider/simulators/mjwp.py \
  examples/run_mjwp.py \
  workspace/core4d/scripts/E087/ \
  workspace/core4d/scripts/train/train_E087.sh \
  workspace/core4d/scripts/run_E087_preprocess.sh \
  workspace/core4d/scripts/run_E087_remote.sh \
  workspace/core4d/scripts/pull_E087_remote_results.sh \
  workspace/core4d/scripts/eval/eval_E087.py \
  workspace/core4d/scripts/eval/eval_E086.py \
  workspace/core4d/scripts/eval/eval_E085.py \
  workspace/core4d/scripts/eval/eval_E084.py \
  workspace/core4d/scripts/eval/eval_E083.py \
  workspace/core4d/scripts/eval/eval_E081.py \
  workspace/core4d/scripts/eval/eval_E079.py \
  workspace/core4d/scripts/eval/eval_E078.py \
  workspace/core4d/scripts/eval/eval_E072.py \
  workspace/core4d/scripts/eval/diagnose_E082_body_fall.py \
  workspace/core4d_collab_retarget/scripts/eval/paper_metrics.py \
  workspace/core4d/results/E084/contact_masks/ \
  workspace/core4d/results/E085/raw_targets/ \
  workspace/core4d/results/E087/mass_audit/ \
  examples/config/override/core4d_E084C_d003_box021_20231018_029_p2_semantic.yaml \
  examples/config/override/core4d_E085A_rawtarget_main.yaml \
  examples/config/override/core4d_E086A_rawtarget_strict_main.yaml \
  examples/config/override/core4d_E087A_m5_rawtarget_main.yaml \
  examples/config/override/core4d_E087B_m10_rawtarget_main.yaml \
  examples/config/override/core4d_E087C_m5_safe_main.yaml \
  example_datasets/processed/core4d/assets/objects/box021/box021_m.obj \
  example_datasets/processed/core4d/unitree_g1/humanoid_object/d003_box021_20231018_029_p2_upperobj_e083/ \
  example_datasets/processed/core4d/unitree_g1/humanoid_object/d003_box021_20231018_029_p2_upperobj_e083_m5_e087/ \
  example_datasets/processed/core4d/unitree_g1/humanoid_object/d003_box021_20231018_029_p2_upperobj_e083_m10_e087/ \
  "${REMOTE_HOST}:${REMOTE_REPO}/"

echo "[$(date '+%H:%M:%S')] launching remote E087 session"
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_REPO' && mkdir -p logs/E087 workspace/core4d/results/E087 && tmux new-session -d -s '$REMOTE_SESSION' 'bash workspace/core4d/scripts/E087/run_remote_inside.sh'"

echo "Remote launched."
echo "Monitor:"
echo "  ssh ${REMOTE_HOST} \"tmux capture-pane -t ${REMOTE_SESSION} -p | tail -40\""
echo "Pull results when done:"
echo "  REMOTE_HOST=${REMOTE_HOST} REMOTE_REPO=${REMOTE_REPO} bash workspace/core4d/scripts/pull_E087_remote_results.sh"

