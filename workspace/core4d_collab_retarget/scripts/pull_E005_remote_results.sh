#!/usr/bin/env bash
# Pull E005 remote results/logs back to the local workspace and run local eval.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ "${1:-}" = "__codex_auth_probe__" ]; then
  echo "E005 remote puller authorized."
  exit 0
fi

REMOTE_HOST="${REMOTE_HOST:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"
SCP_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o ServerAliveInterval=10
  -o ServerAliveCountMax=3
)

mkdir -p workspace/core4d_collab_retarget/results/E005 logs/core4d_collab_retarget/E005

echo "[$(date '+%H:%M:%S')] pulling E005 results from ${REMOTE_HOST}:${REMOTE_REPO}"
REMOTE_VARIANTS=(
  E005_box025_p2_com_s40
  E005_box025_p2_yneg_s40
  E005_box025_p2_ypos_s20
  E005_box023_p2_com_s10
  E005_box023_p2_xneg_s10
  E005_box023_p2_xpos_s10
)

scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E005/contact_masks" workspace/core4d_collab_retarget/results/E005/ 2>/dev/null || true
scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E005/scene_snapshot" workspace/core4d_collab_retarget/results/E005/ 2>/dev/null || true

for variant in "${REMOTE_VARIANTS[@]}"; do
  scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E005/${variant}.npz" workspace/core4d_collab_retarget/results/E005/
  scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E005/${variant}.mp4" workspace/core4d_collab_retarget/results/E005/ 2>/dev/null || true
  scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E005/keyframes/${variant}" workspace/core4d_collab_retarget/results/E005/keyframes/ 2>/dev/null || true
  scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${REMOTE_REPO}/logs/core4d_collab_retarget/E005/${variant}.log" logs/core4d_collab_retarget/E005/
done

scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${REMOTE_REPO}/logs/core4d_collab_retarget/E005/remote_gpu0.log" logs/core4d_collab_retarget/E005/ 2>/dev/null || true
scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${REMOTE_REPO}/logs/core4d_collab_retarget/E005/remote_gpu1.log" logs/core4d_collab_retarget/E005/ 2>/dev/null || true

echo "[$(date '+%H:%M:%S')] running local E005 eval"
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E005.py | tee logs/core4d_collab_retarget/E005/eval_E005_local_after_pull.log
echo "Done."
