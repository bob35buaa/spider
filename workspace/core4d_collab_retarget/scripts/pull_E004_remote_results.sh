#!/usr/bin/env bash
# Pull E004 remote results/logs back to the local workspace and run local eval.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ "${1:-}" = "__codex_auth_probe__" ]; then
  echo "E004 remote puller authorized."
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

mkdir -p workspace/core4d_collab_retarget/results/E004 logs/core4d_collab_retarget/E004

echo "[$(date '+%H:%M:%S')] pulling E004 results from ${REMOTE_HOST}:${REMOTE_REPO}"
REMOTE_VARIANTS=(
  E004_box025_p2_g05
  E004_box025_p2_s10
  E004_box025_p2_s40
  E004_box025_p2_s40_hc
  E004_box023_p2_s10
)

scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E004/contact_masks" workspace/core4d_collab_retarget/results/E004/ 2>/dev/null || true
scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E004/scene_snapshot" workspace/core4d_collab_retarget/results/E004/ 2>/dev/null || true

for variant in "${REMOTE_VARIANTS[@]}"; do
  scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E004/${variant}.npz" workspace/core4d_collab_retarget/results/E004/
  scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E004/${variant}.mp4" workspace/core4d_collab_retarget/results/E004/ 2>/dev/null || true
  scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E004/keyframes/${variant}" workspace/core4d_collab_retarget/results/E004/keyframes/ 2>/dev/null || true
  scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${REMOTE_REPO}/logs/core4d_collab_retarget/E004/${variant}.log" logs/core4d_collab_retarget/E004/
done

scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${REMOTE_REPO}/logs/core4d_collab_retarget/E004/remote_gpu0.log" logs/core4d_collab_retarget/E004/ 2>/dev/null || true
scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${REMOTE_REPO}/logs/core4d_collab_retarget/E004/remote_gpu1.log" logs/core4d_collab_retarget/E004/remote_gpu1_interrupted.log 2>/dev/null || true
scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${REMOTE_REPO}/logs/core4d_collab_retarget/E004/manual_guard_hc.log" logs/core4d_collab_retarget/E004/manual_guard_hc_interrupted.log 2>/dev/null || true

echo "[$(date '+%H:%M:%S')] running local E004 eval"
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E004.py | tee logs/core4d_collab_retarget/E004/eval_E004_local_after_pull.log
echo "Done."
