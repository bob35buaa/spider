#!/usr/bin/env bash
# Pull E008 remote results/logs back to the local workspace and run local eval.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ "${1:-}" = "__codex_auth_probe__" ]; then
  echo "E008 remote puller authorized."
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

mkdir -p workspace/core4d_collab_retarget/results/E008 logs/core4d_collab_retarget/E008

echo "[$(date '+%H:%M:%S')] pulling E008 results from ${REMOTE_HOST}:${REMOTE_REPO}"
REMOTE_VARIANTS=(
  E008_box025_p2_ypos_k20_vmax0
  E008_box025_p2_yneg_k20_vmax2
  E008_box025_p2_ypos_k20_vmax2
  E008_box023_p2_xneg_k10_vmax0
  E008_box023_p2_xpos_k10_vmax0
)

scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E008/contact_masks" workspace/core4d_collab_retarget/results/E008/ 2>/dev/null || true
scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E008/scene_snapshot" workspace/core4d_collab_retarget/results/E008/ 2>/dev/null || true

PULLED_VARIANTS=()
for variant in "${REMOTE_VARIANTS[@]}"; do
  remote_npz="${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E008/${variant}.npz"
  if ! ssh "${SCP_OPTS[@]}" "${REMOTE_HOST}" test -f "${remote_npz}"; then
    echo "[$(date '+%H:%M:%S')] WARN missing remote result, skip ${variant}"
    continue
  fi
  scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${remote_npz}" workspace/core4d_collab_retarget/results/E008/
  scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E008/${variant}.mp4" workspace/core4d_collab_retarget/results/E008/ 2>/dev/null || true
  scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E008/keyframes/${variant}" workspace/core4d_collab_retarget/results/E008/keyframes/ 2>/dev/null || true
  scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${REMOTE_REPO}/logs/core4d_collab_retarget/E008/${variant}.log" logs/core4d_collab_retarget/E008/
  PULLED_VARIANTS+=("${variant}")
done

scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${REMOTE_REPO}/logs/core4d_collab_retarget/E008/remote_gpu0.log" logs/core4d_collab_retarget/E008/ 2>/dev/null || true
scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${REMOTE_REPO}/logs/core4d_collab_retarget/E008/remote_gpu1.log" logs/core4d_collab_retarget/E008/ 2>/dev/null || true

echo "[$(date '+%H:%M:%S')] running local E008 eval"
if [ "${#PULLED_VARIANTS[@]}" -gt 0 ]; then
  .venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E008.py "${PULLED_VARIANTS[@]}" | tee logs/core4d_collab_retarget/E008/eval_E008_local_after_pull.log
else
  echo "No remote E008 variant results found."
fi
echo "Done."
