#!/usr/bin/env bash
# Pull E011 remote results/logs back to the local workspace and run local eval.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ "${1:-}" = "__codex_auth_probe__" ]; then
  echo "E011 remote puller authorized."
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

mkdir -p workspace/core4d_collab_retarget/results/E011 logs/core4d_collab_retarget/E011

echo "[$(date '+%H:%M:%S')] pulling E011 results from ${REMOTE_HOST}:${REMOTE_REPO}"
REMOTE_VARIANTS=(
  E011_box025_p2_com_xyz_k50
  E011_box025_p2_com_xyz_k100
  E011_box025_p2_com_xyz_k50_g1
  E011_box025_p2_com_xyz_k50_rot1
  E011_box023_p2_com_xyz_k50
  E011_box023_p2_com_xyz_k100
)

scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E011/contact_masks" workspace/core4d_collab_retarget/results/E011/ 2>/dev/null || true
scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E011/scene_snapshot" workspace/core4d_collab_retarget/results/E011/ 2>/dev/null || true

PULLED_VARIANTS=()
for variant in "${REMOTE_VARIANTS[@]}"; do
  remote_npz="${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E011/${variant}.npz"
  if ! ssh "${SCP_OPTS[@]}" "${REMOTE_HOST}" test -f "${remote_npz}"; then
    echo "[$(date '+%H:%M:%S')] WARN missing remote result, skip ${variant}"
    continue
  fi
  scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${remote_npz}" workspace/core4d_collab_retarget/results/E011/
  scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E011/${variant}.mp4" workspace/core4d_collab_retarget/results/E011/ 2>/dev/null || true
  scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E011/keyframes/${variant}" workspace/core4d_collab_retarget/results/E011/keyframes/ 2>/dev/null || true
  scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${REMOTE_REPO}/logs/core4d_collab_retarget/E011/${variant}.log" logs/core4d_collab_retarget/E011/
  PULLED_VARIANTS+=("${variant}")
done

scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${REMOTE_REPO}/logs/core4d_collab_retarget/E011/remote_gpu0.log" logs/core4d_collab_retarget/E011/ 2>/dev/null || true
scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${REMOTE_REPO}/logs/core4d_collab_retarget/E011/remote_gpu1.log" logs/core4d_collab_retarget/E011/ 2>/dev/null || true

echo "[$(date '+%H:%M:%S')] running local E011 eval"
if [ "${#PULLED_VARIANTS[@]}" -gt 0 ]; then
  .venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E011.py "${PULLED_VARIANTS[@]}" | tee logs/core4d_collab_retarget/E011/eval_E011_local_after_pull.log
else
  echo "No remote E011 variant results found."
fi
echo "Done."
