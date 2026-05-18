#!/usr/bin/env bash
# Pull E012 remote results/logs back to the local workspace and run local eval.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ "${1:-}" = "__codex_auth_probe__" ]; then
  echo "E012 remote puller authorized."
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

mkdir -p workspace/core4d_collab_retarget/results/E012 logs/core4d_collab_retarget/E012

echo "[$(date '+%H:%M:%S')] pulling E012 results from ${REMOTE_HOST}:${REMOTE_REPO}"
REMOTE_VARIANTS=(
  E012_box025_p2_dualy_x20_k50_g05
  E012_box025_p2_dualy_x30_k100_g05
  E012_box025_p2_dualy_x20_k100_g08
  E012_box025_p2_dualy_x20_k150_g05
  E012_box023_p2_dualx_y10_k50_g05
  E012_box023_p2_dualx_y10_k100_g05
)

scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E012/contact_masks" workspace/core4d_collab_retarget/results/E012/ 2>/dev/null || true
scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E012/scene_snapshot" workspace/core4d_collab_retarget/results/E012/ 2>/dev/null || true

PULLED_VARIANTS=()
for variant in "${REMOTE_VARIANTS[@]}"; do
  remote_npz="${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E012/${variant}.npz"
  if ! ssh "${SCP_OPTS[@]}" "${REMOTE_HOST}" test -f "${remote_npz}"; then
    echo "[$(date '+%H:%M:%S')] WARN missing remote result, skip ${variant}"
    continue
  fi
  scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${remote_npz}" workspace/core4d_collab_retarget/results/E012/
  scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E012/${variant}.mp4" workspace/core4d_collab_retarget/results/E012/ 2>/dev/null || true
  scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E012/keyframes/${variant}" workspace/core4d_collab_retarget/results/E012/keyframes/ 2>/dev/null || true
  scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${REMOTE_REPO}/logs/core4d_collab_retarget/E012/${variant}.log" logs/core4d_collab_retarget/E012/
  PULLED_VARIANTS+=("${variant}")
done

scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${REMOTE_REPO}/logs/core4d_collab_retarget/E012/remote_gpu0.log" logs/core4d_collab_retarget/E012/ 2>/dev/null || true
scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${REMOTE_REPO}/logs/core4d_collab_retarget/E012/remote_gpu1.log" logs/core4d_collab_retarget/E012/ 2>/dev/null || true

echo "[$(date '+%H:%M:%S')] running local E012 eval"
if [ "${#PULLED_VARIANTS[@]}" -gt 0 ]; then
  .venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E012.py "${PULLED_VARIANTS[@]}" | tee logs/core4d_collab_retarget/E012/eval_E012_local_after_pull.log
else
  echo "No remote E012 variant results found."
fi
echo "Done."
