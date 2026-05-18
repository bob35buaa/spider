#!/usr/bin/env bash
# Pull E014 remote results/logs back to the local workspace and run local eval.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ "${1:-}" = "__codex_auth_probe__" ]; then
  echo "E014 remote puller authorized."
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

mkdir -p workspace/core4d_collab_retarget/results/E014 logs/core4d_collab_retarget/E014

echo "[$(date '+%H:%M:%S')] pulling E014 results from ${REMOTE_HOST}:${REMOTE_REPO}"
REMOTE_VARIANTS=(
  E014_box025_p2_jointB_t05
  E014_box025_p2_jointB_t02_hc1
  E014_box025_p2_jointB_t02_g08
  E014_box023_p2_jointB_t02_hc1
)

scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E014/contact_masks" workspace/core4d_collab_retarget/results/E014/ 2>/dev/null || true
scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E014/scene_snapshot" workspace/core4d_collab_retarget/results/E014/ 2>/dev/null || true

PULLED_VARIANTS=()
for variant in "${REMOTE_VARIANTS[@]}"; do
  remote_npz="${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E014/${variant}.npz"
  if ! ssh "${SCP_OPTS[@]}" "${REMOTE_HOST}" test -f "${remote_npz}"; then
    echo "[$(date '+%H:%M:%S')] WARN missing remote result, skip ${variant}"
    continue
  fi
  scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${remote_npz}" workspace/core4d_collab_retarget/results/E014/
  scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E014/${variant}.mp4" workspace/core4d_collab_retarget/results/E014/ 2>/dev/null || true
  scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E014/keyframes/${variant}" workspace/core4d_collab_retarget/results/E014/keyframes/ 2>/dev/null || true
  scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${REMOTE_REPO}/logs/core4d_collab_retarget/E014/${variant}.log" logs/core4d_collab_retarget/E014/
  PULLED_VARIANTS+=("${variant}")
done

scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${REMOTE_REPO}/logs/core4d_collab_retarget/E014/remote_gpu0.log" logs/core4d_collab_retarget/E014/ 2>/dev/null || true
scp "${SCP_OPTS[@]}" "${REMOTE_HOST}:${REMOTE_REPO}/logs/core4d_collab_retarget/E014/remote_gpu1.log" logs/core4d_collab_retarget/E014/ 2>/dev/null || true

echo "[$(date '+%H:%M:%S')] running local E014 eval"
if [ "${#PULLED_VARIANTS[@]}" -gt 0 ]; then
  .venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E014.py "${PULLED_VARIANTS[@]}" | tee logs/core4d_collab_retarget/E014/eval_E014_local_after_pull.log
else
  echo "No remote E014 variant results found."
fi
echo "Done."
