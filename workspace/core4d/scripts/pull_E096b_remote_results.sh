#!/usr/bin/env bash
# Pull E096b remote CEM results/logs from spider-remote.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider_e096b_run}"
REMOTE="${REMOTE:-spider-remote}"
LOCAL_RESULTS="workspace/core4d/results/E096b/cem/${STAGE}"
LOCAL_LOGS="logs/E096b/cem/${STAGE}"
LOCAL_REMOTE_LOGS="logs/E096b/remote"

mkdir -p "$LOCAL_RESULTS" "$LOCAL_LOGS" "$LOCAL_REMOTE_LOGS"

scp "$REMOTE:$REMOTE_REPO/workspace/core4d/results/E096b/cem/${STAGE}/*" "$LOCAL_RESULTS/" 2>/dev/null || true
scp -r "$REMOTE:$REMOTE_REPO/workspace/core4d/results/E096b/cem/${STAGE}/keyframes" "$LOCAL_RESULTS/" 2>/dev/null || true
while IFS= read -r remote_dir; do
  [ -n "$remote_dir" ] || continue
  scp -r "$REMOTE:$REMOTE_REPO/workspace/core4d/results/E096b/cem/${STAGE}/${remote_dir}" "$LOCAL_RESULTS/" 2>/dev/null || true
done < <(
  ssh "$REMOTE" "find '$REMOTE_REPO/workspace/core4d/results/E096b/cem/${STAGE}' -maxdepth 1 -type d -name '*_outdir_${STAGE}' -printf '%f\n'" 2>/dev/null || true
)
scp "$REMOTE:$REMOTE_REPO/logs/E096b/cem/${STAGE}/*.log" "$LOCAL_LOGS/" 2>/dev/null || true
scp "$REMOTE:$REMOTE_REPO/logs/E096b/remote/*.log" "$LOCAL_REMOTE_LOGS/" 2>/dev/null || true

echo "Pulled E096b ${STAGE} results from ${REMOTE}:${REMOTE_REPO}"
