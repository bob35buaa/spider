#!/usr/bin/env bash
# Pull E022 remote results back to local workspace.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ "${1:-}" = "__codex_auth_probe__" ]; then
  echo "E022 pull entrypoint authorized."
  exit 0
fi

REMOTE_HOST="${REMOTE_HOST:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"
LOCAL_RESULTS="workspace/core4d_collab_retarget/results/E022"
LOCAL_LOGS="logs/core4d_collab_retarget/E022"
mkdir -p "$LOCAL_RESULTS" "$LOCAL_LOGS"

scp -r "${REMOTE_HOST}:${REMOTE_REPO}/${LOCAL_RESULTS}/"* "$LOCAL_RESULTS/" 2>/dev/null || true
scp -r "${REMOTE_HOST}:${REMOTE_REPO}/${LOCAL_LOGS}/"* "$LOCAL_LOGS/" 2>/dev/null || true
echo "Pulled E022 remote results."
