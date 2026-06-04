#!/usr/bin/env bash
# Pull E122 CEM results and logs from spider-remote.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-smoke}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"

mkdir -p "workspace/core4d/results/E122/cem/${STAGE}" "logs/E122"
rsync -av "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/results/E122/cem/${STAGE}/" \
  "workspace/core4d/results/E122/cem/${STAGE}/"
rsync -av "${REMOTE}:${REMOTE_ROOT}/logs/E122/" "logs/E122/"

echo "Pulled E122 ${STAGE} results from ${REMOTE}:${REMOTE_ROOT}"
