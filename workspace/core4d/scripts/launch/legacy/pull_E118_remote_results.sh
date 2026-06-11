#!/usr/bin/env bash
# Pull E118 CEM results and logs from spider-remote.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-smoke}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"

mkdir -p "workspace/core4d/results/E118/cem/${STAGE}" "logs/E118"
rsync -av "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/results/E118/cem/${STAGE}/" \
  "workspace/core4d/results/E118/cem/${STAGE}/"
rsync -av "${REMOTE}:${REMOTE_ROOT}/logs/E118/" "logs/E118/"

echo "Pulled E118 ${STAGE} results from ${REMOTE}:${REMOTE_ROOT}"
