#!/usr/bin/env bash
# Pull E117 remote results/logs from spider-remote.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-smoke}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"

mkdir -p "workspace/core4d/results/E117/cem/${STAGE}" "logs/E117"
rsync -av "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/results/E117/cem/${STAGE}/" \
  "workspace/core4d/results/E117/cem/${STAGE}/"
rsync -av "${REMOTE}:${REMOTE_ROOT}/logs/E117/" "logs/E117/"
echo "Pulled E117 ${STAGE} results from ${REMOTE}:${REMOTE_ROOT}"
