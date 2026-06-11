#!/usr/bin/env bash
# Pull E116 remote results/logs from spider-remote.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"

mkdir -p "workspace/core4d/results/E116/cem/${STAGE}" "logs/E116"
rsync -av "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/results/E116/cem/${STAGE}/" \
  "workspace/core4d/results/E116/cem/${STAGE}/"
rsync -av "${REMOTE}:${REMOTE_ROOT}/logs/E116/" "logs/E116/"
echo "Pulled E116 ${STAGE} results from ${REMOTE}:${REMOTE_ROOT}"
