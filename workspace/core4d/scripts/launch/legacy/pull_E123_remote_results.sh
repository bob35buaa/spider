#!/usr/bin/env bash
# Pull E123 CEM results and logs from spider-remote.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-smoke}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"

mkdir -p "workspace/core4d/results/E123/cem/${STAGE}" "workspace/core4d/results/E123/warmstarts" "logs/E123"
rsync -av "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/results/E123/cem/${STAGE}/" \
  "workspace/core4d/results/E123/cem/${STAGE}/"
rsync -av "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/results/E123/warmstarts/" \
  "workspace/core4d/results/E123/warmstarts/"
rsync -av "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/results/E123/preflight/" \
  "workspace/core4d/results/E123/preflight/"
rsync -av "${REMOTE}:${REMOTE_ROOT}/logs/E123/" "logs/E123/"

echo "Pulled E123 ${STAGE} results from ${REMOTE}:${REMOTE_ROOT}"
