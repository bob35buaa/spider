#!/usr/bin/env bash
# Pull E119 CEM results and logs from spider-remote.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-smoke}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"

mkdir -p "workspace/core4d/results/E119/cem/${STAGE}" "logs/E119"
rsync -av "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/results/E119/cem/${STAGE}/" \
  "workspace/core4d/results/E119/cem/${STAGE}/"
rsync -av "${REMOTE}:${REMOTE_ROOT}/logs/E119/" "logs/E119/"

echo "Pulled E119 ${STAGE} results from ${REMOTE}:${REMOTE_ROOT}"
