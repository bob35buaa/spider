#!/usr/bin/env bash
# Pull E106 remote CEM outputs back to local. Results remain out of git.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REMOTE="${REMOTE:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"
STAGE="${1:-full}"

mkdir -p "workspace/core4d/results/E106/cem/${STAGE}" "logs/E106"

rsync -av \
  "${REMOTE}:${REMOTE_REPO}/workspace/core4d/results/E106/cem/${STAGE}/" \
  "workspace/core4d/results/E106/cem/${STAGE}/"

rsync -av \
  "${REMOTE}:${REMOTE_REPO}/logs/E106/" \
  "logs/E106/"

echo "Pulled E106 ${STAGE} results from ${REMOTE}:${REMOTE_REPO}"
