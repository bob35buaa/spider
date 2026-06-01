#!/usr/bin/env bash
# Pull E107 remote CEM outputs back to local. Results remain out of git.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REMOTE="${REMOTE:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"
STAGE="${1:-full}"

mkdir -p "workspace/core4d/results/E107/cem/${STAGE}" "logs/E107"

rsync -av \
  "${REMOTE}:${REMOTE_REPO}/workspace/core4d/results/E107/cem/${STAGE}/" \
  "workspace/core4d/results/E107/cem/${STAGE}/"

rsync -av \
  "${REMOTE}:${REMOTE_REPO}/logs/E107/" \
  "logs/E107/"

echo "Pulled E107 ${STAGE} results from ${REMOTE}:${REMOTE_REPO}"
