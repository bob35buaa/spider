#!/usr/bin/env bash
# Pull E144 remote results and logs back to local workspace.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"

mkdir -p "workspace/core4d/results/E144/cem/${STAGE}" "logs/E144/cem/${STAGE}"

retry() {
  local attempt=1
  local max_attempts="${E144_REMOTE_RETRY_MAX:-4}"
  while true; do
    if "$@"; then
      return 0
    fi
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "Command failed after ${attempt} attempts: $*" >&2
      return 1
    fi
    echo "Retry ${attempt}/${max_attempts}: $*" >&2
    sleep "$((attempt * 3))"
    attempt=$((attempt + 1))
  done
}

retry rsync -az --ignore-missing-args \
  "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/results/E144/cem/${STAGE}/" \
  "workspace/core4d/results/E144/cem/${STAGE}/"
retry rsync -az --ignore-missing-args \
  "${REMOTE}:${REMOTE_ROOT}/logs/E144/cem/${STAGE}/" \
  "logs/E144/cem/${STAGE}/"

echo "Pulled E144 ${STAGE} results and logs from ${REMOTE}"
