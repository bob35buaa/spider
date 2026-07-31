#!/usr/bin/env bash
# E182 remote Ada 0/1 deployment/worker entrypoint. Never alters existing jobs.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

STAGE="${1:-preflight}"
REMOTE_HOST="${ADA_HOST:-spider-remote}"
EXPECTED_GPUS="${ADA_EXPECTED_GPUS:-0 1}"
POLICY_GPUS="${ADA_POLICY_GPUS:-0,1}"
ALLOW_OVERLAP="${ALLOW_EXISTING_COMPUTE_OVERLAP:-1}"
SCRIPT="workspace/core4d/scripts/experiments/E182/deploy_remote_snapshot.py"

if [[ "$EXPECTED_GPUS" != "0 1" || "$POLICY_GPUS" != "0,1" ]]; then
  echo "E182 remote GPU allowlist must be exactly 0,1." >&2
  exit 2
fi
if [[ "$ALLOW_OVERLAP" != "1" ]]; then
  echo "E182 requires authorized compute overlap." >&2
  exit 2
fi

case "$STAGE" in
  preflight)
    uv run python "$SCRIPT" deploy --remote-host "$REMOTE_HOST"
    ;;
  full)
    echo "E182 remote Full worker is not implemented yet; do not launch manually." >&2
    exit 2
    ;;
  *)
    echo "usage: $0 {preflight|full}" >&2
    exit 2
    ;;
esac
