#!/usr/bin/env bash
# E182 local authority/preflight and later local Full entrypoint.
# Existing GPU jobs are coexistence inputs and are never stopped or modified.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

STAGE="${1:-preflight}"
SCRIPT_ROOT="workspace/core4d/scripts/experiments/E182"
RESULT_ROOT="workspace/core4d/results/E182/s0_environment"
LOG_ROOT="logs/E182/s0"
REMOTE_HOST="${ADA_HOST:-spider-remote}"
LOCAL_GPU_ID="${LOCAL_GPU_ID:-0}"
ALLOW_OVERLAP="${ALLOW_EXISTING_COMPUTE_OVERLAP:-1}"
mkdir -p "$LOG_ROOT" "$RESULT_ROOT"

if [[ "$ALLOW_OVERLAP" != "1" ]]; then
  echo "E182 requires ALLOW_EXISTING_COMPUTE_OVERLAP=1 per user authorization." >&2
  exit 2
fi

run_preflight() {
  uv run python "$SCRIPT_ROOT/build_authority.py" \
    2>&1 | tee "$LOG_ROOT/authority.log"
  uv run python "$SCRIPT_ROOT/deploy_remote_snapshot.py" freeze \
    2>&1 | tee "$LOG_ROOT/source_freeze.log"
  uv run python "$SCRIPT_ROOT/deploy_remote_snapshot.py" deploy \
    --remote-host "$REMOTE_HOST" \
    2>&1 | tee "$LOG_ROOT/remote_deploy.log"

  local remote_root
  remote_root="$(uv run python -c \
    'import json; print(json.load(open("workspace/core4d/results/E182/s0_environment/remote_deployment_manifest.json"))["remote_root"])')"
  uv run python "$SCRIPT_ROOT/probe_environment.py" \
    --remote-host "$REMOTE_HOST" \
    --remote-root "$remote_root" \
    --local-gpu-id "$LOCAL_GPU_ID" \
    --allow-existing-compute-overlap \
    2>&1 | tee "$LOG_ROOT/environment.log"
  uv run python "$SCRIPT_ROOT/test_preflight.py" \
    2>&1 | tee "$LOG_ROOT/tests.log"
  uv run python "$SCRIPT_ROOT/audit_preflight.py" \
    2>&1 | tee "$LOG_ROOT/gate0_audit.log"
}

case "$STAGE" in
  preflight)
    run_preflight
    ;;
  query-tape|task-audit|pareto-canary|freeze-production|full)
    echo "E182 stage '$STAGE' is not implemented yet; Gate 0 must pass first." >&2
    exit 2
    ;;
  *)
    echo "usage: $0 {preflight|query-tape|task-audit|pareto-canary|freeze-production|full}" >&2
    exit 2
    ;;
esac
