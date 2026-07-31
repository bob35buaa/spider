#!/usr/bin/env bash
# Pull only E182 artifacts from the immutable Ada run root; never delete local data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT"

STAGE="${1:-query-tape}"
REMOTE_HOST="${ADA_HOST:-spider-remote}"
DEPLOYMENT="workspace/core4d/results/E182/s0_environment/remote_deployment_manifest.json"
RESULT_ROOT="workspace/core4d/results/E182/s1_query_tape"
LOG_ROOT="logs/E182/s1"
SCRIPT_ROOT="workspace/core4d/scripts/experiments/E182"

if [[ ! -f "$DEPLOYMENT" ]]; then
  echo "E182 remote deployment manifest is missing: $DEPLOYMENT" >&2
  exit 2
fi
REMOTE_ROOT="$(uv run python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["remote_root"])' "$DEPLOYMENT")"
if [[ "$REMOTE_ROOT" != /home/xiayb/pHRI_workspace/e182_runs/e182_*/spider ]]; then
  echo "Refusing unsafe E182 remote root: $REMOTE_ROOT" >&2
  exit 2
fi

case "$STAGE" in
  query-tape)
    mkdir -p "$RESULT_ROOT" "$LOG_ROOT"
    rsync --archive --ignore-existing \
      "$REMOTE_HOST:$REMOTE_ROOT/$RESULT_ROOT/" "$RESULT_ROOT/"
    rsync --archive --ignore-existing \
      "$REMOTE_HOST:$REMOTE_ROOT/$LOG_ROOT/" "$LOG_ROOT/"
    uv run python "$SCRIPT_ROOT/runtime_inputs.py" verify \
      --root . --manifest "$RESULT_ROOT/runtime_inputs_manifest.json"
    uv run python "$SCRIPT_ROOT/audit_query_tape_replay.py" \
      --result-root "$RESULT_ROOT"
    echo "E182_PULL_QUERY_TAPE=PASS"
    ;;
  full)
    echo "E182 Full pull is not implemented until production SHA/allocation freeze." >&2
    exit 2
    ;;
  *)
    echo "usage: $0 {query-tape|full}" >&2
    exit 2
    ;;
esac
