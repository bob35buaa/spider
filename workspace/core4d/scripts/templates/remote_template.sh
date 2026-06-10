#!/usr/bin/env bash
# {{EXP_ID}} remote launcher: {{DESCRIPTION}}
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"
SESSION="${SESSION:-{{EXP_ID}}_${STAGE}_$(date +%H%M%S)}"
VARIANTS_FILE="${VARIANTS_FILE:-{{VARIANTS_TSV_PATH}}}"

if [ "$STAGE" != "smoke" ] && [ "$STAGE" != "full" ]; then
  echo "Invalid STAGE=$STAGE (use smoke|full)" >&2
  exit 2
fi

retry() {
  local attempt=1
  local max_attempts="${{{EXP_ID}}_REMOTE_RETRY_MAX:-4}"
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

echo "Syncing {{EXP_ID}} artifacts to ${REMOTE}:${REMOTE_ROOT}"
retry ssh "$REMOTE" "mkdir -p \
  '$REMOTE_ROOT/workspace/core4d/scripts/{{EXP_ID}}' \
  '$REMOTE_ROOT/workspace/core4d/scripts/train' \
  '$REMOTE_ROOT/{{RESULT_ROOT}}/cem/${STAGE}' \
  '$REMOTE_ROOT/logs/{{EXP_ID}}/cem/${STAGE}' \
  '$REMOTE_ROOT/examples/config/override'"

retry rsync -az workspace/core4d/scripts/{{EXP_ID}}/ "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/scripts/{{EXP_ID}}/"
retry rsync -az workspace/core4d/scripts/train/{{TRAIN_SCRIPT}} "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/scripts/train/"
retry rsync -az examples/config/override/{{OVERRIDE_PREFIX}}_*.yaml "${REMOTE}:${REMOTE_ROOT}/examples/config/override/"

retry ssh "$REMOTE" "cd '$REMOTE_ROOT' && \
  bash -n workspace/core4d/scripts/train/{{TRAIN_SCRIPT}} && \
  bash workspace/core4d/scripts/train/{{TRAIN_SCRIPT}} list '$STAGE' 0 remote-gpu0 >/tmp/{{EXP_ID_LOWER}}_gpu0.list && \
  bash workspace/core4d/scripts/train/{{TRAIN_SCRIPT}} list '$STAGE' 0 remote-gpu1 >/tmp/{{EXP_ID_LOWER}}_gpu1.list && \
  echo remote-gpu0=\$(wc -l </tmp/{{EXP_ID_LOWER}}_gpu0.list) remote-gpu1=\$(wc -l </tmp/{{EXP_ID_LOWER}}_gpu1.list)"

retry ssh "$REMOTE" "cd '$REMOTE_ROOT' && \
  tmux new-session -d -s '$SESSION' \"bash -lc 'set -euo pipefail; \
    echo {{EXP_ID}} remote session $SESSION stage=$STAGE; \
    (bash workspace/core4d/scripts/train/{{TRAIN_SCRIPT}} remote-gpu0 $STAGE 0) & PID0=\\\$!; \
    (bash workspace/core4d/scripts/train/{{TRAIN_SCRIPT}} remote-gpu1 $STAGE 1) & PID1=\\\$!; \
    echo launched PID0=\\\$PID0 PID1=\\\$PID1; \
    wait \\\$PID0; echo remote-gpu0 done; \
    wait \\\$PID1; echo remote-gpu1 done; \
    echo {{EXP_ID}} remote complete'\""

echo "Started remote tmux session: ${SESSION}"
echo "Monitor: ssh ${REMOTE} \"tmux capture-pane -t ${SESSION} -p | tail -80\""
