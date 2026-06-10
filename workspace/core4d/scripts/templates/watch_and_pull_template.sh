#!/usr/bin/env bash
# {{EXP_ID}}: Watch local/remote queues, pull remote outputs, then run eval.
# Auto-generated from watch_and_pull_template.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# --- Configuration (override via env or edit below) ---
REMOTE="${REMOTE:-{{REMOTE_HOST}}}"
LOCAL_SESSION="${LOCAL_SESSION:-{{LOCAL_TMUX}}}"
REMOTE_SESSION="${REMOTE_SESSION:-{{REMOTE_TMUX}}}"
EXPECTED_NPZ_COUNT="${EXPECTED_NPZ_COUNT:-{{EXPECTED_NPZ_COUNT}}}"
PULL_SCRIPT="${PULL_SCRIPT:-{{PULL_SCRIPT}}}"
EVAL_SCRIPT="${EVAL_SCRIPT:-{{EVAL_SCRIPT}}}"
RESULT_ROOT="${RESULT_ROOT:-{{RESULT_ROOT}}}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-{{POLL_INTERVAL}}}"

# --- Logging ---
LOG_DIR="logs/{{EXP_ID}}/monitor"
LOG_FILE="$LOG_DIR/watch_and_pull_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

# --- Helper functions ---
local_running() {
  tmux has-session -t "$LOCAL_SESSION" 2>/dev/null
}

remote_reachable() {
  ssh -o ConnectTimeout=12 "$REMOTE" "true" 2>/dev/null
}

remote_running() {
  ssh -o ConnectTimeout=12 "$REMOTE" \
    "tmux has-session -t '$REMOTE_SESSION' 2>/dev/null"
}

# --- Main logic ---
{
  echo "[$(date '+%F %T')] {{EXP_ID}} watch_and_pull start"
  echo "[$(date '+%F %T')] config: remote=$REMOTE local_session=$LOCAL_SESSION remote_session=$REMOTE_SESSION"
  echo "[$(date '+%F %T')] config: expected_npz=$EXPECTED_NPZ_COUNT poll_interval=${INTERVAL_SECONDS}s"

  finished_checks=0

  while true; do
    local_run=0
    remote_run=0
    remote_ok=0

    # Check local tmux
    local_running && local_run=1 || true

    # Check remote: if SSH fails, assume still running (hardened)
    if remote_reachable; then
      remote_ok=1
      remote_running && remote_run=1 || true
    else
      # SSH unreachable → conservative: assume remote is still running
      remote_run=1
    fi

    echo "[$(date '+%F %T')] local_running=${local_run} remote_running=${remote_run} remote_reachable=${remote_ok}"

    # Require both sides done
    if [ "$local_run" -eq 0 ] && [ "$remote_run" -eq 0 ] && [ "$remote_ok" -eq 1 ]; then
      finished_checks=$((finished_checks + 1))
    else
      finished_checks=0
    fi

    # Consecutive 2 checks confirm completion
    if [ "$finished_checks" -ge 2 ]; then
      echo "[$(date '+%F %T')] Both sides confirmed finished (2 consecutive checks)"
      break
    fi

    sleep "$INTERVAL_SECONDS"
  done

  # --- Pull remote results ---
  echo "[$(date '+%F %T')] Pulling remote results via: $PULL_SCRIPT"
  bash "$PULL_SCRIPT"

  # --- Validate artifact count ---
  got_npz="$(find "$RESULT_ROOT" -name '*.npz' 2>/dev/null | wc -l)"
  echo "[$(date '+%F %T')] Artifact count: expected=${EXPECTED_NPZ_COUNT} got_npz=${got_npz}"

  if [ "$got_npz" -lt "$EXPECTED_NPZ_COUNT" ]; then
    echo "[$(date '+%F %T')] ERROR: NPZ count mismatch (got ${got_npz}, expected ${EXPECTED_NPZ_COUNT})" >&2
    exit 2
  fi

  # --- Run eval ---
  echo "[$(date '+%F %T')] Running eval: $EVAL_SCRIPT"
  bash "$EVAL_SCRIPT"

  echo "[$(date '+%F %T')] {{EXP_ID}} watch_and_pull complete"
} 2>&1 | tee "$LOG_FILE"
