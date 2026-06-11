#!/usr/bin/env bash
# Wait for E106 local/remote queues, pull remote outputs, then run one unified eval.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"
LOCAL_SESSION_PREFIX="${LOCAL_SESSION_PREFIX:-e106_local_full_}"
REMOTE_SESSION_PREFIX="${REMOTE_SESSION_PREFIX:-e106_remote_full_}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
LOG_DIR="logs/E106/monitor"
LOG_FILE="$LOG_DIR/wait_pull_eval_${STAGE}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

count_local_npz() {
  find "workspace/core4d/results/E106/cem/${STAGE}" -maxdepth 1 -name 'E106B*.npz' 2>/dev/null | wc -l
}

count_remote_npz() {
  ssh -o ConnectTimeout=12 "$REMOTE" \
    "cd '$REMOTE_REPO' && find 'workspace/core4d/results/E106/cem/${STAGE}' -maxdepth 1 -name 'E106B*.npz' 2>/dev/null | wc -l"
}

local_running() {
  tmux list-sessions 2>/dev/null | grep -q "$LOCAL_SESSION_PREFIX"
}

remote_running() {
  ssh -o ConnectTimeout=12 "$REMOTE" \
    "tmux list-sessions 2>/dev/null | grep -q '$REMOTE_SESSION_PREFIX'"
}

remote_reachable() {
  ssh -o ConnectTimeout=12 "$REMOTE" "true"
}

{
  echo "[$(date '+%F %T')] E106 wait/pull/eval start stage=${STAGE}"
  finished_checks=0
  while true; do
    local_run=0
    remote_run=0
    remote_ok=0
    local_running && local_run=1 || true
    if remote_reachable; then
      remote_ok=1
      remote_running && remote_run=1 || true
    else
      remote_run=1
    fi
    local_count="$(count_local_npz)"
    remote_count="$(count_remote_npz || echo unknown)"
    echo "[$(date '+%F %T')] local_running=${local_run} remote_running=${remote_run} remote_reachable=${remote_ok} local_npz=${local_count} remote_npz=${remote_count}"
    if [ "$local_run" -eq 0 ] && [ "$remote_run" -eq 0 ] && [ "$remote_ok" -eq 1 ]; then
      finished_checks=$((finished_checks + 1))
    else
      finished_checks=0
    fi
    if [ "$finished_checks" -ge 2 ]; then
      break
    fi
    sleep "$INTERVAL_SECONDS"
  done

  echo "[$(date '+%F %T')] queues finished; pulling remote results"
  bash workspace/core4d/scripts/pull_E106_remote_results.sh "$STAGE"

  expected="$(awk -F '\t' 'NF && $1 !~ /^#/ {n++} END {print n+0}' workspace/core4d/scripts/E106/variants.tsv)"
  got_npz="$(find "workspace/core4d/results/E106/cem/${STAGE}" -maxdepth 1 -name 'E106B*.npz' 2>/dev/null | wc -l)"
  got_mp4="$(find "workspace/core4d/results/E106/cem/${STAGE}" -maxdepth 1 -name 'E106B*.mp4' 2>/dev/null | wc -l)"
  echo "[$(date '+%F %T')] pulled counts expected=${expected} npz=${got_npz} mp4=${got_mp4}"
  if [ "$got_npz" -ne "$expected" ] || [ "$got_mp4" -ne "$expected" ]; then
    echo "E106 output count mismatch after pull." >&2
    exit 2
  fi

  echo "[$(date '+%F %T')] running unified eval"
  bash workspace/core4d/scripts/train/train_E106_box026_candidate_batch.sh eval "$STAGE" 0
  echo "[$(date '+%F %T')] E106 wait/pull/eval complete"
} 2>&1 | tee "$LOG_FILE"
