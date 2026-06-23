#!/usr/bin/env bash
# Watch E167 local/remote CEM, pull remote results, then run B2 postprocess/eval.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REMOTE="${REMOTE:-spider-remote}"
LOCAL_SESSION="${LOCAL_SESSION:-}"
REMOTE_SESSION="${REMOTE_SESSION:-}"
REMOTE_SESSIONS="${REMOTE_SESSIONS:-${REMOTE_SESSION}}"
POLL_INTERVAL="${POLL_INTERVAL:-600}"
STABLE_DONE_POLLS="${STABLE_DONE_POLLS:-2}"
EXPECTED_CEM_NPZ="${EXPECTED_CEM_NPZ:-14}"
EXPECTED_B2_NPZ="${EXPECTED_B2_NPZ:-7}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
LOG_DIR="logs/E167/monitor"
RESULTS="workspace/core4d/results/E167/holosoma_zonly/cem/full"
POST="workspace/core4d/results/E167/holosoma_zonly/postprocess/full"

mkdir -p "$LOG_DIR" "$RESULTS" "$POST"
LOG_FILE="$LOG_DIR/watch_E167_zonly_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

local_running() {
  if [ -z "$LOCAL_SESSION" ]; then
    return 1
  fi
  tmux has-session -t "$LOCAL_SESSION" 2>/dev/null
}

remote_running() {
  if [ -z "$REMOTE_SESSIONS" ]; then
    return 1
  fi
  if ! ssh "$REMOTE" "true" >/dev/null 2>&1; then
    log "remote ssh unavailable; conservatively treating remote as running"
    return 0
  fi
  local session
  for session in $REMOTE_SESSIONS; do
    if ssh "$REMOTE" "tmux has-session -t '$session'" >/dev/null 2>&1; then
      return 0
    fi
  done
  return 1
}

count_cem_npz() {
  find "$RESULTS" -maxdepth 1 -name 'E167_*.npz' | wc -l | awk '{print $1}'
}

count_b2_npz() {
  find "$POST" -maxdepth 1 -name 'E167_*_E167A_B2.npz' | wc -l | awk '{print $1}'
}

log "E167 watcher start local_session=${LOCAL_SESSION:-none} remote_sessions=${REMOTE_SESSIONS:-none} poll=${POLL_INTERVAL}s"
stable_done=0
while true; do
  if [ -n "$REMOTE_SESSION" ]; then
    bash workspace/core4d/scripts/launch/active/pull_E167_remote_results.sh full | tee -a "$LOG_FILE" || true
  fi
  cem_count="$(count_cem_npz)"
  b2_count="$(count_b2_npz)"
  lr=0
  rr=0
  local_running && lr=1 || lr=0
  remote_running && rr=1 || rr=0
  log "status local_running=$lr remote_running=$rr cem_npz=${cem_count}/${EXPECTED_CEM_NPZ} b2_npz=${b2_count}/${EXPECTED_B2_NPZ}"

  if [ "$lr" -eq 0 ] && [ "$rr" -eq 0 ]; then
    stable_done=$((stable_done + 1))
  else
    stable_done=0
  fi

  if [ "$cem_count" -ge "$EXPECTED_CEM_NPZ" ]; then
    log "CEM artifacts complete; running E167 B2 postprocess"
    bash workspace/core4d/scripts/launch/active/run_E167_B2_postprocess.sh full 2>&1 | tee -a "$LOG_FILE"
    b2_count="$(count_b2_npz)"
    log "postprocess count=${b2_count}/${EXPECTED_B2_NPZ}; running full eval"
    bash workspace/core4d/scripts/eval/wrappers/eval_E167_zonly.sh full 2>&1 | tee -a "$LOG_FILE"
    log "E167 watcher complete"
    exit 0
  fi

  if [ "$stable_done" -ge "$STABLE_DONE_POLLS" ]; then
    log "sessions ended but CEM artifacts incomplete: ${cem_count}/${EXPECTED_CEM_NPZ}"
    exit 1
  fi

  sleep "$POLL_INTERVAL"
done
