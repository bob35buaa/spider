#!/usr/bin/env bash
# Conservatively monitor local + remote E179 Full and pull after two absences.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
REMOTE="${REMOTE:-batchcom@61.172.170.106}"
REMOTE_SSH_PORT="${REMOTE_SSH_PORT:-30409}"
REMOTE_SSH_KEY="${REMOTE_SSH_KEY:-/home/ubuntu/.ssh/id_rsa_tianyiyun}"
POLL_INTERVAL="${POLL_INTERVAL:-300}"
WATCH_ONCE="${WATCH_ONCE:-0}"
RESULT_ROOT="workspace/core4d/results/E179"
LOCAL_JSON="$RESULT_ROOT/s0_environment/local_full_latest.json"
REMOTE_JSON="$RESULT_ROOT/s0_environment/a100_full_latest.json"
PULL_SCRIPT="workspace/core4d/scripts/launch/active/pull_E179_remote_a100_results.sh"
LOG="logs/E179/monitor/watch_full.log"
mkdir -p "$(dirname "$LOG")"

[ -f "$LOCAL_JSON" ] && [ -f "$REMOTE_JSON" ] || {
  echo "missing E179 local/remote Full execution pointers" >&2
  exit 3
}
mapfile -t SESSIONS < <(
  "$PYTHON_BIN" - "$LOCAL_JSON" "$REMOTE_JSON" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["session"])
print(json.load(open(sys.argv[2], encoding="utf-8"))["session"])
PY
)
LOCAL_SESSION="${SESSIONS[0]}"
REMOTE_SESSION="${SESSIONS[1]}"
SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=4
  -p "$REMOTE_SSH_PORT"
  -i "$REMOTE_SSH_KEY"
)

absent_checks=0
while true; do
  local_running=0
  remote_running=0
  tmux has-session -t "$LOCAL_SESSION" 2>/dev/null && local_running=1
  set +e
  ssh "${SSH_OPTS[@]}" "$REMOTE" \
    "tmux has-session -t '$REMOTE_SESSION' 2>/dev/null"
  remote_rc=$?
  set -e
  if [ "$remote_rc" -eq 0 ]; then
    remote_running=1
  elif [ "$remote_rc" -ne 1 ]; then
    remote_running=1
    echo "[$(date -Is)] SSH unavailable rc=$remote_rc; remote conservatively running" \
      | tee -a "$LOG"
  fi

  if [ "$local_running" -eq 0 ] && [ "$remote_running" -eq 0 ]; then
    absent_checks=$((absent_checks + 1))
  else
    absent_checks=0
  fi
  echo "[$(date -Is)] local=$local_running remote=$remote_running absent=$absent_checks/2" \
    | tee -a "$LOG"

  if [ "$absent_checks" -ge 2 ]; then
    bash "$PULL_SCRIPT" | tee -a "$LOG"
    exit "${PIPESTATUS[0]}"
  fi
  if [ "$WATCH_ONCE" = "1" ]; then
    exit 0
  fi
  sleep "$POLL_INTERVAL"
done
