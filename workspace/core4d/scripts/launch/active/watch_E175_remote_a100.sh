#!/usr/bin/env bash
# Monitor one E175 A100 session; pull and validate after two absent checks.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-canary}"
if [ "$MODE" != "canary" ] && [ "$MODE" != "full" ]; then
  echo "usage: $0 {canary|full}" >&2
  exit 2
fi

REMOTE="${REMOTE:-batchcom@61.172.170.106}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/dataset-assist-0/xiayb/workspace/spider}"
REMOTE_SSH_PORT="${REMOTE_SSH_PORT:-30409}"
REMOTE_SSH_KEY="${REMOTE_SSH_KEY:-/home/ubuntu/.ssh/id_rsa_tianyiyun}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
POLL_INTERVAL="${POLL_INTERVAL:-300}"
WATCH_ONCE="${WATCH_ONCE:-0}"
LATEST_JSON="workspace/core4d/results/E175/s0_environment/a100_${MODE}_latest.json"
PULL_SCRIPT="workspace/core4d/scripts/launch/active/pull_E175_remote_a100_results.sh"

[ -f "$LATEST_JSON" ] || {
  echo "missing E175 execution pointer: $LATEST_JSON" >&2
  exit 3
}
SESSION="$(
  "$PYTHON_BIN" - "$LATEST_JSON" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["session"])
PY
)"
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
  set +e
  ssh "${SSH_OPTS[@]}" "$REMOTE" \
    "tmux has-session -t '$SESSION' 2>/dev/null"
  remote_rc=$?
  set -e
  timestamp="$(date -Is)"

  if [ "$remote_rc" -eq 0 ]; then
    absent_checks=0
    echo "[$timestamp] E175 $MODE session=$SESSION running"
    ssh "${SSH_OPTS[@]}" "$REMOTE" \
      "tmux capture-pane -t '$SESSION' -p 2>/dev/null | tail -20; \
       nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
         --format=csv,noheader,nounits" || true
  elif [ "$remote_rc" -eq 1 ]; then
    absent_checks=$((absent_checks + 1))
    echo "[$timestamp] E175 $MODE session absent confirmation=$absent_checks/2"
    if [ "$absent_checks" -ge 2 ]; then
      bash "$PULL_SCRIPT" "$MODE"
      exit $?
    fi
  else
    absent_checks=0
    echo "[$timestamp] SSH unavailable rc=$remote_rc; conservatively treating session as running" >&2
  fi

  if [ "$WATCH_ONCE" = "1" ]; then
    exit 0
  fi
  sleep "$POLL_INTERVAL"
done
