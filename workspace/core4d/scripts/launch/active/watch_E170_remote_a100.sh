#!/usr/bin/env bash
# Persistent E170 watcher: incremental pull, then strict/partial artifact audit.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
MODE="${1:-full}"; POLL_SECONDS="${E170_WATCH_POLL_SECONDS:-300}"
RESULT_ROOT="workspace/core4d/results/E170"; LATEST="$RESULT_ROOT/s0_environment/latest_${MODE}_session.txt"
SESSION="${SESSION:-$(cat "$LATEST" 2>/dev/null || true)}"; [ -n "$SESSION" ] || { echo "missing session" >&2; exit 1; }
REMOTE="${REMOTE:-batchcom@61.172.170.106}"; PORT="${REMOTE_SSH_PORT:-30409}"; KEY="${REMOTE_SSH_KEY:-$HOME/.ssh/id_rsa_tianyiyun}"
LOG="logs/E170/monitor/watch_${MODE}_${SESSION}.log"; mkdir -p "$(dirname "$LOG")"
while ssh -o BatchMode=yes -o ConnectTimeout=20 -p "$PORT" -i "$KEY" "$REMOTE" "tmux has-session -t '$SESSION'" >/dev/null 2>&1; do
  printf '[%s] session=%s running; incremental pull\n' "$(date -Is)" "$SESSION" | tee -a "$LOG"
  ALLOW_INCOMPLETE=1 SESSION="$SESSION" bash workspace/core4d/scripts/launch/active/pull_E170_remote_a100_results.sh "$MODE" >>"$LOG" 2>&1 || true
  sleep "$POLL_SECONDS"
done
printf '[%s] session=%s ended; final pull\n' "$(date -Is)" "$SESSION" | tee -a "$LOG"
SESSION="$SESSION" bash workspace/core4d/scripts/launch/active/pull_E170_remote_a100_results.sh "$MODE" | tee -a "$LOG"
