#!/usr/bin/env bash
# Monitor one E169 session and periodically pull completed artifacts.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
MODE="${1:-full}"
POLL_SECONDS="${E169_WATCH_POLL_SECONDS:-600}"
SESSION_FILE="workspace/core4d/results/E169/s0_environment/latest_${MODE}_session.txt"
SESSION="${SESSION:-$(cat "$SESSION_FILE")}"
REMOTE="${REMOTE:-batchcom@61.172.170.106}"
PORT="${REMOTE_SSH_PORT:-30409}"
KEY="${REMOTE_SSH_KEY:-$HOME/.ssh/id_rsa_tianyiyun}"
while ssh -o BatchMode=yes -o ConnectTimeout=20 -p "$PORT" -i "$KEY" "$REMOTE" "tmux has-session -t '$SESSION'" >/dev/null 2>&1; do
  ALLOW_INCOMPLETE=1 SESSION="$SESSION" bash workspace/core4d/scripts/launch/active/pull_E169_remote_a100_results.sh "$MODE" || true
  sleep "$POLL_SECONDS"
done
SESSION="$SESSION" bash workspace/core4d/scripts/launch/active/pull_E169_remote_a100_results.sh "$MODE"
