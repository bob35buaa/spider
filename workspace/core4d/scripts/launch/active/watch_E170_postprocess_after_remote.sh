#!/usr/bin/env bash
# Wait for both E170 remote sessions to end, then force a canonical full pull
# before running the strict render/eval/workbook postprocess.  This watcher is
# read-only with respect to remote processes: it never kills, pauses, or sends
# input to another job.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RESULT_ROOT="workspace/core4d/results/E170"
MAIN_SESSION="${MAIN_SESSION:-$(cat "$RESULT_ROOT/s0_environment/latest_full_session.txt" 2>/dev/null || true)}"
RECOVERY_SESSION="${RECOVERY_SESSION:-$(cat "$RESULT_ROOT/s0_environment/latest_recovery_full_session.txt" 2>/dev/null || true)}"
[ -n "$MAIN_SESSION" ] || { echo "missing main full session" >&2; exit 1; }
[ -n "$RECOVERY_SESSION" ] || { echo "missing recovery full session" >&2; exit 1; }

REMOTE="${REMOTE:-batchcom@61.172.170.106}"
PORT="${REMOTE_SSH_PORT:-30409}"
KEY="${REMOTE_SSH_KEY:-$HOME/.ssh/id_rsa_tianyiyun}"
POLL_SECONDS="${E170_FINALIZER_POLL_SECONDS:-300}"
ABSENT_CONFIRMATIONS="${E170_FINALIZER_ABSENT_CONFIRMATIONS:-3}"
LOG="logs/E170/monitor/watch_postprocess_${MAIN_SESSION}_${RECOVERY_SESSION}.log"
mkdir -p "$(dirname "$LOG")"

probe_session() {
  local session="$1" rc
  set +e
  ssh -o BatchMode=yes -o ConnectTimeout=20 -p "$PORT" -i "$KEY" "$REMOTE" \
    "tmux has-session -t '$session'" >/dev/null 2>&1
  rc=$?
  set -e
  return "$rc"
}

absent_count=0
while [ "$absent_count" -lt "$ABSENT_CONFIRMATIONS" ]; do
  any_alive=0
  probe_error=0
  state=()
  for session in "$MAIN_SESSION" "$RECOVERY_SESSION"; do
    if probe_session "$session"; then
      any_alive=1
      state+=("${session}=alive")
    else
      rc=$?
      if [ "$rc" -eq 1 ]; then
        state+=("${session}=ended")
      else
        probe_error=1
        state+=("${session}=probe_error_${rc}")
      fi
    fi
  done
  if [ "$probe_error" -eq 1 ] || [ "$any_alive" -eq 1 ]; then
    absent_count=0
  else
    absent_count=$((absent_count + 1))
  fi
  printf '[%s] %s absent_confirmation=%s/%s\n' \
    "$(date -Is)" "${state[*]}" "$absent_count" "$ABSENT_CONFIRMATIONS" | tee -a "$LOG"
  if [ "$absent_count" -lt "$ABSENT_CONFIRMATIONS" ]; then sleep "$POLL_SECONDS"; fi
done

printf '[%s] both sessions ended; canonical 24-row full pull\n' "$(date -Is)" | tee -a "$LOG"
SESSION="$MAIN_SESSION" bash workspace/core4d/scripts/launch/active/pull_E170_remote_a100_results.sh full | tee -a "$LOG"
printf '[%s] strict pull passed; start postprocess\n' "$(date -Is)" | tee -a "$LOG"
bash workspace/core4d/scripts/launch/active/postprocess_E170_after_full.sh | tee -a "$LOG"
printf '[%s] postprocess complete\n' "$(date -Is)" | tee -a "$LOG"
