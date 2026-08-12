#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SCOPE="${1:-wave0}"
WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-0}"
POLL_SECONDS="${POLL_SECONDS:-30}"
REMOTE="${E196_ADA_REMOTE:-spider-remote}"
SSH_CONFIG_FILE="${E196_ADA_SSH_CONFIG:-}"
SSH_PORT="${E196_ADA_PORT:-}"
SSH_BIND_INTERFACE="${E196_ADA_BIND_INTERFACE:-}"
SSH_BASE=(ssh)
[ -n "$SSH_CONFIG_FILE" ] && SSH_BASE+=(-F "$SSH_CONFIG_FILE")
[ -n "$SSH_PORT" ] && SSH_BASE+=(-p "$SSH_PORT")
[ -n "$SSH_BIND_INTERFACE" ] && SSH_BASE+=(-o "BindInterface=$SSH_BIND_INTERFACE")
SSH_BASE+=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
SSH=("${SSH_BASE[@]}" "$REMOTE")

remote_session_state() {
  local attempt state
  for attempt in 1 2 3 4 5; do
    if state="$("${SSH[@]}" "if tmux has-session -t '$REMOTE_SESSION' 2>/dev/null; then echo running; else echo stopped; fi")"; then
      printf '%s\n' "$state"
      return 0
    fi
    printf '[%s] E196 %s remote state attempt %d/5 unavailable\n' \
      "$(date --iso-8601=seconds)" "$SCOPE" "$attempt" >&2
    [ "$attempt" -lt 5 ] && sleep 3
  done
  return 255
}
case "$SCOPE" in
  wave0) LOCAL_SESSION="E196_reference_wave0_local"; REMOTE_SESSION="E196_reference_wave0_ada" ;;
  full|remaining) LOCAL_SESSION="E196_reference_remaining_local"; REMOTE_SESSION="E196_reference_remaining_ada" ;;
  *) echo "unsupported scope: $SCOPE" >&2; exit 2 ;;
esac

running() {
  tmux has-session -t "$LOCAL_SESSION" 2>/dev/null && return 0
  local remote_state
  if ! remote_state="$(remote_session_state)"; then
    printf '[%s] E196 %s remote state unavailable; retaining wait gate\n' \
      "$(date --iso-8601=seconds)" "$SCOPE" >&2
    return 0
  fi
  [ "$remote_state" = "running" ] && return 0
  [ "$remote_state" = "stopped" ] || {
    echo "unexpected remote session state: $remote_state" >&2
    return 0
  }
  return 1
}

while running; do
  printf '[%s] E196 %s still running\n' "$(date --iso-8601=seconds)" "$SCOPE"
  if [ "$WAIT_FOR_COMPLETION" != "1" ]; then
    exit 3
  fi
  sleep "$POLL_SECONDS"
done

PULL_SCOPE="$SCOPE"
[ "$SCOPE" = "full" ] && PULL_SCOPE="full"
E196_ADA_REMOTE="$REMOTE" \
E196_ADA_SSH_CONFIG="$SSH_CONFIG_FILE" \
E196_ADA_PORT="$SSH_PORT" \
E196_ADA_BIND_INTERFACE="$SSH_BIND_INTERFACE" \
bash workspace/core4d/scripts/launch/active/pull_E196_reference_fix_remote_results.sh "$PULL_SCOPE"
if [ "$SCOPE" = "wave0" ]; then
  "$PYTHON_BIN" workspace/core4d/scripts/experiments/E196/audit_reference_fix.py --scope wave0 --require-all
  bash workspace/core4d/scripts/eval/wrappers/eval_E196_reference_fix.sh wave0
  echo "E196 wave0 landed gate passed; inspect evidence before launching remaining"
else
  "$PYTHON_BIN" workspace/core4d/scripts/experiments/E196/audit_reference_fix.py --scope full --require-all
  bash workspace/core4d/scripts/eval/wrappers/eval_E196_reference_fix.sh full
  bash workspace/core4d/scripts/launch/active/run_E196_reference_fix_render_all.sh
fi
