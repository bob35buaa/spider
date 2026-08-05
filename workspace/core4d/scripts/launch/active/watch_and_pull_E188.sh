#!/usr/bin/env bash
# Fail-closed E188 canary/full monitor, pull, promotion, and remaining-queue launch.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${E188_PYTHON_BIN:-.venv/bin/python}"
RUNNER=workspace/core4d/scripts/experiments/E188/run_full_queue.py
LOCAL_LAUNCH=workspace/core4d/scripts/launch/active/run_E188_local.sh
REMOTE_LAUNCH=workspace/core4d/scripts/launch/active/run_E188_remote_a100.sh
REMOTE_PULL=workspace/core4d/scripts/launch/active/pull_E188_remote_a100_results.sh
HOST="${E188_A100_HOST:-61.172.170.106}"
PORT="${E188_A100_PORT:-30409}"
IDENTITY="${E188_A100_IDENTITY:-/home/ubuntu/.ssh/id_rsa_tianyiyun}"
POLL_SECONDS="${E188_POLL_SECONDS:-30}"
WATCH_MODE="${1:-auto}"
if [[ "$WATCH_MODE" != auto && "$WATCH_MODE" != full-only ]]; then
  echo "usage: $0 {auto|full-only}" >&2
  exit 2
fi
QUEUE_V2=workspace/core4d/results/E188/s5_full/queue_speed_rebalanced_v2
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -p "$PORT" -i "$IDENTITY" "batchcom@$HOST")

log() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"; }

deployment_root() {
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["remote_root"])' "$1"
}

wait_canary() {
  local local_manifest=workspace/core4d/results/E188/s4_canary/rows/bucket003_20231018_003_p1/manifest.json
  local deployment=workspace/core4d/results/E188/s4_canary/deployment/remote_deployment_manifest.json
  local remote_root session_tag remote_count
  remote_root="$(deployment_root "$deployment")"
  session_tag="$(basename "$(dirname "$remote_root")")"
  while true; do
    remote_count="$("${SSH[@]}" "n=0; for case_id in bucket007_20231003_2_021_p1 bucket007_20231020_055_p1; do test -f '$remote_root/workspace/core4d/results/E188/s4_canary/rows/'\"\$case_id\"'/manifest.json' && n=\$((n+1)); done; echo \$n")"
    local local_count=0
    [[ -f "$local_manifest" ]] && local_count=1
    log "canary manifests local=$local_count/1 remote=$remote_count/2"
    if [[ "$local_count" == 1 && "$remote_count" == 2 ]]; then
      break
    fi
    if [[ "$local_count" == 0 ]] && ! tmux has-session -t e188_canary_local0 >/dev/null 2>&1; then
      log "FAIL local canary session exited without manifest"
      return 2
    fi
    for gpu in 4 5; do
      local case_id=bucket007_20231003_2_021_p1
      [[ "$gpu" == 5 ]] && case_id=bucket007_20231020_055_p1
      if ! "${SSH[@]}" "test -f '$remote_root/workspace/core4d/results/E188/s4_canary/rows/$case_id/manifest.json' || tmux has-session -t 'e188_canary_a100_${gpu}_${session_tag}'"; then
        log "FAIL remote canary gpu=$gpu exited without manifest"
        return 2
      fi
    done
    sleep "$POLL_SECONDS"
  done
}

wait_full() {
  local deployment=workspace/core4d/results/E188/s5_full/deployment/remote_deployment_manifest.json
  local remote_root session_tag remote_count local_count
  local -a local_cases remote_cases
  mapfile -t local_cases < <(awk -F '\t' 'NR>1 && $1=="False" {print $2}' "$QUEUE_V2/local-0.tsv")
  mapfile -t remote_cases < <(for worker in a100-4 a100-5; do awk -F '\t' 'NR>1 && $1=="False" {print $2}' "$QUEUE_V2/$worker.tsv"; done)
  local local_expected=${#local_cases[@]}
  local remote_expected=${#remote_cases[@]}
  remote_root="$(deployment_root "$deployment")"
  session_tag="$(basename "$(dirname "$remote_root")")"
  while true; do
    local_count=0
    for case_id in "${local_cases[@]}"; do
      [[ -f "workspace/core4d/results/E188/s5_full/rows/$case_id/manifest.json" ]] && local_count=$((local_count+1))
    done
    remote_count="$("${SSH[@]}" "n=0; for case_id in ${remote_cases[*]}; do test -f '$remote_root/workspace/core4d/results/E188/s5_full/rows/'\"\$case_id\"'/manifest.json' && n=\$((n+1)); done; echo \$n")"
    log "full manifests local=$local_count/$local_expected remote=$remote_count/$remote_expected"
    if [[ "$local_count" == "$local_expected" && "$remote_count" == "$remote_expected" ]]; then
      break
    fi
    if [[ "$local_count" -lt "$local_expected" ]] && ! tmux has-session -t e188_full_local0 >/dev/null 2>&1; then
      log "FAIL local full session exited before $local_expected/$local_expected manifests"
      return 2
    fi
    for gpu in 4 5; do
      if ! "${SSH[@]}" "tmux has-session -t 'e188_full_a100_${gpu}_${session_tag}'"; then
        local expected
        expected="$(awk -F '\t' 'NR>1 && $1=="False" {n++} END{print n+0}' "$QUEUE_V2/a100-$gpu.tsv")"
        local worker_count
        worker_count="$("${SSH[@]}" "test -f '$remote_root/workspace/core4d/results/E188/s5_full/worker_state/a100-$gpu.json' && echo '$expected' || echo 0")"
        if [[ "$worker_count" != "$expected" ]]; then
          log "FAIL remote full gpu=$gpu exited without worker completion state"
          return 2
        fi
      fi
    done
    sleep "$POLL_SECONDS"
  done
}

if [[ "$WATCH_MODE" == auto ]]; then
  log "watching E188 canaries"
  wait_canary
  log "canary compute manifests complete; pulling remote rows"
  bash "$REMOTE_PULL" canary
  PYTHONPATH=workspace/core4d/scripts/experiments/E188 "$PYTHON_BIN" "$RUNNER" freeze-canary-gate
  PYTHONPATH=workspace/core4d/scripts/experiments/E188 "$PYTHON_BIN" "$RUNNER" promote-canaries
  log "canary gate/promotion PASS; launching speed-rebalanced remaining12"
  bash "$REMOTE_LAUNCH" full
  bash "$LOCAL_LAUNCH" full
else
  log "watching already-launched speed-rebalanced remaining12"
fi
wait_full
log "remaining Full manifests complete; pulling remote rows"
bash "$REMOTE_PULL" full
PYTHONPATH=workspace/core4d/scripts/experiments/E188 "$PYTHON_BIN" "$RUNNER" closure
log "E188 CEM closure PASS 15/15; evaluation/rendering remain"
