#!/usr/bin/env bash
# E182 remote Ada 0/1 deployment/worker entrypoint. Never alters existing jobs.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

STAGE="${1:-preflight}"
REMOTE_HOST="${ADA_HOST:-spider-remote}"
EXPECTED_GPUS="${ADA_EXPECTED_GPUS:-0 1}"
POLICY_GPUS="${ADA_POLICY_GPUS:-0,1}"
ALLOW_OVERLAP="${ALLOW_EXISTING_COMPUTE_OVERLAP:-1}"
SCRIPT="workspace/core4d/scripts/experiments/E182/deploy_remote_snapshot.py"
RUNTIME_SCRIPT="workspace/core4d/scripts/experiments/E182/runtime_inputs.py"
S0_RESULT="workspace/core4d/results/E182/s0_environment"
S1_LOG="logs/E182/s1"
REMOTE_PYTHON="/home/xiayb/pHRI_workspace/spider/.venv/bin/python"
MIN_FREE_MIB="${E182_S1_MIN_FREE_MIB:-16000}"
mkdir -p "$S1_LOG"

if [[ "$EXPECTED_GPUS" != "0 1" || "$POLICY_GPUS" != "0,1" ]]; then
  echo "E182 remote GPU allowlist must be exactly 0,1." >&2
  exit 2
fi
if [[ "$ALLOW_OVERLAP" != "1" ]]; then
  echo "E182 requires authorized compute overlap." >&2
  exit 2
fi

deploy_query_runtime() {
  uv run python "$SCRIPT" freeze \
    2>&1 | tee "$S1_LOG/source_freeze.log"
  uv run python "$SCRIPT" deploy --remote-host "$REMOTE_HOST" \
    2>&1 | tee "$S1_LOG/source_deploy.log"
  REMOTE_ROOT="$(uv run python -c \
    'import json; print(json.load(open("workspace/core4d/results/E182/s0_environment/remote_deployment_manifest.json"))["remote_root"])')"
  uv run python "$RUNTIME_SCRIPT" freeze \
    2>&1 | tee "$S1_LOG/runtime_input_freeze.log"
  uv run python "$RUNTIME_SCRIPT" deploy \
    --remote-host "$REMOTE_HOST" \
    --remote-root "$REMOTE_ROOT" \
    2>&1 | tee "$S1_LOG/runtime_input_deploy.log"
}

snapshot_and_validate_gpus() {
  local snapshot line gpu free_mib
  snapshot="$(ssh "$REMOTE_HOST" \
    "nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,memory.free --format=csv,noheader,nounits")"
  printf '%s\n' "$snapshot" | tee "$S1_LOG/remote_gpu_before_query_tape.csv"
  ssh "$REMOTE_HOST" \
    "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits" \
    > "$S1_LOG/remote_compute_processes_before_query_tape.csv" || true
  for gpu in 0 1; do
    line="$(printf '%s\n' "$snapshot" | awk -F ',' -v wanted="$gpu" '$1+0==wanted {print}')"
    if [[ -z "$line" || "$line" != *"RTX 6000 Ada"* ]]; then
      echo "E182 expected RTX 6000 Ada GPU $gpu." >&2
      exit 2
    fi
    free_mib="$(printf '%s\n' "$line" | awk -F ',' '{gsub(/ /,"",$6); print $6+0}')"
    if (( free_mib < MIN_FREE_MIB )); then
      echo "E182 Ada GPU $gpu free=${free_mib}MiB < ${MIN_FREE_MIB}MiB; delaying only E182." >&2
      exit 2
    fi
  done
}

launch_query_worker() {
  local gpu="$1" case_id="$2" execution_id session remote_log worker_command quoted_command remote_launch
  execution_id="$(basename "$(dirname "$REMOTE_ROOT")")"
  session="e182_s1_ada${gpu}_${execution_id#e182_}"
  remote_log="$REMOTE_ROOT/logs/E182/s1/remote_ada${gpu}_${case_id}.log"
  worker_command="E182_PYTHON_BIN=$REMOTE_PYTHON bash workspace/core4d/scripts/launch/active/run_E182_query_tape_worker.sh $case_id $gpu > $remote_log 2>&1"
  if ssh "$REMOTE_HOST" "tmux has-session -t '$session'" >/dev/null 2>&1; then
    echo "E182 query session already exists; leaving it unchanged: $session"
    return
  fi
  printf -v quoted_command '%q' "$worker_command"
  printf -v remote_launch \
    'mkdir -p %q && tmux new-session -d -s %q -c %q %s' \
    "$(dirname "$remote_log")" "$session" "$REMOTE_ROOT" "$quoted_command"
  ssh "$REMOTE_HOST" "$remote_launch"
  echo "E182_QUERY_REMOTE_LAUNCHED session=$session gpu=$gpu case=$case_id"
}

case "$STAGE" in
  preflight)
    uv run python "$SCRIPT" deploy --remote-host "$REMOTE_HOST"
    ;;
  query-tape)
    deploy_query_runtime
    snapshot_and_validate_gpus
    launch_query_worker 0 bucket003_20231018_001_p1
    launch_query_worker 1 bucket004_20231002_021_p1
    ;;
  full)
    echo "E182 remote Full worker is not implemented yet; do not launch manually." >&2
    exit 2
    ;;
  *)
    echo "usage: $0 {preflight|query-tape|full}" >&2
    exit 2
    ;;
esac
