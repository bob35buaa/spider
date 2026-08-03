#!/usr/bin/env bash
# E187 immutable Full CEM workers for RTX 6000 Ada GPU0/1; never alters existing jobs.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-preflight}"
if [[ "$MODE" != "preflight" && "$MODE" != "full" ]]; then
  echo "usage: $0 {preflight|full}" >&2
  exit 2
fi

REMOTE_HOST="${ADA_HOST:-spider-remote}"
LOCAL_PYTHON="${E187_PYTHON_BIN:-.venv/bin/python}"
REMOTE_PYTHON="/home/xiayb/pHRI_workspace/spider/.venv/bin/python"
DEPLOYER="workspace/core4d/scripts/experiments/E187/deploy_remote_full.py"
DEPLOY_TEST="workspace/core4d/scripts/experiments/E187/test_remote_full_deployment.py"
RUNNER="workspace/core4d/scripts/experiments/E187/run_full_queue.py"
RUNNER_TEST="workspace/core4d/scripts/experiments/E187/test_full_queue_runner.py"
DEPLOYMENT="workspace/core4d/results/E187/s5_full/deployment/remote_deployment_manifest.json"
SNAPSHOT_ROOT="workspace/core4d/results/E187/s5_full/process_snapshots/remote"
MIN_FREE_MIB="${E187_REMOTE_MIN_FREE_MIB:-16000}"
mkdir -p "$SNAPSHOT_ROOT"

PYTHONPATH=workspace/core4d/scripts/experiments/E187 "$LOCAL_PYTHON" "$RUNNER_TEST"
PYTHONPATH=workspace/core4d/scripts/experiments/E187 "$LOCAL_PYTHON" "$DEPLOY_TEST"
PYTHONPATH=. "$LOCAL_PYTHON" "$RUNNER" register-promotions
PYTHONPATH=. "$LOCAL_PYTHON" "$DEPLOYER" freeze
PYTHONPATH=. "$LOCAL_PYTHON" "$DEPLOYER" deploy --remote-host "$REMOTE_HOST"

REMOTE_ROOT="$($LOCAL_PYTHON -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["remote_root"])' "$DEPLOYMENT")"
if [[ "$REMOTE_ROOT" != /home/xiayb/pHRI_workspace/e187_runs/e187_full_*/spider ]]; then
  echo "Refusing unsafe E187 Full remote root: $REMOTE_ROOT" >&2
  exit 2
fi

gpu_snapshot="$(ssh "$REMOTE_HOST" \
  'nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits')"
printf '%s\n' "$gpu_snapshot" | tee "$SNAPSHOT_ROOT/gpu_before.csv"
ssh "$REMOTE_HOST" \
  'nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits' \
  > "$SNAPSHOT_ROOT/compute_before.csv" || true
if [[ -s "$SNAPSHOT_ROOT/compute_before.csv" ]]; then
  echo "E187 Full remote compute-app list is non-empty; refusing without waiting or preemption." >&2
  exit 2
fi
for gpu in 0 1; do
  line="$(printf '%s\n' "$gpu_snapshot" | awk -F ',' -v wanted="$gpu" '$1+0==wanted {print}')"
  if [[ -z "$line" || "$line" != *"RTX 6000 Ada"* ]]; then
    echo "E187 Full expected RTX 6000 Ada GPU $gpu." >&2
    exit 2
  fi
  free_mib="$(printf '%s\n' "$line" | awk -F ',' '{gsub(/ /,"",$6); print $6+0}')"
  if (( free_mib < MIN_FREE_MIB )); then
    echo "E187 Full Ada GPU $gpu free=${free_mib}MiB < ${MIN_FREE_MIB}MiB; refusing without waiting." >&2
    exit 2
  fi
done

remote_preflight() {
  local worker="$1" gpu_id="$2"
  ssh "$REMOTE_HOST" \
    "cd '$REMOTE_ROOT' && PYTHONPATH=. MUJOCO_GL=egl '$REMOTE_PYTHON' '$RUNNER' preflight --worker '$worker' --gpu-id '$gpu_id' --python-bin '$REMOTE_PYTHON'"
}
remote_preflight remote-0 0
remote_preflight remote-1 1

if [[ "$MODE" == "preflight" ]]; then
  echo "E187_FULL_REMOTE_PREFLIGHT=PASS root=$REMOTE_ROOT"
  exit 0
fi

launch_worker() {
  local worker="$1" gpu_id="$2" execution_id session log_dir log_path worker_command quoted
  execution_id="$(basename "$(dirname "$REMOTE_ROOT")")"
  session="e187_a4_ada${gpu_id}_${execution_id#e187_full_}"
  log_dir="$REMOTE_ROOT/logs/E187/s5_full"
  log_path="$log_dir/launcher_${worker}.log"
  if ssh "$REMOTE_HOST" "tmux has-session -t '$session'" >/dev/null 2>&1; then
    echo "Exact E187 Full remote session already exists; leaving unchanged: $session"
    return
  fi
  worker_command="cd '$REMOTE_ROOT' && PYTHONPATH=. MUJOCO_GL=egl '$REMOTE_PYTHON' '$RUNNER' run --worker '$worker' --gpu-id '$gpu_id' --python-bin '$REMOTE_PYTHON' --output-root workspace/core4d/results/E187/s5_full > '$log_path' 2>&1"
  printf -v quoted '%q' "$worker_command"
  ssh "$REMOTE_HOST" "mkdir -p '$log_dir' && tmux new-session -d -s '$session' -c '$REMOTE_ROOT' $quoted"
  echo "E187_FULL_REMOTE_LAUNCHED session=$session worker=$worker gpu=$gpu_id"
}

launch_worker remote-0 0
launch_worker remote-1 1
