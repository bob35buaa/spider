#!/usr/bin/env bash
# E187 immutable RTX 6000 Ada GPU0/1 canary entrypoint; never alters existing jobs.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-preflight}"
if [[ "$MODE" == "full" ]]; then
  exec bash workspace/core4d/scripts/launch/active/run_E187_full_remote_a6000.sh full
fi
if [[ "$MODE" != "preflight" && "$MODE" != "canary" ]]; then
  echo "usage: $0 {preflight|canary|full}" >&2
  exit 2
fi

REMOTE_HOST="${ADA_HOST:-spider-remote}"
LOCAL_PYTHON="${E187_PYTHON_BIN:-.venv/bin/python}"
REMOTE_PYTHON="/home/xiayb/pHRI_workspace/spider/.venv/bin/python"
SCRIPT="workspace/core4d/scripts/experiments/E187/deploy_remote_canary.py"
RUNNER="workspace/core4d/scripts/experiments/E187/run_production_canary.py"
TEST="workspace/core4d/scripts/experiments/E187/test_remote_canary_deployment.py"
DEPLOYMENT="workspace/core4d/results/E187/s4_canary/deployment/remote_deployment_manifest.json"
SNAPSHOT_ROOT="workspace/core4d/results/E187/s4_canary/process_snapshots/remote"
MIN_FREE_MIB="${E187_REMOTE_MIN_FREE_MIB:-16000}"
mkdir -p "$SNAPSHOT_ROOT"

PYTHONPATH=. "$LOCAL_PYTHON" "$TEST"
PYTHONPATH=. "$LOCAL_PYTHON" "$SCRIPT" freeze
PYTHONPATH=. "$LOCAL_PYTHON" "$SCRIPT" deploy --remote-host "$REMOTE_HOST"

REMOTE_ROOT="$($LOCAL_PYTHON -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["remote_root"])' "$DEPLOYMENT")"
if [[ "$REMOTE_ROOT" != /home/xiayb/pHRI_workspace/e187_runs/e187_*/spider ]]; then
  echo "Refusing unsafe E187 remote root: $REMOTE_ROOT" >&2
  exit 2
fi

gpu_snapshot="$(ssh "$REMOTE_HOST" \
  'nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits')"
printf '%s\n' "$gpu_snapshot" | tee "$SNAPSHOT_ROOT/gpu_before.csv"
ssh "$REMOTE_HOST" \
  'nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits' \
  > "$SNAPSHOT_ROOT/compute_before.csv" || true
if [[ -s "$SNAPSHOT_ROOT/compute_before.csv" ]]; then
  echo "E187 remote compute-app list is non-empty; refusing launch without waiting or preemption." >&2
  exit 2
fi
for gpu in 0 1; do
  line="$(printf '%s\n' "$gpu_snapshot" | awk -F ',' -v wanted="$gpu" '$1+0==wanted {print}')"
  if [[ -z "$line" || "$line" != *"RTX 6000 Ada"* ]]; then
    echo "E187 expected RTX 6000 Ada GPU $gpu." >&2
    exit 2
  fi
  free_mib="$(printf '%s\n' "$line" | awk -F ',' '{gsub(/ /,"",$6); print $6+0}')"
  if (( free_mib < MIN_FREE_MIB )); then
    echo "E187 Ada GPU $gpu free=${free_mib}MiB < ${MIN_FREE_MIB}MiB; refusing without waiting." >&2
    exit 2
  fi
done

remote_preflight() {
  local case_id="$1" gpu_id="$2"
  ssh "$REMOTE_HOST" \
    "cd '$REMOTE_ROOT' && PYTHONPATH=. MUJOCO_GL=egl '$REMOTE_PYTHON' '$RUNNER' preflight --case-id '$case_id' --gpu-id '$gpu_id' --python-bin '$REMOTE_PYTHON'"
}
remote_preflight bucket004_20231002_021_p1 0
remote_preflight bucket007_20231020_055_p1 1

if [[ "$MODE" == "preflight" ]]; then
  echo "E187_REMOTE_CANARY_PREFLIGHT=PASS root=$REMOTE_ROOT"
  exit 0
fi

launch_worker() {
  local case_id="$1" gpu_id="$2" execution_id session log_dir log_path worker_command quoted
  execution_id="$(basename "$(dirname "$REMOTE_ROOT")")"
  session="e187_a3_ada${gpu_id}_${execution_id#e187_}"
  log_dir="$REMOTE_ROOT/logs/E187/s4_canary"
  log_path="$log_dir/launcher_ada${gpu_id}_${case_id}.log"
  if ssh "$REMOTE_HOST" "tmux has-session -t '$session'" >/dev/null 2>&1; then
    echo "Exact E187 canary session already exists; leaving unchanged: $session"
    return
  fi
  worker_command="cd '$REMOTE_ROOT' && PYTHONPATH=. MUJOCO_GL=egl '$REMOTE_PYTHON' '$RUNNER' run --case-id '$case_id' --gpu-id '$gpu_id' --python-bin '$REMOTE_PYTHON' --output-root workspace/core4d/results/E187/s4_canary > '$log_path' 2>&1"
  printf -v quoted '%q' "$worker_command"
  ssh "$REMOTE_HOST" "mkdir -p '$log_dir' && tmux new-session -d -s '$session' -c '$REMOTE_ROOT' $quoted"
  echo "E187_REMOTE_CANARY_LAUNCHED session=$session gpu=$gpu_id case=$case_id"
}

launch_worker bucket004_20231002_021_p1 0
launch_worker bucket007_20231020_055_p1 1
