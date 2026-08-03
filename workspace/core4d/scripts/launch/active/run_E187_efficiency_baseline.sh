#!/usr/bin/env bash
# Run isolated short E178 baselines on the exact three E187 canary devices.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-preflight}"
if [[ "$MODE" != "preflight" && "$MODE" != "local" && "$MODE" != "remote1" && "$MODE" != "run" ]]; then
  echo "usage: $0 {preflight|local|remote1|run}" >&2
  exit 2
fi

LOCAL_PYTHON="${E187_PYTHON_BIN:-.venv/bin/python}"
REMOTE_PYTHON="/home/xiayb/pHRI_workspace/spider/.venv/bin/python"
REMOTE_HOST="${ADA_HOST:-spider-remote}"
AUDIT="workspace/core4d/scripts/experiments/E187/audit_canary_efficiency.py"
TEST="workspace/core4d/scripts/experiments/E187/test_canary_efficiency.py"
DEPLOYMENT="workspace/core4d/results/E187/s4_canary/deployment/remote_deployment_manifest.json"
RESULT_ROOT="workspace/core4d/results/E187/s4_canary/efficiency_baseline"
SNAPSHOT_ROOT="workspace/core4d/results/E187/s4_canary/process_snapshots/efficiency"

PYTHONPATH=. "$LOCAL_PYTHON" "$TEST"
PYTHONPATH=. "$LOCAL_PYTHON" "$AUDIT" preflight --python-bin "$LOCAL_PYTHON" --repo-root .
if [[ "$MODE" == "preflight" ]]; then
  exit 0
fi

run_local_baseline() {
  local local_case local_row local_command
  local_case="bucket003_20231018_003_p1"
  local_row="$RESULT_ROOT/rows/$local_case"
  mkdir -p "$SNAPSHOT_ROOT/local"
  nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader,nounits > "$SNAPSHOT_ROOT/local/gpu_before.csv"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits > "$SNAPSHOT_ROOT/local/compute_before.csv"
  if [[ -s "$SNAPSHOT_ROOT/local/compute_before.csv" ]]; then
    echo "E187 local efficiency GPU is not free after canary; refusing without waiting/preemption." >&2
    exit 2
  fi
  if [[ "$(nvidia-smi --id=0 --query-gpu=name --format=csv,noheader | xargs)" != "NVIDIA GeForce RTX 5090" ]]; then
    echo "E187 efficiency local baseline requires RTX5090 GPU0." >&2
    exit 2
  fi
  if [[ -e "$local_row" ]]; then
    echo "Refusing to overwrite existing local E187 efficiency row: $local_row" >&2
    exit 2
  fi
  local_command="$($LOCAL_PYTHON "$AUDIT" command --case-id "$local_case" \
    --python-bin "$LOCAL_PYTHON" --gpu-id 0 --repo-root .)"
  mkdir -p "$local_row/outdir"
  printf '%s\n' "# $local_command" > "$local_row/run.log"
  PYTHONPATH=. MUJOCO_GL=egl bash -lc "$local_command" >> "$local_row/run.log" 2>&1
  PYTHONPATH=. "$LOCAL_PYTHON" "$AUDIT" verify-row --root . --case-id "$local_case"
  nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader,nounits > "$SNAPSHOT_ROOT/local/gpu_after.csv"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits > "$SNAPSHOT_ROOT/local/compute_after.csv"
  echo "E187_LOCAL_EFFICIENCY_BASELINE=PASS"
}

if [[ "$MODE" == "local" ]]; then
  run_local_baseline
  exit 0
fi
if [[ ! -f "$DEPLOYMENT" ]]; then
  echo "E187 remote deployment manifest missing: $DEPLOYMENT" >&2
  exit 2
fi
REMOTE_ROOT="$($LOCAL_PYTHON -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["remote_root"])' "$DEPLOYMENT")"
if [[ "$REMOTE_ROOT" != /home/xiayb/pHRI_workspace/e187_runs/e187_*/spider ]]; then
  echo "Refusing unsafe E187 remote root: $REMOTE_ROOT" >&2
  exit 2
fi

mkdir -p "$SNAPSHOT_ROOT/local" "$SNAPSHOT_ROOT/remote"
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,memory.free,utilization.gpu \
  --format=csv,noheader,nounits > "$SNAPSHOT_ROOT/local/gpu_before.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits > "$SNAPSHOT_ROOT/local/compute_before.csv"
ssh "$REMOTE_HOST" \
  'nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits' \
  > "$SNAPSHOT_ROOT/remote/gpu_before.csv"
ssh "$REMOTE_HOST" \
  'nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits' \
  > "$SNAPSHOT_ROOT/remote/compute_before.csv" || true
if [[ -s "$SNAPSHOT_ROOT/local/compute_before.csv" && "$MODE" != "remote1" ]]; then
  echo "E187 efficiency baseline requires completed canaries; refusing without waiting/preemption." >&2
  exit 2
fi
if [[ "$MODE" != "remote1" && "$(nvidia-smi --id=0 --query-gpu=name --format=csv,noheader | xargs)" != "NVIDIA GeForce RTX 5090" ]]; then
  echo "E187 efficiency local baseline requires RTX5090 GPU0." >&2
  exit 2
fi
for gpu in 0 1; do
  name="$(ssh "$REMOTE_HOST" "nvidia-smi --id=$gpu --query-gpu=name --format=csv,noheader" | xargs)"
  if [[ "$name" != "NVIDIA RTX 6000 Ada Generation" ]]; then
    echo "E187 efficiency remote baseline requires Ada GPU $gpu." >&2
    exit 2
  fi
done

launch_remote() {
  local case_id="$1" gpu_id="$2" row_root session command quoted worker
  row_root="$REMOTE_ROOT/$RESULT_ROOT/rows/$case_id"
  session="e187_c9_e178_ada${gpu_id}_$(basename "$(dirname "$REMOTE_ROOT")" | cut -c6-21)"
  if ssh "$REMOTE_HOST" "test -e '$row_root' || tmux has-session -t '$session'" >/dev/null 2>&1; then
    echo "Refusing to overwrite existing E187 efficiency row/session: $case_id" >&2
    exit 2
  fi
  command="$($LOCAL_PYTHON "$AUDIT" command --case-id "$case_id" \
    --python-bin "$REMOTE_PYTHON" --gpu-id "$gpu_id" --repo-root "$REMOTE_ROOT")"
  worker="mkdir -p '$row_root/outdir' && printf '%s\\n' '# $command' > '$row_root/run.log' && cd '$REMOTE_ROOT' && PYTHONPATH=. MUJOCO_GL=egl $command >> '$row_root/run.log' 2>&1"
  printf -v quoted '%q' "$worker"
  ssh "$REMOTE_HOST" "tmux new-session -d -s '$session' -c '$REMOTE_ROOT' $quoted"
  echo "E187_EFFICIENCY_BASELINE_LAUNCHED session=$session case=$case_id gpu=$gpu_id"
}

if [[ "$MODE" == "remote1" ]]; then
  gpu1_uuid="$(ssh "$REMOTE_HOST" 'nvidia-smi --id=1 --query-gpu=uuid --format=csv,noheader' | xargs)"
  if grep -Fq "$gpu1_uuid" "$SNAPSHOT_ROOT/remote/compute_before.csv"; then
    echo "E187 Ada GPU1 still has a compute process; refusing without waiting/preemption." >&2
    exit 2
  fi
  launch_remote bucket007_20231020_055_p1 1
  exit 0
fi

if [[ -s "$SNAPSHOT_ROOT/remote/compute_before.csv" ]]; then
  echo "E187 remote efficiency baselines require completed canaries; refusing without waiting/preemption." >&2
  exit 2
fi

launch_remote bucket004_20231002_021_p1 0
launch_remote bucket007_20231020_055_p1 1

run_local_baseline
