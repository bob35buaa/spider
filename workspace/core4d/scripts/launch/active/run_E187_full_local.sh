#!/usr/bin/env bash
# E187 immutable Full CEM worker for local RTX5090 GPU0.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-preflight}"
if [[ "$MODE" != "preflight" && "$MODE" != "full" ]]; then
  echo "usage: $0 {preflight|full}" >&2
  exit 2
fi

PYTHON_BIN="${E187_PYTHON_BIN:-.venv/bin/python}"
RUNNER="workspace/core4d/scripts/experiments/E187/run_full_queue.py"
TEST="workspace/core4d/scripts/experiments/E187/test_full_queue_runner.py"
QUEUE="workspace/core4d/results/E187/s5_full/queue/local-0.tsv"
RESULT_ROOT="workspace/core4d/results/E187/s5_full"
SNAPSHOT_ROOT="$RESULT_ROOT/process_snapshots/local"
LAUNCH_LOG_ROOT="logs/E187/s5_full"
mkdir -p "$SNAPSHOT_ROOT" "$LAUNCH_LOG_ROOT"

mapfile -t LOCAL_TASKS < <(awk -F $'\t' 'NR > 1 {print $13}' "$QUEUE")
bash workspace/core4d/scripts/convert/snapshot_scenes.sh \
  E187/s5_full/local "${LOCAL_TASKS[@]}"

capture_snapshot() {
  local label="$1"
  nvidia-smi \
    --query-gpu=index,name,uuid,memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu,power.draw \
    --format=csv,noheader,nounits > "$SNAPSHOT_ROOT/gpu_${label}.csv"
  nvidia-smi \
    --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits > "$SNAPSHOT_ROOT/compute_${label}.csv"
}

capture_snapshot before
gpu_name="$(nvidia-smi --id=0 --query-gpu=name --format=csv,noheader | xargs)"
if [[ "$gpu_name" != "NVIDIA GeForce RTX 5090" ]]; then
  echo "E187 Full local worker requires GPU0 RTX5090, got: $gpu_name" >&2
  exit 1
fi
if [[ -s "$SNAPSHOT_ROOT/compute_before.csv" ]]; then
  echo "E187 Full local compute-app list is non-empty; refusing without waiting or preemption." >&2
  exit 2
fi

PYTHONPATH=workspace/core4d/scripts/experiments/E187 "$PYTHON_BIN" "$TEST"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. MUJOCO_GL="${MUJOCO_GL:-egl}" "$PYTHON_BIN" \
  "$RUNNER" preflight --worker local-0 --gpu-id 0 --python-bin "$PYTHON_BIN"

if [[ "$MODE" == "preflight" ]]; then
  echo "E187_FULL_LOCAL_PREFLIGHT=PASS"
  exit 0
fi

PYTHONPATH=. "$PYTHON_BIN" "$RUNNER" register-promotions
queue_sha="$(sha256sum workspace/core4d/results/E187/s5_full/queue/queue_manifest.json | awk '{print substr($1,1,16)}')"
session="e187_a4_local_${queue_sha}"
launcher_log="$LAUNCH_LOG_ROOT/launcher_local-0.log"
if tmux has-session -t "$session" >/dev/null 2>&1; then
  echo "Exact E187 Full local session already exists; leaving unchanged: $session"
  exit 0
fi
worker_command="cd '$(pwd)' && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. MUJOCO_GL='${MUJOCO_GL:-egl}' '$PYTHON_BIN' '$RUNNER' run --worker local-0 --gpu-id 0 --python-bin '$PYTHON_BIN' --output-root '$RESULT_ROOT' > '$launcher_log' 2>&1"
tmux new-session -d -s "$session" -c "$(pwd)" bash -lc "$worker_command"
capture_snapshot after_launch
echo "E187_FULL_LOCAL_LAUNCHED session=$session worker=local-0 gpu=0 full_rows_registered=3"
