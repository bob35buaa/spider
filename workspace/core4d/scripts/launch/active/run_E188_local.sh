#!/usr/bin/env bash
# E188 local RTX5090 GPU0 worker: static preflight, canary, then remaining Full.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-preflight}"
if [[ "$MODE" != "preflight" && "$MODE" != "canary" && "$MODE" != "full" ]]; then
  echo "usage: $0 {preflight|canary|full}" >&2
  exit 2
fi
PYTHON_BIN="${E188_PYTHON_BIN:-.venv/bin/python}"
RUNNER=workspace/core4d/scripts/experiments/E188/run_full_queue.py
TEST=workspace/core4d/scripts/experiments/E188/test_contracts.py
SNAPSHOT=workspace/core4d/results/E188/s5_full/process_snapshots/local
mkdir -p "$SNAPSHOT" logs/E188

nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total --format=csv,noheader,nounits > "$SNAPSHOT/gpu_${MODE}_before.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits > "$SNAPSHOT/compute_${MODE}_before.csv" || true
# 2026-08-05 explicit user authority: telemetry only; do not gate launch on
# memory or existing compute processes, and allow workload overlap on GPU0.

PYTHONPATH=workspace/core4d/scripts/experiments/E188 "$PYTHON_BIN" "$TEST"
PYTHONPATH=workspace/core4d/scripts/experiments/E188 "$PYTHON_BIN" "$RUNNER" preflight --worker local-0 --phase "$([[ "$MODE" == full ]] && echo full || echo canary)" --python-bin "$PYTHON_BIN" --device-id 0 --render-mode inline
if [[ "$MODE" == preflight ]]; then
  echo E188_LOCAL_PREFLIGHT=PASS
  exit 0
fi

session="e188_${MODE}_local0"
log="logs/E188/launcher_${MODE}_local-0.log"
if tmux has-session -t "$session" >/dev/null 2>&1; then
  echo "E188 local session already exists: $session"
  exit 0
fi
worker_command="cd '$(pwd)' && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=workspace/core4d/scripts/experiments/E188 MUJOCO_GL='${MUJOCO_GL:-egl}' '$PYTHON_BIN' '$RUNNER' run --worker local-0 --phase '$MODE' --python-bin '$PYTHON_BIN' --device-id 0 --physical-gpu 0 --render-mode inline > '$log' 2>&1"
tmux new-session -d -s "$session" -c "$(pwd)" bash -lc "$worker_command"
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total --format=csv,noheader,nounits > "$SNAPSHOT/gpu_${MODE}_after_launch.csv"
echo "E188_LOCAL_LAUNCHED session=$session phase=$MODE worker=local-0 gpu=0"
