#!/usr/bin/env bash
# E168 local CEM runner. Canary/recovery only; full production uses remote queues after canary.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-canary}"
LOCAL_GPU="${LOCAL_GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RUNNER="workspace/core4d/scripts/experiments/E168/run_cem_queue.py"
GPU_IDLE_MAX_MEM_MB="${E168_GPU_IDLE_MAX_MEM_MB:-3000}"
GPU_IDLE_MAX_UTIL_PCT="${E168_GPU_IDLE_MAX_UTIL_PCT:-20}"
WAIT_FOR_GPU_IDLE="${WAIT_FOR_GPU_IDLE:-0}"

case "$MODE" in
  canary)
    RUN_MODE="canary"
    ;;
  production-one|recovery)
    RUN_MODE="production"
    ;;
  full|production)
    echo "E168 full production must be launched through remote worker manifests after canary; use production-one/recovery for local tail jobs." >&2
    exit 2
    ;;
  *)
    echo "usage: $0 {canary|production-one|recovery}" >&2
    exit 2
    ;;
esac

check_gpu_driver() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; cannot run local E168 CEM." >&2
    exit 1
  fi
  if ! nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits >/tmp/e168_nvidia_smi.$$ 2>/tmp/e168_nvidia_smi_err.$$; then
    echo "nvidia-smi failed; local GPU is unavailable." >&2
    cat /tmp/e168_nvidia_smi_err.$$ >&2 || true
    rm -f /tmp/e168_nvidia_smi.$$ /tmp/e168_nvidia_smi_err.$$
    exit 1
  fi
  rm -f /tmp/e168_nvidia_smi.$$ /tmp/e168_nvidia_smi_err.$$
}

wait_for_gpu_idle() {
  if [ "$WAIT_FOR_GPU_IDLE" != "1" ]; then
    return 0
  fi
  echo "Waiting for GPU ${LOCAL_GPU}: mem<=${GPU_IDLE_MAX_MEM_MB}MiB util<=${GPU_IDLE_MAX_UTIL_PCT}%"
  while true; do
    local stat mem util
    stat="$(nvidia-smi --id="$LOCAL_GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    mem="${stat%%,*}"
    util="${stat##*,}"
    if [ -n "$mem" ] && [ -n "$util" ] && [ "$mem" -le "$GPU_IDLE_MAX_MEM_MB" ] && [ "$util" -le "$GPU_IDLE_MAX_UTIL_PCT" ]; then
      echo "GPU ${LOCAL_GPU} ready: mem=${mem}MiB util=${util}%"
      return 0
    fi
    echo "GPU ${LOCAL_GPU} busy: mem=${mem:-NA}MiB util=${util:-NA}%"
    sleep "${E168_GPU_IDLE_POLL_SEC:-120}"
  done
}

check_gpu_driver
wait_for_gpu_idle

exec "$PYTHON_BIN" "$RUNNER" \
  --mode "$RUN_MODE" \
  --python-bin "$PYTHON_BIN" \
  --gpu-id "$LOCAL_GPU" \
  "${@:2}"
