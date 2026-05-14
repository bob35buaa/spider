#!/usr/bin/env bash
# E075: E074A ctrl guard plus time-limited hold/contact variants on box023.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-parallel}"
ARG2="${2:-0}"
ARG3="${3:-1}"

RESULTS=workspace/core4d/results/E075
LOGS=logs/E075
mkdir -p "$RESULTS/keyframes" "$LOGS"

variant_to_override() {
  case "$1" in
    E075B) echo "core4d_e075b_box023" ;;
    E075A) echo "core4d_e075a_box023" ;;
    *) echo "Unknown E075 variant: $1" >&2; return 1 ;;
  esac
}

run_one() {
  local variant=$1
  local gpu=$2
  local override
  override="$(variant_to_override "$variant")"
  local name="${variant}_box023"
  local out_dir="$RESULTS/${name}_outdir"
  mkdir -p "$out_dir"

  echo "[$(date '+%H:%M:%S')] === ${name} override=${override} GPU=${gpu} ==="
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
    +override="$override" \
    task=box023_person1 \
    +use_torch_compile=false \
    output_dir="$out_dir" \
    video_output_path="$RESULTS/${name}.mp4" \
    > "$LOGS/${name}.log" 2>&1

  cp "$out_dir/trajectory_mjwp_act.npz" "$RESULTS/${name}.npz"

  if command -v ffmpeg >/dev/null 2>&1 && [ -f "$RESULTS/${name}.mp4" ]; then
    mkdir -p "$RESULTS/keyframes/${variant}"
    for f in 100 115 130 145 160 166 180; do
      ffmpeg -y -loglevel error -i "$RESULTS/${name}.mp4" \
        -vf "select=eq(n\\,$f)" -frames:v 1 -vsync 0 \
        "$RESULTS/keyframes/${variant}/f${f}.jpg"
    done
  fi
  echo "[$(date '+%H:%M:%S')] === ${name} done ==="
}

echo "[$(date '+%H:%M:%S')] === scene snapshot (idempotent) ==="
bash workspace/core4d/scripts/convert/snapshot_scenes.sh E075 box023_person1

case "$MODE" in
  single)
    VARIANT="${ARG2:-E075B}"
    GPU="${ARG3:-0}"
    run_one "$VARIANT" "$GPU"
    .venv/bin/python workspace/core4d/scripts/eval/eval_E075.py "$VARIANT" | tee "$LOGS/eval_${VARIANT}.log"
    ;;
  parallel)
    GPU_B="${ARG2:-0}"
    GPU_A="${ARG3:-1}"
    (run_one E075B "$GPU_B") &
    PID_B=$!
    (run_one E075A "$GPU_A") &
    PID_A=$!
    echo "[$(date '+%H:%M:%S')] launched E075B PID=${PID_B}, E075A PID=${PID_A}"
    wait "$PID_B"
    echo "[$(date '+%H:%M:%S')] E075B complete"
    wait "$PID_A"
    echo "[$(date '+%H:%M:%S')] E075A complete"
    .venv/bin/python workspace/core4d/scripts/eval/eval_E075.py E075B E075A | tee "$LOGS/eval_E075.log"
    ;;
  eval)
    shift || true
    .venv/bin/python workspace/core4d/scripts/eval/eval_E075.py "$@" | tee "$LOGS/eval_E075.log"
    ;;
  *)
    echo "Usage:"
    echo "  $0 single E075B 0"
    echo "  $0 single E075A 1"
    echo "  $0 parallel 0 1"
    echo "  $0 eval [E075B E075A]"
    exit 2
    ;;
esac

echo "=== E075 ${MODE} done ==="
