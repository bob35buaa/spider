#!/usr/bin/env bash
# E078: CORE4D 3cm per-hand contact mask on box023 person1/person2.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-parallel}"
ARG2="${2:-0}"
ARG3="${3:-1}"

RESULTS=workspace/core4d/results/E078
LOGS=logs/E078
mkdir -p "$RESULTS/keyframes" "$LOGS"

variant_to_override() {
  case "$1" in
    E078A) echo "core4d_e078a_box023_p1_3cm" ;;
    E078B) echo "core4d_e078b_box023_p2_3cm" ;;
    *) echo "Unknown E078 variant: $1" >&2; return 1 ;;
  esac
}

variant_to_task() {
  case "$1" in
    E078A) echo "box023_person1" ;;
    E078B) echo "box023_person2" ;;
    *) echo "Unknown E078 variant: $1" >&2; return 1 ;;
  esac
}

variant_to_name() {
  case "$1" in
    E078A) echo "E078A_box023_p1" ;;
    E078B) echo "E078B_box023_p2" ;;
    *) echo "Unknown E078 variant: $1" >&2; return 1 ;;
  esac
}

run_one() {
  local variant=$1
  local gpu=$2
  local override task name out_dir
  override="$(variant_to_override "$variant")"
  task="$(variant_to_task "$variant")"
  name="$(variant_to_name "$variant")"
  out_dir="$RESULTS/${name}_outdir"
  mkdir -p "$out_dir"

  echo "[$(date '+%H:%M:%S')] === ${name} task=${task} override=${override} GPU=${gpu} ==="
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
    +override="$override" \
    task="$task" \
    +use_torch_compile=false \
    output_dir="$out_dir" \
    video_output_path="$RESULTS/${name}.mp4" \
    > "$LOGS/${name}.log" 2>&1

  cp "$out_dir/trajectory_mjwp_act.npz" "$RESULTS/${name}.npz"

  if command -v ffmpeg >/dev/null 2>&1 && [ -f "$RESULTS/${name}.mp4" ]; then
    mkdir -p "$RESULTS/keyframes/${variant}"
    for f in 100 115 120 121 122 123 124 125 130 145 160 166 180; do
      ffmpeg -y -loglevel error -i "$RESULTS/${name}.mp4" \
        -vf "select=eq(n\\,$f)" -frames:v 1 -vsync 0 \
        "$RESULTS/keyframes/${variant}/f${f}.jpg"
    done
  fi
  echo "[$(date '+%H:%M:%S')] === ${name} done ==="
}

echo "[$(date '+%H:%M:%S')] === scene snapshot (idempotent) ==="
bash workspace/core4d/scripts/convert/snapshot_scenes.sh E078 box023_person1 box023_person2

case "$MODE" in
  single)
    VARIANT="${ARG2:-E078A}"
    GPU="${ARG3:-0}"
    run_one "$VARIANT" "$GPU"
    .venv/bin/python workspace/core4d/scripts/eval/eval_E078.py "$VARIANT" | tee "$LOGS/eval_${VARIANT}.log"
    ;;
  parallel)
    GPU_A="${ARG2:-0}"
    GPU_B="${ARG3:-1}"
    (run_one E078A "$GPU_A") &
    PID_A=$!
    (run_one E078B "$GPU_B") &
    PID_B=$!
    echo "[$(date '+%H:%M:%S')] launched E078A PID=${PID_A}, E078B PID=${PID_B}"
    wait "$PID_A"
    echo "[$(date '+%H:%M:%S')] E078A complete"
    wait "$PID_B"
    echo "[$(date '+%H:%M:%S')] E078B complete"
    .venv/bin/python workspace/core4d/scripts/eval/eval_E078.py E078A E078B | tee "$LOGS/eval_E078.log"
    ;;
  eval)
    shift || true
    .venv/bin/python workspace/core4d/scripts/eval/eval_E078.py "$@" | tee "$LOGS/eval_E078.log"
    ;;
  *)
    echo "Usage:"
    echo "  $0 single E078A 0"
    echo "  $0 single E078B 1"
    echo "  $0 parallel 0 1"
    echo "  $0 eval [E078A E078B]"
    exit 2
    ;;
esac

echo "=== E078 ${MODE} done ==="
