#!/usr/bin/env bash
# E087: Box021 mass/reward audit CEM runs.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-local}"
GPU="${2:-0}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d/scripts/E087/variants.tsv}"
RESULTS="${RESULTS:-workspace/core4d/results/E087}"
LOGS="${LOGS:-logs/E087}"
mkdir -p "$RESULTS/keyframes" "$LOGS"

variant_row() {
  local variant=$1
  awk -F '\t' -v v="$variant" 'NF && $1 !~ /^#/ && $1 == v {print; exit}' "$VARIANTS_FILE"
}

field_from_row() {
  local row=$1
  local idx=$2
  awk -F '\t' -v row="$row" -v idx="$idx" 'BEGIN {split(row, a, FS); print a[idx]}'
}

variants_for_split() {
  local split_name=$1
  awk -F '\t' -v want_split="$split_name" 'NF && $1 !~ /^#/ && $7 == want_split {print $1}' "$VARIANTS_FILE"
}

extract_keyframes() {
  local variant=$1
  local video=$2
  [ "${SKIP_KEYFRAMES:-0}" = "1" ] && return 0
  [ -f "$video" ] || return 0
  command -v ffmpeg >/dev/null 2>&1 || return 0
  mkdir -p "$RESULTS/keyframes/$variant"
  for f in 16 32 50 75 90 100 115 120 125 145; do
    timeout "${KEYFRAME_TIMEOUT_SECONDS:-30}" ffmpeg -nostdin -y -loglevel error -i "$video" \
      -vf "select=eq(n\\,$f)" -frames:v 1 -vsync 0 \
      "$RESULTS/keyframes/$variant/f${f}.jpg" || true
  done
}

run_one() {
  local variant=$1
  local gpu=$2
  local row task override out_dir
  row="$(variant_row "$variant")"
  if [ -z "$row" ]; then
    echo "Unknown E087 variant: $variant" >&2
    return 2
  fi
  task="$(field_from_row "$row" 3)"
  override="core4d_${variant}"
  out_dir="$RESULTS/${variant}_outdir"
  mkdir -p "$out_dir"

  echo "[$(date '+%H:%M:%S')] === ${variant} task=${task} override=${override} GPU=${gpu} ==="
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
    +override="$override" \
    task="$task" \
    +use_torch_compile=false \
    output_dir="$out_dir" \
    video_output_path="$RESULTS/${variant}.mp4" \
    > "$LOGS/${variant}.log" 2>&1

  cp "$out_dir/trajectory_mjwp_act.npz" "$RESULTS/${variant}.npz"
  extract_keyframes "$variant" "$RESULTS/${variant}.mp4"
  echo "[$(date '+%H:%M:%S')] === ${variant} done ==="
}

run_smoke_one() {
  local variant=$1
  local gpu=$2
  local row task override out_dir smoke_dir
  row="$(variant_row "$variant")"
  if [ -z "$row" ]; then
    echo "Unknown E087 variant: $variant" >&2
    return 2
  fi
  task="$(field_from_row "$row" 3)"
  override="core4d_${variant}"
  smoke_dir="$RESULTS/smoke"
  out_dir="$smoke_dir/${variant}_outdir"
  mkdir -p "$out_dir" "$LOGS/smoke"
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
    +override="$override" \
    task="$task" \
    +use_torch_compile=false \
    max_sim_steps="${SMOKE_MAX_SIM_STEPS:-4}" \
    num_samples="${SMOKE_NUM_SAMPLES:-32}" \
    max_num_iterations="${SMOKE_MAX_NUM_ITERATIONS:-1}" \
    save_video="${SMOKE_SAVE_VIDEO:-false}" \
    output_dir="$out_dir" \
    video_output_path="$smoke_dir/${variant}.mp4" \
    > "$LOGS/smoke/${variant}.log" 2>&1
  cp "$out_dir/trajectory_mjwp_act.npz" "$smoke_dir/${variant}.npz"
}

case "$MODE" in
  list)
    awk -F '\t' 'NF && $1 !~ /^#/ {print $1}' "$VARIANTS_FILE"
    ;;
  smoke)
    shift 2 || true
    if [ "$#" -gt 0 ]; then
      variants=("$@")
    else
      mapfile -t variants < <(awk -F '\t' 'NF && $1 !~ /^#/ {print $1}' "$VARIANTS_FILE")
    fi
    for variant in "${variants[@]}"; do
      run_smoke_one "$variant" "$GPU"
    done
    ;;
  local|remote-gpu0|remote-gpu1)
    mapfile -t variants < <(variants_for_split "$MODE")
    for variant in "${variants[@]}"; do
      run_one "$variant" "$GPU"
    done
    RESULTS="$RESULTS" VARIANTS_FILE="$VARIANTS_FILE" .venv/bin/python workspace/core4d/scripts/eval/eval_E087.py "${variants[@]}" | tee "$LOGS/eval_${MODE}.log"
    ;;
  single)
    VARIANT="${3:-}"
    if [ -z "$VARIANT" ]; then
      echo "Usage: $0 single <gpu> <variant>" >&2
      exit 2
    fi
    run_one "$VARIANT" "$GPU"
    RESULTS="$RESULTS" VARIANTS_FILE="$VARIANTS_FILE" .venv/bin/python workspace/core4d/scripts/eval/eval_E087.py "$VARIANT" | tee "$LOGS/eval_${VARIANT}.log"
    ;;
  eval)
    shift || true
    RESULTS="$RESULTS" VARIANTS_FILE="$VARIANTS_FILE" .venv/bin/python workspace/core4d/scripts/eval/eval_E087.py "$@" | tee "$LOGS/eval_E087.log"
    ;;
  *)
    echo "Usage: $0 list | smoke <gpu> [variant ...] | local 0 | remote-gpu0 0 | remote-gpu1 1 | single 0 <variant> | eval [variant ...]" >&2
    exit 2
    ;;
esac

echo "=== E087 ${MODE} done ==="
