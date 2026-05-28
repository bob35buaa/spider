#!/usr/bin/env bash
# E085: raw-contact external target gate and CEM runs.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-local}"
GPU="${2:-0}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d/scripts/E085/variants.tsv}"
RESULTS="${RESULTS:-workspace/core4d/results/E085}"
LOGS="${LOGS:-logs/E085}"
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

snapshot_split() {
  local split_name=$1
  mapfile -t tasks < <(awk -F '\t' -v want_split="$split_name" 'NF && $1 !~ /^#/ && $7 == want_split {print $3}' "$VARIANTS_FILE")
  if [ "${#tasks[@]}" -gt 0 ]; then
    echo "[$(date '+%H:%M:%S')] === scene snapshot split=${split_name} ==="
    bash workspace/core4d/scripts/convert/snapshot_scenes.sh E085 "${tasks[@]}"
    for task in "${tasks[@]}"; do
      meta="example_datasets/processed/core4d/unitree_g1/humanoid_object/${task}/upper_object_collision_meta.json"
      if [ -f "$meta" ]; then
        mkdir -p "$RESULTS/scene_snapshot/$task"
        cp "$meta" "$RESULTS/scene_snapshot/$task/upper_object_collision_meta.json"
      fi
    done
  fi
}

extract_keyframes() {
  local variant=$1
  local video=$2
  [ "${SKIP_KEYFRAMES:-0}" = "1" ] && return 0
  [ -f "$video" ] || return 0
  command -v ffmpeg >/dev/null 2>&1 || return 0
  mkdir -p "$RESULTS/keyframes/$variant"
  for f in 16 32 50 75 90 100 115 120 125 145 160 180 204; do
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
    echo "Unknown E085 variant: $variant" >&2
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
    echo "Unknown E085 variant: $variant" >&2
    return 2
  fi
  task="$(field_from_row "$row" 3)"
  override="core4d_${variant}"
  smoke_dir="$RESULTS/smoke"
  out_dir="$smoke_dir/${variant}_outdir"
  mkdir -p "$out_dir" "$LOGS/smoke"

  echo "[$(date '+%H:%M:%S')] === smoke ${variant} task=${task} override=${override} GPU=${gpu} ==="
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
  echo "[$(date '+%H:%M:%S')] === smoke ${variant} done ==="
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
  single)
    VARIANT="${3:-}"
    if [ -z "$VARIANT" ]; then
      echo "Usage: $0 single <gpu> <variant>" >&2
      exit 2
    fi
    snapshot_split "single"
    run_one "$VARIANT" "$GPU"
    RESULTS="$RESULTS" VARIANTS_FILE="$VARIANTS_FILE" .venv/bin/python workspace/core4d/scripts/eval/eval_E085.py "$VARIANT" | tee "$LOGS/eval_${VARIANT}.log"
    ;;
  local|remote-gpu0|remote-gpu1|guard)
    snapshot_split "$MODE"
    mapfile -t variants < <(variants_for_split "$MODE")
    if [ "${#variants[@]}" -eq 0 ]; then
      echo "No E085 variants for split=${MODE}" >&2
      exit 2
    fi
    for variant in "${variants[@]}"; do
      run_one "$variant" "$GPU"
    done
    RESULTS="$RESULTS" VARIANTS_FILE="$VARIANTS_FILE" .venv/bin/python workspace/core4d/scripts/eval/eval_E085.py "${variants[@]}" | tee "$LOGS/eval_${MODE}.log"
    ;;
  eval)
    shift || true
    RESULTS="$RESULTS" VARIANTS_FILE="$VARIANTS_FILE" .venv/bin/python workspace/core4d/scripts/eval/eval_E085.py "$@" | tee "$LOGS/eval_E085.log"
    ;;
  *)
    echo "Usage:"
    echo "  $0 list"
    echo "  $0 smoke <gpu> [variant ...]"
    echo "  $0 local 0"
    echo "  $0 remote-gpu0 0"
    echo "  $0 single 0 E085A_rawtarget_main"
    echo "  $0 eval [variant ...]"
    exit 2
    ;;
esac

echo "=== E085 ${MODE} done ==="
