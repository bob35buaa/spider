#!/usr/bin/env bash
# E079: CORE4D 10+ case generalization CEM runs.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-local}"
GPU="${2:-0}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d/scripts/E079/variants.tsv}"
RESULTS="${RESULTS:-workspace/core4d/results/E079}"
LOGS="${LOGS:-logs/E079}"
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
  awk -F '\t' -v want_split="$split_name" 'NF && $1 !~ /^#/ && $5 == want_split && $6 != "calib" {print $1}' "$VARIANTS_FILE"
}

snapshot_split() {
  local split_name=$1
  mapfile -t tasks < <(awk -F '\t' -v want_split="$split_name" 'NF && $1 !~ /^#/ && $5 == want_split && $6 != "calib" {print $2}' "$VARIANTS_FILE")
  if [ "${#tasks[@]}" -gt 0 ]; then
    echo "[$(date '+%H:%M:%S')] === scene snapshot split=${split_name} ==="
    bash workspace/core4d/scripts/convert/snapshot_scenes.sh E079 "${tasks[@]}"
  fi
}

extract_keyframes() {
  local variant=$1
  local video=$2
  [ -f "$video" ] || return 0
  command -v ffmpeg >/dev/null 2>&1 || return 0
  mkdir -p "$RESULTS/keyframes/$variant"
  for f in 50 75 100 115 120 125 145 160 180; do
    ffmpeg -y -loglevel error -i "$video" \
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
    echo "Unknown E079 variant: $variant" >&2
    return 2
  fi
  task="$(field_from_row "$row" 2)"
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

case "$MODE" in
  single)
    VARIANT="${3:-}"
    if [ -z "$VARIANT" ]; then
      echo "Usage: $0 single <gpu> <variant>" >&2
      exit 2
    fi
    run_one "$VARIANT" "$GPU"
    .venv/bin/python workspace/core4d/scripts/eval/eval_E079.py "$VARIANT" | tee "$LOGS/eval_${VARIANT}.log"
    ;;
  local|remote)
    snapshot_split "$MODE"
    mapfile -t variants < <(variants_for_split "$MODE")
    if [ "${#variants[@]}" -eq 0 ]; then
      echo "No E079 variants for split=${MODE}" >&2
      exit 2
    fi
    for variant in "${variants[@]}"; do
      run_one "$variant" "$GPU"
    done
    .venv/bin/python workspace/core4d/scripts/eval/eval_E079.py "${variants[@]}" | tee "$LOGS/eval_${MODE}.log"
    ;;
  eval)
    shift || true
    .venv/bin/python workspace/core4d/scripts/eval/eval_E079.py "$@" | tee "$LOGS/eval_E079.log"
    ;;
  *)
    echo "Usage:"
    echo "  $0 local 0"
    echo "  $0 remote 1"
    echo "  $0 single 0 E079_box023_p1"
    echo "  $0 eval [variant ...]"
    exit 2
    ;;
esac

echo "=== E079 ${MODE} done ==="
