#!/usr/bin/env bash
# E090 Phase 3: SPIDER smoke for topface-preIK Box021 gate-pass cases.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-local}"
GPU="${2:-0}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d/scripts/E090/variants_smoke.tsv}"
RESULTS="${RESULTS:-workspace/core4d/results/E090/smoke}"
LOGS="${LOGS:-logs/E090/smoke}"
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
    bash workspace/core4d/scripts/convert/snapshot_scenes.sh E090 "${tasks[@]}" > "$LOGS/snapshot_${split_name}.log" 2>&1
    for task in "${tasks[@]}"; do
      meta="example_datasets/processed/core4d/unitree_g1/humanoid_object/${task}/upper_object_collision_meta.json"
      if [ -f "$meta" ]; then
        mkdir -p "workspace/core4d/results/E090/scene_snapshot/$task"
        cp "$meta" "workspace/core4d/results/E090/scene_snapshot/$task/upper_object_collision_meta.json"
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
  for f in 10 25 40 55 70 85 100 120 135; do
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
    echo "Unknown E090 smoke variant: $variant" >&2
    return 2
  fi
  task="$(field_from_row "$row" 3)"
  override="core4d_${variant}"
  out_dir="$RESULTS/${variant}_outdir_smoke"
  mkdir -p "$out_dir"

  echo "[$(date '+%H:%M:%S')] === ${variant} task=${task} override=${override} GPU=${gpu} ==="
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
    +override="$override" \
    task="$task" \
    +use_torch_compile=false \
    max_num_iterations=4 \
    output_dir="$out_dir" \
    video_output_path="$RESULTS/${variant}_smoke.mp4" \
    > "$LOGS/${variant}_smoke.log" 2>&1

  cp "$out_dir/trajectory_mjwp_act.npz" "$RESULTS/${variant}.npz"
  extract_keyframes "$variant" "$RESULTS/${variant}_smoke.mp4"
  echo "[$(date '+%H:%M:%S')] === ${variant} done ==="
}

case "$MODE" in
  list)
    awk -F '\t' 'NF && $1 !~ /^#/ {print $1}' "$VARIANTS_FILE"
    ;;
  single)
    VARIANT="${3:-}"
    if [ -z "$VARIANT" ]; then
      echo "Usage: $0 single <gpu> <variant>" >&2
      exit 2
    fi
    snapshot_split local
    run_one "$VARIANT" "$GPU"
    RESULTS="$RESULTS" VARIANTS_FILE="$VARIANTS_FILE" .venv/bin/python workspace/core4d/scripts/eval/eval_E090.py --stage smoke "$VARIANT" | tee "$LOGS/eval_${VARIANT}.log"
    ;;
  local|remote-gpu0|remote-gpu1)
    snapshot_split "$MODE"
    mapfile -t variants < <(variants_for_split "$MODE")
    if [ "${#variants[@]}" -eq 0 ]; then
      echo "No E090 smoke variants for split=${MODE}" >&2
      exit 2
    fi
    for variant in "${variants[@]}"; do
      run_one "$variant" "$GPU"
    done
    RESULTS="$RESULTS" VARIANTS_FILE="$VARIANTS_FILE" .venv/bin/python workspace/core4d/scripts/eval/eval_E090.py --stage smoke "${variants[@]}" | tee "$LOGS/eval_${MODE}.log"
    ;;
  eval)
    shift || true
    RESULTS="$RESULTS" VARIANTS_FILE="$VARIANTS_FILE" .venv/bin/python workspace/core4d/scripts/eval/eval_E090.py --stage smoke "$@" | tee "$LOGS/eval_E090_smoke.log"
    ;;
  *)
    echo "Usage:"
    echo "  $0 list"
    echo "  $0 local 0"
    echo "  $0 single 0 E090S1_box021_20231011_035_p2_btop"
    echo "  $0 eval [variant ...]"
    exit 2
    ;;
esac

echo "=== E090 smoke ${MODE} done ==="
