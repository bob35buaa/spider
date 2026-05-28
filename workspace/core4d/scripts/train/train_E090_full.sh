#!/usr/bin/env bash
# E090 Phase 4: full CEM for smoke-passing topface-preIK Box021 case(s).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-local}"
GPU="${2:-0}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d/scripts/E090/variants_smoke.tsv}"
RESULTS="${RESULTS:-workspace/core4d/results/E090/full}"
LOGS="${LOGS:-logs/E090/full}"
FULL_VARIANTS="${FULL_VARIANTS:-E090S1_box021_20231011_035_p2_btop}"
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

snapshot_variants() {
  local variants=("$@")
  local tasks=()
  for variant in "${variants[@]}"; do
    local row task
    row="$(variant_row "$variant")"
    [ -n "$row" ] || { echo "Unknown E090 full variant: $variant" >&2; exit 2; }
    task="$(field_from_row "$row" 3)"
    tasks+=("$task")
  done
  if [ "${#tasks[@]}" -gt 0 ]; then
    echo "[$(date '+%H:%M:%S')] === scene snapshot full ==="
    bash workspace/core4d/scripts/convert/snapshot_scenes.sh E090 "${tasks[@]}" > "$LOGS/snapshot_full.log" 2>&1
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
    echo "Unknown E090 full variant: $variant" >&2
    return 2
  fi
  task="$(field_from_row "$row" 3)"
  override="core4d_${variant}"
  out_dir="$RESULTS/${variant}_outdir_full"
  mkdir -p "$out_dir"

  echo "[$(date '+%H:%M:%S')] === FULL ${variant} task=${task} override=${override} GPU=${gpu} ==="
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
    +override="$override" \
    task="$task" \
    +use_torch_compile=false \
    output_dir="$out_dir" \
    video_output_path="$RESULTS/${variant}_full.mp4" \
    > "$LOGS/${variant}_full.log" 2>&1

  cp "$out_dir/trajectory_mjwp_act.npz" "$RESULTS/${variant}.npz"
  extract_keyframes "$variant" "$RESULTS/${variant}_full.mp4"
  echo "[$(date '+%H:%M:%S')] === FULL ${variant} done ==="
}

case "$MODE" in
  list)
    printf '%s\n' $FULL_VARIANTS
    ;;
  single)
    VARIANT="${3:-}"
    if [ -z "$VARIANT" ]; then
      echo "Usage: $0 single <gpu> <variant>" >&2
      exit 2
    fi
    snapshot_variants "$VARIANT"
    run_one "$VARIANT" "$GPU"
    RESULTS="$RESULTS" VARIANTS_FILE="$VARIANTS_FILE" .venv/bin/python workspace/core4d/scripts/eval/eval_E090.py --stage full "$VARIANT" | tee "$LOGS/eval_${VARIANT}.log"
    ;;
  local|remote-gpu0|remote-gpu1)
    # shellcheck disable=SC2206
    variants=($FULL_VARIANTS)
    snapshot_variants "${variants[@]}"
    for variant in "${variants[@]}"; do
      run_one "$variant" "$GPU"
    done
    RESULTS="$RESULTS" VARIANTS_FILE="$VARIANTS_FILE" .venv/bin/python workspace/core4d/scripts/eval/eval_E090.py --stage full "${variants[@]}" | tee "$LOGS/eval_${MODE}.log"
    ;;
  eval)
    shift || true
    RESULTS="$RESULTS" VARIANTS_FILE="$VARIANTS_FILE" .venv/bin/python workspace/core4d/scripts/eval/eval_E090.py --stage full "$@" | tee "$LOGS/eval_E090_full.log"
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

echo "=== E090 full ${MODE} done ==="
