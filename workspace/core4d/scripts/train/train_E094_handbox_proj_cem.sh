#!/usr/bin/env bash
# E094: CEM with adaptive-support handbox-aware external contact targets.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-local}"       # list | local | remote-gpu0 | remote-gpu1 | single | eval
STAGE="${2:-full}"       # smoke | full
GPU="${3:-0}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d/scripts/E094/variants.tsv}"
RESULTS="${RESULTS:-workspace/core4d/results/E094/cem/${STAGE}}"
LOGS="${LOGS:-logs/E094/cem/${STAGE}}"
mkdir -p "$RESULTS/keyframes" "$LOGS"

if [ "$STAGE" != "smoke" ] && [ "$STAGE" != "full" ]; then
  echo "Invalid STAGE=$STAGE (use smoke|full)" >&2
  exit 2
fi

variant_row() {
  local variant=$1
  awk -F '\t' -v v="$variant" 'NF && $1 !~ /^#/ && $1 == "cem_hbproj" && $3 == v {print; exit}' "$VARIANTS_FILE"
}

field_from_row() {
  local row=$1
  local idx=$2
  awk -F '\t' -v row="$row" -v idx="$idx" 'BEGIN {split(row, a, FS); print a[idx]}'
}

variants_for_split() {
  local split_name=$1
  awk -F '\t' -v want="$split_name" 'NF && $1 !~ /^#/ && $1 == "cem_hbproj" && $7 == want {print $3}' "$VARIANTS_FILE"
}

extract_keyframes() {
  local variant=$1
  local video=$2
  [ "${SKIP_KEYFRAMES:-0}" = "1" ] && return 0
  [ -f "$video" ] || return 0
  command -v ffmpeg >/dev/null 2>&1 || return 0
  mkdir -p "$RESULTS/keyframes/$variant"
  for f in 10 25 40 55 70 85 100 120 140 160; do
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
    echo "Unknown E094 variant: $variant" >&2
    return 2
  fi
  task="$(field_from_row "$row" 5)"
  override="core4d_${variant}"
  out_dir="$RESULTS/${variant}_outdir_${STAGE}"
  mkdir -p "$out_dir"
  echo "[$(date '+%H:%M:%S')] === E094 ${STAGE} ${variant} task=${task} GPU=${gpu} ==="
  if [ "$STAGE" = "smoke" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
      +override="$override" \
      task="$task" \
      +use_torch_compile=false \
      video_camera=auto \
      max_num_iterations="${SMOKE_MAX_NUM_ITERATIONS:-4}" \
      output_dir="$out_dir" \
      video_output_path="$RESULTS/${variant}_${STAGE}.mp4" \
      > "$LOGS/${variant}_${STAGE}.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
      +override="$override" \
      task="$task" \
      +use_torch_compile=false \
      video_camera=auto \
      output_dir="$out_dir" \
      video_output_path="$RESULTS/${variant}_${STAGE}.mp4" \
      > "$LOGS/${variant}_${STAGE}.log" 2>&1
  fi
  cp "$out_dir/trajectory_mjwp_act.npz" "$RESULTS/${variant}.npz"
  extract_keyframes "$variant" "$RESULTS/${variant}_${STAGE}.mp4"
  echo "[$(date '+%H:%M:%S')] === E094 ${STAGE} ${variant} done ==="
}

eval_variants() {
  local variants=("$@")
  if [ "${#variants[@]}" -eq 0 ]; then
    return 0
  fi
  RESULTS="$RESULTS" VARIANTS_FILE="$VARIANTS_FILE" .venv/bin/python \
    workspace/core4d/scripts/eval/eval_E094_cem.py --stage "$STAGE" "${variants[@]}" \
    | tee "$LOGS/eval_${MODE}_${STAGE}.log"
}

case "$MODE" in
  list)
    variants_for_split "${4:-local}"
    ;;
  single)
    VARIANT="${4:-}"
    if [ -z "$VARIANT" ]; then
      echo "Usage: $0 single <smoke|full> <gpu> <variant>" >&2
      exit 2
    fi
    run_one "$VARIANT" "$GPU"
    eval_variants "$VARIANT"
    ;;
  local|remote-gpu0|remote-gpu1)
    mapfile -t variants < <(variants_for_split "$MODE")
    if [ "${#variants[@]}" -eq 0 ]; then
      echo "No E094 variants for split=${MODE}; nothing to run."
      exit 0
    fi
    for variant in "${variants[@]}"; do
      run_one "$variant" "$GPU"
    done
    eval_variants "${variants[@]}"
    ;;
  eval)
    shift 3 || true
    RESULTS="$RESULTS" VARIANTS_FILE="$VARIANTS_FILE" .venv/bin/python \
      workspace/core4d/scripts/eval/eval_E094_cem.py --stage "$STAGE" "$@" \
      | tee "$LOGS/eval_E094_cem_${STAGE}.log"
    ;;
  *)
    echo "Usage:"
    echo "  $0 list <smoke|full> <gpu> [split]"
    echo "  $0 local <smoke|full> 0"
    echo "  $0 remote-gpu0 <smoke|full> 0"
    echo "  $0 remote-gpu1 <smoke|full> 1"
    echo "  $0 single <smoke|full> 0 <variant>"
    echo "  $0 eval <smoke|full> 0 [variant ...]"
    exit 2
    ;;
esac

echo "=== E094 CEM ${MODE} ${STAGE} done ==="
