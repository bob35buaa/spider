#!/usr/bin/env bash
# {{EXP_ID}}: {{DESCRIPTION}}
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-list}"       # list | single | {{SPLITS_PIPE}}
STAGE="${2:-full}"      # smoke | full
GPU="${3:-0}"
VARIANTS_FILE="${VARIANTS_FILE:-{{VARIANTS_TSV_PATH}}}"
RESULTS="${RESULTS:-{{RESULT_ROOT}}/cem/${STAGE}}"
LOGS="${LOGS:-logs/{{EXP_ID}}/cem/${STAGE}}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
mkdir -p "$RESULTS/keyframes" "$LOGS"

if [ "$STAGE" != "smoke" ] && [ "$STAGE" != "full" ]; then
  echo "Invalid STAGE=$STAGE (use smoke|full)" >&2
  exit 2
fi

variant_row() {
  local variant=$1
  awk -F '\t' -v v="$variant" 'NF && NR > 1 && $2 == v {print; exit}' "$VARIANTS_FILE"
}

field_from_row() {
  local row=$1
  local idx=$2
  awk -F '\t' -v row="$row" -v idx="$idx" 'BEGIN {split(row, a, FS); print a[idx]}'
}

variants_for_split() {
  local split=$1
  awk -F '\t' -v want="$split" 'NF && NR > 1 && ${{SPLIT_COL}} == want && ${{STATUS_COL}} == "to_run" {print $2}' "$VARIANTS_FILE"
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

outputs_exist() {
  local variant=$1
  [ -f "$RESULTS/${variant}.npz" ] && \
    [ -f "$RESULTS/${variant}_${STAGE}.mp4" ] && \
    [ -f "$RESULTS/${variant}_outdir_${STAGE}/trajectory_mjwp_act.npz" ]
}

run_one() {
  local variant=$1
  local gpu=$2
  local row task run_status override_stem out_dir
  row="$(variant_row "$variant")"
  if [ -z "$row" ]; then
    echo "Unknown {{EXP_ID}} variant: $variant" >&2
    return 2
  fi
  task="$(field_from_row "$row" {{TASK_COL}})"
  run_status="$(field_from_row "$row" {{STATUS_COL}})"
  if [ "$run_status" != "to_run" ]; then
    echo "[$(date '+%H:%M:%S')] === {{EXP_ID}} ${variant} run_status=${run_status}; skip ==="
    return 0
  fi
  if outputs_exist "$variant"; then
    echo "[$(date '+%H:%M:%S')] === {{EXP_ID}} ${variant} outputs exist; skip ==="
    return 0
  fi
  override_stem="$(basename "$(field_from_row "$row" {{OVERRIDE_COL}})" .yaml)"
  out_dir="$RESULTS/${variant}_outdir_${STAGE}"
  mkdir -p "$out_dir"
  echo "[$(date '+%H:%M:%S')] === {{EXP_ID}} ${STAGE} ${variant} task=${task} GPU=${gpu} ==="
  if [ "$STAGE" = "smoke" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u examples/run_mjwp.py \
      +override="$override_stem" task="$task" +use_torch_compile=false video_camera=auto \
      num_samples="${SMOKE_NUM_SAMPLES:-64}" max_num_iterations="${SMOKE_MAX_NUM_ITERATIONS:-4}" \
      output_dir="$out_dir" video_output_path="$RESULTS/${variant}_${STAGE}.mp4" \
      > "$LOGS/${variant}_${STAGE}.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u examples/run_mjwp.py \
      +override="$override_stem" task="$task" +use_torch_compile=false video_camera=auto \
      output_dir="$out_dir" video_output_path="$RESULTS/${variant}_${STAGE}.mp4" \
      > "$LOGS/${variant}_${STAGE}.log" 2>&1
  fi
  cp "$out_dir/trajectory_mjwp_act.npz" "$RESULTS/${variant}.npz"
  extract_keyframes "$variant" "$RESULTS/${variant}_${STAGE}.mp4"
  echo "[$(date '+%H:%M:%S')] === {{EXP_ID}} ${STAGE} ${variant} done ==="
}

case "$MODE" in
  list)
    variants_for_split "${4:-{{DEFAULT_SPLIT}}}"
    ;;
  single)
    VARIANT="${4:-}"
    [ -n "$VARIANT" ] || { echo "Usage: $0 single <smoke|full> <gpu> <variant>" >&2; exit 2; }
    run_one "$VARIANT" "$GPU"
    ;;
  {{SPLITS_PIPE}})
    mapfile -t variants < <(variants_for_split "$MODE")
    echo "{{EXP_ID}} split=${MODE} variants=${#variants[@]} stage=${STAGE} gpu=${GPU}"
    for variant in "${variants[@]}"; do
      run_one "$variant" "$GPU"
    done
    ;;
  *)
    echo "Usage: $0 {list|single|{{SPLITS_PIPE}}} {smoke|full} <gpu> [variant|split]" >&2
    exit 2
    ;;
esac
