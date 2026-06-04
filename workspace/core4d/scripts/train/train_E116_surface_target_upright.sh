#!/usr/bin/env bash
# E116 Phase A: surface-target + upright/safety diagnostic CEM.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-local}"       # list | local | remote-gpu0 | remote-gpu1 | single | eval
STAGE="${2:-smoke}"      # smoke | full
GPU="${3:-0}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d/scripts/E116/variants.tsv}"
RESULTS="${RESULTS:-workspace/core4d/results/E116/cem/${STAGE}}"
LOGS="${LOGS:-logs/E116/cem/${STAGE}}"
mkdir -p "$RESULTS/keyframes" "$LOGS"

if [ "$STAGE" != "smoke" ] && [ "$STAGE" != "full" ]; then
  echo "Invalid STAGE=$STAGE (use smoke|full)" >&2
  exit 2
fi

split_name() {
  case "$1" in
    local) echo "local-gpu0" ;;
    remote-gpu0) echo "remote-gpu0" ;;
    remote-gpu1) echo "remote-gpu1" ;;
    *) echo "$1" ;;
  esac
}

variant_row() {
  local variant=$1
  awk -F '\t' -v v="$variant" 'NF && $1 !~ /^#/ && $2 == v {print; exit}' "$VARIANTS_FILE"
}

field_from_row() {
  local row=$1
  local idx=$2
  awk -F '\t' -v row="$row" -v idx="$idx" 'BEGIN {split(row, a, FS); print a[idx]}'
}

variants_for_split() {
  local split=$1
  awk -F '\t' -v want="$split" 'NF && $1 !~ /^#/ && $6 == want {print $2}' "$VARIANTS_FILE"
}

require_manifest_ready() {
  if [ ! -f "$VARIANTS_FILE" ]; then
    echo "Missing E116 variants file: $VARIANTS_FILE. Run build_surface_target_upright_manifest.py first." >&2
    exit 2
  fi
  local count
  count="$(awk -F '\t' 'NF && $1 !~ /^#/ {n++} END {print n+0}' "$VARIANTS_FILE")"
  if [ "$count" -ne 15 ]; then
    echo "E116 variants.tsv has $count rows, expected 15. Rebuild the manifest." >&2
    exit 2
  fi
  local preflight="workspace/core4d/results/E116/preflight/phaseA_preflight.tsv"
  if [ ! -f "$preflight" ]; then
    echo "Missing E116 preflight: $preflight. Rebuild the manifest." >&2
    exit 2
  fi
  if awk -F '\t' 'NF && $1 !~ /^#/ && $0 ~ /False/ {bad=1} END {exit bad ? 1 : 0}' "$preflight"; then
    :
  else
    echo "E116 preflight contains a False field: $preflight" >&2
    exit 2
  fi
}

snapshot_variants() {
  [ "${E116_SKIP_SCENE_SNAPSHOT:-0}" = "1" ] && return 0
  local variants=("$@")
  [ "${#variants[@]}" -gt 0 ] || return 0
  local tasks=()
  local variant row task
  for variant in "${variants[@]}"; do
    row="$(variant_row "$variant")"
    task="$(field_from_row "$row" 4)"
    tasks+=("$task")
  done
  bash workspace/core4d/scripts/convert/snapshot_scenes.sh E116 "${tasks[@]}"
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
    echo "Unknown E116 variant: $variant" >&2
    return 2
  fi
  task="$(field_from_row "$row" 4)"
  override="core4d_${variant}"
  out_dir="$RESULTS/${variant}_outdir_${STAGE}"
  mkdir -p "$out_dir"
  if [ -f "$RESULTS/${variant}.npz" ] && [ -f "$RESULTS/${variant}_${STAGE}.mp4" ] && [ -f "$out_dir/trajectory_mjwp_act.npz" ]; then
    echo "[$(date '+%H:%M:%S')] === E116 ${STAGE} ${variant} already complete; skip ==="
    return 0
  fi
  echo "[$(date '+%H:%M:%S')] === E116 ${STAGE} ${variant} task=${task} GPU=${gpu} ==="
  if [ "$STAGE" = "smoke" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
      +override="$override" task="$task" +use_torch_compile=false video_camera=auto \
      max_num_iterations="${SMOKE_MAX_NUM_ITERATIONS:-4}" \
      output_dir="$out_dir" video_output_path="$RESULTS/${variant}_${STAGE}.mp4" \
      > "$LOGS/${variant}_${STAGE}.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
      +override="$override" task="$task" +use_torch_compile=false video_camera=auto \
      output_dir="$out_dir" video_output_path="$RESULTS/${variant}_${STAGE}.mp4" \
      > "$LOGS/${variant}_${STAGE}.log" 2>&1
  fi
  cp "$out_dir/trajectory_mjwp_act.npz" "$RESULTS/${variant}.npz"
  extract_keyframes "$variant" "$RESULTS/${variant}_${STAGE}.mp4"
  echo "[$(date '+%H:%M:%S')] === E116 ${STAGE} ${variant} done ==="
}

case "$MODE" in
  list)
    require_manifest_ready
    variants_for_split "$(split_name "${4:-local}")"
    exit 0
    ;;
  single)
    require_manifest_ready
    VARIANT="${4:-}"
    [ -n "$VARIANT" ] || { echo "Usage: $0 single <smoke|full> <gpu> <variant>" >&2; exit 2; }
    snapshot_variants "$VARIANT"
    run_one "$VARIANT" "$GPU"
    ;;
  local|remote-gpu0|remote-gpu1)
    require_manifest_ready
    split="$(split_name "$MODE")"
    mapfile -t variants < <(variants_for_split "$split")
    if [ "${#variants[@]}" -eq 0 ]; then
      echo "No E116 variants for split=${split}; nothing to run."
      exit 0
    fi
    echo "E116 split=${split} variants=${#variants[@]} stage=${STAGE} gpu=${GPU}"
    snapshot_variants "${variants[@]}"
    for variant in "${variants[@]}"; do
      run_one "$variant" "$GPU"
    done
    ;;
  eval)
    shift 3 || true
    bash workspace/core4d/scripts/eval/eval_E116_surface_target_upright.sh "$STAGE" "$@"
    ;;
  *)
    echo "Usage: $0 {list|local|remote-gpu0|remote-gpu1|single|eval} {smoke|full} <gpu> [variant]" >&2
    exit 2
    ;;
esac
