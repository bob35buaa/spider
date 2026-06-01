#!/usr/bin/env bash
# E107 Phase 2: Box021 selected-4 clean ref-FK full CEM.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-local}"       # list | local | remote-gpu0 | remote-gpu1 | single | eval
STAGE="${2:-full}"       # full | smoke
GPU="${3:-0}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d/scripts/E107/selected4_variants.tsv}"
RESULTS="${RESULTS:-workspace/core4d/results/E107/cem/${STAGE}}"
LOGS="${LOGS:-logs/E107/cem/${STAGE}}"
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

snapshot_variants() {
  [ "${E107_SKIP_SCENE_SNAPSHOT:-0}" = "1" ] && return 0
  local variants=("$@")
  [ "${#variants[@]}" -gt 0 ] || return 0
  local tasks=()
  local variant row task
  for variant in "${variants[@]}"; do
    row="$(variant_row "$variant")"
    task="$(field_from_row "$row" 4)"
    tasks+=("$task")
  done
  bash workspace/core4d/scripts/convert/snapshot_scenes.sh E107 "${tasks[@]}"
}

require_manifest_ready() {
  local count
  count="$(awk -F '\t' 'NF && $1 !~ /^#/ {n++} END {print n+0}' "$VARIANTS_FILE")"
  if [ "$count" -ne 4 ]; then
    echo "E107 selected4 variants.tsv has $count rows, expected 4. Run build_selected4_cem_manifest.py first." >&2
    exit 2
  fi
}

pre_cem_gate_ok() {
  local variant=$1
  [ "${E107_REQUIRE_PRE_CEM_GATE:-1}" = "0" ] && return 0
  local review="workspace/core4d/results/E107/pre_cem_visual_review/${variant}/REVIEW.md"
  if [ ! -f "$review" ]; then
    echo "Missing E107 pre-CEM medium review: $review" >&2
    return 1
  fi
  if grep -Eq 'FAIL_PRE_CEM_VISUAL|Status:[[:space:]]*FAIL' "$review"; then
    echo "Pre-CEM review failed for $variant: $review" >&2
    return 1
  fi
  if ! grep -Eq 'PASS_WITH_NOTES|Status:[[:space:]]*PASS|decision:[[:space:]]*PASS|PASS' "$review"; then
    echo "Pre-CEM review lacks PASS/PASS_WITH_NOTES for $variant: $review" >&2
    return 1
  fi
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
    echo "Unknown E107 variant: $variant" >&2
    return 2
  fi
  task="$(field_from_row "$row" 4)"
  if [[ "$task" == *"_e092_"* ]]; then
    echo "Refusing to run polluted legacy task path: $task" >&2
    return 2
  fi
  pre_cem_gate_ok "$variant"
  override="core4d_${variant}"
  out_dir="$RESULTS/${variant}_outdir_${STAGE}"
  mkdir -p "$out_dir"
  echo "[$(date '+%H:%M:%S')] === E107 ${STAGE} ${variant} task=${task} GPU=${gpu} ==="
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
  echo "[$(date '+%H:%M:%S')] === E107 ${STAGE} ${variant} done ==="
}

case "$MODE" in
  list)
    variants_for_split "$(split_name "${4:-local}")"
    exit 0
    ;;
  single)
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
      echo "No E107 variants for split=${split}; nothing to run."
      exit 0
    fi
    echo "E107 split=${split} variants=${#variants[@]} stage=${STAGE} gpu=${GPU}"
    snapshot_variants "${variants[@]}"
    for variant in "${variants[@]}"; do
      run_one "$variant" "$GPU"
    done
    ;;
  eval)
    shift 3 || true
    RESULTS="$RESULTS" VARIANTS_FILE="$VARIANTS_FILE" .venv/bin/python \
      workspace/core4d/scripts/eval/eval_E107_box021_selected4.py --stage "$STAGE" "$@" \
      | tee "$LOGS/eval_E107_${STAGE}.log"
    ;;
  *)
    echo "Usage: $0 {list|local|remote-gpu0|remote-gpu1|single|eval} {smoke|full} <gpu> [variant ...]" >&2
    exit 2
    ;;
esac
