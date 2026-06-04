#!/usr/bin/env bash
# E123 Phase A: two-stage carry curriculum CEM.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-local}"       # list | stage1-list | local | remote-gpu0 | remote-gpu1 | single-stage1 | single | eval
STAGE="${2:-smoke}"      # smoke | full
GPU="${3:-0}"
STAGE1_FILE="${STAGE1_FILE:-workspace/core4d/scripts/E123/stage1.tsv}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d/scripts/E123/variants.tsv}"
RESULTS="${RESULTS:-workspace/core4d/results/E123/cem/${STAGE}}"
LOGS="${LOGS:-logs/E123/cem/${STAGE}}"
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

row_from_file() {
  local file=$1
  local variant=$2
  awk -F '\t' -v v="$variant" 'NF && $1 !~ /^#/ && $2 == v {print; exit}' "$file"
}

field_from_row() {
  local row=$1
  local idx=$2
  awk -F '\t' -v row="$row" -v idx="$idx" 'BEGIN {split(row, a, FS); print a[idx]}'
}

stage1_for_split() {
  local split=$1
  awk -F '\t' -v want="$split" 'NF && $1 !~ /^#/ && $6 == want {print $2}' "$STAGE1_FILE"
}

stage2_for_split() {
  local split=$1
  awk -F '\t' -v want="$split" 'NF && $1 !~ /^#/ && $6 == want {print $2}' "$VARIANTS_FILE"
}

require_manifest_ready() {
  if [ ! -f "$STAGE1_FILE" ] || [ ! -f "$VARIANTS_FILE" ]; then
    echo "Missing E123 manifests. Run build_two_stage_curriculum_manifest.py first." >&2
    exit 2
  fi
  local stage1_count stage2_count
  stage1_count="$(awk -F '\t' 'NF && $1 !~ /^#/ {n++} END {print n+0}' "$STAGE1_FILE")"
  stage2_count="$(awk -F '\t' 'NF && $1 !~ /^#/ {n++} END {print n+0}' "$VARIANTS_FILE")"
  if [ "$stage1_count" -ne 4 ] || [ "$stage2_count" -ne 8 ]; then
    echo "E123 manifest row count mismatch: stage1=$stage1_count stage2=$stage2_count, expected 4/8." >&2
    exit 2
  fi
  local summary="workspace/core4d/results/E123/preflight/phaseA_manifest_summary.json"
  if [ ! -f "$summary" ]; then
    echo "Missing E123 preflight summary: $summary. Rebuild the manifest." >&2
    exit 2
  fi
  local python_bin="${PYTHON_BIN:-.venv/bin/python}"
  if [ ! -x "$python_bin" ]; then
    python_bin="${PYTHON:-python3}"
  fi
  "$python_bin" - <<'PY'
import json
from pathlib import Path
summary = Path("workspace/core4d/results/E123/preflight/phaseA_manifest_summary.json")
data = json.loads(summary.read_text())
if not data.get("all_preflight_ok"):
    raise SystemExit("E123 preflight summary all_preflight_ok is false")
PY
}

snapshot_tasks() {
  [ "${E123_SKIP_SCENE_SNAPSHOT:-0}" = "1" ] && return 0
  local variants=("$@")
  [ "${#variants[@]}" -gt 0 ] || return 0
  local tasks=()
  local variant row task
  for variant in "${variants[@]}"; do
    row="$(row_from_file "$STAGE1_FILE" "$variant")"
    [ -n "$row" ] || row="$(row_from_file "$VARIANTS_FILE" "$variant")"
    task="$(field_from_row "$row" 4)"
    tasks+=("$task")
  done
  bash workspace/core4d/scripts/convert/snapshot_scenes.sh E123 "${tasks[@]}"
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

run_mjwp_variant() {
  local variant=$1
  local task=$2
  local gpu=$3
  local label=$4
  local override="core4d_${variant}"
  local out_dir="$RESULTS/${variant}_outdir_${STAGE}"
  mkdir -p "$out_dir"
  if [ -f "$RESULTS/${variant}.npz" ] && [ -f "$RESULTS/${variant}_${STAGE}.mp4" ] && [ -f "$out_dir/trajectory_mjwp_act.npz" ]; then
    echo "[$(date '+%H:%M:%S')] === E123 ${label} ${STAGE} ${variant} already complete; skip ==="
    return 0
  fi
  echo "[$(date '+%H:%M:%S')] === E123 ${label} ${STAGE} ${variant} task=${task} GPU=${gpu} ==="
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
  echo "[$(date '+%H:%M:%S')] === E123 ${label} ${STAGE} ${variant} done ==="
}

run_stage1_one() {
  local variant=$1
  local gpu=$2
  local row task
  row="$(row_from_file "$STAGE1_FILE" "$variant")"
  [ -n "$row" ] || { echo "Unknown E123 stage1 variant: $variant" >&2; return 2; }
  task="$(field_from_row "$row" 4)"
  run_mjwp_variant "$variant" "$task" "$gpu" "stage1"
}

run_stage2_one() {
  local variant=$1
  local gpu=$2
  local row task warmstart
  row="$(row_from_file "$VARIANTS_FILE" "$variant")"
  [ -n "$row" ] || { echo "Unknown E123 stage2 variant: $variant" >&2; return 2; }
  task="$(field_from_row "$row" 4)"
  warmstart="$(field_from_row "$row" 16)"
  if [ ! -f "$warmstart" ]; then
    echo "Missing E123 stage2 warmstart for ${variant}: ${warmstart}. Run stage1 conversion first." >&2
    return 2
  fi
  run_mjwp_variant "$variant" "$task" "$gpu" "stage2"
}

convert_stage1_for_split() {
  local split=$1
  .venv/bin/python workspace/core4d/scripts/E123/build_two_stage_curriculum_manifest.py \
    --convert-stage1 --stage "$STAGE" --split "$split"
}

case "$MODE" in
  stage1-list)
    require_manifest_ready
    stage1_for_split "$(split_name "${4:-local}")"
    exit 0
    ;;
  list)
    require_manifest_ready
    stage2_for_split "$(split_name "${4:-local}")"
    exit 0
    ;;
  single-stage1)
    require_manifest_ready
    VARIANT="${4:-}"
    [ -n "$VARIANT" ] || { echo "Usage: $0 single-stage1 <smoke|full> <gpu> <variant>" >&2; exit 2; }
    snapshot_tasks "$VARIANT"
    run_stage1_one "$VARIANT" "$GPU"
    ;;
  single)
    require_manifest_ready
    VARIANT="${4:-}"
    [ -n "$VARIANT" ] || { echo "Usage: $0 single <smoke|full> <gpu> <variant>" >&2; exit 2; }
    snapshot_tasks "$VARIANT"
    run_stage2_one "$VARIANT" "$GPU"
    ;;
  local|remote-gpu0|remote-gpu1)
    require_manifest_ready
    split="$(split_name "$MODE")"
    mapfile -t stage1_variants < <(stage1_for_split "$split")
    mapfile -t stage2_variants < <(stage2_for_split "$split")
    if [ "${#stage1_variants[@]}" -eq 0 ] && [ "${#stage2_variants[@]}" -eq 0 ]; then
      echo "No E123 variants for split=${split}; nothing to run."
      exit 0
    fi
    echo "E123 split=${split} stage1=${#stage1_variants[@]} stage2=${#stage2_variants[@]} stage=${STAGE} gpu=${GPU}"
    snapshot_tasks "${stage1_variants[@]}" "${stage2_variants[@]}"
    for variant in "${stage1_variants[@]}"; do
      run_stage1_one "$variant" "$GPU"
    done
    convert_stage1_for_split "$split"
    for variant in "${stage2_variants[@]}"; do
      run_stage2_one "$variant" "$GPU"
    done
    ;;
  eval)
    shift 3 || true
    bash workspace/core4d/scripts/eval/eval_E123_two_stage_curriculum.sh "$STAGE" "$@"
    ;;
  *)
    echo "Usage: $0 {stage1-list|list|local|remote-gpu0|remote-gpu1|single-stage1|single|eval} {smoke|full} <gpu> [variant]" >&2
    exit 2
    ;;
esac
