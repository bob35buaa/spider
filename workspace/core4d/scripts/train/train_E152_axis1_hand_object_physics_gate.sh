#!/usr/bin/env bash
# E152: axis-1 hand-object physics gate experiments.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-list}"       # list | single | local-gpu0 | remote-gpu0 | remote-gpu1
STAGE="${2:-full}"      # smoke | full
GPU="${3:-0}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d/scripts/E152/variants.tsv}"
RESULTS="${RESULTS:-workspace/core4d/results/E152/axis1_hand_object_physics_gate/cem/${STAGE}}"
LOGS="${LOGS:-logs/E152/cem/${STAGE}}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
mkdir -p "$RESULTS/keyframes" "$LOGS"

if [ "$STAGE" != "smoke" ] && [ "$STAGE" != "full" ]; then
  echo "Invalid STAGE=$STAGE (use smoke|full)" >&2
  exit 2
fi

field_for_variant() {
  local variant=$1
  local field=$2
  "$PYTHON_BIN" - <<'PY' "$VARIANTS_FILE" "$variant" "$field"
import csv
import sys
path, variant, field = sys.argv[1:4]
with open(path, encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["variant"] == variant:
            print(row.get(field, ""))
            raise SystemExit(0)
raise SystemExit(f"unknown variant: {variant}")
PY
}

variants_for_split() {
  local split=$1
  "$PYTHON_BIN" - <<'PY' "$VARIANTS_FILE" "$split"
import csv
import sys
path, split = sys.argv[1:3]
with open(path, encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["split"] == split and row["run_status"] == "to_run":
            print(row["variant"])
PY
}

require_manifest_ready() {
  if [ ! -f "$VARIANTS_FILE" ]; then
    echo "Missing E152 variants file: $VARIANTS_FILE" >&2
    exit 2
  fi
  local args=()
  if [ "${E152_ALLOW_MISSING_REUSE:-0}" = "1" ]; then
    args+=(--allow-missing-reuse)
  fi
  "$PYTHON_BIN" workspace/core4d/scripts/E152/check_hand_gate_preflight.py "${args[@]}" >/tmp/e152_preflight_train.log
  echo "E152 manifest ready" >&2
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
  local task run_status override_stem out_dir method min_sdf max_pct
  run_status="$(field_for_variant "$variant" run_status)"
  method="$(field_for_variant "$variant" method)"
  task="$(field_for_variant "$variant" derived_task)"
  if [ "$run_status" != "to_run" ]; then
    echo "[$(date '+%H:%M:%S')] === E152 ${variant} run_status=${run_status}; skip ==="
    return 0
  fi
  if outputs_exist "$variant"; then
    echo "[$(date '+%H:%M:%S')] === E152 ${variant} outputs exist; skip ==="
    return 0
  fi
  override_stem="$(basename "$(field_for_variant "$variant" override)" .yaml)"
  min_sdf="$(field_for_variant "$variant" cem_hand_gate_min_sdf_m)"
  max_pct="$(field_for_variant "$variant" cem_hand_gate_max_violation_pct)"
  out_dir="$RESULTS/${variant}_outdir_${STAGE}"
  mkdir -p "$out_dir"
  echo "[$(date '+%H:%M:%S')] === E152 ${STAGE} ${variant} method=${method} task=${task} GPU=${gpu} hand_gate=${min_sdf}/${max_pct} ==="
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
  echo "[$(date '+%H:%M:%S')] === E152 ${STAGE} ${variant} done ==="
}

case "$MODE" in
  list)
    require_manifest_ready
    variants_for_split "${4:-local-gpu0}"
    ;;
  single)
    require_manifest_ready
    VARIANT="${4:-}"
    [ -n "$VARIANT" ] || { echo "Usage: $0 single <smoke|full> <gpu> <variant>" >&2; exit 2; }
    run_one "$VARIANT" "$GPU"
    ;;
  local-gpu0|remote-gpu0|remote-gpu1)
    require_manifest_ready
    mapfile -t variants < <(variants_for_split "$MODE")
    echo "E152 split=${MODE} variants=${#variants[@]} stage=${STAGE} gpu=${GPU}"
    for variant in "${variants[@]}"; do
      run_one "$variant" "$GPU"
    done
    ;;
  *)
    echo "Usage: $0 {list|single|local-gpu0|remote-gpu0|remote-gpu1} {smoke|full} <gpu> [variant|split]" >&2
    exit 2
    ;;
esac
