#!/usr/bin/env bash
# E124 Phase A: SBTO carry-horizon diagnostic.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-local}"       # list | local | remote-gpu0 | remote-gpu1 | single | eval
STAGE="${2:-smoke}"      # smoke | full
GPU="${3:-0}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d/scripts/E124/variants.tsv}"
RESULTS="${RESULTS:-workspace/core4d/results/E124/cem/${STAGE}}"
LOGS="${LOGS:-logs/E124/cem/${STAGE}}"
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
    echo "Missing E124 variants file: $VARIANTS_FILE. Run build_sbto_carry_horizon_manifest.py first." >&2
    exit 2
  fi
  local count
  count="$(awk -F '\t' 'NF && $1 !~ /^#/ {n++} END {print n+0}' "$VARIANTS_FILE")"
  if [ "$count" -ne 8 ]; then
    echo "E124 variants.tsv has $count rows, expected 8. Rebuild the manifest." >&2
    exit 2
  fi
  local preflight="workspace/core4d/results/E124/preflight/phaseA_preflight.tsv"
  local summary="workspace/core4d/results/E124/preflight/phaseA_manifest_summary.json"
  if [ ! -f "$preflight" ] || [ ! -f "$summary" ]; then
    echo "Missing E124 preflight files. Rebuild the manifest." >&2
    exit 2
  fi
  local python_bin="${PYTHON_BIN:-.venv/bin/python}"
  if [ ! -x "$python_bin" ]; then
    python_bin="${PYTHON:-python3}"
  fi
  "$python_bin" - <<'PY'
import json
from pathlib import Path
summary = Path("workspace/core4d/results/E124/preflight/phaseA_manifest_summary.json")
data = json.loads(summary.read_text())
if not data.get("all_preflight_ok"):
    raise SystemExit("E124 preflight summary all_preflight_ok is false")
PY
}

snapshot_variants() {
  [ "${E124_SKIP_SCENE_SNAPSHOT:-0}" = "1" ] && return 0
  local variants=("$@")
  [ "${#variants[@]}" -gt 0 ] || return 0
  local tasks=()
  local variant row task
  for variant in "${variants[@]}"; do
    row="$(variant_row "$variant")"
    task="$(field_from_row "$row" 4)"
    tasks+=("$task")
  done
  bash workspace/core4d/scripts/convert/snapshot_scenes.sh E124 "${tasks[@]}"
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

copy_result_npz() {
  local src=$1
  local dst=$2
  local python_bin="${PYTHON_BIN:-.venv/bin/python}"
  if [ ! -x "$python_bin" ]; then
    python_bin="${PYTHON:-python3}"
  fi
  "$python_bin" - "$src" "$dst" <<'PY'
import sys
from pathlib import Path
import numpy as np

src, dst = sys.argv[1:3]
src_path = Path(src)
data = np.load(src, allow_pickle=True)
raw = {key: data[key] for key in data.files}
out = {}
changed = False
has_ref_channels = {"qpos_ref", "qvel_ref", "ctrl_ref", "time_ref"}.issubset(data.files)
for key in data.files:
    arr = data[key]
    if key in {"qpos", "qvel", "ctrl", "time"} and getattr(arr, "ndim", 0) >= 2 and arr.shape[0] == 1:
        sim_arr = arr[0]
        ref_key = f"{key}_ref"
        if has_ref_channels and ref_key in data.files:
            ref_arr = data[ref_key][0]
            if key == "time":
                arr = np.stack([sim_arr, ref_arr], axis=-1)
            else:
                arr = np.stack([sim_arr, ref_arr], axis=1)
        else:
            arr = sim_arr
        changed = True
    out[key] = arr
if changed:
    backup = src_path.with_name(f"{src_path.stem}_sbto_raw{src_path.suffix}")
    if not backup.exists():
        np.savez(backup, **raw)
    np.savez(src, **out)
np.savez(dst, **out)
PY
}

run_one() {
  local variant=$1
  local gpu=$2
  local row task override out_dir
  row="$(variant_row "$variant")"
  if [ -z "$row" ]; then
    echo "Unknown E124 variant: $variant" >&2
    return 2
  fi
  task="$(field_from_row "$row" 4)"
  override="core4d_${variant}"
  out_dir="$RESULTS/${variant}_outdir_${STAGE}"
  mkdir -p "$out_dir"
  if [ -f "$RESULTS/${variant}.npz" ] && [ -f "$RESULTS/${variant}_${STAGE}.mp4" ] && [ -f "$out_dir/trajectory_mjwp_act.npz" ]; then
    echo "[$(date '+%H:%M:%S')] === E124 ${STAGE} ${variant} already complete; skip ==="
    return 0
  fi
  echo "[$(date '+%H:%M:%S')] === E124 ${STAGE} ${variant} task=${task} GPU=${gpu} ==="
  if [ "$STAGE" = "smoke" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
      +override="$override" task="$task" +use_torch_compile=false video_camera=auto \
      sbto_max_iter_per_knot="${SMOKE_SBTO_MAX_ITER_PER_KNOT:-1}" \
      num_samples="${SMOKE_NUM_SAMPLES:-512}" \
      sbto_knot_dt="${SMOKE_SBTO_KNOT_DT:-0.35}" \
      output_dir="$out_dir" video_output_path="$RESULTS/${variant}_${STAGE}.mp4" \
      > "$LOGS/${variant}_${STAGE}.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
      +override="$override" task="$task" +use_torch_compile=false video_camera=auto \
      output_dir="$out_dir" video_output_path="$RESULTS/${variant}_${STAGE}.mp4" \
      > "$LOGS/${variant}_${STAGE}.log" 2>&1
  fi
  copy_result_npz "$out_dir/trajectory_mjwp_act.npz" "$RESULTS/${variant}.npz"
  extract_keyframes "$variant" "$RESULTS/${variant}_${STAGE}.mp4"
  echo "[$(date '+%H:%M:%S')] === E124 ${STAGE} ${variant} done ==="
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
      echo "No E124 variants for split=${split}; nothing to run."
      exit 0
    fi
    echo "E124 split=${split} variants=${#variants[@]} stage=${STAGE} gpu=${GPU}"
    snapshot_variants "${variants[@]}"
    for variant in "${variants[@]}"; do
      run_one "$variant" "$GPU"
    done
    ;;
  eval)
    shift 3 || true
    bash workspace/core4d/scripts/eval/eval_E124_sbto_carry_horizon.sh "$STAGE" "$@"
    ;;
  *)
    echo "Usage: $0 {list|local|remote-gpu0|remote-gpu1|single|eval} {smoke|full} <gpu> [variant]" >&2
    exit 2
    ;;
esac
