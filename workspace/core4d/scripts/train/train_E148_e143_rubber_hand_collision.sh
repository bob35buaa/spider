#!/usr/bin/env bash
# E148: rubber_hull hand-collision full CEM for E143 24-case workset.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-list}"       # list | single | remote-gpu0 | remote-gpu1
STAGE="${2:-full}"      # smoke | full
GPU="${3:-0}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d/scripts/E148/variants.tsv}"
RESULTS="${RESULTS:-workspace/core4d/results/E148/e143_24case_rubber_hand_collision/cem/${STAGE}}"
LOGS="${LOGS:-logs/E148/cem/${STAGE}}"
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
  awk -F '\t' -v want="$split" 'NF && NR > 1 && $9 == want && $10 == "to_run" {print $2}' "$VARIANTS_FILE"
}

require_manifest_ready() {
  if [ ! -f "$VARIANTS_FILE" ]; then
    echo "Missing E148 variants file: $VARIANTS_FILE" >&2
    exit 2
  fi
  "$PYTHON_BIN" - <<'PY' "$VARIANTS_FILE"
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
if len(rows) != 24:
    raise SystemExit(f"E148 manifest must contain 24 rows, got {len(rows)}: {path}")
reuse = [r for r in rows if r["run_status"] == "reuse_e147"]
todo = [r for r in rows if r["run_status"] == "to_run"]
if len(reuse) != 8 or len(todo) != 16:
    raise SystemExit(f"E148 expected reuse=8/to_run=16, got reuse={len(reuse)} to_run={len(todo)}")
splits = {r["split"] for r in todo}
if splits != {"remote-gpu0", "remote-gpu1"}:
    raise SystemExit(f"E148 expected remote-gpu0/remote-gpu1 splits for to_run rows, got {sorted(splits)}")
missing = []
for row in todo:
    for key in ("override", "base_scene_act", "rubber_scene_act", "target_scene", "trajectory", "mask_path"):
        p = Path(row[key])
        if not p.is_file():
            missing.append(f"{row['variant']}:{key}:{p}")
if missing:
    raise SystemExit("E148 manifest missing to-run files:\n" + "\n".join(missing))
print(f"E148 manifest ready: rows={len(rows)} reuse={len(reuse)} to_run={len(todo)}", file=sys.stderr)
PY
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
    echo "Unknown E148 variant: $variant" >&2
    return 2
  fi
  task="$(field_from_row "$row" 15)"
  run_status="$(field_from_row "$row" 10)"
  if [ "$run_status" != "to_run" ]; then
    echo "[$(date '+%H:%M:%S')] === E148 ${variant} run_status=${run_status}; skip ==="
    return 0
  fi
  if outputs_exist "$variant"; then
    echo "[$(date '+%H:%M:%S')] === E148 ${variant} outputs exist; skip ==="
    return 0
  fi
  override_stem="$(basename "$(field_from_row "$row" 27)" .yaml)"
  out_dir="$RESULTS/${variant}_outdir_${STAGE}"
  mkdir -p "$out_dir"
  echo "[$(date '+%H:%M:%S')] === E148 ${STAGE} ${variant} task=${task} GPU=${gpu} ==="
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
  echo "[$(date '+%H:%M:%S')] === E148 ${STAGE} ${variant} done ==="
}

case "$MODE" in
  list)
    require_manifest_ready
    variants_for_split "${4:-remote-gpu0}"
    ;;
  single)
    require_manifest_ready
    VARIANT="${4:-}"
    [ -n "$VARIANT" ] || { echo "Usage: $0 single <smoke|full> <gpu> <variant>" >&2; exit 2; }
    run_one "$VARIANT" "$GPU"
    ;;
  remote-gpu0|remote-gpu1)
    require_manifest_ready
    mapfile -t variants < <(variants_for_split "$MODE")
    echo "E148 split=${MODE} variants=${#variants[@]} stage=${STAGE} gpu=${GPU}"
    for variant in "${variants[@]}"; do
      run_one "$variant" "$GPU"
    done
    ;;
  *)
    echo "Usage: $0 {list|single|remote-gpu0|remote-gpu1} {smoke|full} <gpu> [variant|split]" >&2
    exit 2
    ;;
esac
