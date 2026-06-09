#!/usr/bin/env bash
# E151: route-B rubber-hand surface contact reward experiments.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-list}"       # list | single | local-gpu0 | remote-gpu0 | remote-gpu1
STAGE="${2:-full}"      # smoke | full
GPU="${3:-0}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d/scripts/E151/variants.tsv}"
RESULTS="${RESULTS:-workspace/core4d/results/E151/route_b_hand_surface_contact/cem/${STAGE}}"
LOGS="${LOGS:-logs/E151/cem/${STAGE}}"
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
  awk -F '\t' -v want="$split" 'NF && NR > 1 && $12 == want && $13 == "to_run" {print $2}' "$VARIANTS_FILE"
}

require_manifest_ready() {
  if [ ! -f "$VARIANTS_FILE" ]; then
    echo "Missing E151 variants file: $VARIANTS_FILE" >&2
    exit 2
  fi
  "$PYTHON_BIN" - <<'PY' "$VARIANTS_FILE"
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
reuse = [r for r in rows if r["run_status"] == "reuse_e148"]
todo = [r for r in rows if r["run_status"] == "to_run"]
if len(rows) != 12 or len(reuse) != 3 or len(todo) != 9:
    raise SystemExit(f"E151 expected rows=12 reuse=3 to_run=9, got rows={len(rows)} reuse={len(reuse)} to_run={len(todo)}")
split_counts = {}
for row in todo:
    split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
if split_counts != {"local-gpu0": 3, "remote-gpu0": 3, "remote-gpu1": 3}:
    raise SystemExit(f"E151 expected split counts 3/3/3, got {split_counts}")
missing = []
for row in todo:
    for key in ("override", "target_scene", "trajectory", "rubber_scene_act", "mask_path"):
        if not Path(row[key]).is_file():
            missing.append(f"{row['variant']}:{key}:{row[key]}")
    if row["method"] in {"b2_sup", "b2_tip"} and not Path(row["target_npz"]).is_file():
        missing.append(f"{row['variant']}:target_npz:{row['target_npz']}")
if missing:
    raise SystemExit("E151 missing run inputs:\n" + "\n".join(missing))
print(f"E151 manifest ready: rows={len(rows)} reuse={len(reuse)} to_run={len(todo)}", file=sys.stderr)
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
  local row task run_status override_stem out_dir method
  row="$(variant_row "$variant")"
  if [ -z "$row" ]; then
    echo "Unknown E151 variant: $variant" >&2
    return 2
  fi
  method="$(field_from_row "$row" 4)"
  task="$(field_from_row "$row" 21)"
  run_status="$(field_from_row "$row" 13)"
  if [ "$run_status" != "to_run" ]; then
    echo "[$(date '+%H:%M:%S')] === E151 ${variant} run_status=${run_status}; skip ==="
    return 0
  fi
  if outputs_exist "$variant"; then
    echo "[$(date '+%H:%M:%S')] === E151 ${variant} outputs exist; skip ==="
    return 0
  fi
  override_stem="$(basename "$(field_from_row "$row" 27)" .yaml)"
  out_dir="$RESULTS/${variant}_outdir_${STAGE}"
  mkdir -p "$out_dir"
  echo "[$(date '+%H:%M:%S')] === E151 ${STAGE} ${variant} method=${method} task=${task} GPU=${gpu} ==="
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
  echo "[$(date '+%H:%M:%S')] === E151 ${STAGE} ${variant} done ==="
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
    echo "E151 split=${MODE} variants=${#variants[@]} stage=${STAGE} gpu=${GPU}"
    for variant in "${variants[@]}"; do
      run_one "$variant" "$GPU"
    done
    ;;
  *)
    echo "Usage: $0 {list|single|local-gpu0|remote-gpu0|remote-gpu1} {smoke|full} <gpu> [variant|split]" >&2
    exit 2
    ;;
esac
