#!/usr/bin/env bash
# E147 rubber_hull hand-collision full-CEM runner.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-list}"       # list | single | remote-gpu0 | remote-gpu1
STAGE="${2:-full}"      # smoke | full
GPU="${3:-0}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d/scripts/E147/variants.tsv}"
RESULTS="${RESULTS:-workspace/core4d/results/E147/rubber_hand_collision/cem/${STAGE}}"
LOGS="${LOGS:-logs/E147/cem/${STAGE}}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
mkdir -p "$RESULTS/keyframes" "$LOGS"

if [ "$STAGE" != "smoke" ] && [ "$STAGE" != "full" ]; then
  echo "Invalid STAGE=$STAGE (use smoke|full)" >&2
  exit 2
fi

variant_row() {
  local variant=$1
  awk -F '\t' -v v="$variant" 'NF && $1 !~ /^#/ && NR > 1 && $2 == v {print; exit}' "$VARIANTS_FILE"
}

field_from_row() {
  local row=$1
  local idx=$2
  awk -F '\t' -v row="$row" -v idx="$idx" 'BEGIN {split(row, a, FS); print a[idx]}'
}

variants_for_split() {
  local split=$1
  awk -F '\t' -v want="$split" 'NF && $1 !~ /^#/ && NR > 1 && $8 == want && $30 != "already_done" {print $2}' "$VARIANTS_FILE"
}

require_manifest_ready() {
  if [ ! -f "$VARIANTS_FILE" ]; then
    echo "Missing E147 variants file: $VARIANTS_FILE" >&2
    exit 2
  fi
  "$PYTHON_BIN" - <<'PY' "$VARIANTS_FILE"
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
if len(rows) != 10:
    raise SystemExit(f"E147 manifest must contain 10 rows, got {len(rows)}: {path}")
missing = []
for row in rows:
    for key in ("trajectory", "base_scene_act", "rubber_scene_act", "override"):
        p = Path(row[key])
        if not p.is_file():
            missing.append(f"{row['variant']}:{key}:{p}")
if missing:
    raise SystemExit("E147 manifest missing files:\n" + "\n".join(missing))
splits = {row["split"] for row in rows}
if splits != {"remote-gpu0", "remote-gpu1"}:
    raise SystemExit(f"E147 expected remote-gpu0/remote-gpu1 splits, got {sorted(splits)}")
print(f"E147 manifest ready: rows={len(rows)}", file=sys.stderr)
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
  local row task override out_dir
  row="$(variant_row "$variant")"
  if [ -z "$row" ]; then
    echo "Unknown E147 variant: $variant" >&2
    return 2
  fi
  task="$(field_from_row "$row" 13)"
  override="core4d_${variant}"
  out_dir="$RESULTS/${variant}_outdir_${STAGE}"
  mkdir -p "$out_dir"
  if outputs_exist "$variant"; then
    echo "[$(date '+%H:%M:%S')] === E147 ${variant} outputs exist; skip ==="
    return 0
  fi

  echo "[$(date '+%H:%M:%S')] === E147 ${STAGE} ${variant} task=${task} GPU=${gpu} ==="
  if [ "$STAGE" = "smoke" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u examples/run_mjwp.py \
      +override="$override" task="$task" +use_torch_compile=false video_camera=auto \
      num_samples="${SMOKE_NUM_SAMPLES:-64}" max_num_iterations="${SMOKE_MAX_NUM_ITERATIONS:-4}" \
      output_dir="$out_dir" video_output_path="$RESULTS/${variant}_${STAGE}.mp4" \
      > "$LOGS/${variant}_${STAGE}.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u examples/run_mjwp.py \
      +override="$override" task="$task" +use_torch_compile=false video_camera=auto \
      output_dir="$out_dir" video_output_path="$RESULTS/${variant}_${STAGE}.mp4" \
      > "$LOGS/${variant}_${STAGE}.log" 2>&1
  fi
  cp "$out_dir/trajectory_mjwp_act.npz" "$RESULTS/${variant}.npz"
  extract_keyframes "$variant" "$RESULTS/${variant}_${STAGE}.mp4"
  echo "[$(date '+%H:%M:%S')] === E147 ${STAGE} ${variant} done ==="
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
    echo "E147 split=${MODE} variants=${#variants[@]} stage=${STAGE} gpu=${GPU}"
    for variant in "${variants[@]}"; do
      run_one "$variant" "$GPU"
    done
    ;;
  *)
    echo "Usage: $0 {list|single|remote-gpu0|remote-gpu1} {smoke|full} <gpu> [variant|split]" >&2
    exit 2
    ;;
esac
