#!/usr/bin/env bash
# E145 Phase2 high-priority full-CEM runner. Consumes the 9-case high-priority manifest by default.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-local}"       # list | local | remote-gpu0 | remote-gpu1 | single | eval
STAGE="${2:-full}"       # smoke | full
GPU="${3:-0}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d/scripts/E145/variants_phase2_high.tsv}"
PREFLIGHT="${PREFLIGHT:-workspace/core4d/results/E145/full_nonbox_to_rl_ready/cem_ready/raw_mask_ref_fk_cem_ready_preflight.tsv}"
RESULTS="${RESULTS:-workspace/core4d/results/E145/full_nonbox_to_rl_ready/cem/${STAGE}}"
LOGS="${LOGS:-logs/E145/cem/${STAGE}}"
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
  awk -F '\t' -v want="$split" 'NF && $1 !~ /^#/ && NR > 2 && $8 == want && $24 != "already_done" {print $2}' "$VARIANTS_FILE"
}

require_manifest_ready() {
  if [ ! -f "$VARIANTS_FILE" ]; then
    echo "Missing E145 high-priority variants file: $VARIANTS_FILE" >&2
    exit 2
  fi
  if [ ! -f "$PREFLIGHT" ]; then
    echo "Missing E145 preflight: $PREFLIGHT" >&2
    exit 2
  fi
  .venv/bin/python - <<'PY' "$VARIANTS_FILE" "$PREFLIGHT"
import csv
import os
import sys
from pathlib import Path

variants_path = Path(sys.argv[1])
preflight_path = Path(sys.argv[2])
e144_variants_path = Path("workspace/core4d/scripts/E144/variants.tsv")

def rows(path):
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))

variants = rows(variants_path)
if len(variants) != 9:
    raise SystemExit(f"E145 Phase2 high-priority manifest must contain 9 rows, got {len(variants)}: {variants_path}")
selected = {row["variant"] for row in variants}
preflight = {row["variant"]: row for row in rows(preflight_path)}
missing = sorted(selected - set(preflight))
bad = sorted(v for v in selected if preflight.get(v, {}).get("preflight_ok") != "True")
if missing or bad:
    raise SystemExit(f"E145 high-priority preflight failed; missing={missing}; bad={bad}")
if os.environ.get("E145_ALLOW_E144_OVERLAP", "0") != "1" and e144_variants_path.is_file():
    e144_cases = {row["case_id"] for row in rows(e144_variants_path)}
    overlap = sorted(row["case_id"] for row in variants if row["case_id"] in e144_cases)
    if overlap:
        raise SystemExit(
            "E145 high-priority manifest overlaps E144 full-CEM cases; "
            f"overlap={overlap}. Set E145_ALLOW_E144_OVERLAP=1 only for an intentional rerun."
        )
print(f"E145 high-priority manifest ready: rows={len(variants)}")
PY
}

snapshot_variants() {
  [ "${E145_SKIP_SCENE_SNAPSHOT:-0}" = "1" ] && return 0
  local variants=("$@")
  [ "${#variants[@]}" -gt 0 ] || return 0
  local tasks=()
  local variant row task
  for variant in "${variants[@]}"; do
    row="$(variant_row "$variant")"
    task="$(field_from_row "$row" 12)"
    tasks+=("$task")
  done
  bash workspace/core4d/scripts/convert/snapshot_scenes.sh E145 "${tasks[@]}"
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
    echo "Unknown E145 variant: $variant" >&2
    return 2
  fi
  task="$(field_from_row "$row" 12)"
  if outputs_exist "$variant"; then
    echo "[$(date '+%H:%M:%S')] === E145 ${variant} outputs exist; skip ==="
    return 0
  fi
  override="core4d_${variant}"
  out_dir="$RESULTS/${variant}_outdir_${STAGE}"
  mkdir -p "$out_dir"
  echo "[$(date '+%H:%M:%S')] === E145 ${STAGE} ${variant} task=${task} GPU=${gpu} ==="
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
  echo "[$(date '+%H:%M:%S')] === E145 ${STAGE} ${variant} done ==="
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
      echo "No E145 variants for split=${split}; nothing to run."
      exit 0
    fi
    echo "E145 split=${split} variants=${#variants[@]} stage=${STAGE} gpu=${GPU}"
    snapshot_variants "${variants[@]}"
    for variant in "${variants[@]}"; do
      run_one "$variant" "$GPU"
    done
    ;;
  eval)
    shift 3 || true
    bash workspace/core4d/scripts/eval/eval_E145_full_nonbox_raw_mask_ref_fk.sh "$STAGE" "$@"
    ;;
  *)
    echo "Usage: $0 {list|local|remote-gpu0|remote-gpu1|single|eval} {smoke|full} <gpu> [variant]" >&2
    exit 2
    ;;
esac
