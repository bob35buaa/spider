#!/usr/bin/env bash
# E029: dynamic D6 support-body retarget entrypoint.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-smoke}"
GPU="${2:-0}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d_collab_retarget/results/E029/d6/manifest.tsv}"
RESULTS="${RESULTS:-workspace/core4d_collab_retarget/results/E029/d6}"
LOGS="${LOGS:-logs/core4d_collab_retarget/E029}"
mkdir -p "$RESULTS/keyframes" "$RESULTS/online_video" "$LOGS"
RUN_EXTRA_ARGS=()

variant_row() {
  local variant=$1
  awk -F '\t' -v v="$variant" 'NR > 1 && NF && $1 == v {print; exit}' "$VARIANTS_FILE"
}

field_from_row() {
  local row=$1
  local idx=$2
  awk -F '\t' -v row="$row" -v idx="$idx" 'BEGIN {split(row, a, FS); print a[idx]}'
}

all_compliant_variants() {
  awk -F '\t' 'NR > 1 && NF && $6 == "d6_compliant" {print $1}' "$VARIANTS_FILE"
}

variants_for_queue() {
  local queue=$1
  awk -F '\t' -v want_queue="$queue" 'NR > 1 && NF && $7 == want_queue {print $1}' "$VARIANTS_FILE"
}

override_for_variant() {
  echo "core4d_collab_${1}"
}

extract_keyframes() {
  local variant=$1
  local video=$2
  [ -f "$video" ] || return 0
  command -v ffmpeg >/dev/null 2>&1 || return 0
  mkdir -p "$RESULTS/keyframes/$variant"
  for f in 16 32 50 75 90 115 160 204; do
    ffmpeg -y -loglevel error -i "$video" \
      -vf "select=eq(n\\,$f)" -frames:v 1 -vsync 0 \
      "$RESULTS/keyframes/$variant/f${f}.jpg" || true
  done
}

run_one() {
  local variant=$1
  local gpu=$2
  shift 2
  local extra_args=("$@")
  local row task override out_dir npz_out variant_log run_pid run_status
  row="$(variant_row "$variant")"
  if [ -z "$row" ]; then
    echo "Unknown E029 variant: $variant" >&2
    return 2
  fi
  task="$(field_from_row "$row" 4)"
  override="$(override_for_variant "$variant")"
  out_dir="$RESULTS/${variant}_outdir"
  npz_out="$out_dir/trajectory_mjwp.npz"
  variant_log="$LOGS/${variant}.log"
  mkdir -p "$out_dir"

  echo "[$(date '+%H:%M:%S')] === ${variant} task=${task} override=${override} GPU=${gpu} ==="
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
    +override="$override" \
    task="$task" \
    +use_torch_compile=false \
    output_dir="$out_dir" \
    video_output_path="$RESULTS/online_video/${variant}.mp4" \
    "${extra_args[@]}" \
    > "$variant_log" 2>&1 &
  run_pid=$!
  wait "$run_pid" || run_status=$?
  run_status="${run_status:-0}"
  if [ "$run_status" -ne 0 ]; then
    echo "[$(date '+%H:%M:%S')] ERROR ${variant} exited with status ${run_status}" >&2
    return "$run_status"
  fi
  if [ ! -f "$npz_out" ]; then
    echo "Expected output missing: $npz_out" >&2
    return 1
  fi
  cp "$npz_out" "$RESULTS/${variant}.npz"
  extract_keyframes "$variant" "$RESULTS/online_video/${variant}.mp4"
}

run_variants() {
  local gpu=$1
  shift
  local -a variants=("$@")
  local status=0
  if [ "${#variants[@]}" -eq 0 ]; then
    echo "No E029 variants selected" >&2
    exit 2
  fi
  for variant in "${variants[@]}"; do
    if ! run_one "$variant" "$gpu" "${RUN_EXTRA_ARGS[@]}"; then
      echo "[$(date '+%H:%M:%S')] WARN ${variant} failed; continuing E029 queue" | tee -a "$LOGS/failed_${MODE}.log" >&2
      status=1
    fi
  done
  return "$status"
}

case "$MODE" in
  __codex_auth_probe__)
    echo "E029 train entrypoint authorized."
    ;;
  smoke)
    mapfile -t variants < <(all_compliant_variants)
    RUN_EXTRA_ARGS=(
      num_samples=8
      max_num_iterations=1
      max_sim_steps=4
      save_video=false
      viewer=none
    )
    run_variants "$GPU" "${variants[@]}"
    ;;
  local)
    mapfile -t variants < <(variants_for_queue local)
    run_variants "$GPU" "${variants[@]}"
    ;;
  one|single)
    VARIANT="${3:-}"
    if [ -z "$VARIANT" ]; then
      echo "Usage: $0 one <gpu> <variant>" >&2
      exit 2
    fi
    run_one "$VARIANT" "$GPU"
    ;;
  sanity)
    .venv/bin/python workspace/core4d_collab_retarget/scripts/E029/check_d6_support_load_path.py \
      --subset representative --mode d6-compliant-support --target-mode smooth_final --render-video
    ;;
  *)
    echo "Usage: $0 smoke 0 | local 0 | one 0 <variant> | sanity 0" >&2
    exit 2
    ;;
esac

echo "=== E029 ${MODE} done ==="
