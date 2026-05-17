#!/usr/bin/env bash
# E003: true-freejoint object mass/friction feasibility sweep.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-local}"
GPU="${2:-0}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d_collab_retarget/scripts/E003/variants.tsv}"
RESULTS="${RESULTS:-workspace/core4d_collab_retarget/results/E003}"
LOGS="${LOGS:-logs/core4d_collab_retarget/E003}"
mkdir -p "$RESULTS/keyframes" "$RESULTS/scene_snapshot" "$LOGS"

variant_row() {
  local variant=$1
  awk -F '\t' -v v="$variant" 'NF && $1 !~ /^#/ && $1 == v {print; exit}' "$VARIANTS_FILE"
}

field_from_row() {
  local row=$1
  local idx=$2
  awk -F '\t' -v row="$row" -v idx="$idx" 'BEGIN {split(row, a, FS); print a[idx]}'
}

all_variants() {
  awk -F '\t' 'NF && $1 !~ /^#/ {print $1}' "$VARIANTS_FILE"
}

variants_for_group() {
  local group=$1
  awk -F '\t' -v want_group="$group" 'NF && $1 !~ /^#/ && $7 == want_group {print $1}' "$VARIANTS_FILE"
}

override_for_variant() {
  local variant=$1
  echo "core4d_collab_${variant}"
}

snapshot_task() {
  local task=$1
  local src="example_datasets/processed/core4d/unitree_g1/humanoid_object/${task}"
  local dst="$RESULTS/scene_snapshot/${task}"
  mkdir -p "$dst/0"
  cp "$src/scene.xml" "$dst/scene.xml"
  cp "$src/task_info.json" "$dst/task_info.json"
  cp "$src/freejoint_leg_object_collision_meta.json" "$dst/freejoint_leg_object_collision_meta.json"
  cp "$src/physics_sweep_meta.json" "$dst/physics_sweep_meta.json"
  cp "$src/0/trajectory_kinematic.npz" "$dst/0/trajectory_kinematic.npz"
}

snapshot_variants() {
  local -a variants=("$@")
  if [ "${#variants[@]}" -gt 0 ]; then
    echo "[$(date '+%H:%M:%S')] === E003 scene snapshot variants=${variants[*]} ==="
    local variant row task
    for variant in "${variants[@]}"; do
      row="$(variant_row "$variant")"
      task="$(field_from_row "$row" 3)"
      snapshot_task "$task"
    done
  fi
}

extract_keyframes() {
  local variant=$1
  local video=$2
  [ -f "$video" ] || return 0
  command -v ffmpeg >/dev/null 2>&1 || return 0
  mkdir -p "$RESULTS/keyframes/$variant"
  for f in 50 75 100 115 120 125 145 160 180 204; do
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
  local row task override out_dir npz_out
  row="$(variant_row "$variant")"
  if [ -z "$row" ]; then
    echo "Unknown E003 variant: $variant" >&2
    return 2
  fi
  task="$(field_from_row "$row" 3)"
  override="$(override_for_variant "$variant")"
  out_dir="$RESULTS/${variant}_outdir"
  mkdir -p "$out_dir"

  echo "[$(date '+%H:%M:%S')] === ${variant} task=${task} override=${override} GPU=${gpu} ==="
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
    +override="$override" \
    task="$task" \
    +use_torch_compile=false \
    output_dir="$out_dir" \
    video_output_path="$RESULTS/${variant}.mp4" \
    "${extra_args[@]}" \
    > "$LOGS/${variant}.log" 2>&1

  npz_out="$out_dir/trajectory_mjwp.npz"
  if [ ! -f "$npz_out" ]; then
    echo "Expected freejoint output missing: $npz_out" >&2
    echo "Available output files:" >&2
    find "$out_dir" -maxdepth 1 -type f -print >&2
    return 1
  fi
  cp "$npz_out" "$RESULTS/${variant}.npz"
  extract_keyframes "$variant" "$RESULTS/${variant}.mp4"
  echo "[$(date '+%H:%M:%S')] === ${variant} done ==="
}

run_variants() {
  local gpu=$1
  shift
  local -a variants=("$@")
  if [ "${#variants[@]}" -eq 0 ]; then
    echo "No E003 variants selected" >&2
    exit 2
  fi
  snapshot_variants "${variants[@]}"
  for variant in "${variants[@]}"; do
    run_one "$variant" "$gpu"
  done
  if [ "${SKIP_EVAL:-0}" != "1" ]; then
    .venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E003.py "${variants[@]}" | tee "$LOGS/eval_${MODE}.log"
  fi
}

case "$MODE" in
  smoke)
    mapfile -t variants < <(all_variants)
    snapshot_variants "${variants[@]}"
    for variant in "${variants[@]}"; do
      run_one "$variant" "$GPU" \
        num_samples=8 \
        max_num_iterations=1 \
        max_sim_steps=4 \
        save_video=false \
        viewer=none
    done
    ;;
  single)
    VARIANT="${3:-}"
    if [ -z "$VARIANT" ]; then
      echo "Usage: $0 single <gpu> <variant>" >&2
      exit 2
    fi
    snapshot_task "$(field_from_row "$(variant_row "$VARIANT")" 3)"
    run_one "$VARIANT" "$GPU"
    .venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E003.py "$VARIANT" | tee "$LOGS/eval_${VARIANT}.log"
    ;;
  local)
    mapfile -t variants < <(all_variants)
    run_variants "$GPU" "${variants[@]}"
    ;;
  gpu0|gpu1)
    mapfile -t variants < <(variants_for_group "$MODE")
    run_variants "$GPU" "${variants[@]}"
    ;;
  eval)
    shift || true
    .venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E003.py "$@" | tee "$LOGS/eval_E003.log"
    ;;
  *)
    echo "Usage:"
    echo "  $0 smoke 0"
    echo "  $0 local 0"
    echo "  $0 gpu0 0"
    echo "  $0 gpu1 1"
    echo "  $0 single 0 E003_box025_p2_m1"
    echo "  $0 eval [variant ...]"
    exit 2
    ;;
esac

echo "=== E003 ${MODE} done ==="
