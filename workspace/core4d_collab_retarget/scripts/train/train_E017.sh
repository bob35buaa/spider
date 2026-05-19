#!/usr/bin/env bash
# E017: anchor audit/selection candidates.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-quick_subset}"
GPU="${2:-0}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d_collab_retarget/results/E017/manifest.tsv}"
RESULTS="${RESULTS:-workspace/core4d_collab_retarget/results/E017}"
LOGS="${LOGS:-logs/core4d_collab_retarget/E017}"
mkdir -p "$RESULTS/keyframes" "$RESULTS/scene_snapshot" "$LOGS"
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

all_variants() {
  awk -F '\t' 'NR > 1 && NF {print $1}' "$VARIANTS_FILE"
}

variants_for_queue() {
  local queue=$1
  awk -F '\t' -v want_queue="$queue" 'NR > 1 && NF && $7 == want_queue {print $1}' "$VARIANTS_FILE"
}

variants_for_subset() {
  awk -F '\t' 'NR > 1 && NF && ($8 == "anchor_debug" || $8 == "manual_seed") {print $1}' "$VARIANTS_FILE"
}

variants_for_queue_subset() {
  local queue=$1
  awk -F '\t' -v want_queue="$queue" 'NR > 1 && NF && $7 == want_queue && ($8 == "anchor_debug" || $8 == "manual_seed") {print $1}' "$VARIANTS_FILE"
}

override_for_variant() {
  echo "core4d_collab_${1}"
}

snapshot_task() {
  local task=$1
  local src="example_datasets/processed/core4d/unitree_g1/humanoid_object/${task}"
  local dst="$RESULTS/scene_snapshot/${task}"
  mkdir -p "$dst/0"
  cp "$src/scene.xml" "$dst/scene.xml"
  cp "$src"/scene_e017_jointB_*.xml "$dst/" 2>/dev/null || true
  cp "$src/task_info.json" "$dst/task_info.json" 2>/dev/null || true
  cp "$src/scene_act_meta.json" "$dst/scene_act_meta.json" 2>/dev/null || true
  cp "$src/freejoint_leg_object_collision_meta.json" "$dst/freejoint_leg_object_collision_meta.json" 2>/dev/null || true
  cp "$src/0/trajectory_kinematic.npz" "$dst/0/trajectory_kinematic.npz"
}

snapshot_variants() {
  local -a variants=("$@")
  local variant row task
  for variant in "${variants[@]}"; do
    row="$(variant_row "$variant")"
    task="$(field_from_row "$row" 3)"
    snapshot_task "$task"
  done
}

run_one() {
  local variant=$1
  local gpu=$2
  shift 2
  local extra_args=("$@")
  local row task override out_dir variant_log npz_out run_pid start_ts timeout_s stall_s now_ts mtime_ts run_status
  row="$(variant_row "$variant")"
  if [ -z "$row" ]; then
    echo "Unknown E017 variant: $variant" >&2
    return 2
  fi
  task="$(field_from_row "$row" 3)"
  override="$(override_for_variant "$variant")"
  out_dir="$RESULTS/${variant}_outdir"
  variant_log="$LOGS/${variant}.log"
  mkdir -p "$out_dir"

  echo "[$(date '+%H:%M:%S')] === ${variant} task=${task} override=${override} GPU=${gpu} ==="
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
    +override="$override" \
    task="$task" \
    +use_torch_compile=false \
    output_dir="$out_dir" \
    video_output_path="$RESULTS/${variant}.mp4" \
    "${extra_args[@]}" \
    > "$variant_log" 2>&1 &
  run_pid=$!
  start_ts="$(date +%s)"
  timeout_s="${RUN_TIMEOUT_SECONDS:-0}"
  stall_s="${RUN_STALL_TIMEOUT_SECONDS:-0}"
  while kill -0 "$run_pid" 2>/dev/null; do
    sleep 30
    now_ts="$(date +%s)"
    if [ "$timeout_s" -gt 0 ] && [ $((now_ts - start_ts)) -gt "$timeout_s" ]; then
      echo "[$(date '+%H:%M:%S')] ERROR ${variant} timed out after ${timeout_s}s" | tee -a "$variant_log" >&2
      kill "$run_pid" 2>/dev/null || true
      sleep 10
      kill -9 "$run_pid" 2>/dev/null || true
      wait "$run_pid" 2>/dev/null || true
      return 124
    fi
    if [ "$stall_s" -gt 0 ] && [ -f "$variant_log" ]; then
      mtime_ts="$(stat -c %Y "$variant_log")"
      if [ $((now_ts - mtime_ts)) -gt "$stall_s" ]; then
        echo "[$(date '+%H:%M:%S')] ERROR ${variant} stalled: no log update for ${stall_s}s" | tee -a "$variant_log" >&2
        kill "$run_pid" 2>/dev/null || true
        sleep 10
        kill -9 "$run_pid" 2>/dev/null || true
        wait "$run_pid" 2>/dev/null || true
        return 124
      fi
    fi
  done
  set +e
  wait "$run_pid"
  run_status=$?
  set -e
  if [ "$run_status" -ne 0 ]; then
    echo "[$(date '+%H:%M:%S')] ERROR ${variant} exited with status ${run_status}" >&2
    return "$run_status"
  fi
  npz_out="$out_dir/trajectory_mjwp.npz"
  if [ ! -f "$npz_out" ]; then
    echo "Expected output missing: $npz_out" >&2
    return 1
  fi
  cp "$npz_out" "$RESULTS/${variant}.npz"
  echo "[$(date '+%H:%M:%S')] === ${variant} done ==="
}

run_variants() {
  local gpu=$1
  shift
  local -a variants=("$@")
  if [ "${#variants[@]}" -eq 0 ]; then
    echo "No E017 variants selected" >&2
    exit 2
  fi
  local status=0
  snapshot_variants "${variants[@]}"
  for variant in "${variants[@]}"; do
    if ! run_one "$variant" "$gpu" "${RUN_EXTRA_ARGS[@]}"; then
      echo "[$(date '+%H:%M:%S')] WARN ${variant} failed; continuing E017 queue" | tee -a "$LOGS/failed_${MODE}.log" >&2
      status=1
    fi
  done
  if [ "${SKIP_EVAL:-0}" != "1" ]; then
    .venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E017.py "${variants[@]}" | tee "$LOGS/eval_${MODE}.log"
  fi
  return "$status"
}

case "$MODE" in
  __codex_auth_probe__)
    echo "E017 train entrypoint authorized."
    ;;
  smoke)
    mapfile -t variants < <(all_variants)
    RUN_EXTRA_ARGS=(
      num_samples=8
      max_num_iterations=1
      max_sim_steps=4
      save_video=false
      viewer=none
    )
    run_variants "$GPU" "${variants[@]}"
    ;;
  quick)
    mapfile -t variants < <(all_variants)
    RUN_EXTRA_ARGS=(
      num_samples="${E017_QUICK_NUM_SAMPLES:-128}"
      max_num_iterations="${E017_QUICK_MAX_ITERS:-4}"
      save_video=false
      viewer=none
    )
    run_variants "$GPU" "${variants[@]}"
    ;;
  quick_subset|subset_quick)
    mapfile -t variants < <(variants_for_subset)
    RUN_EXTRA_ARGS=(
      num_samples="${E017_QUICK_NUM_SAMPLES:-128}"
      max_num_iterations="${E017_QUICK_MAX_ITERS:-4}"
      save_video=false
      viewer=none
    )
    run_variants "$GPU" "${variants[@]}"
    ;;
  full|local|local_wave)
    mapfile -t variants < <(variants_for_queue local)
    run_variants "$GPU" "${variants[@]}"
    ;;
  local_quick)
    mapfile -t variants < <(variants_for_queue local)
    RUN_EXTRA_ARGS=(
      num_samples="${E017_QUICK_NUM_SAMPLES:-128}"
      max_num_iterations="${E017_QUICK_MAX_ITERS:-4}"
      save_video=false
      viewer=none
    )
    run_variants "$GPU" "${variants[@]}"
    ;;
  remote_gpu0|remote_gpu1)
    mapfile -t variants < <(variants_for_queue "$MODE")
    RUN_EXTRA_ARGS=(
      num_samples="${E017_QUICK_NUM_SAMPLES:-128}"
      max_num_iterations="${E017_QUICK_MAX_ITERS:-4}"
      save_video=false
      viewer=none
    )
    run_variants "$GPU" "${variants[@]}"
    ;;
  remote_subset_gpu0)
    mapfile -t variants < <(variants_for_queue_subset remote_gpu0)
    RUN_EXTRA_ARGS=(
      num_samples="${E017_QUICK_NUM_SAMPLES:-128}"
      max_num_iterations="${E017_QUICK_MAX_ITERS:-4}"
      save_video=false
      viewer=none
    )
    run_variants "$GPU" "${variants[@]}"
    ;;
  remote_subset_gpu1)
    mapfile -t variants < <(variants_for_queue_subset remote_gpu1)
    RUN_EXTRA_ARGS=(
      num_samples="${E017_QUICK_NUM_SAMPLES:-128}"
      max_num_iterations="${E017_QUICK_MAX_ITERS:-4}"
      save_video=false
      viewer=none
    )
    run_variants "$GPU" "${variants[@]}"
    ;;
  one|single)
    VARIANT="${3:-}"
    if [ -z "$VARIANT" ]; then
      echo "Usage: $0 one <gpu> <variant>" >&2
      exit 2
    fi
    run_one "$VARIANT" "$GPU"
    .venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E017.py "$VARIANT" | tee "$LOGS/eval_${VARIANT}.log"
    ;;
  eval)
    shift || true
    .venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E017.py "$@" | tee "$LOGS/eval_E017.log"
    ;;
  *)
    echo "Unknown E017 mode: $MODE" >&2
    exit 2
    ;;
esac
