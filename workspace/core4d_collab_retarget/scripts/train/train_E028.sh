#!/usr/bin/env bash
# E028: hard no-penetration / surface feasibility.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-local}"
GPU="${2:-0}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d_collab_retarget/results/E028/manifest.tsv}"
RESULTS="${RESULTS:-workspace/core4d_collab_retarget/results/E028}"
LOGS="${LOGS:-logs/core4d_collab_retarget/E028}"
mkdir -p "$RESULTS/keyframes" "$RESULTS/scene_snapshot" "$RESULTS/online_video" "$LOGS"
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
  if [ "$queue" = "local" ]; then
    awk -F '\t' 'NR > 1 && NF && ($7 == "local" || $7 == "local_after_main") {print $1}' "$VARIANTS_FILE"
  else
    awk -F '\t' -v want_queue="$queue" 'NR > 1 && NF && $7 == want_queue {print $1}' "$VARIANTS_FILE"
  fi
}

override_for_variant() {
  echo "core4d_collab_${1}"
}

snapshot_variants() {
  local -a variants=("$@")
  if [ "${#variants[@]}" -eq 0 ]; then
    return 0
  fi
  local -A seen=()
  local -a tasks=()
  local variant row task
  for variant in "${variants[@]}"; do
    row="$(variant_row "$variant")"
    task="$(field_from_row "$row" 3)"
    if [ -z "${seen[$task]:-}" ]; then
      seen[$task]=1
      tasks+=("$task")
    fi
  done
  echo "[$(date '+%H:%M:%S')] === E028 scene snapshot tasks=${tasks[*]} ==="
  bash workspace/core4d_collab_retarget/scripts/convert/snapshot_scenes.sh E028 "${tasks[@]}"
}

extract_keyframes() {
  local variant=$1
  local video=$2
  [ -f "$video" ] || return 0
  command -v ffmpeg >/dev/null 2>&1 || return 0
  mkdir -p "$RESULTS/keyframes/$variant"
  for f in 32 50 75 90 100 115 130 160 180 204 250; do
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
  local row task override out_dir npz_out variant_log run_pid start_ts
  local timeout_s stall_s now_ts mtime_ts run_status
  row="$(variant_row "$variant")"
  if [ -z "$row" ]; then
    echo "Unknown E028 variant: $variant" >&2
    return 2
  fi
  task="$(field_from_row "$row" 3)"
  override="$(override_for_variant "$variant")"
  out_dir="$RESULTS/${variant}_outdir"
  npz_out="$out_dir/trajectory_mjwp.npz"
  mkdir -p "$out_dir"
  variant_log="$LOGS/${variant}.log"

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
    if [ -f "$npz_out" ]; then
      echo "[$(date '+%H:%M:%S')] WARN ${variant} exited with status ${run_status}, but ${npz_out} exists; preserving saved rollout" | tee -a "$variant_log" >&2
    else
      echo "[$(date '+%H:%M:%S')] ERROR ${variant} exited with status ${run_status}" >&2
      return "$run_status"
    fi
  fi

  if [ ! -f "$npz_out" ]; then
    echo "Expected freejoint output missing: $npz_out" >&2
    find "$out_dir" -maxdepth 1 -type f -print >&2
    return 1
  fi
  cp "$npz_out" "$RESULTS/${variant}.npz"
  extract_keyframes "$variant" "$RESULTS/online_video/${variant}.mp4"
  echo "[$(date '+%H:%M:%S')] === ${variant} done ==="
}

run_variants() {
  local gpu=$1
  shift
  local -a variants=("$@")
  if [ "${#variants[@]}" -eq 0 ]; then
    echo "No E028 variants selected" >&2
    exit 2
  fi
  local status=0
  snapshot_variants "${variants[@]}"
  for variant in "${variants[@]}"; do
    if ! run_one "$variant" "$gpu" "${RUN_EXTRA_ARGS[@]}"; then
      echo "[$(date '+%H:%M:%S')] WARN ${variant} failed; continuing E028 queue" | tee -a "$LOGS/failed_${MODE}.log" >&2
      status=1
    fi
  done
  if [ "${SKIP_EVAL:-0}" != "1" ]; then
    .venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E028.py "${variants[@]}" | tee "$LOGS/eval_${MODE}.log"
  fi
  return "$status"
}

case "$MODE" in
  __codex_auth_probe__)
    echo "E028 train entrypoint authorized."
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
  local|remote_gpu0|remote_gpu1)
    mapfile -t variants < <(variants_for_queue "$MODE")
    run_variants "$GPU" "${variants[@]}"
    ;;
  one|single)
    VARIANT="${3:-}"
    if [ -z "$VARIANT" ]; then
      echo "Usage: $0 one <gpu> <variant>" >&2
      exit 2
    fi
    snapshot_variants "$VARIANT"
    run_one "$VARIANT" "$GPU"
    if [ "${SKIP_EVAL:-0}" != "1" ]; then
      .venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E028.py "$VARIANT" | tee "$LOGS/eval_${VARIANT}.log"
    fi
    ;;
  eval)
    shift || true
    .venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E028.py "$@" | tee "$LOGS/eval_E028.log"
    ;;
  *)
    echo "Usage:"
    echo "  $0 smoke 0"
    echo "  $0 local 0"
    echo "  $0 remote_gpu0 0"
    echo "  $0 remote_gpu1 1"
    echo "  $0 one 0 E028_bucket007_p1_barrier_quad_m02"
    echo "  $0 eval [variant ...]"
    exit 2
    ;;
esac

echo "=== E028 ${MODE} done ==="
