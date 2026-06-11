#!/usr/bin/env bash
# E153 shared sweep helper: run one case's gateA_b1 ablation grid.
#   min_sdf_m   ∈ {-0.005, -0.010, -0.015}
#   max_viol    ∈ {0.05, 0.10}
#   hard_floor  = -0.020 (fixed; activates max_violation_pct after Stage0 decouple)
#
# REQUIRES Stage0 (plan 161 §2): config field `cem_hand_gate_hard_floor_m` must exist,
# else Hydra rejects the CLI override. Implement Stage0 before running.
#
# Sourced by run_case_{box021,box004,box023}.sh, which set CASE/TASK/OVERRIDE then call:
#   e153_run_case <gpu>
set -euo pipefail

E153_HARD_FLOOR_M="${E153_HARD_FLOOR_M:--0.020}"
E153_MIN_SDF_LIST="${E153_MIN_SDF_LIST:--0.005 -0.010 -0.015}"
E153_MAX_VIOL_LIST="${E153_MAX_VIOL_LIST:-0.05 0.10}"
STAGE="${STAGE:-full}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RESULTS="${RESULTS:-workspace/core4d/results/E153/gate_threshold_sweep/cem/${STAGE}}"
LOGS="${LOGS:-logs/E153/cem/${STAGE}}"

# sdf -0.005 -> sdf005 ; viol 0.05 -> v05
_sdf_tag() { awk -v x="$1" 'BEGIN{printf "sdf%03d", int((-x)*1000 + 0.5)}'; }
_viol_tag() { awk -v x="$1" 'BEGIN{printf "v%02d", int(x*100 + 0.5)}'; }

e153_run_one() {
  local case_id=$1 task=$2 override=$3 gpu=$4 min_sdf=$5 max_viol=$6
  local variant="E153_${case_id}_gateA_b1_$(_sdf_tag "$min_sdf")_$(_viol_tag "$max_viol")"
  local out_dir="$RESULTS/${variant}_outdir_${STAGE}"
  if [ -f "$out_dir/trajectory_mjwp_act.npz" ] && [ -f "$RESULTS/${variant}.npz" ] \
     && [ -f "$RESULTS/${variant}_${STAGE}.mp4" ]; then
    echo "[$(date '+%H:%M:%S')] === ${variant} outputs exist; skip ==="
    return 0
  fi
  mkdir -p "$out_dir" "$RESULTS/keyframes" "$LOGS"
  echo "[$(date '+%H:%M:%S')] === E153 ${STAGE} ${variant} GPU=${gpu} min_sdf=${min_sdf} max_viol=${max_viol} hard_floor=${E153_HARD_FLOOR_M} ==="
  local extra=()
  if [ "$STAGE" = "smoke" ]; then
    extra=(num_samples="${SMOKE_NUM_SAMPLES:-64}" max_num_iterations="${SMOKE_MAX_NUM_ITERATIONS:-4}")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u examples/run_mjwp.py \
    +override="$override" task="$task" +use_torch_compile=false video_camera=auto \
    cem_hand_gate_min_sdf_m="$min_sdf" \
    cem_hand_gate_max_violation_pct="$max_viol" \
    +cem_hand_gate_hard_floor_m="$E153_HARD_FLOOR_M" \
    "${extra[@]}" \
    output_dir="$out_dir" video_output_path="$RESULTS/${variant}_${STAGE}.mp4" \
    > "$LOGS/${variant}_${STAGE}.log" 2>&1
  cp "$out_dir/trajectory_mjwp_act.npz" "$RESULTS/${variant}.npz"
  if command -v ffmpeg >/dev/null 2>&1 && [ "${SKIP_KEYFRAMES:-0}" != "1" ]; then
    mkdir -p "$RESULTS/keyframes/$variant"
    for f in 25 55 85; do
      timeout 30 ffmpeg -nostdin -y -loglevel error -i "$RESULTS/${variant}_${STAGE}.mp4" \
        -vf "select=eq(n\\,$f)" -frames:v 1 -vsync 0 "$RESULTS/keyframes/$variant/f${f}.jpg" || true
    done
  fi
  echo "[$(date '+%H:%M:%S')] === ${variant} done ==="
}

# e153_run_case <gpu> : run the full 3x2 grid for the case defined by caller (CASE/TASK/OVERRIDE).
e153_run_case() {
  local gpu=$1
  : "${CASE:?set CASE}" "${TASK:?set TASK}" "${OVERRIDE:?set OVERRIDE}"
  cd "$(git rev-parse --show-toplevel)"
  # Per-experiment scene snapshot (experiment.md §7) before training.
  if [ -f workspace/core4d/scripts/convert/snapshot_scenes.sh ] && [ "${E153_SKIP_SNAPSHOT:-0}" != "1" ]; then
    bash workspace/core4d/scripts/convert/snapshot_scenes.sh E153 "$TASK" || true
  fi
  echo "=== E153 sweep case=${CASE} task=${TASK} override=${OVERRIDE} gpu=${gpu} stage=${STAGE} ==="
  for min_sdf in $E153_MIN_SDF_LIST; do
    for max_viol in $E153_MAX_VIOL_LIST; do
      e153_run_one "$CASE" "$TASK" "$OVERRIDE" "$gpu" "$min_sdf" "$max_viol"
    done
  done
  echo "=== E153 sweep case=${CASE} complete ==="
}
