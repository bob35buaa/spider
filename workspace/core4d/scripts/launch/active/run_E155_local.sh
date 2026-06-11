#!/usr/bin/env bash
# E155 local runner for selected release-smoothing variants.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
LOCAL_GPU="${LOCAL_GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RESULTS="workspace/core4d/results/E155/cem/${STAGE}"
LOGS="logs/E155/cem/${STAGE}"

if [ "$STAGE" != "smoke" ] && [ "$STAGE" != "full" ]; then
  echo "Invalid STAGE=$STAGE (use smoke|full)" >&2
  exit 2
fi

mkdir -p "$RESULTS" "$LOGS"

declare -A METHOD_TAG METHOD_EXTRA
METHOD_TAG[ramp5]="ramp5"
METHOD_EXTRA[ramp5]="+contact_hdmi_mask_ramp_frames=5"
METHOD_TAG[ramp10]="ramp10"
METHOD_EXTRA[ramp10]="+contact_hdmi_mask_ramp_frames=10"
METHOD_TAG[decay]="decay"
METHOD_EXTRA[decay]="+hand_support_decay_frac=0.15"
METHOD_TAG[neutral]="neutral"
METHOD_EXTRA[neutral]="+hand_support_neutral_baseline=1.0"

declare -A CASE_TASK CASE_OVERRIDE
CASE_TASK[box021_029_p2]="d003_box021_20231018_029_p2_e107_clean"
CASE_OVERRIDE[box021_029_p2]="core4d_E152_box021_029_p2_gateA_b1"
CASE_TASK[box004_083_p2]="e091_box004_20231003_2_083_p2_e092_dyn"
CASE_OVERRIDE[box004_083_p2]="core4d_E152_box004_083_p2_gateA_b1"
CASE_TASK[box023_person2]="box023_person2_legobj"
CASE_OVERRIDE[box023_person2]="core4d_E152_box023_person2_gateA_b1"

GATE_ARGS="cem_hand_gate_min_sdf_m=-0.010 cem_hand_gate_max_violation_pct=0.10 +cem_hand_gate_hard_floor_m=-0.020"
DEFAULT_WORK="box004_083_p2:ramp5"
CASE_METHODS="${CASE_METHODS:-$DEFAULT_WORK}"

is_complete() {
  local variant=$1
  [ -f "$RESULTS/${variant}.npz" ] \
    && [ -f "$RESULTS/${variant}_${STAGE}.mp4" ] \
    && [ -f "$RESULTS/${variant}_outdir_${STAGE}/trajectory_mjwp_act.npz" ]
}

run_one() {
  local case_id=$1 method=$2
  local task="${CASE_TASK[$case_id]:-}"
  local override="${CASE_OVERRIDE[$case_id]:-}"
  local tag="${METHOD_TAG[$method]:-}"
  local extra="${METHOD_EXTRA[$method]:-}"

  if [ -z "$task" ] || [ -z "$override" ] || [ -z "$tag" ] || [ -z "$extra" ]; then
    echo "Unknown E155 case/method: ${case_id}:${method}" >&2
    exit 2
  fi

  local variant="E155_${case_id}_${tag}"
  local out_dir="$RESULTS/${variant}_outdir_${STAGE}"

  if is_complete "$variant"; then
    echo "[$(date '+%H:%M:%S')] === ${variant} complete; skip ==="
    return 0
  fi

  local smoke_args=()
  if [ "$STAGE" = "smoke" ]; then
    smoke_args=(num_samples=64 max_num_iterations=4)
  fi

  mkdir -p "$out_dir"
  echo "[$(date '+%H:%M:%S')] === START ${variant} GPU=${LOCAL_GPU} stage=${STAGE} ==="

  CUDA_VISIBLE_DEVICES="$LOCAL_GPU" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -u examples/run_mjwp.py \
      +override="$override" task="$task" \
      +use_torch_compile=false video_camera=auto \
      $GATE_ARGS \
      +contact_hdmi_mask_carry_union=true \
      $extra \
      "${smoke_args[@]}" \
      output_dir="$out_dir" \
      video_output_path="$RESULTS/${variant}_${STAGE}.mp4" \
      > "$LOGS/${variant}.log" 2>&1

  cp "$out_dir/trajectory_mjwp_act.npz" "$RESULTS/${variant}.npz"
  echo "[$(date '+%H:%M:%S')] === DONE ${variant} ==="
}

echo "=== E155 ${STAGE} local selected run on GPU ${LOCAL_GPU} ==="
echo "CASE_METHODS=${CASE_METHODS}"
for spec in $CASE_METHODS; do
  case_id="${spec%%:*}"
  method="${spec##*:}"
  run_one "$case_id" "$method"
done

echo "=== E155 ${STAGE} selected local run complete ==="
