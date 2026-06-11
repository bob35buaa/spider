#!/usr/bin/env bash
# E155: hand_support_rew 放手平滑过渡实验
# 4 方法 × 3 case = 12 runs, 本机双卡并行 (GPU0: 6 runs, GPU1: 6 runs)
#
# GPU0: box021 × 4 methods + box004 × 2 methods (ramp5, ramp10)
# GPU1: box004 × 2 methods (decay, neutral) + box023 × 4 methods
#
# 预计: ~36 min per GPU (6 runs × ~6 min each)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${STAGE:-full}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RESULTS="workspace/core4d/results/E155/cem/${STAGE}"
LOGS="logs/E155/cem/${STAGE}"
mkdir -p "$RESULTS" "$LOGS"

# ── Methods ─────────────────────────────────────────────────────────────────
# Each method has a tag and extra CLI overrides
declare -A METHOD_TAG METHOD_EXTRA
METHOD_TAG[ramp5]="ramp5"
METHOD_EXTRA[ramp5]="+contact_hdmi_mask_ramp_frames=5"

METHOD_TAG[ramp10]="ramp10"
METHOD_EXTRA[ramp10]="+contact_hdmi_mask_ramp_frames=10"

METHOD_TAG[decay]="decay"
METHOD_EXTRA[decay]="+hand_support_decay_frac=0.15"

METHOD_TAG[neutral]="neutral"
METHOD_EXTRA[neutral]="+hand_support_neutral_baseline=1.0"

METHODS=(ramp5 ramp10 decay neutral)

# ── Cases ───────────────────────────────────────────────────────────────────
declare -A CASE_TASK CASE_OVERRIDE
CASE_TASK[box021_029_p2]="d003_box021_20231018_029_p2_e107_clean"
CASE_OVERRIDE[box021_029_p2]="core4d_E152_box021_029_p2_gateA_b1"

CASE_TASK[box004_083_p2]="e091_box004_20231003_2_083_p2_e092_dyn"
CASE_OVERRIDE[box004_083_p2]="core4d_E152_box004_083_p2_gateA_b1"

CASE_TASK[box023_person2]="box023_person2_legobj"
CASE_OVERRIDE[box023_person2]="core4d_E152_box023_person2_gateA_b1"

# ── Common gate params ──────────────────────────────────────────────────────
GATE_ARGS="cem_hand_gate_min_sdf_m=-0.010 cem_hand_gate_max_violation_pct=0.10 +cem_hand_gate_hard_floor_m=-0.020"

# ── Run function ────────────────────────────────────────────────────────────
run_one() {
    local case_id=$1 method=$2 gpu=$3
    local task="${CASE_TASK[$case_id]}"
    local override="${CASE_OVERRIDE[$case_id]}"
    local tag="${METHOD_TAG[$method]}"
    local extra="${METHOD_EXTRA[$method]}"
    local variant="E155_${case_id}_${tag}"
    local out_dir="$RESULTS/${variant}_outdir_${STAGE}"

    # Skip if already done
    if [ -f "$RESULTS/${variant}.npz" ] && [ -f "$RESULTS/${variant}_${STAGE}.mp4" ]; then
        echo "[$(date '+%H:%M:%S')] === ${variant} exists; skip ==="
        return 0
    fi

    mkdir -p "$out_dir"
    echo "[$(date '+%H:%M:%S')] === START ${variant} GPU=${gpu} ==="

    local smoke_args=""
    if [ "$STAGE" = "smoke" ]; then
        smoke_args="num_samples=64 max_num_iterations=4"
    fi

    CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
        "$PYTHON_BIN" -u examples/run_mjwp.py \
        +override="$override" task="$task" \
        +use_torch_compile=false video_camera=auto \
        $GATE_ARGS \
        +contact_hdmi_mask_carry_union=true \
        $extra \
        $smoke_args \
        output_dir="$out_dir" \
        video_output_path="$RESULTS/${variant}_${STAGE}.mp4" \
        > "$LOGS/${variant}.log" 2>&1

    cp "$out_dir/trajectory_mjwp_act.npz" "$RESULTS/${variant}.npz"
    echo "[$(date '+%H:%M:%S')] === DONE ${variant} ==="
}

# ── GPU0: box021 × 4 + box004 × 2 ──────────────────────────────────────────
gpu0_work() {
    for method in "${METHODS[@]}"; do
        run_one box021_029_p2 "$method" 0
    done
    run_one box004_083_p2 ramp5 0
    run_one box004_083_p2 ramp10 0
}

# ── GPU1: box004 × 2 + box023 × 4 ──────────────────────────────────────────
gpu1_work() {
    run_one box004_083_p2 decay 1
    run_one box004_083_p2 neutral 1
    for method in "${METHODS[@]}"; do
        run_one box023_person2 "$method" 1
    done
}

# ── Launch ──────────────────────────────────────────────────────────────────
echo "=== E155 ${STAGE} — 12 runs on 2 GPUs ==="
echo "=== GPU0: box021×4 + box004×2 | GPU1: box004×2 + box023×4 ==="
echo "=== Started at $(date '+%Y-%m-%d %H:%M:%S') ==="

gpu0_work &
PID0=$!

gpu1_work &
PID1=$!

echo "GPU0 PID=$PID0 | GPU1 PID=$PID1"
wait $PID0; echo "[$(date '+%H:%M:%S')] GPU0 complete"
wait $PID1; echo "[$(date '+%H:%M:%S')] GPU1 complete"

echo "=== E155 ${STAGE} ALL DONE at $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "Results: $RESULTS/"
echo "Logs: $LOGS/"
ls "$RESULTS"/*.npz 2>/dev/null | wc -l | xargs -I{} echo "NPZ count: {}/12"
