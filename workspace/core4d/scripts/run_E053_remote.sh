#!/bin/bash
# E053: Collision Box Margin Sweep
# 3 margins (0.90, 0.95, 1.00) × 3 cases (box025, bucket010, desk005) = 9 experiments
#
# GPU allocation:
#   GPU0: box025 × 3 margins (sequential), then desk005_m090
#   GPU1: bucket010 × 3 margins (sequential), then desk005_m095, desk005_m100
#
# Each experiment: set margin → run retarget → copy results
# ~30min per experiment, ~2h total with 2 GPUs
set -e

RESULTS=workspace/core4d/results/E053
LOGS_DIR=logs/E053
MARGIN_TOOL=workspace/core4d/scripts/convert/set_collision_margin.py
mkdir -p "$RESULTS" "$LOGS_DIR"

run_experiment() {
    local name=$1 gpu=$2 task=$3 margin=$4
    local case_name="${task}"

    echo "[$(date '+%H:%M:%S')] Setting collision margin=${margin} for ${case_name}"
    uv run "$MARGIN_TOOL" --cases "$case_name" --margin "$margin" \
        > "$LOGS_DIR/${name}_margin.log" 2>&1

    echo "[$(date '+%H:%M:%S')] Starting ${name} on GPU${gpu} (margin=${margin})"
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl uv run examples/run_mjwp.py \
        +override=core4d_e041c task=$task \
        video_output_path="$RESULTS/${name}.mp4" \
        > "$LOGS_DIR/${name}.log" 2>&1

    # Copy trajectory
    local task_dir="example_datasets/processed/core4d/unitree_g1/humanoid_object/${task}/0"
    cp "${task_dir}/trajectory_mjwp_act.npz" "$RESULTS/${name}.npz" 2>/dev/null || true
    echo "[$(date '+%H:%M:%S')] Finished ${name}"
}

# Phase 1: box025 (GPU0) and bucket010 (GPU1) in parallel — no shared scene.xml
(
    run_experiment "E053a_box025_m090" 0 box025_person1 0.90
    run_experiment "E053b_box025_m095" 0 box025_person1 0.95
    run_experiment "E053c_box025_m100" 0 box025_person1 1.00
) &
PID0=$!

(
    run_experiment "E053a_bucket010_m090" 1 bucket010_person1 0.90
    run_experiment "E053b_bucket010_m095" 1 bucket010_person1 0.95
    run_experiment "E053c_bucket010_m100" 1 bucket010_person1 1.00
) &
PID1=$!

echo "[$(date '+%H:%M:%S')] Phase 1 launched (box025+bucket010). PIDs: GPU0=$PID0, GPU1=$PID1"
wait $PID0
echo "[$(date '+%H:%M:%S')] GPU0 (box025) done"
wait $PID1
echo "[$(date '+%H:%M:%S')] GPU1 (bucket010) done"

# Phase 2: desk005 — sequential on single GPU to avoid scene.xml race
# Use both GPUs but margin by margin (set once, run once)
echo "[$(date '+%H:%M:%S')] Phase 2: desk005 × 3 margins (sequential, GPU0)"
run_experiment "E053a_desk005_m090" 0 desk005_person2 0.90
run_experiment "E053b_desk005_m095" 0 desk005_person2 0.95
run_experiment "E053c_desk005_m100" 0 desk005_person2 1.00

echo ""
echo "=== All E053 experiments complete ==="
echo "Results: $RESULTS/"
ls -la "$RESULTS"/*.mp4 2>/dev/null || echo "(no mp4 files found)"

# Restore collision boxes to 1.05x (current default) after sweep
echo ""
echo "Restoring collision boxes to margin=1.05 (default)..."
uv run "$MARGIN_TOOL" --cases box025_person1 bucket010_person1 desk005_person2 --margin 1.05
echo "Done."
