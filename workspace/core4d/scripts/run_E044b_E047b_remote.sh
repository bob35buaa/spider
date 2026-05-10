#!/bin/bash
# E044b + E047b parallel sweep
# GPU0: E044b (wrist_weight=3.0) box025, bucket010, desk005
# GPU1: E047b (SBTO relaxed: momentum=0.5, σ_min=0.03) box025, bucket010, desk005
set -e

RESULTS_044=workspace/core4d/results/E044
RESULTS_047=workspace/core4d/results/E047
LOGS_DIR=logs/E044_E047b
mkdir -p "$RESULTS_044" "$RESULTS_047" "$LOGS_DIR"

run_experiment() {
    local name=$1 gpu=$2 override=$3 task=$4 results_dir=$5
    echo "[$(date '+%H:%M:%S')] Starting $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl uv run examples/run_mjwp.py \
        +override=$override task=$task \
        video_output_path="$results_dir/${name}.mp4" \
        > "$LOGS_DIR/${name}.log" 2>&1
    local task_dir="example_datasets/processed/core4d/unitree_g1/humanoid_object/${task}/0"
    cp "${task_dir}/trajectory_mjwp_act.npz" "$results_dir/${name}.npz" 2>/dev/null || true
    echo "[$(date '+%H:%M:%S')] Finished $name"
}

# GPU 0: E044b (wrist_weight=3.0)
(
    run_experiment "E044b_box025" 0 core4d_e044b box025_person1 "$RESULTS_044"
    run_experiment "E044b_bucket010" 0 core4d_e044b bucket010_person1 "$RESULTS_044"
    run_experiment "E044b_desk005" 0 core4d_e044b desk005_person2 "$RESULTS_044"
) &
PID0=$!

# GPU 1: E047b (SBTO relaxed)
(
    run_experiment "E047b_box025" 1 core4d_e047b box025_person1 "$RESULTS_047"
    run_experiment "E047b_bucket010" 1 core4d_e047b bucket010_person1 "$RESULTS_047"
    run_experiment "E047b_desk005" 1 core4d_e047b desk005_person2 "$RESULTS_047"
) &
PID1=$!

echo "=== Launched: GPU0(E044b) PID=$PID0 | GPU1(E047b) PID=$PID1 ==="
wait $PID0; echo "GPU0 (E044b) done"
wait $PID1; echo "GPU1 (E047b) done"
echo "=== All E044b+E047b experiments complete ==="
