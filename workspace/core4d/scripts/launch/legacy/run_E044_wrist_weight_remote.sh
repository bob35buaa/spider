#!/bin/bash
# E044: Wrist weight boost sweep
# GPU0: E044a (wrist_weight=2.0) box025, bucket010, desk005
# GPU1: E044b (wrist_weight=3.0) box025, bucket010, desk005
set -e

RESULTS_DIR=workspace/core4d/results/E044
LOGS_DIR=logs/E044
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

run_experiment() {
    local name=$1 gpu=$2 override=$3 task=$4
    echo "[$(date '+%H:%M:%S')] Starting $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl uv run examples/run_mjwp.py \
        +override=$override task=$task \
        video_output_path="$RESULTS_DIR/${name}.mp4" \
        > "$LOGS_DIR/${name}.log" 2>&1
    local task_dir="example_datasets/processed/core4d/unitree_g1/humanoid_object/${task}/0"
    cp "${task_dir}/trajectory_mjwp_act.npz" "$RESULTS_DIR/${name}.npz" 2>/dev/null || true
    echo "[$(date '+%H:%M:%S')] Finished $name"
}

# GPU 0: E044a (wrist_weight=2.0)
(
    run_experiment "E044a_box025" 0 core4d_e044a box025_person1
    run_experiment "E044a_bucket010" 0 core4d_e044a bucket010_person1
    run_experiment "E044a_desk005" 0 core4d_e044a desk005_person2
) &
PID0=$!

# GPU 1: E044b (wrist_weight=3.0)
(
    run_experiment "E044b_box025" 1 core4d_e044b box025_person1
    run_experiment "E044b_bucket010" 1 core4d_e044b bucket010_person1
    run_experiment "E044b_desk005" 1 core4d_e044b desk005_person2
) &
PID1=$!

echo "=== Launched: GPU0 PID=$PID0 | GPU1 PID=$PID1 ==="
wait $PID0; echo "GPU0 done"
wait $PID1; echo "GPU1 done"
echo "=== All E044 experiments complete ==="
