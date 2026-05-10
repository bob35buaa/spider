#!/bin/bash
# E045: Local-frame sigma sweep — tighter tracking
# GPU0: E045a (sigma=0.3) box025, bucket010, desk005
# GPU1: E045b (sigma=0.15) box025, bucket010, desk005
set -e

RESULTS_DIR=workspace/core4d/results/E045
LOGS_DIR=logs/E045
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

run_experiment() {
    local name=$1 gpu=$2 override=$3 task=$4
    echo "[$(date '+%H:%M:%S')] Starting $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl uv run examples/run_mjwp.py \
        +override=$override task=$task \
        video_output_path="$RESULTS_DIR/${name}.mp4" \
        > "$LOGS_DIR/${name}.log" 2>&1
    # Copy trajectory result
    local task_dir="example_datasets/processed/core4d/unitree_g1/humanoid_object/${task}/0"
    cp "${task_dir}/trajectory_mjwp_act.npz" "$RESULTS_DIR/${name}.npz" 2>/dev/null || true
    echo "[$(date '+%H:%M:%S')] Finished $name"
}

# GPU 0: E045a (sigma=0.3) — 3 cases sequential
(
    run_experiment "E045a_box025" 0 core4d_e045a box025_person1
    run_experiment "E045a_bucket010" 0 core4d_e045a bucket010_person1
    run_experiment "E045a_desk005" 0 core4d_e045a desk005_person2
) &
PID0=$!

# GPU 1: E045b (sigma=0.15) — 3 cases sequential
(
    run_experiment "E045b_box025" 1 core4d_e045b box025_person1
    run_experiment "E045b_bucket010" 1 core4d_e045b bucket010_person1
    run_experiment "E045b_desk005" 1 core4d_e045b desk005_person2
) &
PID1=$!

echo "=== Launched: GPU0 PID=$PID0 | GPU1 PID=$PID1 ==="
wait $PID0; echo "GPU0 done"
wait $PID1; echo "GPU1 done"
echo "=== All E045 experiments complete ==="
