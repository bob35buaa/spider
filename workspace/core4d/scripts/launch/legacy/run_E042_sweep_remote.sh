#!/bin/bash
# E042: Wrist freeze sweep — parallel execution on 2-GPU remote machine
# E042a: E040 + wrist freeze | E042b: E041c + wrist freeze | E042c: no contact + wrist freeze
# All on box025 + bucket010, 6 experiments total
set -e

RESULTS_DIR=workspace/core4d/results/E042
LOGS_DIR=logs/E042
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

run_experiment() {
    local name=$1 gpu=$2 override=$3 task=$4
    local task_dir
    if [ "$task" = "box025_person1" ]; then
        task_dir="example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/0"
    elif [ "$task" = "bucket010_person1" ]; then
        task_dir="example_datasets/processed/core4d/unitree_g1/humanoid_object/bucket010_person1/0"
    elif [ "$task" = "desk005_person2" ]; then
        task_dir="example_datasets/processed/core4d/unitree_g1/humanoid_object/desk005_person2/0"
    fi
    echo "[$(date '+%H:%M:%S')] Starting $name on GPU $gpu ($override, $task)"
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl uv run examples/run_mjwp.py \
        +override=$override task=$task \
        video_output_path="$RESULTS_DIR/${name}.mp4" \
        > "$LOGS_DIR/${name}.log" 2>&1
    cp "$task_dir/trajectory_mjwp_act.npz" "$RESULTS_DIR/${name}.npz"
    echo "[$(date '+%H:%M:%S')] Finished $name"
}

# GPU 0: E042a box025, E042a bucket010, E042c box025 (3 sequential)
(
    run_experiment "E042a_box025"    0 core4d_e042a box025_person1
    run_experiment "E042a_bucket010" 0 core4d_e042a bucket010_person1
    run_experiment "E042c_box025"    0 core4d_e042c box025_person1
) &
PID0=$!

# GPU 1: E042b box025, E042b bucket010, E042c bucket010 (3 sequential)
(
    run_experiment "E042b_box025"    1 core4d_e042b box025_person1
    run_experiment "E042b_bucket010" 1 core4d_e042b bucket010_person1
    run_experiment "E042c_bucket010" 1 core4d_e042c bucket010_person1
) &
PID1=$!

echo "=== Launched: GPU0 PID=$PID0 | GPU1 PID=$PID1 ==="
echo "=== GPU0: E042a_box025 → E042a_bucket010 → E042c_box025 ==="
echo "=== GPU1: E042b_box025 → E042b_bucket010 → E042c_bucket010 ==="
wait $PID0; echo "GPU0 done"
wait $PID1; echo "GPU1 done"
echo "=== All complete ==="
