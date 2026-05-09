#!/bin/bash
# E041 variants: parallel execution on 2-GPU remote machine
# GPU0: E041b (gain=3) + E041c (additive) on box025
# GPU1: E041d (near_field) on box025 + E041c on bucket010
set -e

RESULTS_DIR=workspace/core4d/results/E041_sweep
LOGS_DIR=logs/E041_sweep
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

run_experiment() {
    local name=$1 gpu=$2 override=$3 task=$4
    local task_dir
    if [ "$task" = "box025_person1" ]; then
        task_dir="example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/0"
    elif [ "$task" = "bucket010_person1" ]; then
        task_dir="example_datasets/processed/core4d/unitree_g1/humanoid_object/bucket010_person1/0"
    fi
    echo "[$(date '+%H:%M:%S')] Starting $name on GPU $gpu ($override, $task)"
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl uv run examples/run_mjwp.py \
        +override=$override task=$task \
        video_output_path="$RESULTS_DIR/${name}.mp4" \
        > "$LOGS_DIR/${name}.log" 2>&1
    cp "$task_dir/trajectory_mjwp_act.npz" "$RESULTS_DIR/${name}.npz"
    echo "[$(date '+%H:%M:%S')] Finished $name"
}

# GPU 0: E041b (gain=3) box025, then E041c (additive) box025
(
    run_experiment "E041b_box025" 0 core4d_e041b box025_person1
    run_experiment "E041c_box025" 0 core4d_e041c box025_person1
) &
PID0=$!

# GPU 1: E041d (near_field) box025, then E041c (additive) bucket010
(
    run_experiment "E041d_box025" 1 core4d_e041d box025_person1
    run_experiment "E041c_bucket010" 1 core4d_e041c bucket010_person1
) &
PID1=$!

echo "=== Launched: GPU0 PID=$PID0 | GPU1 PID=$PID1 ==="
echo "=== GPU0: E041b_box025 → E041c_box025 ==="
echo "=== GPU1: E041d_box025 → E041c_bucket010 ==="
wait $PID0; echo "GPU0 done"
wait $PID1; echo "GPU1 done"
echo "=== All complete ==="
