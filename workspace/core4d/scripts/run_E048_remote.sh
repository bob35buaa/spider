#!/bin/bash
# E048: Collision-box-fixed baseline + new cases
# GPU0: box025_person1, bucket010_person1, box001_person1
# GPU1: desk005_person2, box023_person1, box024_person1
set -e

RESULTS=workspace/core4d/results/E048
LOGS_DIR=logs/E048
mkdir -p "$RESULTS" "$LOGS_DIR"

run_experiment() {
    local name=$1 gpu=$2 task=$3
    echo "[$(date '+%H:%M:%S')] Starting $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl uv run examples/run_mjwp.py \
        +override=core4d_e041c task=$task \
        video_output_path="$RESULTS/${name}.mp4" \
        > "$LOGS_DIR/${name}.log" 2>&1
    local task_dir="example_datasets/processed/core4d/unitree_g1/humanoid_object/${task}/0"
    cp "${task_dir}/trajectory_mjwp_act.npz" "$RESULTS/${name}.npz" 2>/dev/null || true
    echo "[$(date '+%H:%M:%S')] Finished $name"
}

# GPU 0: baseline re-runs + box001
(
    run_experiment "E048_box025_baseline" 0 box025_person1
    run_experiment "E048_bucket010_baseline" 0 bucket010_person1
    run_experiment "E048_box001" 0 box001_person1
) &
PID0=$!

# GPU 1: baseline re-run + new cases
(
    run_experiment "E048_desk005_baseline" 1 desk005_person2
    run_experiment "E048_box023" 1 box023_person1
    run_experiment "E048_box024" 1 box024_person1
) &
PID1=$!

echo "=== Launched: GPU0 PID=$PID0 | GPU1 PID=$PID1 ==="
wait $PID0; echo "GPU0 done"
wait $PID1; echo "GPU1 done"
echo "=== All E048 experiments complete ==="
ls -la "$RESULTS"/*.npz 2>/dev/null || echo "No NPZ results"
ls -la "$RESULTS"/*.mp4 2>/dev/null || echo "No videos"
