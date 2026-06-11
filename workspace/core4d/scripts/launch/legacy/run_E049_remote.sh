#!/bin/bash
# E049: E041c + 3 HDMI optimizations (PD + wrist damping + wrist noise zero)
# GPU0: box023, box025
# GPU1: bucket010, desk005
set -e
cd /home/xiayb/pHRI_workspace/spider
source .venv/bin/activate

RESULTS=workspace/core4d/results/E049
LOGS_DIR=logs/E049
mkdir -p "$RESULTS" "$LOGS_DIR"

run_experiment() {
    local name=$1 gpu=$2 task=$3
    echo "[$(date '+%H:%M:%S')] Starting $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl python examples/run_mjwp.py \
        +override=core4d_e049 task=$task \
        video_output_path="$RESULTS/${name}.mp4" \
        > "$LOGS_DIR/${name}.log" 2>&1
    local task_dir="example_datasets/processed/core4d/unitree_g1/humanoid_object/${task}/0"
    cp "${task_dir}/trajectory_mjwp_act.npz" "$RESULTS/${name}.npz" 2>/dev/null || true
    echo "[$(date '+%H:%M:%S')] Finished $name"
}

(
    run_experiment "E049_box023" 0 box023_person1
    run_experiment "E049_box025" 0 box025_person1
) &
PID0=$!

(
    run_experiment "E049_bucket010" 1 bucket010_person1
    run_experiment "E049_desk005" 1 desk005_person2
) &
PID1=$!

echo "=== Launched: GPU0 PID=$PID0 | GPU1 PID=$PID1 ==="
wait $PID0; echo "GPU0 done"
wait $PID1; echo "GPU1 done"
echo "=== All E049 experiments complete ==="
ls -la "$RESULTS"/*.npz 2>/dev/null || echo "No NPZ results"
