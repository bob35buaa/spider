#!/bin/bash
# E037c: Run S3 best config (gain=3.5, sigma=0.15) on all cases
# + S3b variant (gain=3.5, sigma=0.15, baseline=1.0) for stability comparison
#
# GPU0: S3 on desk005 + bucket010 (sequential)
# GPU1: S3b (baseline=1.0) on desk005 + box025 + bucket010 (sequential)
#
# Usage: bash workspace/core4d/scripts/run_e037c_remote.sh

set -e

OVERRIDE=core4d_e037
RESULTS_DIR=workspace/core4d/results/E037c
LOGS_DIR=logs/E037c

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

run_experiment() {
    local name=$1
    local gpu=$2
    local task=$3
    local gain=$4
    local sigma=$5
    local baseline=$6

    local output_dir="example_datasets/processed/core4d/unitree_g1/humanoid_object/${task}/0"
    echo "[$(date '+%H:%M:%S')] Starting $name on GPU $gpu (task=$task, gain=$gain, sigma=$sigma, baseline=$baseline)"
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl uv run examples/run_mjwp.py \
        +override=$OVERRIDE task=$task \
        contact_mask_rew_scale=$gain \
        contact_mask_rew_sigma=$sigma \
        contact_mask_rew_baseline=$baseline \
        video_output_path="$RESULTS_DIR/${name}.mp4" \
        > "$LOGS_DIR/${name}.log" 2>&1
    cp "$output_dir/trajectory_mjwp_act.npz" "$RESULTS_DIR/${name}.npz"
    echo "[$(date '+%H:%M:%S')] Finished $name"
}

# GPU 0: S3 (gain=3.5, sigma=0.15, baseline=0) — best config from sweep
(
    run_experiment "S3_desk005" 0 desk005_person2 3.5 0.15 0.0
    run_experiment "S3_bucket010" 0 bucket010_person1 3.5 0.15 0.0
) &
PID_GPU0=$!

# GPU 1: S3b (gain=3.5, sigma=0.15, baseline=1.0) — stability variant
(
    run_experiment "S3b_desk005" 1 desk005_person2 3.5 0.15 1.0
    run_experiment "S3b_box025" 1 box025_person1 3.5 0.15 1.0
    run_experiment "S3b_bucket010" 1 bucket010_person1 3.5 0.15 1.0
) &
PID_GPU1=$!

echo "=== E037c launched: GPU0 (S3 desk+bucket) PID=$PID_GPU0 | GPU1 (S3b all) PID=$PID_GPU1 ==="
wait $PID_GPU0
echo "[$(date '+%H:%M:%S')] GPU0 done"
wait $PID_GPU1
echo "[$(date '+%H:%M:%S')] GPU1 done"

echo ""
echo "=== All E037c experiments complete ==="
echo "Results in: $RESULTS_DIR/"
echo "Logs in: $LOGS_DIR/"
