#!/bin/bash
# E037 Parameter Sweep — Remote Machine (2 GPUs)
# Runs 4 configs: S2,S3 on GPU0; S4,S5 on GPU1
#
# 参数网格设计:
#   S2: gain=3.5, sigma=0.3, baseline=0.0 — max gain = tracking max
#   S3: gain=3.5, sigma=0.15, baseline=0.0 — max gain + steeper kernel
#   S4: gain=3.5, sigma=0.3, baseline=1.0 — max gain + HDMI baseline
#   S5: gain=5.0, sigma=0.3, baseline=1.0 — exceed tracking (HDMI exact params)
#
# Usage: bash workspace/core4d/scripts/run_e037_sweep_remote.sh
# Prerequisites: same repo checkout with E037 code changes applied

set -e

TASK=box025_person1
OVERRIDE=core4d_e037
RESULTS_DIR=workspace/core4d/results/E037_sweep
LOGS_DIR=logs/E037_sweep

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

OUTPUT_DIR=example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/0

run_experiment() {
    local name=$1
    local gpu=$2
    local gain=$3
    local sigma=$4
    local baseline=$5

    echo "[$(date '+%H:%M:%S')] Starting $name on GPU $gpu (gain=$gain, sigma=$sigma, baseline=$baseline)"
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl uv run examples/run_mjwp.py \
        +override=$OVERRIDE task=$TASK \
        contact_mask_rew_scale=$gain \
        contact_mask_rew_sigma=$sigma \
        contact_mask_rew_baseline=$baseline \
        > "$LOGS_DIR/${name}_box025.log" 2>&1
    cp "$OUTPUT_DIR/trajectory_mjwp_act.npz" "$RESULTS_DIR/${name}_box025.npz"
    echo "[$(date '+%H:%M:%S')] Finished $name"
}

# GPU 0: S2 → S3 (sequential on same GPU)
(
    run_experiment "S2_gain3.5_sigma0.3_base0.0" 0 3.5 0.3 0.0
    run_experiment "S3_gain3.5_sigma0.15_base0.0" 0 3.5 0.15 0.0
) &
PID_GPU0=$!

# GPU 1: S4 → S5 (sequential on same GPU)
(
    run_experiment "S4_gain3.5_sigma0.3_base1.0" 1 3.5 0.3 1.0
    run_experiment "S5_gain5.0_sigma0.3_base1.0" 1 5.0 0.3 1.0
) &
PID_GPU1=$!

echo "=== E037 Sweep launched: GPU0 (S2,S3) PID=$PID_GPU0 | GPU1 (S4,S5) PID=$PID_GPU1 ==="
echo "=== Logs: $LOGS_DIR/ ==="

wait $PID_GPU0
echo "[$(date '+%H:%M:%S')] GPU0 done (S2, S3)"
wait $PID_GPU1
echo "[$(date '+%H:%M:%S')] GPU1 done (S4, S5)"

echo ""
echo "=== All sweep experiments complete ==="
echo "Results in: $RESULTS_DIR/"
echo "Logs in: $LOGS_DIR/"
echo ""
echo "Run evaluation:"
echo "  uv run workspace/core4d/scripts/eval/eval_comprehensive.py box025_person1 $RESULTS_DIR/S2_gain3.5_sigma0.3_base0.0_box025.npz"
echo "  uv run workspace/core4d/scripts/eval/eval_comprehensive.py box025_person1 $RESULTS_DIR/S3_gain3.5_sigma0.15_base0.0_box025.npz"
echo "  uv run workspace/core4d/scripts/eval/eval_comprehensive.py box025_person1 $RESULTS_DIR/S4_gain3.5_sigma0.3_base1.0_box025.npz"
echo "  uv run workspace/core4d/scripts/eval/eval_comprehensive.py box025_person1 $RESULTS_DIR/S5_gain5.0_sigma0.3_base1.0_box025.npz"
