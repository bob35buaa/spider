#!/bin/bash
# Run IK fast + retargeting fast for oakink lift_board with a specific seed
# Usage: robustness_test.sh <seed> <gpu_id>
SEED=${1:-0}
GPU_ID=${2:-0}

export PATH=$HOME/.local/bin:$PATH
export MUJOCO_GL=egl
cd ~/spider

RESULTS_FILE="robustness_seed${SEED}.csv"
echo "seed,step,pos_err,quat_err,ik_time_s,retarget_time_s,retry_warnings,fail_errors" > "$RESULTS_FILE"

echo "=== Seed $SEED on GPU $GPU_ID ==="

# Step 1: Run IK fast (no seed control — we set numpy seed manually via env)
echo "Running IK fast..."
IK_START=$(date +%s)
CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONHASHSEED=$SEED uv run python -c "
import numpy as np; np.random.seed($SEED)
import spider.preprocess.ik_fast as ik
ik.main(
    dataset_name='oakink',
    robot_type='xhand',
    embodiment_type='bimanual',
    task='lift_board',
    data_id=0,
    show_viewer=False,
    save_video=False,
)
" > "ik_fast_seed${SEED}.log" 2>&1
IK_END=$(date +%s)
IK_ELAPSED=$((IK_END - IK_START))
echo "IK fast done in ${IK_ELAPSED}s"

# Step 2: Run retargeting fast
echo "Running retargeting fast with seed=$SEED..."
RET_START=$(date +%s)
CUDA_VISIBLE_DEVICES=$GPU_ID uv run examples/run_mjwp_fast.py +override=oakink_fast_new \
    task=lift_board data_id=0 robot_type=xhand embodiment_type=bimanual \
    viewer=none save_video=false save_info=true seed=$SEED \
    > "retarget_fast_seed${SEED}.log" 2>&1
RET_END=$(date +%s)
RET_ELAPSED=$((RET_END - RET_START))

POS_ERR=$(grep -oP 'pos=\K[0-9.]+' "retarget_fast_seed${SEED}.log" | tail -1)
QUAT_ERR=$(grep -oP 'quat=\K[0-9.]+' "retarget_fast_seed${SEED}.log" | tail -1)
RETRY_WARNS=$(grep -c "tracking error exceeded" "retarget_fast_seed${SEED}.log")
FAIL_ERRS=$(grep -c "Failed to find feasible" "retarget_fast_seed${SEED}.log")
EARLY_EXIT=$(grep -c "Early exit" "retarget_fast_seed${SEED}.log")

echo "$SEED,retarget,${POS_ERR:-NA},${QUAT_ERR:-NA},$IK_ELAPSED,$RET_ELAPSED,$RETRY_WARNS,$FAIL_ERRS" >> "$RESULTS_FILE"
echo "  Seed $SEED: pos=$POS_ERR, quat=$QUAT_ERR, ik=${IK_ELAPSED}s, retarget=${RET_ELAPSED}s, retries=$RETRY_WARNS, failures=$FAIL_ERRS, early_exit=$EARLY_EXIT"
