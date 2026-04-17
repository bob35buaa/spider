#!/bin/bash
# Run robustness tests for oakink lift_board with seeds 0-4 using 4 GPUs
export PATH=$HOME/.local/bin:$PATH
export MUJOCO_GL=egl
cd ~/spider

RESULTS_FILE="robustness_results.csv"
echo "seed,pos_err,quat_err,ik_time_s,retarget_time_s,retry_warnings,fail_errors,early_exits" > "$RESULTS_FILE"

run_seed() {
    local seed=$1
    local gpu=$2

    echo "=== Seed $seed on GPU $gpu ==="

    # IK fast
    IK_START=$(date +%s)
    CUDA_VISIBLE_DEVICES=$gpu PYTHONHASHSEED=$seed uv run python -c "
import numpy as np; np.random.seed($seed)
from spider.preprocess.ik_fast import main
main(
    dataset_name='oakink',
    robot_type='xhand',
    embodiment_type='bimanual',
    task='lift_board',
    data_id=0,
    show_viewer=False,
    save_video=False,
)
" > "robustness_ik_seed${seed}.log" 2>&1
    IK_END=$(date +%s)

    # Retargeting fast
    RET_START=$(date +%s)
    CUDA_VISIBLE_DEVICES=$gpu uv run examples/run_mjwp_fast.py +override=oakink_fast_new \
        task=lift_board data_id=0 robot_type=xhand embodiment_type=bimanual \
        viewer=none save_video=false save_info=false seed=$seed \
        > "robustness_ret_seed${seed}.log" 2>&1
    RET_END=$(date +%s)

    POS_ERR=$(grep -oP 'pos=\K[0-9.]+' "robustness_ret_seed${seed}.log" | tail -1)
    QUAT_ERR=$(grep -oP 'quat=\K[0-9.]+' "robustness_ret_seed${seed}.log" | tail -1)
    RETRIES=$(grep -c "tracking error exceeded" "robustness_ret_seed${seed}.log")
    FAILURES=$(grep -c "Failed to find feasible" "robustness_ret_seed${seed}.log")
    EARLY=$(grep -c "Early exit" "robustness_ret_seed${seed}.log")

    echo "$seed,${POS_ERR:-NA},${QUAT_ERR:-NA},$((IK_END-IK_START)),$((RET_END-RET_START)),$RETRIES,$FAILURES,$EARLY" >> "$RESULTS_FILE"
    echo "  Seed $seed: pos=$POS_ERR, quat=$QUAT_ERR, retries=$RETRIES, failures=$FAILURES, early=$EARLY"
}

# Batch 1: seeds 0-3 on GPUs 0-3
run_seed 0 0 &
run_seed 1 1 &
run_seed 2 2 &
run_seed 3 3 &
wait

# Batch 2: seed 4 on GPU 0
run_seed 4 0
wait

echo ""
echo "=== ROBUSTNESS RESULTS ==="
column -t -s',' "$RESULTS_FILE"
