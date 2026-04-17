#!/bin/bash
# Test IK fast + retargeting fast with revert logic on lift_board and p36-tea
export PATH=$HOME/.local/bin:$PATH
export MUJOCO_GL=egl
cd ~/spider

run_task() {
    local gpu=$1
    local override=$2
    local dataset=$3
    local task=$4
    local data_id=$5
    local logprefix="/tmp/revert_${dataset}_${task}"

    echo "=== [GPU$gpu] $dataset/$task ==="

    # Step 1: IK fast
    echo "  Running IK fast..."
    CUDA_VISIBLE_DEVICES=$gpu uv run python -c "
from spider.preprocess.ik_fast import main
main(
    dataset_name='$dataset',
    robot_type='xhand',
    embodiment_type='bimanual',
    task='$task',
    data_id=$data_id,
    show_viewer=False,
    save_video=False,
)
" > "${logprefix}_ik.log" 2>&1
    echo "  IK fast done."

    # Step 2: Retargeting fast
    echo "  Running retargeting fast..."
    CUDA_VISIBLE_DEVICES=$gpu uv run examples/run_mjwp_fast.py +override=$override \
        task=$task data_id=$data_id robot_type=xhand embodiment_type=bimanual \
        viewer=none save_video=false save_info=true \
        > "${logprefix}_ret.log" 2>&1

    POS=$(grep -oP 'pos=\K[0-9.]+' "${logprefix}_ret.log" | tail -1)
    QUAT=$(grep -oP 'quat=\K[0-9.]+' "${logprefix}_ret.log" | tail -1)
    echo "  Results: pos=$POS, quat=$QUAT"
    echo "  Revert events:"
    grep "Reverting\|Forward attempt\|Recovered\|Unrecoverable\|Retrying with 2x\|Proceeding" "${logprefix}_ret.log" | sed 's/^/    /'
    echo ""
}

# Run lift_board and tea in parallel on different GPUs
run_task 0 oakink_fast_new oakink lift_board 0 &
run_task 1 gigahand_fast_new gigahand p36-tea 0 &
wait

echo "=== ALL DONE ==="
