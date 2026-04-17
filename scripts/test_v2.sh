#!/bin/bash
# Test v2: IK fast + retargeting fast with rot_threshold=0.5, boosted retry noise
# Tests both tasks with video, then viewer compatibility
export PATH=$HOME/.local/bin:$PATH
export MUJOCO_GL=egl
cd ~/spider

echo "=== Phase 1: IK fast for both tasks ==="
# IK fast for lift_board and tea in parallel
uv run python -c "
from spider.preprocess.ik_fast import main
main(dataset_name='oakink', robot_type='xhand', embodiment_type='bimanual',
     task='lift_board', data_id=0, show_viewer=False, save_video=False)
" > /tmp/ik_lift.log 2>&1 &
P1=$!

uv run python -c "
from spider.preprocess.ik_fast import main
main(dataset_name='gigahand', robot_type='xhand', embodiment_type='bimanual',
     task='p36-tea', data_id=0, show_viewer=False, save_video=False)
" > /tmp/ik_tea.log 2>&1 &
P2=$!
wait $P1 $P2
echo "IK fast done for both tasks."

echo ""
echo "=== Phase 2: Retargeting fast with save_video ==="
CUDA_VISIBLE_DEVICES=0 uv run examples/run_mjwp_fast.py +override=oakink_fast_new \
    task=lift_board data_id=0 robot_type=xhand embodiment_type=bimanual \
    viewer=none save_video=true save_info=true \
    > /tmp/ret_lift.log 2>&1 &
P1=$!

CUDA_VISIBLE_DEVICES=1 uv run examples/run_mjwp_fast.py +override=gigahand_fast_new \
    task=p36-tea data_id=0 robot_type=xhand embodiment_type=bimanual \
    viewer=none save_video=true save_info=true \
    > /tmp/ret_tea.log 2>&1 &
P2=$!
wait $P1 $P2

echo ""
echo "=== RESULTS ==="
for task in lift tea; do
    log="/tmp/ret_${task}.log"
    echo "--- ${task} ---"
    grep "Final object" "$log" 2>/dev/null
    echo "Revert events:"
    grep "Reverting\|Forward attempt\|Recovered\|Unrecoverable\|Retrying with boosted\|first step" "$log" 2>/dev/null | sed 's/^/  /'
    echo ""
done

echo "=== Phase 3: Viewer compatibility (lift_board, 2 control ticks each) ==="
# Test rerun viewer
echo "Testing rerun viewer..."
CUDA_VISIBLE_DEVICES=2 uv run examples/run_mjwp_fast.py +override=oakink_fast_new \
    task=lift_board data_id=0 robot_type=xhand embodiment_type=bimanual \
    viewer=rerun rerun_spawn=false max_sim_steps=300 save_video=false save_info=false \
    > /tmp/viewer_rerun.log 2>&1
RERUN_EXIT=$?
echo "  rerun exit code: $RERUN_EXIT"
grep -i "error\|exception\|traceback" /tmp/viewer_rerun.log | head -3

# Test viser viewer
echo "Testing viser viewer..."
CUDA_VISIBLE_DEVICES=3 uv run examples/run_mjwp_fast.py +override=oakink_fast_new \
    task=lift_board data_id=0 robot_type=xhand embodiment_type=bimanual \
    viewer=viser wait_on_finish=false max_sim_steps=300 save_video=false save_info=false \
    > /tmp/viewer_viser.log 2>&1
VISER_EXIT=$?
echo "  viser exit code: $VISER_EXIT"
grep -i "error\|exception\|traceback" /tmp/viewer_viser.log | head -3

echo ""
echo "=== ALL DONE ==="
