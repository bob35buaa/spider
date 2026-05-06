#!/bin/bash
# E022: Run anchored trajectories with object reward
# Usage: bash workspace/core4d/scripts/retarget/retarget_e022_anchored_objrew.sh

set -e
export MUJOCO_GL=egl

PROCESSED="/home/ubuntu/Workspace/spider/example_datasets/processed/core4d/unitree_g1/humanoid_object"
RESULTS="/home/ubuntu/Workspace/spider/workspace/core4d/results/E022_anchored_objrew"

declare -A CASE_TASK=(
    ["box025"]="box025_person1"
    ["bucket010"]="bucket010_person1"
    ["chair022"]="chair022_person1"
    ["desk005"]="desk005_person2"
)

declare -A CASE_OVERRIDE=(
    ["box025"]="core4d_box025"
    ["bucket010"]="core4d_bucket010"
    ["chair022"]="core4d_chair022"
    ["desk005"]="core4d_desk005"
)

for case in box025 bucket010 chair022 desk005; do
    task="${CASE_TASK[$case]}"
    override="${CASE_OVERRIDE[$case]}"
    DATA_PATH="$PROCESSED/$task/0/trajectory_kinematic_anchored.npz"

    echo "=== $case: baseline (pos=3.0, contact=1.0) ==="
    uv run examples/run_mjwp.py +override=$override \
        viewer=none save_video=true \
        base_pos_rew_scale=10.0 pos_rew_scale=3.0 contact_rew_scale=1.0 \
        data_path="$DATA_PATH" 2>&1 | grep "Total time"
    cp "$PROCESSED/$task/0/trajectory_mjwp.npz" "$RESULTS/$case/a1_baseline.npz"
    cp "$PROCESSED/$task/0/visualization_mjwp.mp4" "$RESULTS/$case/a1_baseline.mp4" 2>/dev/null

    echo "=== $case: strong (pos=5.0, contact=2.0) ==="
    uv run examples/run_mjwp.py +override=$override \
        viewer=none save_video=true \
        base_pos_rew_scale=10.0 pos_rew_scale=5.0 contact_rew_scale=2.0 \
        data_path="$DATA_PATH" 2>&1 | grep "Total time"
    cp "$PROCESSED/$task/0/trajectory_mjwp.npz" "$RESULTS/$case/a2_strong.npz"
    cp "$PROCESSED/$task/0/visualization_mjwp.mp4" "$RESULTS/$case/a2_strong.mp4" 2>/dev/null

    echo "Done: $case"
    echo
done

echo "All E022 runs complete."
