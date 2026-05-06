#!/bin/bash
# E013: Intra-rollout Mocap Partner (fixes E011 architecture limitation)
# Key: partner mocap updates WITHIN CEM rollouts, not just between MPC steps
# exp_name: core4d | 对应实验: E013
set -euo pipefail

TASK="${1:-box025_person1}"
DATA_ID="${2:-0}"
VIEWER="${3:-none}"
VARIANT="${4:-full}"  # full | bodyonly | contact_body | balanced

case "${VARIANT}" in
    full)
        # E013-r1: Replicate E011 full reward (should improve with intra-rollout)
        OVERRIDE="+override=core4d_box025_e013"
        ;;
    bodyonly)
        # E013-r2: Body only + contact (no direct object tracking)
        OVERRIDE="+override=core4d_box025_e013 pos_rew_scale=0.0 rot_rew_scale=0.0 base_pos_rew_scale=5.0 contact_rew_scale=1.0"
        ;;
    contact_body)
        # E013-r3: Weak object + strong body + strong contact
        OVERRIDE="+override=core4d_box025_e013 pos_rew_scale=1.5 rot_rew_scale=0.5 base_pos_rew_scale=5.0 contact_rew_scale=2.0"
        ;;
    balanced)
        # E013-r4: Stronger body than E011c
        OVERRIDE="+override=core4d_box025_e013 pos_rew_scale=2.0 rot_rew_scale=0.5 base_pos_rew_scale=7.0 contact_rew_scale=1.0"
        ;;
    nointra)
        # E013-ctrl: Control — intra-step disabled (replicate E011)
        OVERRIDE="+override=core4d_box025_e013 mocap_partner_intra_step=false"
        ;;
    *)
        echo "Unknown variant: ${VARIANT}. Use: full|bodyonly|contact_body|balanced|nointra"
        exit 1
        ;;
esac

echo "=== E013 (${VARIANT}) ==="
echo "Config: ${OVERRIDE}"

uv run examples/run_mjwp.py \
    ${OVERRIDE} \
    task="${TASK}" \
    data_id="${DATA_ID}" \
    viewer="${VIEWER}"

echo "Output: example_datasets/processed/core4d/unitree_g1/humanoid_object/${TASK}/${DATA_ID}/trajectory_mjwp.npz"
