#!/bin/bash
# E006f: SPIDER MJWP 前臂接触重定向 (最优配置)
# scene_forearm.xml: 3-box 手/前臂碰撞 + elbow-object contact pair + 0.5kg 物体
# config: contact_rew_scale=1.0, pos_rew_scale=3.0, base_pos_rew_scale=3.0
# exp_name: core4d | 对应实验: E006
set -euo pipefail

TASK="${1:-box025_person1}"
DATA_ID="${2:-0}"
VIEWER="${3:-none}"

uv run examples/run_mjwp.py \
    +override=core4d_box025_forearm \
    task="${TASK}" \
    data_id="${DATA_ID}" \
    viewer="${VIEWER}"

echo "Output: example_datasets/processed/core4d/unitree_g1/humanoid_object/${TASK}/${DATA_ID}/trajectory_mjwp.npz"
