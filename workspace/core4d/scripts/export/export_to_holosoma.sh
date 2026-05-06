#!/bin/bash
# 混合轨迹导出: SPIDER 机器人运动 + 运动学物体 → Holosoma RL 格式
# exp_name: core4d | 对应实验: E005
set -euo pipefail

TASK="${1:-box025_person1}"
OBJECT_NAME="${2:-Box025}"
SCENE_XML="${3:-/home/ubuntu/Workspace/holosoma/src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof_w_${OBJECT_NAME}.xml}"

SPIDER_NPZ="example_datasets/processed/core4d/unitree_g1/humanoid_object/${TASK}/0/trajectory_mjwp.npz"
KIN_NPZ="example_datasets/processed/core4d/unitree_g1/humanoid_object/${TASK}/0/trajectory_kinematic.npz"
OUTPUT_NPZ="workspace/core4d/results/${TASK}_spider_holosoma.npz"

python3 workspace/core4d/scripts/export_spider_to_holosoma.py \
    --spider-npz "${SPIDER_NPZ}" \
    --kinematic-npz "${KIN_NPZ}" \
    --scene-xml "${SCENE_XML}" \
    --output-npz "${OUTPUT_NPZ}" \
    --object-name "${OBJECT_NAME}"

echo "Exported: ${OUTPUT_NPZ}"
