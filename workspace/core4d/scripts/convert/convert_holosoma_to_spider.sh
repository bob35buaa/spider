#!/bin/bash
# 数据转换: holosoma retarget qpos(T,43) → SPIDER trajectory_kinematic.npz
# exp_name: core4d | 对应实验: E001
set -euo pipefail

SOURCE_DIR="/home/ubuntu/Workspace/holosoma/workspace/v2/results/retarget_replace_batch_trimmed"
TASKS=(
    "20231011-048-person1-Box025_with_obj_original.npz:box025_person1:0"
    "20231011-048-person2-Box025_with_obj_original.npz:box025_person2:0"
    "20231002-003-person1-bucket005_with_obj_original.npz:bucket005_person1:0"
    "20231002-003-person2-bucket005_with_obj_original.npz:bucket005_person2:0"
    "20231003_2-059-person1-bucket010_with_obj_original.npz:bucket010_person1:0"
    "20231003_2-059-person2-bucket010_with_obj_original.npz:bucket010_person2:0"
    "20231020-084-person1-chair022_with_obj_original.npz:chair022_person1:0"
    "20231020-084-person2-chair022_with_obj_original.npz:chair022_person2:0"
    "20231023-030-person2-desk005_with_obj_original.npz:desk005_person2:0"
)

for entry in "${TASKS[@]}"; do
    IFS=':' read -r npz_file task data_id <<< "$entry"
    echo "=== Converting: ${task} ==="
    uv run python spider/process_datasets/core4d.py \
        --source-npz "${SOURCE_DIR}/${npz_file}" \
        --task "${task}" \
        --data-id "${data_id}" \
        --no-show-viewer \
        --save-video
done
