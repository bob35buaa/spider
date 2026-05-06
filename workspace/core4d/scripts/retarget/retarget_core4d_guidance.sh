#!/bin/bash
# SPIDER MJWP 物理重定向 (带 contact guidance)
# exp_name: core4d | 对应实验: E003/E004
set -euo pipefail

TASK="${1:-box025_person1}"
DATA_ID="${2:-0}"
OUTPUT_DIR="${3:-workspace/core4d/results/E003}"
OVERRIDE="core4d_box025_act"

# Contact guidance 参数 (可通过环境变量覆盖)
GUIDANCE_DECAY="${GUIDANCE_DECAY:-0.85}"
POS_GAIN="${POS_GAIN:-20.0}"
ROT_GAIN="${ROT_GAIN:-0.3}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

uv run examples/run_mjwp.py \
    +override=${OVERRIDE} \
    task=${TASK} \
    data_id=${DATA_ID} \
    viewer=none \
    save_video=true \
    save_info=true \
    output_dir=${OUTPUT_DIR} \
    guidance_decay_ratio=${GUIDANCE_DECAY} \
    init_pos_actuator_gain=${POS_GAIN} \
    init_pos_actuator_bias=${POS_GAIN} \
    init_rot_actuator_gain=${ROT_GAIN} \
    init_rot_actuator_bias=${ROT_GAIN}
