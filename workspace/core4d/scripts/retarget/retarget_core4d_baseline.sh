#!/bin/bash
# SPIDER MJWP 物理重定向 (无 contact guidance)
# exp_name: core4d | 对应实验: E002
set -euo pipefail

TASK="${1:-box025_person1}"
DATA_ID="${2:-0}"
OUTPUT_DIR="${3:-workspace/core4d/results/E002}"
OVERRIDE="core4d_box025"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

uv run examples/run_mjwp.py \
    +override=${OVERRIDE} \
    task=${TASK} \
    data_id=${DATA_ID} \
    viewer=none \
    save_video=true \
    save_info=true \
    output_dir=${OUTPUT_DIR}
