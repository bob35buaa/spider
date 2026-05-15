#!/usr/bin/env bash
# E077: Build SPIDER box023_person2 from CORE4D raw 20231008/045.
set -euo pipefail

REPO="/home/ubuntu/Workspace/spider"
HOLOSOMA_DIR="/home/ubuntu/Workspace/holosoma"
CORE4D_DIR="/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real/human_object_motions"
SMPLX_MODEL_DIR="/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/smplx"
OUT_DIR="$REPO/workspace/core4d/results/E077/holosoma_box023_person2"
CONVERTED_DIR="$OUT_DIR/converted"
RETARGETED_DIR="$OUT_DIR/retargeted"
TRIMMED="$OUT_DIR/trimmed/20231008-045-person2-Box023_with_obj_original.npz"
TASK_NAME="20231008-045-person2-Box023_with_obj"

mkdir -p "$CONVERTED_DIR" "$RETARGETED_DIR"

source "$HOLOSOMA_DIR/scripts/source_retargeting_setup.sh"

cd "$HOLOSOMA_DIR"
python "$HOLOSOMA_DIR/workspace/pipeline/convert_core4d_to_omniretarget.py" \
  --core4d_dir "$CORE4D_DIR" \
  --smplx_model_dir "$SMPLX_MODEL_DIR" \
  --output_dir "$CONVERTED_DIR" \
  --date 20231008 \
  --seq 045 \
  --person person2 \
  --with_object \
  --replace_wrist_with_fingertip

cd "$HOLOSOMA_DIR/src/holosoma_retargeting/holosoma_retargeting"
python examples/robot_retarget.py \
  --data_path "$CONVERTED_DIR" \
  --task-type object_interaction \
  --task-name "$TASK_NAME" \
  --data_format smplx \
  --task-config.object-name Box023 \
  --save_dir "$RETARGETED_DIR"

cd "$REPO"
env UV_CACHE_DIR=/tmp/uv-cache uv run python workspace/core4d/scripts/E077/trim_box023_person2.py
env UV_CACHE_DIR=/tmp/uv-cache uv run python workspace/core4d/scripts/E077/create_box023_person2_scene.py
env UV_CACHE_DIR=/tmp/uv-cache uv run python spider/process_datasets/core4d.py \
  --source-npz "$TRIMMED" \
  --task box023_person2 \
  --data-id 0 \
  --no-show-viewer \
  --no-save-video
env UV_CACHE_DIR=/tmp/uv-cache uv run python workspace/core4d/scripts/E077/create_box023_person2_scene.py \
  --generate-scene-act
