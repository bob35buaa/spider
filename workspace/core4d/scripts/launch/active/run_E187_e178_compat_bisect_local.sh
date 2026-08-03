#!/usr/bin/env bash
# E187 Gate S0 diagnostic: first-CEM replay under E178 and current source.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
E178_SOURCE_COMMIT="${E178_SOURCE_COMMIT:-64f9a33}"
CASE_ID="bucket007_20231020_055_p1"
OVERRIDE="core4d_E178_bucket007_20231020_055_p1_contactAlignedTop"
TASK="dcv3_omnirt_v2_ref_fk_bucket007_20231020_055_p1"
ROOT="workspace/core4d/results/E187/s0_environment/e178_compat/bisect_v1"
LOG_ROOT="logs/E187/s0/e178_compat/bisect_v1"
OLD_OUT="$ROOT/historical_source_outdir"
CURRENT_OUT="$ROOT/current_source_outdir"
OLD_NPZ="$OLD_OUT/trajectory_mjwp_act.npz"
CURRENT_NPZ="$CURRENT_OUT/trajectory_mjwp_act.npz"
HISTORICAL_FULL="workspace/core4d/results/E178/s6_downstream/cem/full/E178_bucket007_20231020_055_p1_contactAlignedTop_full.npz"
REPORT="$ROOT/short_replay_comparison.json"
SOURCE_TMP="$(mktemp -d -t e187_e178_source_XXXXXX)"
trap 'rm -rf -- "$SOURCE_TMP"' EXIT

if [[ -e "$OLD_NPZ" || -e "$CURRENT_NPZ" || -e "$REPORT" ]]; then
  echo "E187 bisect_v1 output already exists; refusing to overwrite." >&2
  exit 2
fi
mkdir -p "$OLD_OUT" "$CURRENT_OUT" "$LOG_ROOT"

git archive --format=tar "$E178_SOURCE_COMMIT" | tar -xf - -C "$SOURCE_TMP"
test "$(git show "$E178_SOURCE_COMMIT:spider/optimizers/sampling.py" | sha256sum | awk '{print $1}')" = \
  "$(sha256sum "$SOURCE_TMP/spider/optimizers/sampling.py" | awk '{print $1}')"

COMMON_ARGS=(
  "+override=$OVERRIDE"
  "task=$TASK"
  "+use_torch_compile=false"
  "save_video=false"
  "video_camera=auto"
  "num_samples=1024"
  "max_num_iterations=32"
  "seed=0"
  "max_sim_steps=16"
)

nvidia-smi \
  --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader,nounits > "$ROOT/local_gpu_before.csv"
nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits > "$ROOT/local_compute_before.csv"

CUDA_VISIBLE_DEVICES="$GPU_ID" MUJOCO_GL=egl PYTHONPATH="$SOURCE_TMP" \
  "$PYTHON_BIN" -u "$SOURCE_TMP/examples/run_mjwp.py" \
  "${COMMON_ARGS[@]}" \
  "output_dir=$OLD_OUT" \
  "video_output_path=$ROOT/historical_source.mp4" \
  > "$LOG_ROOT/historical_source.log" 2>&1

CUDA_VISIBLE_DEVICES="$GPU_ID" MUJOCO_GL=egl PYTHONPATH=. \
  "$PYTHON_BIN" -u examples/run_mjwp.py \
  "${COMMON_ARGS[@]}" \
  "output_dir=$CURRENT_OUT" \
  "video_output_path=$ROOT/current_source.mp4" \
  > "$LOG_ROOT/current_source.log" 2>&1

PYTHONPATH=. "$PYTHON_BIN" \
  workspace/core4d/scripts/experiments/E187/compare_e178_short_replay.py \
  --historical-full "$HISTORICAL_FULL" \
  --historical-source-short "$OLD_NPZ" \
  --current-source-short "$CURRENT_NPZ" \
  --output "$REPORT"

nvidia-smi \
  --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader,nounits > "$ROOT/local_gpu_after.csv"
nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits > "$ROOT/local_compute_after.csv"
