#!/usr/bin/env bash
# E089 B4: smoke (4-iter CEM) for two B-path repaired box021 cases. Sequential on GPU0.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
GPU="${1:-0}"
RESULTS="${RESULTS:-workspace/core4d/results/E089/B4}"
LOGS="${LOGS:-logs/E089/B4}"
mkdir -p "$RESULTS" "$LOGS"

bash workspace/core4d/scripts/convert/snapshot_scenes.sh E089_B4 \
  d003_box021_20231018_031_p2_btop_upperobj_e089b \
  d003_box021_20231020_020_p1_btop_upperobj_e089b \
  > "$LOGS/snapshot.log" 2>&1

for VARIANT in E089B1_box021_20231018_031_p2_btop E089B2_box021_20231020_020_p1_btop; do
  OUT="$RESULTS/${VARIANT}_outdir_smoke"
  mkdir -p "$OUT"
  echo "[$(date '+%H:%M:%S')] === ${VARIANT} smoke ==="
  CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
    +override="core4d_${VARIANT}" \
    +use_torch_compile=false \
    max_num_iterations=4 \
    output_dir="$OUT" \
    video_output_path="$RESULTS/${VARIANT}_smoke.mp4" \
    > "$LOGS/${VARIANT}_smoke.log" 2>&1
  echo "[$(date '+%H:%M:%S')] === ${VARIANT} done ==="
done
