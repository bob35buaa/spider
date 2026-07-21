#!/usr/bin/env bash
# Evaluate the completed Box021 20231020_022 p1/p2 pair in isolation.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

MANIFEST="workspace/core4d/results/E168/s6_downstream/cem/manifests/cem_production_manifest.tsv"
MANUAL_REVIEW="workspace/core4d/results/E168/s6_downstream/cem/eval/manual_review/e168_user_visual_review.tsv"
OUT_DIR="workspace/core4d/results/E168/s6_downstream/cem/eval/box021_20231020_022_pair"
OUTPUT_XLSX="$OUT_DIR/E168_box021_20231020_022_pair_case_metrics.xlsx"
CASES=(
  "box021_20231020_022_p1"
  "box021_20231020_022_p2"
)

MUJOCO_GL="${MUJOCO_GL:-egl}" .venv/bin/python \
  workspace/core4d/scripts/eval/runners/eval_E168_e167a_metrics.py \
  --manifest "$MANIFEST" \
  --manual-review "$MANUAL_REVIEW" \
  --out-dir "$OUT_DIR" \
  --scope-label "E168 Box021 20231020_022 p1/p2 独立评测" \
  --cases "${CASES[@]}" \
  --require-all

python workspace/core4d/scripts/eval/reports/gen_E168_available_metrics_xlsx.py \
  --eval-dir "$OUT_DIR" \
  --output "$OUTPUT_XLSX"
