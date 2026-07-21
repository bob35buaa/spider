#!/usr/bin/env bash
# Evaluate Box021 rows not present in the canonical user-review snapshot.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

MANIFEST="workspace/core4d/results/E168/s6_downstream/cem/manifests/cem_production_manifest.tsv"
MANUAL_REVIEW="workspace/core4d/results/E168/s6_downstream/cem/eval/manual_review/e168_user_visual_review.tsv"
OUT_DIR="workspace/core4d/results/E168/s6_downstream/cem/eval/box021_remaining12_available"
OUTPUT_XLSX="$OUT_DIR/E168_box021_remaining12_available_case_metrics.xlsx"
SCOPE_LABEL="E168 Box021 剩余12条（排除上一批人工核验16条）"

CASES=(
  "box021_20231018_030_p2"
  "box021_20231018_031_p2"
  "box021_20231018_032_p1"
  "box021_20231018_032_p2"
  "box021_20231018_034_p2"
  "box021_20231018_035_p2"
  "box021_20231020_019_p1"
  "box021_20231020_019_p2"
  "box021_20231020_022_p1"
  "box021_20231020_022_p2"
  "box021_20231020_023_p1"
  "box021_20231020_023_p2"
)

if [ "${#CASES[@]}" -ne 12 ]; then
  echo "expected 12 remaining Box021 cases, got ${#CASES[@]}" >&2
  printf '%s\n' "${CASES[@]}" >&2
  exit 2
fi

MUJOCO_GL="${MUJOCO_GL:-egl}" .venv/bin/python \
  workspace/core4d/scripts/eval/runners/eval_E168_e167a_metrics.py \
  --manifest "$MANIFEST" \
  --manual-review "$MANUAL_REVIEW" \
  --out-dir "$OUT_DIR" \
  --scope-label "$SCOPE_LABEL" \
  --cases "${CASES[@]}"

python workspace/core4d/scripts/eval/reports/gen_E168_available_metrics_xlsx.py \
  --eval-dir "$OUT_DIR" \
  --output "$OUTPUT_XLSX"
