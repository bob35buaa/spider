#!/usr/bin/env bash
# Evaluate all 28 E168 Box021 production rows with canonical user review.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

MANIFEST="workspace/core4d/results/E168/s6_downstream/cem/manifests/cem_production_manifest.tsv"
MANUAL_REVIEW="workspace/core4d/results/E168/s6_downstream/cem/eval/manual_review/e168_user_visual_review.tsv"
OUT_DIR="workspace/core4d/results/E168/s6_downstream/cem/eval/box021_all28_reviewed"
OUTPUT_XLSX="$OUT_DIR/E168_box021_all28_reviewed_case_metrics.xlsx"

mapfile -t CASES < <(
  awk -F '\t' 'NR==1 {for(i=1;i<=NF;i++) h[$i]=i; next} $h["object_key"]=="box021" {print $h["case_id"]}' "$MANIFEST" | sort
)
if [ "${#CASES[@]}" -ne 28 ]; then
  echo "expected 28 Box021 cases, got ${#CASES[@]}" >&2
  exit 2
fi

MUJOCO_GL="${MUJOCO_GL:-egl}" .venv/bin/python \
  workspace/core4d/scripts/eval/runners/eval_E168_e167a_metrics.py \
  --manifest "$MANIFEST" \
  --manual-review "$MANUAL_REVIEW" \
  --out-dir "$OUT_DIR" \
  --scope-label "E168 Box021 全28条人工核验冻结快照" \
  --cases "${CASES[@]}" \
  --require-all

python workspace/core4d/scripts/eval/reports/gen_E168_available_metrics_xlsx.py \
  --eval-dir "$OUT_DIR" \
  --output "$OUTPUT_XLSX"
