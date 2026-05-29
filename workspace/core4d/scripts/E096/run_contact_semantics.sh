#!/usr/bin/env bash
# E096 contact semantic audit/rendering for the three E095 first-batch cases.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OUT_ROOT="${OUT_ROOT:-workspace/core4d/results/E096/contact_semantics}"
PROJ_ROOT="${PROJ_ROOT:-workspace/core4d/results/E096/adaptive_support_projection}"
SAMPLE_COUNT="${SAMPLE_COUNT:-12000}"
VIDEO_FRAMES="${VIDEO_FRAMES:-48}"

.venv/bin/python workspace/core4d/scripts/E096/build_contact_manifest.py --out-root "$OUT_ROOT"
.venv/bin/python workspace/core4d/scripts/E093/audit_contact_geometry.py \
  --manifest "$OUT_ROOT/case_manifest.tsv" \
  --out-root "$OUT_ROOT" \
  --sample-count "$SAMPLE_COUNT"
.venv/bin/python workspace/core4d/scripts/E093/render_contact_geometry_mujoco.py \
  --manifest "$OUT_ROOT/case_manifest.tsv" \
  --points "$OUT_ROOT/per_frame_points.csv" \
  --out-root "$OUT_ROOT" \
  --camera auto \
  --video-frames "$VIDEO_FRAMES"

mapfile -t ready_case_ids < <(awk -F '\t' 'NR > 1 && tolower($16) == "true" {print $1}' "$OUT_ROOT/case_manifest.tsv")
if [ "${#ready_case_ids[@]}" -gt 0 ]; then
  .venv/bin/python workspace/core4d/scripts/E094/build_handbox_target_projection.py \
    --manifest "$OUT_ROOT/case_manifest.tsv" \
    --out-root "$PROJ_ROOT" \
    --case-ids "${ready_case_ids[@]}" \
    --force
  .venv/bin/python workspace/core4d/scripts/E094/render_projection_mujoco.py \
    --manifest "$OUT_ROOT/case_manifest.tsv" \
    --points "$PROJ_ROOT/per_frame_projection.csv" \
    --out-root "$PROJ_ROOT" \
    --tasks "${ready_case_ids[@]}" \
    --video-frames "$VIDEO_FRAMES"
fi

echo "=== E096 contact semantics done ==="
