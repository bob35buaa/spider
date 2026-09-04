#!/usr/bin/env bash
# E207 (PRG+G1 / G1only) bucket007 -- SPIDER-side partner-complete RL export (dcv3).
#
# Two steps, both in-repo (no holosoma, no GPU):
#   1. export_bucket007_rl_input.py   -> rl_export_input.tsv  (74-col dcv3 source rows)
#   2. dcv3 finalize_reused_partner_rl -> partner manifest + paired_rl_export_input.tsv
#
# Partner = the opposite person's OmniRetarget Stage2b *kinematic* motion from E174
# (never a CEM rollout), aligned on the common raw-frame window. All 6 bucket007
# partners resolve from omnirt_v1; no direct-retarget backfill is needed.
#
# SELECTION AUTHORITY: unlike E178, E207 has NO manual review. Every bucket007 case
# is exported and the numeric outcome rides along as provenance. `--manual-review-
# snapshot` is deliberately NOT passed, so the audit records
# approved_source_set_exact=null instead of claiming an approval that never happened.
#
# Usage:
#   bash workspace/core4d/scripts/launch/active/run_E207_bucket007_partner_rl_export.sh
#   DRY_RUN=1 bash .../run_E207_bucket007_partner_rl_export.sh
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

PY="${PY:-.venv/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-disable}"
OUT_DIR="workspace/core4d/results/E207/s6_downstream/rl_export"
INPUT_TSV="${OUT_DIR}/rl_export_input.tsv"
E174="workspace/core4d/results/E174/s3_retarget"
DCV3="workspace/core4d/scripts/data_construction_v3/stages/s6_downstream/finalize_reused_partner_rl.py"
EXPECTED_ROWS="${EXPECTED_ROWS:-6}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exec "$PY" workspace/core4d/scripts/experiments/E207/export_bucket007_rl_input.py --dry-run
fi

echo "[E207-export] step 1/2 — source rows"
"$PY" workspace/core4d/scripts/experiments/E207/export_bucket007_rl_input.py --out-dir "$OUT_DIR"

echo "[E207-export] step 2/2 — dcv3 partner pairing"
"$PY" "$DCV3" \
  --repo "$PWD" \
  --experiment-id E207 \
  --object-key bucket007 \
  --rl-export-input-tsv "$INPUT_TSV" \
  --out-dir "$OUT_DIR" \
  --stage2b-manifest-tsv "${E174}/omnirt_v1/ref_fk/stage2b_manifest_omnirt_v1_ref_fk.tsv" \
  --stage2b-manifest-tsv "${E174}/omnirt_v2/ref_fk/stage2b_manifest_omnirt_v2_ref_fk.tsv" \
  --expected-rows "$EXPECTED_ROWS"

echo "[E207-export] done -> ${OUT_DIR}/paired_rl_export_input.tsv"
echo "[E207-export] downstream must filter on paired_rl_export_decision (NOT rl_export_decision)"
