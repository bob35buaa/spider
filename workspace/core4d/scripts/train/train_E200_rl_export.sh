#!/usr/bin/env bash
# E200 three-version RL export driver: select -> generate partner aug -> build.
#
# Produces three paired RL-export bundles under
#   workspace/core4d/results/E200/s6_downstream/rl_export/{box001_only,5object_all,5object_box001clean}/
# from the reviewed E200 three-arm master workbook. Per-object arm口径:
#   box001/box024/box004 -> PRG+G1+A2 ; box023 -> noPRG ; box021 -> PRG.
# orig rows reuse E198/E190/E170 paired manifests; aug rows are synthesized from
# the E199/E200 priority manifests with augmented partners (generated as needed).
#
# Usage: bash workspace/core4d/scripts/train/train_E200_rl_export.sh [--skip-generate]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO"
PY="${PYTHON_BIN:-$REPO/.venv/bin/python}"
E200="workspace/core4d/scripts/experiments/E200"
RLROOT="workspace/core4d/results/E200/s6_downstream/rl_export"
REVIEW="workspace/core4d/results/E200/s6_downstream/eval/E200_three_arm_master-review.xlsx"
IDX="$RLROOT/partner_aug_motion_index.tsv"
SKIP_GEN=0
[ "${1:-}" = "--skip-generate" ] && SKIP_GEN=1

echo "== Step 1/3: select cases (numeric gates 47/109/97) =="
"$PY" "$E200/select_rl_export_cases.py" --review-xlsx "$REVIEW" --out-root "$RLROOT"

echo "== Step 2/3: resolve + generate augmented partner motions =="
GEN_ARGS=(--selection-tsv "$RLROOT/5object_all/selection.tsv" --out-index "$IDX")
[ "$SKIP_GEN" -eq 0 ] && GEN_ARGS+=(--generate --object-parallel --max-workers "${RETARGET_MAX_WORKERS:-6}")
"$PY" "$E200/generate_partner_aug.py" "${GEN_ARGS[@]}"

echo "== Step 3/3: build the three paired RL-export versions =="
"$PY" "$E200/build_rl_export.py" --repo "$REPO" --rl-export-root "$RLROOT" --partner-index "$IDX"

echo "== done: $RLROOT/{box001_only,5object_all,5object_box001clean}/paired_rl_export_input.tsv =="
