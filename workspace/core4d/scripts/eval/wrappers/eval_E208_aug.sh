#!/usr/bin/env bash
# E208 (R294) · score the 105 aug full-CEM runs against their same-case E206 orig.
#
# Runs on CPU only -- safe to run while the CEM queue is still going, but the
# pairing is only complete once every row is `cem_ok`, so a mid-queue run gives a
# partial (and non-stratified-complete) picture.  Check first:
#   awk -F'\t' 'NR>1 && $36=="cem_ok"' \
#     workspace/core4d/results/E208/s6_downstream/manifests/e208_priority_manifest.tsv | wc -l
#
# --rescore-orig is ON by default: E206's own 21 PRG rollouts are re-scored
# through this script's core_metrics and asserted equal to E206's published
# numbers within 1e-6 (C7c).  That is SCORING reproducibility, not retarget
# reproducibility -- F15 was removed by construction in P1.
#
# Usage:
#   bash workspace/core4d/scripts/eval/wrappers/eval_E208_aug.sh
#   NO_RESCORE=1 bash .../eval_E208_aug.sh          # trust E206's published numbers
#   LIMIT=8 bash .../eval_E208_aug.sh               # quick partial pass
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PY="${PY:-.venv/bin/python}"
RUNNER=workspace/core4d/scripts/eval/runners/eval_E208_aug.py

args=()
[ "${NO_RESCORE:-0}" = "1" ] && args+=(--no-rescore-orig)
[ -n "${LIMIT:-}" ] && args+=(--limit "$LIMIT")

exec "$PY" "$RUNNER" "${args[@]}" "$@"
