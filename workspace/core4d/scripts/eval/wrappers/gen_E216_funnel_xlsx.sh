#!/usr/bin/env bash
# Generate the E216 trans-0.1m vs orig 14-gate funnel workbook + the 0.1m-vs-0.2m
# comparison sheet, from the finished eval.
#
# Reads results/E216/s6_downstream/eval/aug/{e216_trans_rollout.tsv,
# e216_0p1m_vs_0p2m_summary.json} and re-scores each rollout through the E201
# 14-gate wide/narrow funnel (funnel_config.py). All cell values are precomputed
# in python, so there are no spreadsheet formulas to recalc.
#
# Usage:
#   bash workspace/core4d/scripts/eval/wrappers/gen_E216_funnel_xlsx.sh [--out PATH]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
exec "$PYTHON_BIN" workspace/core4d/scripts/eval/reports/gen_E216_trans_vs_orig_funnel_xlsx.py "$@"
