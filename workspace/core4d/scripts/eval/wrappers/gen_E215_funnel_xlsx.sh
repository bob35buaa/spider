#!/usr/bin/env bash
# Generate the E215 rot-vs-orig 14-gate funnel workbook from the finished eval.
#
# Reads results/E215/s6_downstream/eval/aug/e215_rot_rollout.tsv (102 rows) and
# re-scores each rollout through the E201 14-gate wide/narrow funnel
# (funnel_config.py). All cell values are precomputed in python, so there are no
# spreadsheet formulas to recalc.
#
# Usage:
#   bash workspace/core4d/scripts/eval/wrappers/gen_E215_funnel_xlsx.sh [--out PATH]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
exec "$PYTHON_BIN" workspace/core4d/scripts/eval/reports/gen_E215_rot_vs_orig_funnel_xlsx.py "$@"
