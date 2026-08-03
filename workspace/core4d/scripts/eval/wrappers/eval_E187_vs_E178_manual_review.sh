#!/usr/bin/env bash
# Read-only E187 vs E178 manual-review comparison on frozen keep22.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
[ -x "$PYTHON_BIN" ] || {
  echo "missing project python: $PYTHON_BIN" >&2
  exit 2
}

exec "$PYTHON_BIN" \
  workspace/core4d/scripts/eval/reports/gen_E187_vs_E178_manual_review_comparison.py
