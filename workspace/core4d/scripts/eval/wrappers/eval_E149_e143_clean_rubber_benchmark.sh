#!/usr/bin/env bash
# Fixed entry for E149 clean benchmark view of E148 rubber hand comparison.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

.venv/bin/python workspace/core4d/scripts/eval/eval_E149_e143_clean_rubber_benchmark.py
