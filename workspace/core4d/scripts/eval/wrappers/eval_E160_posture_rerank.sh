#!/usr/bin/env bash
# Fixed entry for E160 gateA + surfaceBand-A2 + postureRerankA 3-case evaluation.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
"${PYTHON_BIN}" workspace/core4d/scripts/eval/eval_E160_posture_rerank.py "$@"
