#!/usr/bin/env bash
# E185 CPU-only offline re-aggregation. No SDF, CoACD, GPU, or Full CEM access.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RUNNER="workspace/core4d/scripts/experiments/E185/reaggregate_gate060.py"

"${PYTHON_BIN}" "${RUNNER}" protocol
"${PYTHON_BIN}" "${RUNNER}" aggregate
"${PYTHON_BIN}" "${RUNNER}" visual
"${PYTHON_BIN}" "${RUNNER}" validate
