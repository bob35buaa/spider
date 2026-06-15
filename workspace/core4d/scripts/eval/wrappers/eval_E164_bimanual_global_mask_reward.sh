#!/usr/bin/env bash
# Fixed entry for E164 bimanual global mask/reward evaluation.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
"${PYTHON_BIN}" workspace/core4d/scripts/eval/runners/eval_E164_bimanual_global_mask_reward.py "$@"
