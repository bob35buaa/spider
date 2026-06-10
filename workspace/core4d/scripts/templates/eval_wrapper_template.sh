#!/usr/bin/env bash
# {{EXP_ID}} evaluation wrapper: {{DESCRIPTION}}
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
"${PYTHON_BIN}" workspace/core4d/scripts/eval/eval_{{EXP_ID}}_{{EXP_SLUG}}.py "$@"
