#!/usr/bin/env bash
# E127: no-GPU Holosoma training-contract preflight for E126 fragment exports.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

"${PYTHON_BIN}" "${REPO_ROOT}/workspace/core4d/scripts/E127/check_holosoma_training_contract.py" "$@"
