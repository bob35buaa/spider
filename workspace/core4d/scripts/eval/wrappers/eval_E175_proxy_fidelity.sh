#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

exec "${PYTHON_BIN}" \
  workspace/core4d/scripts/eval/runners/eval_E175_proxy_fidelity.py \
  "$@"
