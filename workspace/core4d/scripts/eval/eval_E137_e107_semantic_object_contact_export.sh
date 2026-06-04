#!/usr/bin/env bash
# E137: write isolated E107 qpos-style semantic object_contact candidates.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
EXPORT_PY="${REPO_ROOT}/workspace/core4d/scripts/E137/export_e107_semantic_object_contact.py"

if [ ! -f "${EXPORT_PY}" ]; then
    echo "ERROR: missing E137 exporter: ${EXPORT_PY}" >&2
    exit 1
fi

"${PYTHON_BIN}" "${EXPORT_PY}"
