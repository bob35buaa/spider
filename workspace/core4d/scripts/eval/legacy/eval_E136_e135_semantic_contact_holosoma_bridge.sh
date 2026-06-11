#!/usr/bin/env bash
# E136: audit E135 raw-contact masks against Holosoma export time axes.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
AUDIT_PY="${REPO_ROOT}/workspace/core4d/scripts/E136/audit_e135_semantic_contact_holosoma_bridge.py"

if [ ! -f "${AUDIT_PY}" ]; then
    echo "ERROR: missing E136 audit script: ${AUDIT_PY}" >&2
    exit 1
fi

"${PYTHON_BIN}" "${AUDIT_PY}"
