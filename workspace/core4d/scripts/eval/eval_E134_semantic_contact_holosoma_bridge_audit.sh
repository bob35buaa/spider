#!/usr/bin/env bash
# E134: read-only semantic contact to Holosoma bridge audit.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
PROBE_PY="${REPO_ROOT}/workspace/core4d/scripts/E134/audit_semantic_contact_holosoma_bridge.py"

if [ ! -f "${PROBE_PY}" ]; then
    echo "ERROR: missing E134 audit script: ${PROBE_PY}" >&2
    exit 1
fi

"${PYTHON_BIN}" "${PROBE_PY}"
