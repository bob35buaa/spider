#!/usr/bin/env bash
# E140: static Holosoma semantic ref-mask reward/config readiness audit.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
SCRIPT="${REPO_ROOT}/workspace/core4d/scripts/E140/audit_holosoma_semantic_ref_mask_reward_readiness.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python3"
fi
if [ ! -f "${SCRIPT}" ]; then
    echo "ERROR: missing E140 audit script: ${SCRIPT}" >&2
    exit 1
fi

"${PYTHON_BIN}" "${SCRIPT}"
