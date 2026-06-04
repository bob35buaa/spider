#!/usr/bin/env bash
# E129: no-GPU main box021_029_p2 carry-state constraint audit.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/workspace/core4d/scripts/E129/audit_main_carry_state.py"
