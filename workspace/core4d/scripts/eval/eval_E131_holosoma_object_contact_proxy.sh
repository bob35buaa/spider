#!/usr/bin/env bash
# E131: no-GPU object_contact proxy export for Holosoma fragment inspection.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/workspace/core4d/scripts/E131/add_holosoma_object_contact_proxy.py"
