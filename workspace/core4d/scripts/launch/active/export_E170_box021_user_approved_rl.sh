#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

cd "${REPO_ROOT}"
.venv/bin/python workspace/core4d/scripts/experiments/E170/export_box021_user_approved_rl.py "$@"
.venv/bin/python workspace/core4d/scripts/experiments/E170/audit_box021_user_approved_rl.py
