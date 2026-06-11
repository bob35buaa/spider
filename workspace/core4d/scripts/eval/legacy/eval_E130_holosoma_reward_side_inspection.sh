#!/usr/bin/env bash
# E130: no-GPU Holosoma reward-side inspection for E126 paired fragments.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/workspace/core4d/scripts/E130/inspect_holosoma_reward_side.py"
