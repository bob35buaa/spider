#!/usr/bin/env bash
# E138: convert E137 semantic object_contact exports into Holosoma WBT format and probe MotionLoader.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
HOLOSOMA_ROOT="${HOLOSOMA_ROOT:-/home/ubuntu/Workspace/holosoma}"
SCRIPT="${REPO_ROOT}/workspace/core4d/scripts/E138/convert_e137_semantic_contact_to_holosoma.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python3"
fi
if [ ! -f "${SCRIPT}" ]; then
    echo "ERROR: missing E138 script: ${SCRIPT}" >&2
    exit 1
fi
if [ ! -f "${HOLOSOMA_ROOT}/scripts/source_isaacsim_setup.sh" ]; then
    echo "ERROR: missing Holosoma setup script: ${HOLOSOMA_ROOT}/scripts/source_isaacsim_setup.sh" >&2
    exit 1
fi

"${PYTHON_BIN}" "${SCRIPT}" --stage convert --input-fps 30 --output-fps 50

(
    cd "${HOLOSOMA_ROOT}"
    # shellcheck disable=SC1091
    source "${HOLOSOMA_ROOT}/scripts/source_isaacsim_setup.sh"
    export PYTHONPATH="${HOLOSOMA_ROOT}/src/holosoma:${HOLOSOMA_ROOT}/src/holosoma_retargeting:${PYTHONPATH:-}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export PYTHONUNBUFFERED=1
    CUDA_VISIBLE_DEVICES="" python "${SCRIPT}" --stage motionloader
)
