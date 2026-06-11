#!/usr/bin/env bash
# E132: CPU-only Holosoma MotionLoader object_contact runtime contract probe.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
HOLOSOMA_ROOT="${HOLOSOMA_ROOT:-/home/ubuntu/Workspace/holosoma}"
PROBE_PY="${REPO_ROOT}/workspace/core4d/scripts/E132/probe_holosoma_motionloader_object_contact.py"

if [ ! -f "${HOLOSOMA_ROOT}/scripts/source_isaacsim_setup.sh" ]; then
    echo "ERROR: missing Holosoma setup script: ${HOLOSOMA_ROOT}/scripts/source_isaacsim_setup.sh" >&2
    exit 1
fi
if [ ! -f "${PROBE_PY}" ]; then
    echo "ERROR: missing E132 probe script: ${PROBE_PY}" >&2
    exit 1
fi

(
    cd "${HOLOSOMA_ROOT}"
    # shellcheck disable=SC1091
    source "${HOLOSOMA_ROOT}/scripts/source_isaacsim_setup.sh"
    export PYTHONPATH="${HOLOSOMA_ROOT}/src/holosoma:${HOLOSOMA_ROOT}/src/holosoma_retargeting:${PYTHONPATH:-}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export PYTHONUNBUFFERED=1
    CUDA_VISIBLE_DEVICES="" python "${PROBE_PY}"
)
