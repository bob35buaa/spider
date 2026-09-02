#!/usr/bin/env bash
# E202-export (plan235): USE13 bucket translation-aug -> partner-paired RL-ready assets.
# Steps: partner aug build (2 new / 10 reuse) -> C2 object-traj parity -> paired RL export
#        -> C6 USE13 stratified eval. Holosoma downstream export is a separate step (see MODE=holosoma).
#
# Env notes (this node): e202_common import is ~5-6 min (mujoco+scipy+geometry, thread
# oversubscription). Cap BLAS threads and do NOT set MUJOCO_GL (osmesa can stall import).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
PY="${REPO}/.venv/bin/python"
EXP_DIR="${REPO}/workspace/core4d/scripts/experiments/E202"
MODE="${1:-all}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1

step_partner() { cd "${EXP_DIR}"; "${PY}" build_partner_aug.py --max-workers "${MAX_WORKERS:-2}" "$@"; }
step_parity()  { cd "${EXP_DIR}"; "${PY}" check_object_traj_parity.py "$@"; }
step_export()  { cd "${EXP_DIR}"; "${PY}" export_aug_partner_rl.py "$@"; }
step_eval()    { cd "${REPO}";    "${PY}" workspace/core4d/scripts/eval/reports/gen_E202_export_use13_stratified.py; }

case "${MODE}" in
    partner)  step_partner "${@:2}" ;;
    parity)   step_parity  "${@:2}" ;;
    export)   step_export  "${@:2}" ;;
    eval)     step_eval ;;
    all)
        step_partner
        step_parity
        step_export
        step_eval
        ;;
    *) echo "Usage: $0 [all|partner|parity|export|eval]" >&2; exit 2 ;;
esac
