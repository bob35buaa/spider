#!/usr/bin/env bash
# E142: join E110 OmniRetarget contact metrics with E112/E113 contact-aware candidates.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python3"
fi

E110_BAND="${REPO_ROOT}/workspace/core4d/results/E110/contact_metric_audit/contact_band_metrics.tsv"
E112_FULL="${REPO_ROOT}/workspace/core4d/results/E112/cem/full/full_eval_summary.csv"
E113_FULL="${REPO_ROOT}/workspace/core4d/results/E113/cem/full/full_method_metrics.csv"
OUT_DIR="${REPO_ROOT}/workspace/core4d/results/E142/omniretarget_contact_exceedance_audit"
SCRIPT="${REPO_ROOT}/workspace/core4d/scripts/E142/audit_omniretarget_contact_exceedance.py"

for f in "${SCRIPT}" "${E110_BAND}" "${E112_FULL}" "${E113_FULL}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: missing required file: ${f}" >&2
        exit 1
    fi
done

"${PYTHON_BIN}" "${SCRIPT}" \
    --e110-band "${E110_BAND}" \
    --e112-full "${E112_FULL}" \
    --e113-full "${E113_FULL}" \
    --out-dir "${OUT_DIR}"
