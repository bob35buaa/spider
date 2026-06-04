#!/usr/bin/env bash
# E128: bounded Holosoma replay.py startup preflight for E126 fragment exports.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
HOLOSOMA_ROOT="${HOLOSOMA_ROOT:-/home/ubuntu/Workspace/holosoma}"
GPU_ID="${GPU_ID:-0}"
TIMEOUT_SECONDS="${E128_TIMEOUT_SECONDS:-300}"
MAX_STEPS="${E128_MAX_STEPS:-20}"
OUT_DIR="${REPO_ROOT}/workspace/core4d/results/E128/holosoma_runtime_startup_preflight"
LOG_DIR="${OUT_DIR}/logs"
MANIFEST="${OUT_DIR}/e128_runtime_startup_manifest.tsv"
SUMMARY_JSON="${OUT_DIR}/e128_runtime_startup_summary.json"
SUMMARY_MD="${OUT_DIR}/e128_runtime_startup_summary.md"

EXP_ALIAS="exp:g1-29dof-wbt-w-object-r135-box021-handbox-exp0601-v4-3"
OBJECT_URDF="${HOLOSOMA_ROOT}/src/holosoma_retargeting/holosoma_retargeting/models/Box021/Box021.urdf"
REPLAY_PY="${HOLOSOMA_ROOT}/src/holosoma/holosoma/replay.py"
PROBE_PY="${REPO_ROOT}/workspace/core4d/scripts/E128/holosoma_replay_no_debug_probe.py"

MOTION_P1="${REPO_ROOT}/workspace/core4d/results/E126/holosoma_fragment_adapter_preflight/exports/E126_box021_035_p1_with_partner_box021_035_p2_mj_w_obj_w_partner.npz"
MOTION_P2="${REPO_ROOT}/workspace/core4d/results/E126/holosoma_fragment_adapter_preflight/exports/E126_box021_035_p2_with_partner_box021_035_p1_mj_w_obj_w_partner.npz"

mkdir -p "${LOG_DIR}"

for f in "${REPLAY_PY}" "${PROBE_PY}" "${OBJECT_URDF}" "${MOTION_P1}" "${MOTION_P2}" "${HOLOSOMA_ROOT}/scripts/source_isaacsim_setup.sh"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: missing required file: ${f}" >&2
        exit 1
    fi
done

printf "case_id\tpartner_case_id\tmotion_path\tgpu_id\ttimeout_seconds\tmax_steps\treturn_code\ttimeout_status\tlog_path\ttraining_launched\trl_smoke_allowed\trl_ready\tstartup_status\tfailure_mode\tnotes\n" > "${MANIFEST}"

run_one() {
    local case_id="$1"
    local partner_case_id="$2"
    local motion_path="$3"
    local log_path="${LOG_DIR}/${case_id}_runtime_startup.log"
    local rc=0
    local timeout_status="no"
    local status="pass"
    local failure_mode=""
    local notes="no-debug replay startup probe completed; fragment-only label still blocks RL smoke"

    set +e
    (
        cd "${HOLOSOMA_ROOT}"
        export VIRTUAL_ENV="${VIRTUAL_ENV:-}"
        # shellcheck disable=SC1091
        source "${HOLOSOMA_ROOT}/scripts/source_isaacsim_setup.sh"
        export PYTHONPATH="${HOLOSOMA_ROOT}/src/holosoma:${HOLOSOMA_ROOT}/src/holosoma_retargeting:${PYTHONPATH:-}"
        export WANDB_MODE="${WANDB_MODE:-offline}"
        export PYTHONUNBUFFERED=1
        export E128_MAX_STEPS="${MAX_STEPS}"
        echo "E128 runtime startup: ${case_id} with partner ${partner_case_id}"
        echo "motion=${motion_path}"
        echo "object=${OBJECT_URDF}"
        echo "gpu=${GPU_ID}"
        echo "max_steps=${E128_MAX_STEPS}"
        CUDA_VISIBLE_DEVICES="${GPU_ID}" timeout --kill-after=30s "${TIMEOUT_SECONDS}s" \
            python "${PROBE_PY}" \
                "${EXP_ALIAS}" \
                simulator:isaacsim \
                logger:disabled \
                --training.headless=True \
                --training.num-envs=1 \
                --command.setup_terms.motion_command.params.motion_config.motion_file="${motion_path}" \
                --robot.object.object_urdf_path="${OBJECT_URDF}" \
                --simulator.config.debug_viz=False \
                --simulator.config.scene.env_spacing=2.0
    ) >"${log_path}" 2>&1
    rc=$?
    set -e

    if [ "${rc}" -eq 124 ] || [ "${rc}" -eq 137 ]; then
        timeout_status="yes"
        status="fail"
        failure_mode="timeout"
        notes="no-debug replay probe did not finish before timeout"
    elif [ "${rc}" -ne 0 ]; then
        status="fail"
        failure_mode="nonzero_exit"
        notes="no-debug replay probe exited non-zero"
    elif grep -Eq "Traceback|AttributeError|RuntimeError|AssertionError|Exception" "${log_path}"; then
        status="fail"
        failure_mode="runtime_exception_in_log"
        notes="no-debug replay probe log contains an exception despite zero return code"
    elif ! grep -Fq "${motion_path}" "${log_path}"; then
        status="fail"
        failure_mode="motion_path_not_in_log"
        notes="startup log did not contain expected motion path"
    elif ! grep -Fq "E128_PROBE_LOADED has_object=True has_partner=True" "${log_path}"; then
        status="fail"
        failure_mode="probe_missing_object_or_partner"
        notes="probe did not confirm object and partner motion loading"
    elif ! grep -Fq "E128_PROBE_DONE" "${log_path}"; then
        status="fail"
        failure_mode="probe_missing_done_marker"
        notes="probe did not emit completion marker"
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tfalse\tfalse\tfalse\t%s\t%s\t%s\n" \
        "${case_id}" "${partner_case_id}" "${motion_path}" "${GPU_ID}" "${TIMEOUT_SECONDS}" "${MAX_STEPS}" "${rc}" \
        "${timeout_status}" "${log_path}" "${status}" "${failure_mode}" "${notes}" >> "${MANIFEST}"
}

run_one "box021_035_p1" "box021_035_p2" "${MOTION_P1}"
run_one "box021_035_p2" "box021_035_p1" "${MOTION_P2}"

"${REPO_ROOT}/.venv/bin/python" - "${MANIFEST}" "${SUMMARY_JSON}" "${SUMMARY_MD}" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
summary_json = Path(sys.argv[2])
summary_md = Path(sys.argv[3])
rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8"), delimiter="\t"))
pass_rows = [row for row in rows if row["startup_status"] == "pass"]
summary = {
    "experiment": "E128",
    "rows": len(rows),
    "startup_pass_rows": len(pass_rows),
    "rl_smoke_allowed_rows": 0,
    "rl_ready_rows": 0,
    "training_launched": False,
    "status": "pass" if len(pass_rows) == len(rows) else "fail",
}
summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = [
    "# E128 Holosoma Runtime Startup Preflight Summary",
    "",
    f"- rows: `{summary['rows']}`",
    f"- startup pass rows: `{summary['startup_pass_rows']}`",
    f"- RL smoke allowed rows: `{summary['rl_smoke_allowed_rows']}`",
    f"- RL-ready rows: `{summary['rl_ready_rows']}`",
    f"- training launched: `{summary['training_launched']}`",
    f"- status: `{summary['status']}`",
    "",
    "| case | partner | startup | return code | timeout | log | notes |",
    "|---|---|---|---:|---|---|---|",
]
for row in rows:
    lines.append(
        f"| `{row['case_id']}` | `{row['partner_case_id']}` | `{row['startup_status']}` | "
        f"{row['return_code']} | `{row['timeout_status']}` | `{row['log_path']}` | {row['notes']} |"
    )
lines.extend(
    [
        "",
        "Interpretation: E128 is a bounded runtime startup gate only. "
        "Fragment-only labels still block PPO smoke, RL-ready handoff, and release claims.",
        "",
    ]
)
summary_md.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {summary_md} rows={summary['rows']} status={summary['status']}")
if summary["status"] != "pass":
    raise SystemExit(1)
PY
