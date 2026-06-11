#!/usr/bin/env bash
# E133: bounded Holosoma env probe for motion_command.ref_object_contact.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
HOLOSOMA_ROOT="${HOLOSOMA_ROOT:-/home/ubuntu/Workspace/holosoma}"
GPU_ID="${GPU_ID:-0}"
TIMEOUT_SECONDS="${E133_TIMEOUT_SECONDS:-600}"
MAX_STEPS="${E133_MAX_STEPS:-250}"
OUT_DIR="${REPO_ROOT}/workspace/core4d/results/E133/holosoma_ref_object_contact_env_probe"
LOG_DIR="${OUT_DIR}/logs"
MANIFEST="${OUT_DIR}/e133_ref_object_contact_manifest.tsv"
SUMMARY_JSON="${OUT_DIR}/e133_ref_object_contact_summary.json"
SUMMARY_MD="${OUT_DIR}/e133_ref_object_contact_summary.md"

EXP_ALIAS="exp:g1-29dof-wbt-w-object-r135-box021-handbox-exp0601-v4-3"
OBJECT_URDF="${HOLOSOMA_ROOT}/src/holosoma_retargeting/holosoma_retargeting/models/Box021/Box021.urdf"
PROBE_PY="${REPO_ROOT}/workspace/core4d/scripts/E133/holosoma_ref_object_contact_no_debug_probe.py"

MOTION_P1="${REPO_ROOT}/workspace/core4d/results/E131/holosoma_object_contact_proxy/exports/E126_box021_035_p1_with_partner_box021_035_p2_object_contact_proxy5cm.npz"
MOTION_P2="${REPO_ROOT}/workspace/core4d/results/E131/holosoma_object_contact_proxy/exports/E126_box021_035_p2_with_partner_box021_035_p1_object_contact_proxy5cm.npz"

mkdir -p "${LOG_DIR}"

for f in "${PROBE_PY}" "${OBJECT_URDF}" "${MOTION_P1}" "${MOTION_P2}" "${HOLOSOMA_ROOT}/scripts/source_isaacsim_setup.sh"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: missing required file: ${f}" >&2
        exit 1
    fi
done

run_one() {
    local case_id="$1"
    local partner_case_id="$2"
    local motion_path="$3"
    local log_path="${LOG_DIR}/${case_id}_ref_object_contact.log"
    local rc=0

    set +e
    (
        cd "${HOLOSOMA_ROOT}"
        export VIRTUAL_ENV="${VIRTUAL_ENV:-}"
        # shellcheck disable=SC1091
        source "${HOLOSOMA_ROOT}/scripts/source_isaacsim_setup.sh"
        export PYTHONPATH="${HOLOSOMA_ROOT}/src/holosoma:${HOLOSOMA_ROOT}/src/holosoma_retargeting:${PYTHONPATH:-}"
        export WANDB_MODE="${WANDB_MODE:-offline}"
        export PYTHONUNBUFFERED=1
        export E133_MAX_STEPS="${MAX_STEPS}"
        export E133_DISABLE_REPLAY_SLEEP="${E133_DISABLE_REPLAY_SLEEP:-1}"
        echo "E133 ref object contact env probe: ${case_id} with partner ${partner_case_id}"
        echo "motion=${motion_path}"
        echo "object=${OBJECT_URDF}"
        echo "gpu=${GPU_ID}"
        echo "max_steps=${E133_MAX_STEPS}"
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
    printf "%s\t%s\t%s\t%s\t%s\n" "${case_id}" "${partner_case_id}" "${motion_path}" "${log_path}" "${rc}"
}

TMP_RUNS="${OUT_DIR}/e133_raw_run_rows.tsv"
{
    printf "case_id\tpartner_case_id\tmotion_path\tlog_path\treturn_code\n"
    run_one "box021_035_p1" "box021_035_p2" "${MOTION_P1}"
    run_one "box021_035_p2" "box021_035_p1" "${MOTION_P2}"
} > "${TMP_RUNS}"

"${REPO_ROOT}/.venv/bin/python" - "${TMP_RUNS}" "${MANIFEST}" "${SUMMARY_JSON}" "${SUMMARY_MD}" "${MAX_STEPS}" "${TIMEOUT_SECONDS}" "${GPU_ID}" <<'PY'
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

raw_runs = Path(sys.argv[1])
manifest = Path(sys.argv[2])
summary_json = Path(sys.argv[3])
summary_md = Path(sys.argv[4])
max_steps = int(sys.argv[5])
timeout_seconds = int(sys.argv[6])
gpu_id = sys.argv[7]

fields = [
    "case_id",
    "partner_case_id",
    "motion_path",
    "gpu_id",
    "timeout_seconds",
    "max_steps",
    "return_code",
    "timeout_status",
    "log_path",
    "has_object",
    "has_partner",
    "has_object_contact",
    "motion_contact_total",
    "motion_left_active",
    "motion_right_active",
    "motion_both_active",
    "ref_contact_total",
    "ref_left_active",
    "ref_right_active",
    "ref_both_active",
    "ref_either_active",
    "observed_time_step_min",
    "observed_time_step_max",
    "startup_status",
    "ref_contact_status",
    "semantic_ref_mask_ready",
    "rl_ready",
    "training_launched",
    "cem_launched",
    "remote_jobs_launched",
    "failure_mode",
    "notes",
]


def extract_summary(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"^E133_PROBE_SUMMARY (\{.*\})$", text, flags=re.MULTILINE)
    if not matches:
        return {}
    return json.loads(matches[-1])


rows = []
for run in csv.DictReader(raw_runs.open(newline="", encoding="utf-8"), delimiter="\t"):
    log_path = Path(run["log_path"])
    rc = int(run["return_code"])
    timeout_status = "yes" if rc in {124, 137} else "no"
    failure_mode = ""
    notes = "bounded no-debug env stepping probe completed; proxy mask remains non-semantic"
    startup_status = "pass"
    ref_contact_status = "diagnostic"
    summary = extract_summary(log_path) if log_path.exists() else {}
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    if timeout_status == "yes":
        startup_status = "fail"
        failure_mode = "timeout"
        notes = "probe timed out"
    elif rc != 0:
        startup_status = "fail"
        failure_mode = "nonzero_exit"
        notes = "probe exited non-zero"
    elif re.search(r"Traceback|AttributeError|RuntimeError|AssertionError", text):
        startup_status = "fail"
        failure_mode = "runtime_error_marker_in_log"
        notes = "probe log contains runtime error marker"
    elif not summary:
        startup_status = "fail"
        failure_mode = "missing_summary_marker"
        notes = "probe did not emit E133_PROBE_SUMMARY"
    elif not (summary.get("has_object") and summary.get("has_partner") and summary.get("has_object_contact")):
        startup_status = "fail"
        failure_mode = "missing_object_partner_or_contact"
        notes = "probe did not confirm object, partner, and object_contact"
    elif int(summary.get("motion_contact_total", 0)) <= 0:
        startup_status = "fail"
        failure_mode = "zero_motion_contact"
        notes = "full motion object_contact is zero"
    if summary and int(summary.get("ref_contact_total", 0)) > 0:
        ref_contact_status = "pass"
    elif summary:
        ref_contact_status = "zero_ref_contact"
    row = {
        "case_id": run["case_id"],
        "partner_case_id": run["partner_case_id"],
        "motion_path": run["motion_path"],
        "gpu_id": gpu_id,
        "timeout_seconds": timeout_seconds,
        "max_steps": max_steps,
        "return_code": rc,
        "timeout_status": timeout_status,
        "log_path": run["log_path"],
        "has_object": str(bool(summary.get("has_object", False))).lower(),
        "has_partner": str(bool(summary.get("has_partner", False))).lower(),
        "has_object_contact": str(bool(summary.get("has_object_contact", False))).lower(),
        "motion_contact_total": int(summary.get("motion_contact_total", 0)),
        "motion_left_active": int(summary.get("motion_left_active", 0)),
        "motion_right_active": int(summary.get("motion_right_active", 0)),
        "motion_both_active": int(summary.get("motion_both_active", 0)),
        "ref_contact_total": int(summary.get("ref_contact_total", 0)),
        "ref_left_active": int(summary.get("ref_left_active", 0)),
        "ref_right_active": int(summary.get("ref_right_active", 0)),
        "ref_both_active": int(summary.get("ref_both_active", 0)),
        "ref_either_active": int(summary.get("ref_either_active", 0)),
        "observed_time_step_min": summary.get("observed_time_step_min", ""),
        "observed_time_step_max": summary.get("observed_time_step_max", ""),
        "startup_status": startup_status,
        "ref_contact_status": ref_contact_status,
        "semantic_ref_mask_ready": "false",
        "rl_ready": "false",
        "training_launched": "false",
        "cem_launched": "false",
        "remote_jobs_launched": "false",
        "failure_mode": failure_mode,
        "notes": notes,
    }
    rows.append(row)

p2_rows = [row for row in rows if row["case_id"] == "box021_035_p2"]
startup_pass_rows = [row for row in rows if row["startup_status"] == "pass"]
p2_ref_positive = bool(p2_rows and int(p2_rows[0]["ref_contact_total"]) > 0)
status = "pass" if len(startup_pass_rows) == len(rows) and p2_ref_positive else "fail"
summary = {
    "experiment": "E133",
    "status": status,
    "rows": len(rows),
    "startup_pass_rows": len(startup_pass_rows),
    "p2_ref_positive": p2_ref_positive,
    "ref_positive_rows": sum(1 for row in rows if int(row["ref_contact_total"]) > 0),
    "structural_ref_mask_runtime_ready_rows": len(startup_pass_rows),
    "semantic_ref_mask_ready_rows": 0,
    "rl_ready_rows": 0,
    "training_launched": False,
    "cem_launched": False,
    "remote_jobs_launched": False,
    "notes": [
        "E133 proves runtime command plumbing for E131 geometry proxy object_contact masks.",
        "E133 does not prove raw/trimmed semantic contact, reward value quality, PPO readiness, or main-case release readiness.",
    ],
}

with manifest.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})

summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = [
    "# E133 Holosoma Ref-Object-Contact Env Probe Summary",
    "",
    f"- rows: `{summary['rows']}`",
    f"- startup pass rows: `{summary['startup_pass_rows']}`",
    f"- ref-positive rows: `{summary['ref_positive_rows']}`",
    f"- p2 ref positive: `{summary['p2_ref_positive']}`",
    "- semantic ref-mask ready rows: `0`",
    "- RL-ready rows: `0`",
    "- training launched: `false`",
    "- CEM launched: `false`",
    f"- status: `{summary['status']}`",
    "",
    "| case | startup | has object contact | motion total | ref total | ref either | observed step max | log | notes |",
    "|---|---|---|---:|---:|---:|---:|---|---|",
]
for row in rows:
    lines.append(
        f"| `{row['case_id']}` | `{row['startup_status']}` | `{row['has_object_contact']}` | "
        f"{row['motion_contact_total']} | {row['ref_contact_total']} | {row['ref_either_active']} | "
        f"{row['observed_time_step_max']} | `{row['log_path']}` | {row['notes']} |"
    )
lines.extend(
    [
        "",
        "Interpretation: E133 is a bounded runtime plumbing probe. The observed masks are E131 geometry proxies, not semantic raw-contact masks, so PPO/RL and main release remain blocked.",
        "",
    ]
)
summary_md.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {summary_md} rows={summary['rows']} status={summary['status']}")
if status != "pass":
    raise SystemExit(1)
PY
