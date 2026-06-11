#!/usr/bin/env bash
# E141: bounded Holosoma semantic ref-mask reward probe. No PPO/CEM.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
HOLOSOMA_ROOT="${HOLOSOMA_ROOT:-/home/ubuntu/Workspace/holosoma}"
GPU_ID="${GPU_ID:-0}"
TIMEOUT_SECONDS="${E141_TIMEOUT_SECONDS:-600}"
MAX_STEPS="${E141_MAX_STEPS:-250}"
OUT_DIR="${REPO_ROOT}/workspace/core4d/results/E141/box021_semantic_ref_mask_reward_probe"
LOG_DIR="${OUT_DIR}/logs"
E139_OUT="${REPO_ROOT}/workspace/core4d/results/E139/e138_semantic_ref_object_contact_env_probe"
PARTNER_DIR="${E139_OUT}/partner_semantic_motions"
ROWS_TSV="${OUT_DIR}/e141_reward_probe_rows.tsv"
SUMMARY_JSON="${OUT_DIR}/e141_reward_probe_summary.json"
SUMMARY_MD="${OUT_DIR}/e141_reward_probe_summary.md"
RAW_RUNS="${OUT_DIR}/e141_raw_run_rows.tsv"

EXP_ALIAS="exp:g1-29dof-wbt-w-object-e141-box021-semantic-refmask-v4-3"
OBJECT_URDF="${HOLOSOMA_ROOT}/src/holosoma_retargeting/holosoma_retargeting/models/Box021/Box021.urdf"
PROBE_PY="${REPO_ROOT}/workspace/core4d/scripts/E141/holosoma_semantic_ref_mask_reward_probe.py"
PREPARE_PY="${REPO_ROOT}/workspace/core4d/scripts/E139/build_e138_partner_semantic_motions.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

mkdir -p "${LOG_DIR}"

MOTIONS=(
    "box021_029_p1:${PARTNER_DIR}/box021_029_p1_with_partner_box021_029_p2_e138_semantic_mj_w_obj_w_partner.npz"
    "box021_029_p2:${PARTNER_DIR}/box021_029_p2_with_partner_box021_029_p1_e138_semantic_mj_w_obj_w_partner.npz"
    "box021_035_p1:${PARTNER_DIR}/box021_035_p1_with_partner_box021_035_p2_e138_semantic_mj_w_obj_w_partner.npz"
    "box021_035_p2:${PARTNER_DIR}/box021_035_p2_with_partner_box021_035_p1_e138_semantic_mj_w_obj_w_partner.npz"
)

if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python3"
fi

for f in "${PROBE_PY}" "${PREPARE_PY}" "${OBJECT_URDF}" "${HOLOSOMA_ROOT}/scripts/source_isaacsim_setup.sh"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: missing required file: ${f}" >&2
        exit 1
    fi
done

for item in "${MOTIONS[@]}"; do
    motion_path="${item#*:}"
    if [ ! -f "${motion_path}" ]; then
        "${PYTHON_BIN}" "${PREPARE_PY}"
        break
    fi
done

for item in "${MOTIONS[@]}"; do
    motion_path="${item#*:}"
    if [ ! -f "${motion_path}" ]; then
        echo "ERROR: missing E139 partner semantic motion: ${motion_path}" >&2
        exit 1
    fi
done

run_one() {
    local case_id="$1"
    local motion_path="$2"
    local log_path="${LOG_DIR}/${case_id}_semantic_ref_mask_reward.log"
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
        export E141_MAX_STEPS="${MAX_STEPS}"
        export E141_DISABLE_REPLAY_SLEEP="${E141_DISABLE_REPLAY_SLEEP:-1}"
        echo "E141 semantic ref-mask reward probe: ${case_id}"
        echo "motion=${motion_path}"
        echo "object=${OBJECT_URDF}"
        echo "gpu=${GPU_ID}"
        echo "max_steps=${E141_MAX_STEPS}"
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
    printf "%s\t%s\t%s\t%s\n" "${case_id}" "${motion_path}" "${log_path}" "${rc}"
}

{
    printf "case_id\tmotion_path\tlog_path\treturn_code\n"
    for item in "${MOTIONS[@]}"; do
        run_one "${item%%:*}" "${item#*:}"
    done
} > "${RAW_RUNS}"

"${REPO_ROOT}/.venv/bin/python" - "${RAW_RUNS}" "${ROWS_TSV}" "${SUMMARY_JSON}" "${SUMMARY_MD}" "${MAX_STEPS}" "${TIMEOUT_SECONDS}" "${GPU_ID}" <<'PY'
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

raw_runs = Path(sys.argv[1])
rows_tsv = Path(sys.argv[2])
summary_json = Path(sys.argv[3])
summary_md = Path(sys.argv[4])
max_steps = int(sys.argv[5])
timeout_seconds = int(sys.argv[6])
gpu_id = sys.argv[7]

fields = [
    "case_id",
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
    "active_reward_terms",
    "motion_shape",
    "motion_contact_total",
    "ref_shape",
    "ref_contact_total",
    "ref_either_active",
    "ref_both_active",
    "reward_compute_steps",
    "r099_ref_contact_frac_max",
    "r099_ref_contact_frac_mean",
    "r099_ref_masked_hand_contact_reward_mean_max",
    "r099_ref_masked_hand_contact_reward_mean_mean",
    "r099_ref_masked_two_hand_contact_reward_mean_max",
    "r099_ref_masked_two_hand_contact_reward_mean_mean",
    "r099_hand_gate_ref_true_mean",
    "r099_two_hand_gate_ref_both_mean",
    "reward_keys_present",
    "reward_keys_finite",
    "ref_contact_frac_positive",
    "ref_mask_reward_positive",
    "startup_status",
    "reward_probe_status",
    "ppo_smoke_allowed",
    "rl_ready",
    "training_launched",
    "cem_launched",
    "remote_jobs_launched",
    "failure_mode",
    "notes",
]


def extract_summary(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"^E141_PROBE_SUMMARY (\{.*\})$", text, flags=re.MULTILINE)
    if not matches:
        return {}
    return json.loads(matches[-1])


def val(summary: dict, key: str, subkey: str, default=None):
    item = summary.get(key)
    if isinstance(item, dict):
        return item.get(subkey, default)
    return default


rows = []
for run in csv.DictReader(raw_runs.open(newline="", encoding="utf-8"), delimiter="\t"):
    log_path = Path(run["log_path"])
    rc = int(run["return_code"])
    timeout_status = "yes" if rc in {124, 137} else "no"
    summary = extract_summary(log_path) if log_path.exists() else {}
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""

    startup_status = "pass"
    reward_probe_status = "pass"
    failure_mode = ""
    notes = "bounded no-PPO reference replay plus explicit reward_manager.compute"
    if timeout_status == "yes":
        startup_status = "fail"
        reward_probe_status = "not_run"
        failure_mode = "timeout"
        notes = "probe timed out"
    elif rc != 0:
        startup_status = "fail"
        reward_probe_status = "not_run"
        failure_mode = "nonzero_exit"
        notes = "probe exited non-zero"
    elif re.search(r"Traceback|AttributeError|RuntimeError|AssertionError", text):
        startup_status = "fail"
        reward_probe_status = "not_run"
        failure_mode = "runtime_error_marker_in_log"
        notes = "probe log contains runtime error marker"
    elif not summary:
        startup_status = "fail"
        reward_probe_status = "not_run"
        failure_mode = "missing_summary_marker"
        notes = "probe did not emit E141_PROBE_SUMMARY"
    elif not (summary.get("has_object") and summary.get("has_partner") and summary.get("has_object_contact")):
        startup_status = "fail"
        reward_probe_status = "not_run"
        failure_mode = "missing_object_partner_or_contact"
        notes = "probe did not confirm object, partner, and object_contact"
    elif not summary.get("reward_keys_present"):
        reward_probe_status = "fail"
        failure_mode = "missing_reward_log_key"
        notes = "expected r099 ref-mask reward log keys were not all present"
    elif not summary.get("reward_keys_finite"):
        reward_probe_status = "fail"
        failure_mode = "nonfinite_reward_log_key"
        notes = "expected r099 ref-mask reward log keys were not finite"
    elif not summary.get("ref_contact_frac_positive"):
        reward_probe_status = "fail"
        failure_mode = "zero_ref_contact_frac"
        notes = "bounded probe did not observe positive ref_contact_frac"
    elif not summary.get("ref_mask_reward_positive"):
        reward_probe_status = "zero_reward_signal"
        failure_mode = "zero_ref_mask_reward"
        notes = "reward keys are finite and ref mask is positive, but actual ref-mask contact reward is zero"

    ppo_smoke_allowed = startup_status == "pass" and reward_probe_status == "pass"
    active_terms = ",".join(summary.get("active_reward_terms", [])) if summary else ""
    row = {
        "case_id": run["case_id"],
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
        "active_reward_terms": active_terms,
        "motion_shape": summary.get("motion_shape", ""),
        "motion_contact_total": summary.get("motion_contact_total", 0),
        "ref_shape": summary.get("ref_shape", ""),
        "ref_contact_total": summary.get("ref_contact_total", 0),
        "ref_either_active": summary.get("ref_either_active", 0),
        "ref_both_active": summary.get("ref_both_active", 0),
        "reward_compute_steps": summary.get("reward_compute_steps", 0),
        "r099_ref_contact_frac_max": val(summary, "r099_ref_contact_frac", "max", ""),
        "r099_ref_contact_frac_mean": val(summary, "r099_ref_contact_frac", "mean", ""),
        "r099_ref_masked_hand_contact_reward_mean_max": val(
            summary, "r099_ref_masked_hand_contact_reward_mean", "max", ""
        ),
        "r099_ref_masked_hand_contact_reward_mean_mean": val(
            summary, "r099_ref_masked_hand_contact_reward_mean", "mean", ""
        ),
        "r099_ref_masked_two_hand_contact_reward_mean_max": val(
            summary, "r099_ref_masked_two_hand_contact_reward_mean", "max", ""
        ),
        "r099_ref_masked_two_hand_contact_reward_mean_mean": val(
            summary, "r099_ref_masked_two_hand_contact_reward_mean", "mean", ""
        ),
        "r099_hand_gate_ref_true_mean": val(summary, "r099_hand_gate_ref_true", "mean", ""),
        "r099_two_hand_gate_ref_both_mean": val(summary, "r099_two_hand_gate_ref_both", "mean", ""),
        "reward_keys_present": str(bool(summary.get("reward_keys_present", False))).lower(),
        "reward_keys_finite": str(bool(summary.get("reward_keys_finite", False))).lower(),
        "ref_contact_frac_positive": str(bool(summary.get("ref_contact_frac_positive", False))).lower(),
        "ref_mask_reward_positive": str(bool(summary.get("ref_mask_reward_positive", False))).lower(),
        "startup_status": startup_status,
        "reward_probe_status": reward_probe_status,
        "ppo_smoke_allowed": str(bool(ppo_smoke_allowed)).lower(),
        "rl_ready": "false",
        "training_launched": str(bool(summary.get("training_launched", False))).lower(),
        "cem_launched": str(bool(summary.get("cem_launched", False))).lower(),
        "remote_jobs_launched": str(bool(summary.get("remote_jobs_launched", False))).lower(),
        "failure_mode": failure_mode,
        "notes": notes,
    }
    rows.append(row)

rows_tsv.parent.mkdir(parents=True, exist_ok=True)
with rows_tsv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

pass_rows = sum(1 for row in rows if row["reward_probe_status"] == "pass")
startup_pass_rows = sum(1 for row in rows if row["startup_status"] == "pass")
positive_ref_rows = sum(1 for row in rows if row["ref_contact_frac_positive"] == "true")
positive_reward_rows = sum(1 for row in rows if row["ref_mask_reward_positive"] == "true")
ppo_smoke_allowed_rows = sum(1 for row in rows if row["ppo_smoke_allowed"] == "true")
status = "pass" if rows and pass_rows == len(rows) else "blocked"
recommendation = "ppo_smoke_allowed" if ppo_smoke_allowed_rows > 0 else "keep_no_ppo_block"

summary = {
    "experiment": "E141",
    "status": status,
    "recommendation": recommendation,
    "row_count": len(rows),
    "startup_pass_rows": startup_pass_rows,
    "reward_probe_pass_rows": pass_rows,
    "positive_ref_contact_frac_rows": positive_ref_rows,
    "positive_ref_mask_reward_rows": positive_reward_rows,
    "ppo_smoke_allowed_rows": ppo_smoke_allowed_rows,
    "rl_ready_rows": 0,
    "training_launched": False,
    "cem_launched": False,
    "remote_jobs_launched": False,
    "rows_tsv": str(rows_tsv),
    "raw_runs": str(raw_runs),
}
summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = [
    "# E141 Reward Probe Summary",
    "",
    f"- status: `{status}`",
    f"- recommendation: `{recommendation}`",
    f"- rows: {len(rows)}",
    f"- startup pass rows: {startup_pass_rows}",
    f"- reward probe pass rows: {pass_rows}",
    f"- positive ref_contact_frac rows: {positive_ref_rows}",
    f"- positive ref-mask reward rows: {positive_reward_rows}",
    f"- ppo smoke allowed rows: {ppo_smoke_allowed_rows}",
    f"- rl_ready_rows: 0",
    f"- training_launched: false",
    f"- cem_launched: false",
    f"- remote_jobs_launched: false",
    "",
    "## Rows",
    "",
    "| case_id | startup | reward_probe | ref_frac_max | hand_reward_max | two_hand_reward_max | ppo_smoke_allowed | failure_mode |",
    "|---|---|---|---:|---:|---:|---|---|",
]
for row in rows:
    lines.append(
        "| {case_id} | {startup_status} | {reward_probe_status} | {r099_ref_contact_frac_max} | "
        "{r099_ref_masked_hand_contact_reward_mean_max} | "
        "{r099_ref_masked_two_hand_contact_reward_mean_max} | {ppo_smoke_allowed} | {failure_mode} |".format(
            **row
        )
    )
summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(json.dumps(summary, indent=2, sort_keys=True))
PY
