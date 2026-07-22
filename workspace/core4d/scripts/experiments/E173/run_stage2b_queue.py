#!/usr/bin/env python3
"""E173: execute a Stage2b manifest one case at a time, retaining failure evidence.

Adopted from the generic E168 queue runner (no object-specific logic). Normalizes
CVXPY solver-infeasible to stage2b_status=omniretarget_infeasible so only that status
feeds the E173 v2 rescue queue.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[5]
CASE_HEADER = [
    "# enabled",
    "date",
    "seq",
    "person",
    "object_name",
    "object_model_rel",
    "source_scene_task",
    "target_task",
    "trim_start",
    "trim_frames",
    "data_id",
    "mask_slug",
]
EXECUTION_FIELDS = [
    "execution_log",
    "execution_returncode",
    "execution_command",
    "execution_attempt",
]
REQUIRED_OUTPUT_FIELDS = [
    "converted_npz",
    "omniretarget_output_npz",
    "trimmed_npz",
    "spider_trajectory",
    "verify_summary",
]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def read_tsv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader), list(reader.fieldnames or [])


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    temp.replace(path)


def write_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def write_case_file(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [
        "1",
        row["date"],
        row["seq"],
        row["person"],
        row["object_name"],
        row["object_model_rel"],
        row["source_scene_task"],
        row["target_task"],
        row.get("trim_start", "auto"),
        row.get("trim_frames", "auto"),
        row.get("data_id", "0"),
        row["mask_slug"],
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(CASE_HEADER)
        writer.writerow(values)


def flag(params: dict[str, Any], key: str) -> str:
    return "1" if bool(params.get(key, False)) else "0"


def command_for(
    row: dict[str, str],
    *,
    case_file: Path,
    holosoma_repo: Path,
    raw_root: Path,
    smplx_model_dir: Path,
    python_bin: Path,
    retarget_python_bin: Path,
) -> tuple[list[str], dict[str, str]]:
    params = json.loads(row.get("params_json", "") or "{}")
    env = os.environ.copy()
    env.update(
        {
            "REPO": str(REPO),
            "HOLOSOMA_DIR": str(holosoma_repo),
            "CORE4D_REAL_ROOT": str(raw_root),
            "SMPLX_MODEL_DIR": str(smplx_model_dir),
            "RESULT_ROOT": os.path.relpath(Path(row["result_root"]), REPO),
            "PYTHON_BIN": str(python_bin),
            "RETARGET_PYTHON_BIN": str(retarget_python_bin),
            "REPLACE_WRIST_WITH_FINGERTIP": flag(
                params, "replace_wrist_with_fingertip"
            ),
            "RETARGET_ENABLE_CONSTRAINT_RELAXATION": flag(
                params, "enable_constraint_relaxation"
            ),
            "RETARGET_ENABLE_FOOT_Z_CONSTRAINT": flag(
                params, "enable_foot_z_constraint"
            ),
            "RETARGET_FOOT_SLIDE_PENALTY_WEIGHT": str(
                float(params.get("foot_slide_penalty_weight", 0.0))
            ),
            "RETARGET_ENABLE_CONTACT_PRESERVATION": flag(
                params, "enable_contact_preservation"
            ),
            "RETARGET_OBJECT_PENETRATION_TOLERANCE_SCALE": str(
                float(params.get("object_penetration_tolerance_scale", 1.0))
            ),
            "TARGET_VARIANT_ID": row["target_variant_id"],
        }
    )
    command = [
        "bash",
        "workspace/core4d/data_preprocess/pipeline.sh",
        "--case-file",
        os.path.relpath(case_file, REPO),
    ]
    return command, env


def output_status(row: dict[str, str]) -> tuple[bool, list[str]]:
    missing = [
        field
        for field in REQUIRED_OUTPUT_FIELDS
        if not Path(row.get(field, "")).is_file()
    ]
    contact_mask = Path(
        row.get("stage2b_contact_mask_npz_expected")
        or row.get("contact_mask_npz", "")
    )
    if not contact_mask.is_file():
        missing.append("contact_mask_npz")
    return not missing, missing


def classify_failure(log_text: str) -> str:
    lowered = log_text.lower()
    infeasible_tokens = [
        "infeasible",
        "solvererror",
        "solver failed",
        "optimization failed",
    ]
    if any(token in lowered for token in infeasible_tokens):
        return "omniretarget_infeasible"
    return "preprocess_fail"


def persist(
    manifest_path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
    summary_path: Path,
) -> None:
    write_tsv(manifest_path, rows, fields)
    write_json(manifest_path.with_suffix(".json"), rows)
    status_counts = Counter(row.get("stage2b_status", "") for row in rows)
    summary = {
        "created_at": now(),
        "rows": len(rows),
        "retarget_variant_id": sorted(
            {row.get("retarget_variant_id", "") for row in rows}
        ),
        "status_counts": dict(status_counts),
        "pass_rows": [
            row["case_id"] for row in rows if row.get("stage2b_status") == "pass"
        ],
        "infeasible_rows": [
            row["case_id"]
            for row in rows
            if row.get("stage2b_status") == "omniretarget_infeasible"
        ],
        "failed_rows": [
            row["case_id"]
            for row in rows
            if row.get("stage2b_status")
            in {"omniretarget_infeasible", "preprocess_fail"}
        ],
    }
    write_json(summary_path, summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-tsv", type=Path, required=True)
    parser.add_argument("--holosoma-repo", type=Path, required=True)
    parser.add_argument("--core4d-raw-root", type=Path, required=True)
    parser.add_argument("--smplx-model-dir", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, default=REPO / ".venv/bin/python")
    parser.add_argument("--retarget-python-bin", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    manifest_path = args.manifest_tsv.expanduser().resolve()
    rows, fields = read_tsv(manifest_path)
    for field in EXECUTION_FIELDS:
        if field not in fields:
            fields.append(field)
    out_dir = manifest_path.parent
    case_dir = out_dir / "execution_cases"
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "stage2b_queue_summary.json"

    eligible = [
        row
        for row in rows
        if row.get("pipeline_enabled") == "1"
        and row.get("stage2b_status")
        not in {"pass", "omniretarget_infeasible"}
    ]
    if args.max_cases is not None:
        eligible = eligible[: args.max_cases]
    eligible_ids = {id(row) for row in eligible}

    for row in rows:
        if id(row) not in eligible_ids:
            continue
        case_id = row["case_id"]
        case_file = case_dir / f"{safe_id(case_id)}.tsv"
        log_path = log_dir / f"{safe_id(case_id)}.log"
        write_case_file(case_file, row)
        command, env = command_for(
            row,
            case_file=case_file,
            holosoma_repo=args.holosoma_repo.expanduser().resolve(),
            raw_root=args.core4d_raw_root.expanduser().resolve(),
            smplx_model_dir=args.smplx_model_dir.expanduser().resolve(),
            python_bin=args.python_bin.expanduser().absolute(),
            retarget_python_bin=args.retarget_python_bin.expanduser().absolute(),
        )
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                command,
                cwd=REPO,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        complete, missing = output_status(row)
        row.update(
            {
                "execution_log": str(log_path),
                "execution_returncode": str(completed.returncode),
                "execution_command": " ".join(command),
                "execution_attempt": str(int(row.get("execution_attempt") or "0") + 1),
                "updated_at": now(),
            }
        )
        if completed.returncode == 0 and complete:
            row.update(
                {
                    "stage2b_status": "pass",
                    "failure_mode": "",
                    "decision_notes": "per-case Stage2b execution completed",
                    "contact_mask_npz": row["stage2b_contact_mask_npz_expected"],
                    "contact_mask_status": "trimmed_mask_available",
                    "contact_mask_time_axis": "trimmed_stage2b_output",
                }
            )
        else:
            failure = classify_failure(log_text)
            row.update(
                {
                    "stage2b_status": failure,
                    "failure_mode": failure,
                    "decision_notes": (
                        f"returncode={completed.returncode}; "
                        f"missing_outputs={','.join(missing)}; log={log_path}"
                    ),
                }
            )
        persist(manifest_path, rows, fields, summary_path)
        print(
            json.dumps(
                {
                    "case_id": case_id,
                    "status": row["stage2b_status"],
                    "returncode": completed.returncode,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    persist(manifest_path, rows, fields, summary_path)
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
