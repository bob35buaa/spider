#!/usr/bin/env python3
"""Continue E145 Stage2b after per-case pipeline failures.

The legacy Stage2b wrapper exits on the first failing case.  This helper keeps
the standard E145 manifest/case-file layout, filters out already completed
cases, records the first case that blocked each retry, and refreshes terminal
manifest status from concrete output files.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[4]
RUN_ROOT = REPO / "workspace/core4d/results/E145/full_nonbox_to_rl_ready"
OUT_DIR = RUN_ROOT / "s3_retarget/omnirt_v1/ref_fk"
RUN_SLUG = "omnirt_v1_ref_fk"
TASK_PREFIX = f"dcv3_{RUN_SLUG}_"
CASE_FILE = OUT_DIR / f"cases_stage2b_ready_{RUN_SLUG}.tsv"
MANIFEST = OUT_DIR / f"stage2b_manifest_{RUN_SLUG}.tsv"
RUN_SCRIPT = OUT_DIR / f"run_stage2b_{RUN_SLUG}.sh"
RESULT_ROOT = OUT_DIR / "results" / RUN_SLUG
CONTINUE_CASE_FILE = OUT_DIR / f"cases_stage2b_continue_pending_{RUN_SLUG}.tsv"
FAILURES_TSV = OUT_DIR / f"stage2b_continue_failures_{RUN_SLUG}.tsv"
SUMMARY_JSON = OUT_DIR / f"stage2b_continue_summary_{RUN_SLUG}.json"

CASE_FIELDS = [
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

REQUIRED_OUTPUT_FIELDS = [
    "converted_npz",
    "omniretarget_output_npz",
    "trimmed_npz",
    "spider_trajectory",
    "verify_summary",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        lines = [line for line in f if line.strip()]
    if not lines:
        return []
    return list(csv.DictReader(lines, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def case_id_from_task(target_task: str) -> str:
    if target_task.startswith(TASK_PREFIX):
        return target_task[len(TASK_PREFIX) :]
    return target_task


def completed_target_tasks(result_root: Path) -> set[str]:
    return {
        path.name.removesuffix("_verify_summary.json")
        for path in result_root.glob(f"{TASK_PREFIX}*_verify_summary.json")
        if path.is_file()
    }


def read_failures(path: Path) -> dict[str, dict[str, str]]:
    return {row["target_task"]: row for row in read_tsv(path) if row.get("target_task")}


def append_failure(path: Path, row: dict[str, str], returncode: int, retry_index: int) -> None:
    fields = ["retry_index", "case_id", "target_task", "returncode", "failure_mode"]
    failures = read_tsv(path)
    existing = {item.get("target_task") for item in failures}
    if row["target_task"] in existing:
        return
    failures.append(
        {
            "retry_index": retry_index,
            "case_id": case_id_from_task(row["target_task"]),
            "target_task": row["target_task"],
            "returncode": returncode,
            "failure_mode": "legacy_stage2b_pipeline_failed_before_verify_summary",
        }
    )
    write_tsv(path, failures, fields)


def write_case_file(path: Path, rows: list[dict[str, str]]) -> None:
    write_tsv(path, rows, CASE_FIELDS)


def command_from_run_script(run_script: Path, case_file: Path) -> list[str]:
    lines = [line.strip() for line in run_script.read_text(encoding="utf-8").splitlines() if line.strip()]
    command_line = lines[-1]
    cmd = shlex.split(command_line)
    for idx, item in enumerate(cmd[:-1]):
        if item == "--case-file":
            cmd[idx + 1] = str(case_file.relative_to(REPO))
            break
    else:
        raise RuntimeError(f"--case-file not found in {run_script}")
    return cmd


def refresh_manifest(manifest: Path, failures_tsv: Path) -> dict[str, Any]:
    rows = read_tsv(manifest)
    fields = list(rows[0].keys()) if rows else []
    failures = read_failures(failures_tsv)
    for row in rows:
        if row.get("pipeline_enabled") != "1":
            continue
        target_task = row.get("target_task", "")
        failure = failures.get(target_task)
        missing = [field for field in REQUIRED_OUTPUT_FIELDS if not Path(row.get(field, "")).is_file()]
        contact_mask = Path(row.get("stage2b_contact_mask_npz_expected") or row.get("contact_mask_npz", ""))
        if not contact_mask.is_file():
            missing.append("contact_mask_npz")
        if not missing:
            row["stage2b_status"] = "pass"
            row["failure_mode"] = ""
            row["decision_notes"] = "Stage2b execute completed and expected outputs exist"
            row["contact_mask_npz"] = str(contact_mask)
            row["contact_mask_status"] = "trimmed_mask_available"
            row["contact_mask_time_axis"] = "trimmed_stage2b_output"
        elif failure:
            row["stage2b_status"] = "fail"
            row["failure_mode"] = failure.get("failure_mode", "legacy_stage2b_pipeline_failed")
            row["decision_notes"] = "Stage2b legacy wrapper failed; no verify summary was produced"
        else:
            row["stage2b_status"] = "fail"
            row["failure_mode"] = "stage2b_outputs_missing_after_continue"
            row["decision_notes"] = "missing executed outputs after continue: " + ",".join(missing)
    write_tsv(manifest, rows, fields)
    manifest.with_suffix(".json").write_text(json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "rows": len(rows),
        "stage2b_status_counts": dict(Counter(row.get("stage2b_status", "") for row in rows)),
        "failure_mode_counts": dict(Counter(row.get("failure_mode", "") for row in rows if row.get("failure_mode"))),
        "object_counts": dict(Counter(row.get("object_key", "") for row in rows)),
        "failure_rows": list(read_failures(failures_tsv).values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-file", type=Path, default=CASE_FILE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--run-script", type=Path, default=RUN_SCRIPT)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--continue-case-file", type=Path, default=CONTINUE_CASE_FILE)
    parser.add_argument("--failures-tsv", type=Path, default=FAILURES_TSV)
    parser.add_argument("--summary-json", type=Path, default=SUMMARY_JSON)
    parser.add_argument("--max-retries", type=int, default=128)
    parser.add_argument("--refresh-only", action="store_true")
    args = parser.parse_args()

    if not args.refresh_only:
        base_rows = read_tsv(args.case_file)
        if not base_rows:
            raise SystemExit(f"empty case file: {args.case_file}")
        command = command_from_run_script(args.run_script, args.continue_case_file)
        retry_index = 0
        while retry_index < args.max_retries:
            completed = completed_target_tasks(args.result_root)
            failures = read_failures(args.failures_tsv)
            pending = [
                row
                for row in base_rows
                if row["target_task"] not in completed and row["target_task"] not in failures
            ]
            if not pending:
                break
            before = len(completed)
            retry_index += 1
            write_case_file(args.continue_case_file, pending)
            proc = subprocess.run(command, cwd=REPO, check=False)
            if proc.returncode == 0:
                break
            after_completed = completed_target_tasks(args.result_root)
            blocked = next((row for row in pending if row["target_task"] not in after_completed), pending[0])
            append_failure(args.failures_tsv, blocked, proc.returncode, retry_index)
            print(
                json.dumps(
                    {
                        "retry": retry_index,
                        "returncode": proc.returncode,
                        "completed_before": before,
                        "completed_after": len(after_completed),
                        "recorded_failure": case_id_from_task(blocked["target_task"]),
                    },
                    ensure_ascii=False,
                )
            )
        else:
            raise SystemExit(f"exceeded max retries: {args.max_retries}")

    summary = refresh_manifest(args.manifest, args.failures_tsv)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
