#!/usr/bin/env python3
"""Validate E175 CEM artifacts and write the canary/full runtime hard gate."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from run_cem_queue import MODULE, validate_runtime_outputs


REPO = Path(__file__).resolve().parents[5]
RESULT_ROOT = REPO / "workspace/core4d/results/E175/s6_downstream"


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def defaults(mode: str) -> tuple[Path, Path, int, str]:
    if mode == "canary":
        return (
            RESULT_ROOT / "manifests/nonbox_multigeom_canary_manifest.tsv",
            RESULT_ROOT / "cem/canary/canary_runtime_gate.json",
            6,
            "canary",
        )
    return (
        RESULT_ROOT / "manifests/nonbox_multigeom_full_manifest.tsv",
        RESULT_ROOT / "cem/full/full_runtime_gate.json",
        39,
        "production",
    )


def validate_row(
    row: dict[str, str],
    *,
    expected_execution_mode: str,
) -> list[str]:
    failures: list[str] = []
    if row.get("execution_mode") != expected_execution_mode:
        failures.append(
            "manifest_mismatch:execution_mode:"
            f"{row.get('execution_mode', '')}"
        )
    if row.get("status") != "run_complete_pending_eval":
        failures.append(f"status:{row.get('status', '')}")
    if row.get("failure_mode"):
        failures.append(f"failure_mode:{row['failure_mode']}")

    failures.extend(MODULE.validate_inputs(row))
    complete, missing = MODULE.output_complete(row)
    if not complete:
        failures.extend(f"missing_artifact:{key}" for key in missing)
    else:
        failures.extend(validate_runtime_outputs(row))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("canary", "full"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    default_manifest, default_output, expected_rows, execution_mode = defaults(
        args.mode
    )
    manifest = repo_path(args.manifest or default_manifest)
    output = repo_path(args.output or default_output)
    rows = read_tsv(manifest)

    errors: list[dict[str, Any]] = []
    case_ids = [row.get("case_id", "") for row in rows]
    if len(rows) != expected_rows:
        errors.append(
            {
                "scope": "manifest",
                "failure": f"row_count:{len(rows)}!=expected:{expected_rows}",
            }
        )
    duplicate_cases = sorted(
        case_id
        for case_id, count in Counter(case_ids).items()
        if not case_id or count != 1
    )
    if duplicate_cases:
        errors.append(
            {
                "scope": "manifest",
                "failure": "duplicate_or_empty_case_id",
                "case_ids": duplicate_cases,
            }
        )

    passed_rows = 0
    for row in rows:
        failures = validate_row(
            row,
            expected_execution_mode=execution_mode,
        )
        if failures:
            errors.append(
                {
                    "scope": "row",
                    "case_id": row.get("case_id", ""),
                    "variant": row.get("variant", ""),
                    "failures": failures,
                }
            )
        else:
            passed_rows += 1

    status = (
        "pass"
        if not errors and passed_rows == expected_rows
        else "fail"
    )
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "experiment_id": "E175",
        "mode": args.mode,
        "status": status,
        "manifest": str(manifest.relative_to(REPO)),
        "expected_rows": expected_rows,
        "manifest_rows": len(rows),
        "passed_rows": passed_rows,
        "failed_rows": expected_rows - passed_rows,
        "status_counts": dict(
            Counter(row.get("status", "") for row in rows)
        ),
        "errors": errors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "status": status,
                "passed_rows": passed_rows,
                "expected_rows": expected_rows,
                "output": str(output.relative_to(REPO))
                if output.is_relative_to(REPO)
                else str(output),
            },
            sort_keys=True,
        )
    )
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
