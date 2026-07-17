#!/usr/bin/env python3
"""Build isolated E168 OmniRetarget v2 canary and rescue inputs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


ELIGIBLE_V1_STATUS = "omniretarget_infeasible"
CANARY_V1_STATUS = "pass"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_tsv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader), list(reader.fieldnames or [])


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-input-tsv", type=Path, required=True)
    parser.add_argument("--v1-manifest-tsv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--canary-case-id", default="bucket004_20231002_018_p1"
    )
    args = parser.parse_args()

    source_rows, fields = read_tsv(args.s3_input_tsv.expanduser().resolve())
    v1_rows, _ = read_tsv(args.v1_manifest_tsv.expanduser().resolve())
    source_by_case = {row["case_id"]: row for row in source_rows}
    if len(source_by_case) != len(source_rows):
        raise SystemExit("S3 input contains duplicate case_id rows")

    rescue_v1_rows = [
        row for row in v1_rows if row.get("stage2b_status") == ELIGIBLE_V1_STATUS
    ]
    rescue_ids = sorted(row["case_id"] for row in rescue_v1_rows)
    missing = [case_id for case_id in rescue_ids if case_id not in source_by_case]
    if missing:
        raise SystemExit(f"v1 infeasible rows missing from S3 input: {missing}")
    if not rescue_ids:
        raise SystemExit("no v1 omniretarget_infeasible rows; v2 rescue is not eligible")

    v1_by_case = {row["case_id"]: row for row in v1_rows}
    canary_id = args.canary_case_id
    if canary_id not in source_by_case:
        raise SystemExit(f"canary row missing from S3 input: {canary_id}")
    canary_v1 = v1_by_case.get(canary_id)
    if not canary_v1 or canary_v1.get("stage2b_status") != CANARY_V1_STATUS:
        status = canary_v1.get("stage2b_status", "missing") if canary_v1 else "missing"
        raise SystemExit(f"canary must be a v1 pass row, got {canary_id}={status}")
    if canary_id in rescue_ids:
        raise SystemExit("canary row cannot also be a production rescue row")

    out_dir = args.out_dir.expanduser().resolve()
    rescue_path = out_dir / "omnirt_v2_rescue_input.tsv"
    canary_path = out_dir / "omnirt_v2_canary_input.tsv"
    write_tsv(rescue_path, [source_by_case[x] for x in rescue_ids], fields)
    write_tsv(canary_path, [source_by_case[canary_id]], fields)

    summary = {
        "created_at": now(),
        "status": "pass",
        "eligibility_rule": "v1 stage2b_status=omniretarget_infeasible only",
        "production_execution_mode": "rescue",
        "canary_execution_mode": "canary",
        "canary_excluded_from_production_yield": True,
        "source_s3_input_tsv": str(args.s3_input_tsv.expanduser().resolve()),
        "source_v1_manifest_tsv": str(args.v1_manifest_tsv.expanduser().resolve()),
        "rescue_input_tsv": str(rescue_path),
        "rescue_rows": len(rescue_ids),
        "rescue_case_ids": rescue_ids,
        "canary_input_tsv": str(canary_path),
        "canary_rows": 1,
        "canary_case_id": canary_id,
        "canary_source_v1_status": CANARY_V1_STATUS,
    }
    summary_path = out_dir / "omnirt_v2_rescue_input_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
