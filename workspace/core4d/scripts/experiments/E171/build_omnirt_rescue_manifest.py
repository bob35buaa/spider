#!/usr/bin/env python3
"""E171: build the isolated omnirt_v2 canary and rescue inputs from fresh v1 results.

Only rows whose fresh v1 status is exactly ``omniretarget_infeasible`` are eligible
for v2 rescue. General preprocess/env/shape failures are NOT rescued. Unlike the
E168 builder this:
  * auto-selects an in-scope v1-pass canary (Box022/026), since E168's hardcoded
    bucket004 canary is out of E171 scope;
  * treats "0 infeasible rows" as a clean no-op (exit 0), because an all-v1-pass
    run is a valid E171 outcome, not an error.

Inputs
  --s3-input-tsv : the raw_contact_pass_*.tsv fed to run_stage2b.py (same schema).
  --v1-manifest-tsv : the completed omnirt_v1 stage2b manifest (has stage2b_status).

Outputs (in --out-dir)
  omnirt_v2_rescue_input.tsv  : s3-input rows for v1-infeasible case_ids (v2 rescue queue)
  omnirt_v2_canary_input.tsv  : one v1-pass row for the v2 adapter canary (if rescue non-empty)
  omnirt_v2_rescue_input_summary.json
"""

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
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(out_dir: Path, summary: dict) -> None:
    (out_dir / "omnirt_v2_rescue_input_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-input-tsv", type=Path, required=True)
    parser.add_argument("--v1-manifest-tsv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--canary-case-id",
        default="",
        help="explicit v1-pass canary case_id; default = first v1-pass case_id (sorted)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source_rows, fields = read_tsv(args.s3_input_tsv.expanduser().resolve())
    v1_rows, _ = read_tsv(args.v1_manifest_tsv.expanduser().resolve())
    source_by_case = {row["case_id"]: row for row in source_rows}
    if len(source_by_case) != len(source_rows):
        raise SystemExit("S3 input contains duplicate case_id rows")

    rescue_ids = sorted(
        row["case_id"] for row in v1_rows if row.get("stage2b_status") == ELIGIBLE_V1_STATUS
    )
    missing = [case_id for case_id in rescue_ids if case_id not in source_by_case]
    if missing:
        raise SystemExit(f"v1 infeasible rows missing from S3 input: {missing}")

    # 0 infeasible -> valid no-op: all v1 rows are pass/other-terminal, no v2 rescue.
    if not rescue_ids:
        write_summary(
            out_dir,
            {
                "created_at": now(),
                "status": "no_rescue_needed",
                "eligibility_rule": "v1 stage2b_status=omniretarget_infeasible only",
                "rescue_rows": 0,
                "rescue_case_ids": [],
                "canary_rows": 0,
                "canary_case_id": "",
                "source_s3_input_tsv": str(args.s3_input_tsv.expanduser().resolve()),
                "source_v1_manifest_tsv": str(args.v1_manifest_tsv.expanduser().resolve()),
            },
        )
        return 0

    # deterministic in-scope canary: an explicit v1-pass row not in the rescue set.
    v1_by_case = {row["case_id"]: row for row in v1_rows}
    v1_pass_ids = sorted(
        cid for cid, row in v1_by_case.items()
        if row.get("stage2b_status") == CANARY_V1_STATUS and cid in source_by_case
    )
    if args.canary_case_id:
        canary_id = args.canary_case_id
        if canary_id not in source_by_case:
            raise SystemExit(f"canary row missing from S3 input: {canary_id}")
        canary_status = v1_by_case.get(canary_id, {}).get("stage2b_status", "missing")
        if canary_status != CANARY_V1_STATUS:
            raise SystemExit(f"canary must be a v1 pass row, got {canary_id}={canary_status}")
        if canary_id in rescue_ids:
            raise SystemExit("canary row cannot also be a production rescue row")
    else:
        eligible_canaries = [cid for cid in v1_pass_ids if cid not in rescue_ids]
        if not eligible_canaries:
            raise SystemExit(
                "rescue set non-empty but no v1-pass row available for a v2 adapter canary"
            )
        canary_id = eligible_canaries[0]

    rescue_path = out_dir / "omnirt_v2_rescue_input.tsv"
    canary_path = out_dir / "omnirt_v2_canary_input.tsv"
    write_tsv(rescue_path, [source_by_case[x] for x in rescue_ids], fields)
    write_tsv(canary_path, [source_by_case[canary_id]], fields)

    write_summary(
        out_dir,
        {
            "created_at": now(),
            "status": "pass",
            "eligibility_rule": "v1 stage2b_status=omniretarget_infeasible only",
            "production_execution_mode": "rescue",
            "canary_execution_mode": "canary",
            "canary_excluded_from_production_yield": True,
            "canary_auto_selected": not bool(args.canary_case_id),
            "source_s3_input_tsv": str(args.s3_input_tsv.expanduser().resolve()),
            "source_v1_manifest_tsv": str(args.v1_manifest_tsv.expanduser().resolve()),
            "rescue_input_tsv": str(rescue_path),
            "rescue_rows": len(rescue_ids),
            "rescue_case_ids": rescue_ids,
            "canary_input_tsv": str(canary_path),
            "canary_rows": 1,
            "canary_case_id": canary_id,
            "canary_source_v1_status": CANARY_V1_STATUS,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
