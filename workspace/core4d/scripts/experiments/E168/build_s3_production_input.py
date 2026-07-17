#!/usr/bin/env python3
"""Exclude direct imports from the E168 new-production Stage2b input."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


DIRECT_IMPORT_CASES = {"box004_20231003_2_082_p1"}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-contact-pass-tsv", type=Path, required=True)
    parser.add_argument("--out-tsv", type=Path, required=True)
    args = parser.parse_args()

    source = args.raw_contact_pass_tsv.expanduser().resolve()
    with source.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    imported = [row for row in rows if row.get("case_id") in DIRECT_IMPORT_CASES]
    production = [row for row in rows if row.get("case_id") not in DIRECT_IMPORT_CASES]
    if [row["case_id"] for row in imported] != sorted(DIRECT_IMPORT_CASES):
        raise SystemExit(
            f"direct import exclusion mismatch: {[row['case_id'] for row in imported]}"
        )
    if len(rows) != 41 or len(production) != 40:
        raise SystemExit(
            f"E168 S3 expected raw pass/new production 41/40, got {len(rows)}/{len(production)}"
        )

    out = args.out_tsv.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(production)
    summary = {
        "created_at": now(),
        "status": "pass",
        "raw_contact_pass_rows": len(rows),
        "direct_import_rows": len(imported),
        "new_production_rows": len(production),
        "direct_import_cases": sorted(DIRECT_IMPORT_CASES),
        "source_tsv": str(source),
        "output_tsv": str(out),
    }
    out.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
