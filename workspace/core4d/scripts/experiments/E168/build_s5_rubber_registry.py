#!/usr/bin/env python3
"""Build the E168 S5 registry slice on the rubber_hull collision axis."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-state-registry", type=Path, required=True)
    parser.add_argument("--out-tsv", type=Path, required=True)
    args = parser.parse_args()

    with args.case_state_registry.expanduser().resolve().open(
        "r", encoding="utf-8", newline=""
    ) as f:
        reader = csv.DictReader(f, delimiter="\t")
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    eligible = [
        row
        for row in rows
        if row.get("retarget_variant_id") in {"omnirt_v1", "omnirt_v2"}
        and row.get("target_variant_id") == "ref_fk"
        and row.get("stage2b_status") == "pass"
        and row.get("target_gate_status") == "pass"
        and row.get("visual_qc_status") == "pass"
    ]
    by_case: dict[str, list[dict[str, str]]] = {}
    for row in eligible:
        by_case.setdefault(row["case_id"], []).append(row)
    selected: list[dict[str, str]] = []
    for case_id, case_rows in sorted(by_case.items()):
        if len(case_rows) != 1:
            variants = [row["retarget_variant_id"] for row in case_rows]
            raise SystemExit(f"ambiguous effective S4 variant for {case_id}: {variants}")
        row = dict(case_rows[0])
        row["hand_collision_variant_id"] = "rubber_hull"
        row["source_type"] = "E168_S5_collision_axis_projection"
        row["source_ref"] = "E168_S4_effective_pass_to_rubber_hull"
        row["updated_at"] = now()
        selected.append(row)
    if len(selected) != 40:
        raise SystemExit(f"expected 40 effective S4 rows, got {len(selected)}")

    out = args.out_tsv.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)
    summary = {
        "created_at": now(),
        "status": "pass",
        "rows": len(selected),
        "retarget_variant_counts": dict(
            Counter(row["retarget_variant_id"] for row in selected)
        ),
        "hand_collision_variant_counts": dict(
            Counter(row["hand_collision_variant_id"] for row in selected)
        ),
        "source_registry": str(args.case_state_registry.expanduser().resolve()),
        "output_tsv": str(out),
    }
    out.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
