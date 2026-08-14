#!/usr/bin/env python3
"""Build the 2-row raw-contact subset that enables the box001 partner Stage2b.

plan227 / E198 RL export needs the opposite-person (partner) Stage2b for two
box001 source cases whose p2 was never retargeted:

  box001_20231003_2_039_p1  -> partner box001_20231003_2_039_p2
  box001_20231003_2_041_p1  -> partner box001_20231003_2_041_p2

Both p2 rows exist in E173's 5cm raw-contact candidates. `2_041_p2` is a genuine
``raw_contact_pass`` at 5cm; `2_039_p2` is ``raw_contact_review`` (weak contact,
active~0.29) and is force-overridden to ``raw_contact_pass`` here so
``run_stage2b.py`` will enable it. The override is recorded in ``raw_contact_notes``
and echoed to stdout -- it is a partner (second-person context), not a training
target. Output is written under E198 so E173 artifacts stay untouched.
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
SRC = (
    REPO
    / "workspace/core4d/results/E173/s1_raw_contact/raw_contact/raw_contact_candidates_5cm.tsv"
)
OUT = (
    REPO
    / "workspace/core4d/results/E198/s3_retarget/box001_partner_raw_contact_5cm.tsv"
)

PARTNERS = ["box001_20231003_2_039_p2", "box001_20231003_2_041_p2"]
FORCE_PASS = {"box001_20231003_2_039_p2"}  # raw_contact_review -> pass (documented)


def main() -> int:
    with SRC.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = list(reader.fieldnames or [])
        by_case = {r["case_id"]: r for r in reader if r.get("case_id") in PARTNERS}

    missing = [c for c in PARTNERS if c not in by_case]
    if missing:
        raise SystemExit(f"partner rows missing from 5cm candidates: {missing}")

    rows = []
    for case_id in PARTNERS:
        row = dict(by_case[case_id])
        if case_id in FORCE_PASS:
            original = row.get("raw_contact_decision", "")
            row["raw_contact_decision"] = "raw_contact_pass"
            note = (
                f"E198_partner_override: raw_contact_decision {original}->raw_contact_pass "
                "to build partner Stage2b for RL export; weak-contact partner context only"
            )
            existing = row.get("raw_contact_notes", "")
            row["raw_contact_notes"] = f"{existing}; {note}" if existing else note
            print(f"[override] {case_id}: {original} -> raw_contact_pass")
        else:
            print(f"[keep] {case_id}: {row.get('raw_contact_decision', '')}")
        rows.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[wrote] {OUT.relative_to(REPO)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
