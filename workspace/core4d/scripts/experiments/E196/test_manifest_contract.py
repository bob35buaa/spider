#!/usr/bin/env python3
"""Direct-entry frozen set and queue tests for E196."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e196_reference_fix_common as C  # noqa: E402


def test_authority_and_queues() -> None:
    audit = C.read_tsv(C.E194_AUDIT)
    mismatch = {
        row["case_id"]
        for row in audit
        if row.get("runtime_convention_matches_xml_axes", "").lower() == "false"
    }
    C.validate_frozen_queues(mismatch)
    assert len(mismatch) == 29
    assert C.case_set_sha256(mismatch) == C.CASE_SET_SHA256
    assert Counter(case_id.split("_", 1)[0] for case_id in mismatch) == Counter(
        {"box001": 21, "box023": 8}
    )


def test_worker_order_and_counts() -> None:
    totals = {}
    for worker in C.WORKERS:
        queue = (*C.WAVE0[worker], *C.REMAINING[worker])
        totals[worker] = len(queue)
        first_box023 = next(
            (index for index, case_id in enumerate(queue) if case_id.startswith("box023_")),
            len(queue),
        )
        assert all(case_id.startswith("box001_") for case_id in queue[:first_box023])
        assert all(case_id.startswith("box023_") for case_id in queue[first_box023:])
    assert totals == {"local-gpu0": 10, "ada-gpu0": 10, "ada-gpu1": 9}
    assert sum(len(rows) for rows in C.WAVE0.values()) == 3
    assert sum(len(rows) for rows in C.REMAINING.values()) == 26


def main() -> None:
    test_authority_and_queues()
    test_worker_order_and_counts()
    print("PASS: E196 frozen manifest contract")


if __name__ == "__main__":
    main()
