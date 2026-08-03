#!/usr/bin/env python3
"""Direct-main contracts for E187 three-worker LPT queues."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).with_name("build_full_queues.py")
SPEC = importlib.util.spec_from_file_location("e187_full_queues", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_lpt_keeps_canaries_first_and_rows_unique() -> None:
    """Three promoted canaries remain first and every remaining row appears once."""
    rows = [
        {
            "case_id": case_id,
            "ordinal": index,
            "predicted_wall_seconds": float(30 - index),
        }
        for index, case_id in enumerate(
            (
                "bucket003_20231018_003_p1",
                "bucket004_20231002_021_p1",
                "bucket007_20231020_055_p1",
                "case4",
                "case5",
                "case6",
            ),
            start=1,
        )
    ]
    walls = {"local-0": 10.0, "remote-0": 20.0, "remote-1": 15.0}
    queues, loads = MODULE.lpt_assign(rows, MODULE.CANARY_BY_WORKER, walls)
    assert set(queues) == set(MODULE.WORKERS)
    for worker, canary in MODULE.CANARY_BY_WORKER.items():
        assert queues[worker][0]["case_id"] == canary
        assert queues[worker][0]["promoted_canary"] is True
        assert queues[worker][0]["queue_position"] == 1
    cases = [row["case_id"] for queue in queues.values() for row in queue]
    assert len(cases) == len(set(cases)) == 6
    assert all(load > 0.0 for load in loads.values())


def test_authority_is_exact_keep22() -> None:
    """Preflight authority remains ordered 1..22 with fixed representative rows."""
    rows, overrides = MODULE.authority_rows()
    assert len(rows) == overrides["row_count"] == 22
    assert [row["ordinal"] for row in rows] == list(range(1, 23))
    assert set(MODULE.CANARY_BY_WORKER.values()).issubset(
        {row["case_id"] for row in rows}
    )


def main() -> int:
    """Run direct-main queue contracts."""
    tests = (
        test_lpt_keeps_canaries_first_and_rows_unique,
        test_authority_is_exact_keep22,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E187_FULL_QUEUE_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
