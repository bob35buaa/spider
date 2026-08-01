#!/usr/bin/env python3
"""Direct-main contracts for the post-result v8 visual diagnostic."""

from __future__ import annotations

from render_task_aware_preseg_v8 import (
    EXPECTED_DIAGNOSTIC_IDS,
    candidate_parts,
    load_static_rows,
    select_diagnostic_rows,
)


def test_v8_visual_rows_cover_frozen_tradeoff_representatives() -> None:
    """Diagnostics must deterministically cover balance, recall, and TP19 precision."""
    rows = select_diagnostic_rows(load_static_rows())
    assert [row["candidate_id"] for row in rows] == list(EXPECTED_DIAGNOSTIC_IDS)
    assert [row["diagnostic_label"] for row in rows] == [
        "best_balance",
        "high_recall",
        "tp19_precision_best",
    ]
    assert all(row["static_p_gate"]["status"] == "FAIL" for row in rows)


def test_v8_visual_reconstructs_exact_frozen_candidates() -> None:
    """Each rendered candidate must reproduce its frozen hull inventory exactly."""
    for row in select_diagnostic_rows(load_static_rows()):
        parts = candidate_parts(row)
        assert len(parts) == row["actual_hulls"]
        assert all(part.is_convex and part.is_watertight for part in parts)


def main() -> int:
    """Run v8 visual contracts without pytest discovery."""
    tests = (
        test_v8_visual_rows_cover_frozen_tradeoff_representatives,
        test_v8_visual_reconstructs_exact_frozen_candidates,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E182_TASK_AWARE_PRESEG_V8_VISUAL_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
