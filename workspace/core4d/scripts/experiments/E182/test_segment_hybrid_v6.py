#!/usr/bin/env python3
"""Direct-main contracts for task-conditioned segment hybrid attempt2-v6."""

from __future__ import annotations

import numpy as np
from search_segment_hybrid_v6 import (
    SEGMENT_COUNT,
    THRESHOLDS_M,
    _parent_screen_evidence,
    enumerate_hybrids,
    parent_source_rows,
)


def test_v6_parent_authority_contract() -> None:
    """V6 may consume only the complete failed v5 screen and its nine candidates."""
    evidence = _parent_screen_evidence()
    assert evidence["status"] == "V5_SCREEN_COMPLETE_NO_LAUNCHABLE_GROUP"
    assert evidence["candidate_count"] == 9
    assert evidence["launch_floor_pass_count"] == 0
    assert evidence["selected_group_count"] == 0
    rows = parent_source_rows()
    assert len(rows) == 9
    assert {row["max_hulls"] for row in rows} == {8, 16, 32}
    for max_hulls in (8, 16, 32):
        group = [row for row in rows if row["max_hulls"] == max_hulls]
        assert len(group) == 3
        assert {row["threshold_m"] for row in group} == set(THRESHOLDS_M)


def test_hybrid_enumeration_obeys_budget_and_p_floor() -> None:
    """Enumeration must find complementary segments without relaxing the 0.70 P floor."""
    oracle = np.array([True, True, False, False], dtype=bool)
    contacts = np.zeros((3, SEGMENT_COUNT, len(oracle)), dtype=bool)
    # Threshold 0 recovers the first true contact without a phantom.
    contacts[0, 0, 0] = True
    # Threshold 2 recovers the second true contact; another segment creates a phantom.
    contacts[2, 1, 1] = True
    contacts[2, 2, 2] = True
    hull_counts = np.ones((3, SEGMENT_COUNT), dtype=np.int32)
    rows = enumerate_hybrids(
        oracle_contact=oracle,
        segment_contacts=contacts,
        segment_hull_counts=hull_counts,
        max_hulls=8,
    )
    assert rows
    assert all(row["actual_hulls"] <= 8 for row in rows)
    assert all(row["precision"] >= 0.70 and row["recall"] >= 0.70 for row in rows)
    best = rows[0]
    assert best["precision"] == 1.0
    assert best["recall"] == 1.0
    assert best["threshold_indices_by_segment"][:2] == [0, 2]
    assert len(best["threshold_indices_by_segment"]) == SEGMENT_COUNT


def test_hybrid_enumeration_rejects_over_budget_assignments() -> None:
    """A P-perfect assignment cannot survive if its selected parts exceed K."""
    oracle = np.array([True, False], dtype=bool)
    contacts = np.zeros((3, SEGMENT_COUNT, len(oracle)), dtype=bool)
    contacts[2, 0, 0] = True
    hull_counts = np.ones((3, SEGMENT_COUNT), dtype=np.int32)
    hull_counts[2, 0] = 2
    rows = enumerate_hybrids(
        oracle_contact=oracle,
        segment_contacts=contacts,
        segment_hull_counts=hull_counts,
        max_hulls=8,
    )
    assert rows == []


def main() -> int:
    """Run v6 contracts without pytest discovery."""
    tests = (
        test_v6_parent_authority_contract,
        test_hybrid_enumeration_obeys_budget_and_p_floor,
        test_hybrid_enumeration_rejects_over_budget_assignments,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E182_SEGMENT_HYBRID_V6_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
