#!/usr/bin/env python3
"""Direct-main tests for tolerance-corrected global-budget attempt2-v5."""

from __future__ import annotations

from build_segmented_coacd_attempt2_v4 import _load_base_parts
from build_segmented_coacd_attempt2_v5 import (
    ABSOLUTE_VOLUME_TOLERANCE_M3,
    RELATIVE_VOLUME_TOLERANCE,
    _v4_failure_evidence,
    candidate_family,
    reduce_segmented_parts,
)


def test_v5_candidate_family_contract() -> None:
    """V5 must retain the frozen 3×K family but use independent identities."""
    rows = candidate_family()
    assert len(rows) == 9
    assert len({row["candidate_id"] for row in rows}) == 9
    assert {row["threshold_m"] for row in rows} == {0.005, 0.010, 0.020}
    assert {row["max_hulls"] for row in rows} == {8, 16, 32}
    assert all("_gmerge5_" in row["candidate_id"] for row in rows)
    assert all(row["method"] == "ISOLATED_COACD_GLOBAL_BUDGET_MERGE_V2" for row in rows)
    assert ABSOLUTE_VOLUME_TOLERANCE_M3 == 1e-9
    assert RELATIVE_VOLUME_TOLERANCE == 1e-6


def test_v4_numeric_failure_is_preserved_and_quantified() -> None:
    """V5 must bind all valid v4 prefix outputs and the exact numeric failure."""
    evidence = _v4_failure_evidence()
    assert evidence["status"] == "V4_REDUCER_NUMERIC_TOLERANCE_FAILED"
    assert evidence["successful_candidate_count"] == 3
    assert evidence["failed_threshold_m"] == 0.010
    assert evidence["failed_target_hulls"] == 8
    assert evidence["segment_index"] == 7
    assert evidence["left_lineage"] == [1]
    assert evidence["right_lineage"] == [2]
    assert -7e-10 < evidence["raw_added_volume_m3"] < -6e-10
    assert -1.3e-7 < evidence["relative_difference"] < -1.1e-7
    assert evidence["changed_contract"] == "ABSOLUTE_OR_RELATIVE_NUMERIC_TOLERANCE"


def test_real_t010_negative_roundoff_is_clamped_and_recorded() -> None:
    """The observed t010 pair must no longer block K8 and must stay auditable."""
    reduced, trace = reduce_segmented_parts(
        _load_base_parts(0.010),
        target_hulls=8,
    )
    assert sum(len(parts) for parts in reduced.values()) == 8
    clamped = [step for step in trace if step["negative_roundoff_clamped"]]
    assert len(clamped) == 1
    assert clamped[0]["segment_index"] == 7
    assert clamped[0]["left_lineage"] == [1]
    assert clamped[0]["right_lineage"] == [2]
    assert clamped[0]["added_volume_m3"] == 0.0
    assert clamped[0]["raw_added_volume_m3"] < 0.0
    assert clamped[0]["tolerance_m3"] > abs(clamped[0]["raw_added_volume_m3"])


def test_real_reduction_hierarchy_is_nested_for_all_thresholds() -> None:
    """K32→K16→K8 must be prefixes of one deterministic hierarchy per base."""
    for threshold_m, base_count in ((0.005, 32), (0.010, 29), (0.020, 18)):
        traces = {}
        for target in (32, 16, 8):
            reduced, trace = reduce_segmented_parts(
                _load_base_parts(threshold_m),
                target_hulls=target,
            )
            assert sum(len(parts) for parts in reduced.values()) <= target
            assert sum(len(parts) for parts in reduced.values()) >= 8
            traces[target] = trace
        assert len(traces[32]) == max(0, base_count - 32)
        assert len(traces[16]) == max(0, base_count - 16)
        assert len(traces[8]) == base_count - 8
        assert traces[8][: len(traces[16])] == traces[16]
        assert traces[16][: len(traces[32])] == traces[32]


def main() -> int:
    """Run v5 tests without pytest discovery."""
    tests = (
        test_v5_candidate_family_contract,
        test_v4_numeric_failure_is_preserved_and_quantified,
        test_real_t010_negative_roundoff_is_clamped_and_recorded,
        test_real_reduction_hierarchy_is_nested_for_all_thresholds,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E182_SEGMENTED_COACD_V5_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
