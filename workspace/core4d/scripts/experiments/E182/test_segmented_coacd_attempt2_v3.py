#!/usr/bin/env python3
"""Direct-main tests for isolated segmented CoACD attempt2-v3."""

from __future__ import annotations

import os

from build_segmented_coacd_attempt2_v3 import (
    CHILD_ROOT,
    SEGMENT_PARALLEL_WORKERS,
    _v2_failure_evidence,
    candidate_family,
)


def test_v3_candidate_family_contract() -> None:
    """V3 must reuse K8 and provide six uniquely identified isolated CoACD rows."""
    rows = candidate_family()
    assert len(rows) == 7
    assert len({row["candidate_id"] for row in rows}) == 7
    assert [row["max_hulls"] for row in rows].count(8) == 1
    assert [row["max_hulls"] for row in rows].count(16) == 3
    assert [row["max_hulls"] for row in rows].count(32) == 3
    k8 = rows[0]
    assert k8["method"] == "REUSE_V2_EXACT_SEGMENT_CONVEX_HULL"
    assert k8["candidate_id"] == "segx2y4_hull_k08_v160"
    coacd_rows = rows[1:]
    assert all(row["method"] == "ISOLATED_SEGMENTED_COACD" for row in coacd_rows)
    assert all("_iso_coacd_" in row["candidate_id"] for row in coacd_rows)
    assert {row["per_segment_max_hulls"] for row in coacd_rows} == {2, 4}
    assert {row["max_vertices"] for row in coacd_rows} == {32}


def test_v2_same_process_failure_is_preserved() -> None:
    """V3 must bind K8 success and the exact six-part K16 failure prefix."""
    evidence = _v2_failure_evidence()
    assert evidence["status"] == "K8_PASS_K16_SAME_PROCESS_BUILD_FAILED"
    assert evidence["signature"] == "segment 3 hull cap violated"
    assert evidence["successful_prefix_part_count"] == 6
    assert (
        evidence["root_cause"] == "COACD_NATIVE_STATE_NOT_ISOLATED_ACROSS_SEGMENT_CALLS"
    )
    assert evidence["changed_execution_contract"] == "ONE_FRESH_PROCESS_PER_SEGMENT"


def test_isolated_child_resource_contract() -> None:
    """Four unique child roots may run concurrently while each native backend stays single-threaded."""
    assert SEGMENT_PARALLEL_WORKERS == 4
    assert all(
        os.environ[name] == "1"
        for name in ("OMP_NUM_THREADS", "TBB_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    )
    sample = "segx2y4_iso_coacd_t005_k16_v032"
    paths = [CHILD_ROOT / sample / f"segment_{index:03d}" for index in range(8)]
    assert len(set(paths)) == 8
    assert all(path.parent.name == sample for path in paths)


def main() -> int:
    """Run v3 tests without pytest discovery."""
    tests = (
        test_v3_candidate_family_contract,
        test_v2_same_process_failure_is_preserved,
        test_isolated_child_resource_contract,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E182_SEGMENTED_COACD_V3_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
