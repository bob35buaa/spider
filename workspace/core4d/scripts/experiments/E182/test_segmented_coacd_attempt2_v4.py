#!/usr/bin/env python3
"""Direct-main tests for deterministic global-budget CoACD attempt2-v4."""

from __future__ import annotations

import numpy as np
import trimesh
from build_segmented_coacd_attempt2_v4 import (
    BASE_MAX_HULLS_PER_SEGMENT,
    FINAL_MAX_VERTICES,
    SEGMENT_COUNT,
    ReducerPart,
    _v3_failure_evidence,
    candidate_family,
    reduce_segmented_parts,
)


def _box(
    *, segment_index: int, local_index: int, x: float, size: float = 1.0
) -> ReducerPart:
    mesh = trimesh.creation.box(extents=(size, size, size))
    mesh.apply_translation((x, 2.0 * segment_index, 0.0))
    return ReducerPart(
        mesh=mesh,
        segment_index=segment_index,
        lineage=(local_index,),
    )


def test_v4_candidate_family_contract() -> None:
    """V4 must derive K8/16/32 from one base decomposition per threshold."""
    rows = candidate_family()
    assert len(rows) == 9
    assert len({row["candidate_id"] for row in rows}) == 9
    assert {row["threshold_m"] for row in rows} == {0.005, 0.010, 0.020}
    assert {row["max_hulls"] for row in rows} == {8, 16, 32}
    assert all(row["method"] == "ISOLATED_COACD_GLOBAL_BUDGET_MERGE" for row in rows)
    assert all(row["base_max_hulls_per_segment"] == 4 for row in rows)
    assert all(row["max_vertices"] == FINAL_MAX_VERTICES for row in rows)
    assert BASE_MAX_HULLS_PER_SEGMENT == 4
    assert SEGMENT_COUNT == 8


def test_v3_cap_failure_is_preserved() -> None:
    """V4 must bind the exact isolated cap failure rather than overwrite it."""
    evidence = _v3_failure_evidence()
    assert evidence["status"] == "V3_ISOLATED_EQUAL_CAP_BUILD_FAILED"
    assert evidence["candidate_id"] == "segx2y4_iso_coacd_t005_k16_v032"
    assert evidence["failed_segment_index"] == 3
    assert evidence["actual_hulls"] == 3
    assert evidence["requested_cap"] == 2
    assert evidence["successful_segment_count"] == 7
    assert evidence["changed_assumption"] == "GLOBAL_BUDGET_NOT_EQUAL_PER_SEGMENT_CAP"


def test_reducer_enforces_global_budget_without_cross_segment_merge() -> None:
    """The reducer must hit K while retaining at least one hull in each segment."""
    parts = {
        0: [_box(segment_index=0, local_index=i, x=0.8 * i) for i in range(3)],
        1: [_box(segment_index=1, local_index=i, x=0.8 * i) for i in range(2)],
    }
    reduced, trace = reduce_segmented_parts(parts, target_hulls=3)
    assert sum(len(group) for group in reduced.values()) == 3
    assert set(reduced) == {0, 1}
    assert all(group for group in reduced.values())
    assert len(trace) == 2
    assert all(step["segment_index"] in {0, 1} for step in trace)
    for segment_index, group in reduced.items():
        assert all(part.segment_index == segment_index for part in group)
        assert all(part.mesh.is_convex for part in group)
        assert all(part.mesh.is_watertight for part in group)
    try:
        reduce_segmented_parts(parts, target_hulls=1)
    except ValueError as error:
        assert "segment count" in str(error)
    else:
        raise AssertionError("target below segment count must fail")


def test_reducer_is_deterministic_and_conservative() -> None:
    """Stable lineage tie-breaks must produce identical conservative merged hulls."""
    parts = {
        0: [_box(segment_index=0, local_index=i, x=0.75 * i) for i in range(3)],
        1: [_box(segment_index=1, local_index=i, x=0.75 * i) for i in range(3)],
    }
    first, first_trace = reduce_segmented_parts(parts, target_hulls=2)
    second, second_trace = reduce_segmented_parts(parts, target_hulls=2)
    assert first_trace == second_trace
    assert [part.lineage for group in first.values() for part in group] == [
        part.lineage for group in second.values() for part in group
    ]
    for segment_index in (0, 1):
        merged = first[segment_index][0].mesh
        source_vertices = np.vstack(
            [part.mesh.vertices for part in parts[segment_index]]
        )
        assert merged.is_convex and merged.is_watertight
        assert len(merged.vertices) <= FINAL_MAX_VERTICES
        assert trimesh.proximity.signed_distance(merged, source_vertices).min() >= -1e-8


def main() -> int:
    """Run v4 tests without pytest discovery."""
    tests = (
        test_v4_candidate_family_contract,
        test_v3_cap_failure_is_preserved,
        test_reducer_enforces_global_budget_without_cross_segment_merge,
        test_reducer_is_deterministic_and_conservative,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E182_SEGMENTED_COACD_V4_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
