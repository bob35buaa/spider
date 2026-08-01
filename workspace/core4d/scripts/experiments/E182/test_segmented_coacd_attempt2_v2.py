#!/usr/bin/env python3
"""Direct-main tests for E182 segmented CoACD attempt2-v2 contracts."""

from __future__ import annotations

import trimesh
from build_segmented_coacd_attempt2_v2 import (
    COACD_MAX_VERTICES,
    K8_MAX_VERTICES,
    _load_v1_segments,
    _v1_failure_evidence,
    candidate_family,
)
from e182_common import repo_path


def test_v2_candidate_family_contract() -> None:
    """V2 must contain one K8 hull fallback and three CoACD rows for K16/K32."""
    rows = candidate_family()
    assert len(rows) == 7
    assert len({row["candidate_id"] for row in rows}) == 7
    assert [row["max_hulls"] for row in rows].count(8) == 1
    assert [row["max_hulls"] for row in rows].count(16) == 3
    assert [row["max_hulls"] for row in rows].count(32) == 3
    k8 = [row for row in rows if row["max_hulls"] == 8][0]
    assert k8 == {
        "candidate_id": "segx2y4_hull_k08_v160",
        "method": "EXACT_SEGMENT_CONVEX_HULL",
        "threshold_m": 0.0,
        "max_hulls": 8,
        "per_segment_max_hulls": 1,
        "max_vertices": K8_MAX_VERTICES,
    }
    assert K8_MAX_VERTICES == 160
    assert COACD_MAX_VERTICES == 32
    assert all(
        row["max_vertices"] == 32 for row in rows if row["method"] == "SEGMENTED_COACD"
    )


def test_v1_failure_is_preserved_before_v2() -> None:
    """V2 must bind the exact v1 partial prefix rather than overwrite its failure."""
    failure = _v1_failure_evidence()
    assert failure["status"] == "BUILD_FAILED_BEFORE_CANDIDATE_MANIFEST"
    assert failure["signature"] == "segment hull cap violated"
    assert failure["successful_prefix_part_count"] == 3
    assert failure["failed_segment_index"] == 3
    assert failure["failed_per_segment_cap"] == 1


def test_real_k8_segment_hulls_fit_explicit_v160_ceiling() -> None:
    """All eight exact segment hulls must be valid and fit v160 but not hidden v32."""
    segments = _load_v1_segments()["segments"]
    hulls = [
        trimesh.load(
            repo_path(segment["path"]), force="mesh", process=False
        ).convex_hull
        for segment in segments
    ]
    vertex_counts = [len(hull.vertices) for hull in hulls]
    assert len(hulls) == 8
    assert max(vertex_counts) <= K8_MAX_VERTICES
    assert max(vertex_counts) > COACD_MAX_VERTICES
    assert all(
        hull.is_convex and hull.is_watertight and hull.is_winding_consistent
        for hull in hulls
    )


def main() -> int:
    """Run v2 tests without pytest discovery."""
    tests = (
        test_v2_candidate_family_contract,
        test_v1_failure_is_preserved_before_v2,
        test_real_k8_segment_hulls_fit_explicit_v160_ceiling,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E182_SEGMENTED_COACD_V2_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
