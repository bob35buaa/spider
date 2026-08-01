#!/usr/bin/env python3
"""Direct-main tests for the E182 bucket003 segmented CoACD attempt2."""

from __future__ import annotations

from build_segmented_coacd_attempt2 import (
    MAX_HULLS,
    MAX_VERTICES,
    SEGMENT_COUNT,
    THRESHOLDS_M,
    _candidate_asset_sha256,
    _candidate_id,
    _load_oracle,
    partition_mesh,
)
from evaluate_task_queries import _candidate_asset_sha256 as evaluator_asset_sha256


def test_candidate_family_contract() -> None:
    """The attempt2 family must remain exactly three thresholds by K8/16/32."""
    identifiers = [
        _candidate_id(threshold_m, max_hulls)
        for threshold_m in THRESHOLDS_M
        for max_hulls in MAX_HULLS
    ]
    assert len(identifiers) == 9
    assert len(set(identifiers)) == 9
    assert MAX_HULLS == (8, 16, 32)
    assert MAX_VERTICES == 32
    assert all(max_hulls % SEGMENT_COUNT == 0 for max_hulls in MAX_HULLS)
    assert identifiers[0] == "segx2y4_t005_k08_v032"
    assert identifiers[-1] == "segx2y4_t020_k32_v032"


def test_real_oracle_partition_contract() -> None:
    """The real bucket003 solid must close exactly across eight valid intersections."""
    _, _, mesh = _load_oracle()
    cells = partition_mesh(mesh)
    assert len(cells) == SEGMENT_COUNT
    assert [(cell[0], cell[1]) for cell in cells] == [
        (x_index, y_index) for x_index in range(2) for y_index in range(4)
    ]
    segments = [cell[4] for cell in cells]
    assert all(
        not segment.is_empty
        and segment.is_watertight
        and segment.is_winding_consistent
        and float(segment.volume) > 0.0
        for segment in segments
    )
    assert abs(sum(float(segment.volume) for segment in segments) - mesh.volume) < 1e-8


def test_candidate_digest_matches_evaluator() -> None:
    """Attempt2 manifests must use the evaluator's exact ordered identity digest."""
    candidate = {
        "object_key": "bucket003",
        "parameters": {"threshold_m": 0.005, "max_convex_hull": 8},
        "parts": [
            {
                "part_index": 0,
                "sha256": "a" * 64,
                "vertex_count": 8,
                "face_count": 12,
            },
            {
                "part_index": 1,
                "sha256": "b" * 64,
                "vertex_count": 10,
                "face_count": 16,
            },
        ],
    }
    assert _candidate_asset_sha256(candidate) == evaluator_asset_sha256(candidate)


def main() -> int:
    """Run attempt2 tests without pytest discovery."""
    tests = (
        test_candidate_family_contract,
        test_real_oracle_partition_contract,
        test_candidate_digest_matches_evaluator,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E182_SEGMENTED_COACD_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
