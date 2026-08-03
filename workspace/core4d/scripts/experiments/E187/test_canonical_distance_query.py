#!/usr/bin/env python3
"""Direct-main contracts for shared exact convex-union query primitives."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts"))

from eval.core.canonical_distance_query import (  # noqa: E402
    build_exact_scene,
    build_exact_union_mesh,
    convex_union_halfspaces,
    exact_signed_distance,
    group_min,
)


def test_exact_signed_distance_sign_and_magnitude() -> None:
    """One convex authority has negative-inside and finite metric magnitude."""
    box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    parts = [box]
    union = build_exact_union_mesh(parts)
    distance = exact_signed_distance(
        build_exact_scene(union),
        convex_union_halfspaces(parts),
        np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    np.testing.assert_allclose(distance, [-1.0, 0.5, 0.0], atol=1e-6, rtol=0.0)


def test_overlapping_union_containment() -> None:
    """Analytic sign treats either convex part as inside the union."""
    left = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    right = left.copy()
    right.apply_translation((1.0, 0.0, 0.0))
    volumes = convex_union_halfspaces([left, right])
    union = build_exact_union_mesh([left, right])
    distance = exact_signed_distance(
        build_exact_scene(union),
        volumes,
        np.array([[-0.5, 0.0, 0.0], [1.5, 0.0, 0.0], [2.5, 0.0, 0.0]]),
    )
    assert distance[0] < 0.0
    assert distance[1] < 0.0
    assert distance[2] > 0.0


def test_group_min_preserves_row_order() -> None:
    """Configured groups reduce only their exact per-geom columns."""
    values = np.array([[3.0, 1.0, 2.0], [-2.0, 4.0, -1.0]])
    np.testing.assert_array_equal(group_min(values, [8, 3, 5], [5, 8]), [2.0, -2.0])


def main() -> int:
    """Run all contracts without pytest collection dependencies."""
    tests = (
        test_exact_signed_distance_sign_and_magnitude,
        test_overlapping_union_containment,
        test_group_min_preserves_row_order,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E187_CANONICAL_DISTANCE_QUERY_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
