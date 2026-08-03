#!/usr/bin/env python3
"""Static contracts for E187's unlocked bucket007 2.5mm candidate."""

from __future__ import annotations

from bake_bucket007_grid_candidate import (
    DEFAULT_OUTPUT_ROOT,
    FORMAL_POINTS,
    MARGIN_M,
    OBJECT_KEY,
    SMOKE_POINTS,
    VOXEL_SIZE_M,
    preflight,
)


def test_frozen_candidate_identity() -> None:
    """The coarse-to-fine tree unlocks only bucket007 at 2.5mm."""
    assert OBJECT_KEY == "bucket007"
    assert VOXEL_SIZE_M == 0.0025
    assert MARGIN_M == 0.120
    assert SMOKE_POINTS == 50_000
    assert FORMAL_POINTS == 1_000_000


def test_preflight_closes_frozen_collider() -> None:
    """Preflight resolves the E186 frozen collider without creating artifacts."""
    result = preflight(DEFAULT_OUTPUT_ROOT)
    assert result["status"] == "PASS"
    assert len(result["candidate_asset_sha256"]) == 64
    assert len(result["ordered_parts_sha256"]) == 64


def main() -> int:
    """Run direct-main contracts."""
    tests = (test_frozen_candidate_identity, test_preflight_closes_frozen_collider)
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E187_BUCKET007_GRID_CANDIDATE_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
