#!/usr/bin/env python3
"""Direct-main contracts for the post-search v7 visual diagnostic."""

from __future__ import annotations

from render_segment3_partition_v7 import (
    EXPECTED_SIGNATURES,
    _load_search,
    contact_signature,
    hybrid_parts,
    select_diagnostic_rows,
)


def test_v7_visual_rows_cover_both_discrete_signatures() -> None:
    """The diagnostic must explain both and only the observed P work points."""
    rows = select_diagnostic_rows(_load_search())
    assert tuple(contact_signature(row) for row in rows) == EXPECTED_SIGNATURES
    assert [row["diagnostic_label"] for row in rows] == [
        "precision_side",
        "recall_side",
    ]
    assert all(row["p_floor_pass"] is False for row in rows)


def test_v7_visual_reconstructs_frozen_hull_counts() -> None:
    """Each display must rebuild the exact K16 hybrid context plus its partition."""
    rows = select_diagnostic_rows(_load_search())
    for row in rows:
        parts = hybrid_parts(row)
        assert len(parts) == row["actual_hulls"]
        assert all(part.is_convex and part.is_watertight for part in parts)


def main() -> int:
    """Run v7 visual contracts without pytest discovery."""
    tests = (
        test_v7_visual_rows_cover_both_discrete_signatures,
        test_v7_visual_reconstructs_frozen_hull_counts,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E182_SEGMENT3_PARTITION_V7_VISUAL_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
