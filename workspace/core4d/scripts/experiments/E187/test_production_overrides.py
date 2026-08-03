#!/usr/bin/env python3
"""Static contracts for E187 production overrides."""

from __future__ import annotations

from build_production_overrides import (
    ALLOWED_EFFECTIVE_DIFF,
    METHOD_ID,
    override_id,
    override_payload,
    read_rows,
)
from freeze_reward_grid_lock import build_payload as build_lock


def test_override_identity_and_axes() -> None:
    """Only the frozen reward/grid axes are available to E187 overrides."""
    assert METHOD_ID == "E187_canonical_distance_continuation_reward_r1"
    assert {
        "surface_band_score_mode",
        "surface_band_continuation_far_weight",
        "surface_band_continuation_near_weight",
        "surface_band_continuation_far_scale_m",
        "surface_band_continuation_near_scale_m",
        "surface_band_continuation_smooth_delta_m",
        "object_distance_manifest",
        "object_distance_error_bound_m",
    } == ALLOWED_EFFECTIVE_DIFF
    rows = read_rows()
    assert len(rows) == 22
    assert len({override_id(row["case_id"]) for row in rows}) == 22


def test_payload_uses_object_lock() -> None:
    """Every row inherits E186 and selects its object-specific frozen grid."""
    lock = build_lock()
    for row in read_rows():
        payload = override_payload(row, lock)
        assert payload["defaults"] == [row["override_id"], "_self_"]
        assert payload["surface_band_score_mode"] == "distance_continuation"
        selected = lock["objects"][row["object_key"]]
        assert payload["object_distance_manifest"] == selected["grid_manifest"]["path"]
        assert payload["object_distance_error_bound_m"] == selected["epsilon_grid_m"]


def main() -> int:
    """Run direct-main contracts."""
    tests = (test_override_identity_and_axes, test_payload_uses_object_lock)
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E187_PRODUCTION_OVERRIDE_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
