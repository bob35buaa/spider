#!/usr/bin/env python3
"""Direct-main contracts for the E187 A1 reward/grid lock."""

from __future__ import annotations

from freeze_reward_grid_lock import build_payload


def test_lock_builds_from_passed_evidence() -> None:
    """All three chosen grids and all global gates close."""
    payload = build_payload()
    assert payload["status"] == "FROZEN"
    assert set(payload["objects"]) == {"bucket003", "bucket004", "bucket007"}
    assert all(payload["global_gates"].values())


def test_resolution_and_waiver_contract() -> None:
    """The lock preserves selected resolutions and honest Gate0 semantics."""
    payload = build_payload()
    assert payload["gate0_technical_status"] == "FAIL"
    assert payload["progression_authority"] == "USER_WAIVED"
    assert {key: row["resolution_m"] for key, row in payload["objects"].items()} == {
        "bucket003": 0.005,
        "bucket004": 0.0025,
        "bucket007": 0.0025,
    }
    assert payload["objects"]["bucket007"]["rejected_5mm_topk_overlap_frac"] < 0.90


def main() -> int:
    """Run all lock contracts."""
    tests = (test_lock_builds_from_passed_evidence, test_resolution_and_waiver_contract)
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E187_REWARD_GRID_LOCK_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
