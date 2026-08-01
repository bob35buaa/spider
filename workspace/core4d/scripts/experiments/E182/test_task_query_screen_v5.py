#!/usr/bin/env python3
"""Direct-main contracts for the v5 one-case task-query screen adapter."""

from __future__ import annotations

import numpy as np
from build_segmented_coacd_attempt2_v5 import FIXTURE_PATH
from evaluate_task_queries_v5 import (
    BOUNDED_CANDIDATE_ID,
    load_v5_case_indices,
    load_v5_fixture,
)
from run_task_query_screen import SCORING_CONTRACT
from run_task_query_screen_v5 import _load_fixture


def test_v5_fixture_authority_contract() -> None:
    """The adapter must accept only the frozen bucket003 nine-candidate fixture."""
    fixture, fixture_sha = load_v5_fixture(FIXTURE_PATH)
    assert fixture["status"] == "FROZEN"
    assert fixture["heldout_access"] == "NOT_ACCESSED_DEV3_ONLY"
    assert fixture["candidate_count"] == 9
    assert len(fixture["cases"]) == 1
    assert fixture["cases"][0]["case_id"] == "bucket003_20231018_001_p1"
    assert len(fixture_sha) == 64
    assert _load_fixture(FIXTURE_PATH) == fixture


def test_v5_index_adapter_reuses_frozen_original_indices() -> None:
    """Derived fixtures must resolve the unchanged nested index NPZ by source SHA."""
    fixture, _ = load_v5_fixture(FIXTURE_PATH)
    case = fixture["cases"][0]
    screen, screen_sha = load_v5_case_indices(case, "screen")
    finalist, finalist_sha = load_v5_case_indices(case, "finalist")
    full, full_sha = load_v5_case_indices(case, "production_canary")
    assert screen_sha == finalist_sha == full_sha == case["indices"]["sha256"]
    assert len(screen) == 197
    assert len(finalist) == 581
    assert len(full) == 1669
    assert np.isin(screen, finalist).all()
    assert np.isin(finalist, full).all()


def test_v5_screen_group_and_scoring_contract() -> None:
    """The screen must form three 3-candidate K groups without changing scoring."""
    fixture = _load_fixture(FIXTURE_PATH)
    assert SCORING_CONTRACT["name"] == "P_R_G_WORST_NORMALIZED_LAUNCH_FLOOR_ERROR_V1"
    assert SCORING_CONTRACT["worst"] == "MAX(P,R,G)"
    for max_hulls in (8, 16, 32):
        rows = [row for row in fixture["candidates"] if row["max_hulls"] == max_hulls]
        assert len(rows) == 3
        assert {row["threshold_m"] for row in rows} == {0.005, 0.010, 0.020}
    assert BOUNDED_CANDIDATE_ID in {
        candidate["candidate_id"] for candidate in fixture["candidates"]
    }


def main() -> int:
    """Run v5 adapter contracts without pytest discovery."""
    tests = (
        test_v5_fixture_authority_contract,
        test_v5_index_adapter_reuses_frozen_original_indices,
        test_v5_screen_group_and_scoring_contract,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E182_TASK_QUERY_SCREEN_V5_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
