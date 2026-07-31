#!/usr/bin/env python3
"""Direct-main tests for the pre-score E182 S2 query-density fixture."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from build_task_query_fixture import (
    DEFAULT_GATE1_AUDIT,
    build_task_query_fixture,
)


def test_real_dev3_progressive_fixture() -> None:
    """The frozen S2 fixture must retain every pose and nested point-density tiers."""
    with tempfile.TemporaryDirectory(prefix="e182_s2_fixture_") as directory:
        payload = build_task_query_fixture(Path(directory))
        assert payload["status"] == "FROZEN"
        assert payload["heldout_access"] == "NOT_ACCESSED_DEV3_ONLY"
        assert payload["gate1"]["status"] == "PASS"
        assert payload["tier_contract"] == {
            "screen": "ALL_54_CANDIDATES_MESH64_ALL_POSES",
            "finalist": "BEST_PER_OBJECT_K_MESH256_ALL_POSES",
            "production_canary": "SELECTED_CANDIDATE_FULL_POINTS",
        }
        assert len(payload["cases"]) == 3
        for case in payload["cases"]:
            assert case["cem_pose_policy"] == "ALL_CHUNKS_SAMPLES_HORIZON"
            assert case["full_point_count"] == 1669
            assert case["screen_point_count"] == 197
            assert case["finalist_point_count"] == 581
            assert case["screen_query_count"] < case["finalist_query_count"]
            assert case["finalist_query_count"] < case["full_query_count"]
            with np.load(Path(directory) / case["indices"]["relative_path"]) as values:
                full = values["full_point_indices"]
                screen = values["screen_point_indices"]
                finalist = values["finalist_point_indices"]
                primitive = values["primitive_point_indices"]
                assert np.array_equal(full, np.arange(1669))
                assert set(screen).issubset(set(finalist))
                assert set(finalist).issubset(set(full))
                assert set(primitive).issubset(set(screen))
                assert len(screen) == len(np.unique(screen))
                assert len(finalist) == len(np.unique(finalist))
            assert all(
                value > 0 for value in case["screen_consumer_point_counts"].values()
            )


def test_gate1_must_be_pass_before_fixture_freeze() -> None:
    """S2 fixture construction must reject any non-PASS Gate1 identity."""
    with tempfile.TemporaryDirectory(prefix="e182_s2_gate1_") as directory:
        root = Path(directory)
        changed = json.loads(DEFAULT_GATE1_AUDIT.read_text())
        changed["status"] = "FAIL"
        gate1 = root / "gate1.json"
        gate1.write_text(json.dumps(changed))
        try:
            build_task_query_fixture(root / "out", gate1_path=gate1)
        except RuntimeError as error:
            assert "Gate1" in str(error)
        else:
            raise AssertionError("S2 fixture accepted a failed Gate1")


def main() -> int:
    """Run tests without pytest discovery."""
    tests = (
        test_real_dev3_progressive_fixture,
        test_gate1_must_be_pass_before_fixture_freeze,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E182_TASK_QUERY_FIXTURE_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
