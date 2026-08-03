#!/usr/bin/env python3
"""Static authority tests for E187 same-tape R/G efficiency."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/eval/runners"))

from eval_E187_reward_grid_efficiency import CASE_SPECS, authority  # noqa: E402


def test_three_object_routes() -> None:
    """Exactly three formal representative cases enter A1 efficiency."""
    assert tuple(CASE_SPECS) == (
        "bucket003_20231018_003_p1",
        "bucket004_20231002_021_p1",
        "bucket007_20231020_055_p1",
    )


def test_authority_paths_exist() -> None:
    """Every route closes scene/config/tape/baseline/chosen inputs."""
    for case_id in CASE_SPECS:
        result = authority(case_id)
        assert set(result["paths"]) == {
            "scene",
            "config",
            "tape",
            "baseline_grid",
            "chosen_grid",
        }


def main() -> int:
    """Run direct-main contracts."""
    tests = (test_three_object_routes, test_authority_paths_exist)
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E187_REWARD_GRID_EFFICIENCY_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
