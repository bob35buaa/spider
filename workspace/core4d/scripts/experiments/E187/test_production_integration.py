#!/usr/bin/env python3
"""Direct-main contracts for the E187 A2 production integration audit."""

from __future__ import annotations

import numpy as np
from audit_production_integration import (
    EXPECTED_CASES_BY_OBJECT,
    ROBOT_OBJECT_GEOMS,
    load_authority,
    recorder_off_contract,
)
from audit_reference_final_gate import false_safe_counts


def test_authority_is_exact_keep22() -> None:
    """The integration audit consumes only the frozen 5/4/13 authority."""
    overrides, scenes, lock = load_authority()
    assert len(overrides["rows"]) == len(scenes) == 22
    assert EXPECTED_CASES_BY_OBJECT == {"bucket003": 5, "bucket004": 4, "bucket007": 13}
    assert len(ROBOT_OBJECT_GEOMS) == 18
    assert (
        overrides["reward_grid_lock"]["sha256"]
        == "2cc949e19cb313e58883fd5d0446872220188ea9ecce06cb23d39e2a6b42194f"
    )
    assert (
        lock["file_sha256"]
        == "6a20df7c2b47d0d5a5ea104bdd18ab129753db7c3a14e5f82aa0ce93a1a66065"
    )


def test_recorder_is_default_off() -> None:
    """Production integration must not create query-tape artifacts."""
    contract = recorder_off_contract()
    assert contract == {
        "query_tape_enabled": False,
        "query_tape_record_geometry_state": False,
        "raw_chunks_exists": False,
        "status": "PASS",
    }


def test_conservative_gate_has_no_false_safe() -> None:
    """Subtracting epsilon may reject, but cannot falsely accept this boundary tape."""
    exact = {
        "body": np.array([0.01, 0.02, 0.03]),
        "hand": np.array([0.00, 0.01, 0.02]),
        "leg": np.array([-0.01, 0.00, 0.01]),
    }
    grid = {key: value + 0.002 for key, value in exact.items()}
    counts = false_safe_counts(
        exact, grid, epsilon_grid_m=0.002, thresholds=dict.fromkeys(exact, 0.0)
    )
    assert counts == {"body": 0, "hand": 0, "leg": 0, "combined": 0}


def main() -> int:
    """Run direct-main contracts."""
    tests = (
        test_authority_is_exact_keep22,
        test_recorder_is_default_off,
        test_conservative_gate_has_no_false_safe,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E187_PRODUCTION_INTEGRATION_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
