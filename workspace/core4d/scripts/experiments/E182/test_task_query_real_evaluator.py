#!/usr/bin/env python3
"""Artifact-backed direct-main tests for the E182 S2 bounded real evaluator."""

from __future__ import annotations

import json

from e182_common import REPO_ROOT, sha256_file

BOUNDED_ROOT = (
    REPO_ROOT
    / "workspace/core4d/results/E182/s2_task_query_eval/bounded_real"
    / "bucket007_20231020_055_p1__t005_k08_v032"
)


def test_bounded_real_artifact_contract() -> None:
    """The fixed one-chunk artifact must close schema/count/SHA without selection use."""
    audit_path = BOUNDED_ROOT / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert audit["selection_eligible"] is False
    assert audit["heldout_access"] == "NOT_ACCESSED_DEV3_ONLY"
    assert audit["expected_query_count"] == 1_242_742
    assert all(audit["checks"].values())
    for key in ("oracle_manifest", "candidate_result"):
        entry = audit[key]
        path = REPO_ROOT / entry["path"]
        assert path.is_file()
        assert sha256_file(path) == entry["sha256"]


def test_bounded_real_prg_metrics_are_finite_and_complete() -> None:
    """The real result must expose P/R/G outputs while remaining non-selection evidence."""
    result = json.loads(
        (BOUNDED_ROOT / "candidate_result.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "COMPLETE"
    assert result["selection_eligible"] is False
    assert result["point_metrics"]["query_count"] == 1_242_742
    assert result["point_metrics"]["nonfinite_count"] == 0
    assert result["p_pose_contact"]["pose_count"] == 382
    assert result["r_shadow"]["chunk_count"] == 1
    assert result["g_shadow"]["chunk_count"] == 1
    assert result["g_shadow"]["optimizer_combined"]["sample_count"] == 64
    assert result["launch_floor"]["total"] == 12


def main() -> int:
    """Run artifact-backed tests without pytest discovery."""
    tests = (
        test_bounded_real_artifact_contract,
        test_bounded_real_prg_metrics_are_finite_and_complete,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E182_TASK_QUERY_REAL_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
