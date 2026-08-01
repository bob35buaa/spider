#!/usr/bin/env python3
"""Synthetic direct-main tests for E182 S2 screen scoring and selection."""

from __future__ import annotations

import copy

import numpy as np
from run_task_query_screen import _select_group, normalized_task_score


def synthetic_result() -> dict:
    """Return a minimal result exactly on every launch-floor denominator."""
    return {
        "point_metrics": {
            "sign_disagreement_fraction": 0.15,
            "deep_sign_mismatch_fraction": 0.05,
            "absolute_error_p90_m": 0.03,
        },
        "p_pose_contact": {"precision": 0.70, "recall": 0.70},
        "r_shadow": {
            "normalized_error_p90": 0.30,
            "geometry_spearman_median": 0.80,
        },
        "g_shadow": {
            "optimizer_combined": {
                "mask_flip_fraction": 0.15,
                "false_reject_fraction": 0.15,
                "false_accept_fraction": 0.15,
            },
            "candidate_fallback_fraction": 0.30,
            "raw_fallback_fraction": 0.10,
            "topk_overlap_mean": 0.0,
        },
    }


def test_normalized_task_score_contract() -> None:
    """Every launch denominator must map to one without hidden weights."""
    score = normalized_task_score(synthetic_result())
    assert all(np.isclose(score[name], 1.0) for name in ("P", "R", "G", "worst"))
    improved = copy.deepcopy(synthetic_result())
    improved["g_shadow"]["candidate_fallback_fraction"] = 0.0
    improved["g_shadow"]["topk_overlap_mean"] = 1.0
    improved["g_shadow"]["optimizer_combined"] = {
        "mask_flip_fraction": 0.0,
        "false_reject_fraction": 0.0,
        "false_accept_fraction": 0.0,
    }
    assert normalized_task_score(improved)["G"] == 0.0


def test_group_selection_near_tie_prefers_actual_hulls() -> None:
    """A <=10% score near-tie must prefer fewer actual hulls before runtime."""
    common = {
        "launch_floor_status": "PASS",
        "threshold_m": 0.01,
        "max_vertices": 32,
    }
    rows = [
        {
            **common,
            "candidate_id": "best_but_many",
            "score_worst": 0.90,
            "actual_hulls": 8,
            "wall_seconds": 1.0,
        },
        {
            **common,
            "candidate_id": "near_fewer",
            "score_worst": 0.99,
            "actual_hulls": 7,
            "wall_seconds": 2.0,
        },
        {
            **common,
            "candidate_id": "outside_tie",
            "score_worst": 1.01,
            "actual_hulls": 6,
            "wall_seconds": 0.5,
        },
    ]
    selected = _select_group(rows)
    assert selected["status"] == "SELECTED_FOR_FINALIST_REVIEW"
    assert selected["near_tie_count"] == 2
    assert selected["selected"]["candidate_id"] == "near_fewer"


def test_group_selection_rejects_failed_floor() -> None:
    """Score cannot rescue a candidate that fails the catastrophic launch floor."""
    rows = [
        {
            "candidate_id": "failed",
            "launch_floor_status": "FAIL",
            "score_worst": 0.0,
            "actual_hulls": 1,
            "wall_seconds": 0.0,
            "threshold_m": 0.0,
            "max_vertices": 1,
        }
    ]
    selected = _select_group(rows)
    assert selected["status"] == "NO_LAUNCH_FLOOR_CANDIDATE"
    assert selected["selected"] is None


def main() -> int:
    """Run screen-contract tests without pytest discovery."""
    tests = (
        test_normalized_task_score_contract,
        test_group_selection_near_tie_prefers_actual_hulls,
        test_group_selection_rejects_failed_floor,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E182_TASK_QUERY_SCREEN_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
