#!/usr/bin/env python3
"""Synthetic direct-main tests for E182 exact-C and D_M query geometry."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import trimesh
from evaluate_task_queries import (
    build_exact_union_mesh,
    compare_gate_validity,
    compare_signed_distance_pair,
    compute_gate_samples,
    consumer_min_clearance,
    geometry_reward_components,
    mesh_signed_distance,
)


def translated_box(extents: tuple[float, float, float], x: float) -> trimesh.Trimesh:
    """Create a watertight axis-aligned box with one x translation."""
    transform = trimesh.transformations.translation_matrix([x, 0.0, 0.0])
    return trimesh.creation.box(extents=extents, transform=transform)


def test_overlapping_parts_use_true_union_boundary() -> None:
    """Internal part faces must not shorten negative exact-union distance."""
    parts = [
        translated_box((2.0, 2.0, 2.0), -0.5),
        translated_box((2.0, 2.0, 2.0), 0.5),
    ]
    union = build_exact_union_mesh(parts)
    assert union.is_watertight
    assert union.is_winding_consistent
    assert np.isclose(union.volume, 12.0)

    queries = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.9, 0.0],
            [1.5, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    signed = mesh_signed_distance(union, queries)
    assert np.allclose(signed, [-1.0, 0.5, -0.1, 0.0], atol=2e-5)

    # A per-part minimum would incorrectly return -0.5 at the overlap center.
    part_center = [mesh_signed_distance(part, queries[:1])[0] for part in parts]
    assert np.isclose(min(part_center), -0.5, atol=2e-5)
    assert signed[0] < min(part_center)


def test_pair_metrics_apply_query_radius_before_sign_gates() -> None:
    """P/R/G sign comparisons must operate on surface clearance, not center SDF."""
    oracle = np.asarray([-0.10, 0.01, 0.03, -0.04], dtype=np.float64)
    candidate = np.asarray([0.10, -0.01, 0.04, 0.01], dtype=np.float64)
    radii = np.asarray([0.0, 0.0, 0.02, 0.0], dtype=np.float64)
    metrics = compare_signed_distance_pair(
        candidate,
        oracle,
        radii_m=radii,
        deep_margin_m=0.02,
    )
    assert metrics["query_count"] == 4
    assert metrics["finite_count"] == 4
    assert metrics["nonfinite_count"] == 0
    assert metrics["sign_disagreement_count"] == 3
    assert np.isclose(metrics["sign_disagreement_fraction"], 0.75)
    assert metrics["false_reject_count"] == 2
    assert metrics["false_accept_count"] == 1
    assert metrics["deep_query_count"] == 2
    assert metrics["deep_sign_mismatch_count"] == 2
    assert np.isclose(metrics["deep_sign_mismatch_fraction"], 1.0)
    assert np.isclose(metrics["absolute_error_p90_m"], 0.155)


def test_pair_metrics_reject_shape_or_nonfinite_inputs() -> None:
    """Coverage and finite gates must fail loudly instead of silently filtering rows."""
    try:
        compare_signed_distance_pair(
            np.zeros(2),
            np.zeros(3),
            radii_m=np.zeros(3),
        )
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("shape mismatch was accepted")

    metrics = compare_signed_distance_pair(
        np.asarray([0.0, np.nan]),
        np.asarray([0.0, 0.0]),
        radii_m=np.zeros(2),
    )
    assert metrics["query_count"] == 2
    assert metrics["finite_count"] == 1
    assert metrics["nonfinite_count"] == 1
    assert metrics["coverage_fraction"] == 0.5
    assert metrics["status"] == "NONFINITE"


def test_consumer_reduction_and_gate_semantics() -> None:
    """Per-geom clearance must reduce exactly like the 48-step CEM gate."""
    clearances = np.asarray(
        [
            [[0.02, -0.004, 0.03], [0.01, -0.006, 0.04]],
            [[0.03, 0.02, 0.01], [0.02, 0.01, 0.00]],
        ]
    )
    mask = np.asarray(
        [
            [True, False],
            [True, True],
            [False, True],
        ]
    )
    reduced = consumer_min_clearance(clearances, mask, ["body", "hand"])
    assert np.allclose(reduced["body"], [[-0.004, -0.006], [0.02, 0.01]])
    assert np.allclose(reduced["hand"], [[-0.004, -0.006], [0.01, 0.00]])

    legacy = compute_gate_samples(
        reduced["body"],
        min_sdf_m=-0.005,
        max_violation_pct=0.5,
        hard_floor_m=float("nan"),
    )
    assert legacy["valid_mask"].tolist() == [False, True]
    tolerant = compute_gate_samples(
        reduced["body"],
        min_sdf_m=-0.005,
        max_violation_pct=0.5,
        hard_floor_m=-0.01,
    )
    assert tolerant["valid_mask"].tolist() == [True, True]
    compared = compare_gate_validity(tolerant["valid_mask"], legacy["valid_mask"])
    assert compared["mask_flip_count"] == 1
    assert compared["false_accept_count"] == 1
    assert compared["false_reject_count"] == 0


def test_geometry_reward_component_contract() -> None:
    """R shadow components must reuse frozen hinge and symmetric-band formulas."""
    config = SimpleNamespace(
        robot_object_penalty_scale=2.0,
        robot_object_penalty_margin_m=0.02,
        robot_object_penalty_deep_threshold_m=0.0,
        leg_object_penalty_scale=2.0,
        leg_object_penalty_margin_m=0.02,
        surface_band_rew_scale=1.5,
        surface_band_penalty_scale=0.0,
        surface_band_width_m=0.003,
        surface_band_min_sdf_m=-0.001,
        surface_band_sigma=0.0015,
        surface_band_score_mode="symmetric_abs",
        surface_band_penetration_tol_m=0.003,
    )
    values = {
        "R_robot_penalty": np.asarray([0.01, 0.03]),
        "R_leg_penalty": np.asarray([0.00, 0.02]),
        "R_surface_band": np.asarray([0.00, 0.004]),
    }
    components = geometry_reward_components(values, config)
    assert np.allclose(components["robot_object_penalty"], [-0.02, 0.0])
    assert np.allclose(components["leg_object_penalty"], [-0.04, 0.0])
    assert np.allclose(components["surface_band_rew"], [1.5, 0.0])
    assert np.allclose(components["total"], [1.44, 0.0])


def main() -> int:
    """Run tests without pytest discovery."""
    tests = (
        test_overlapping_parts_use_true_union_boundary,
        test_pair_metrics_apply_query_radius_before_sign_gates,
        test_pair_metrics_reject_shape_or_nonfinite_inputs,
        test_consumer_reduction_and_gate_semantics,
        test_geometry_reward_component_contract,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E182_TASK_QUERY_GEOMETRY_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
