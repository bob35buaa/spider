#!/usr/bin/env python3
"""Direct-main contracts for E187 continuation fidelity primitives."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts"))

from eval.core.distance_fidelity import (  # noqa: E402
    conservative_grid_distances,
    continuation_score,
    elite_fidelity_metrics,
    geometry_gates,
    object_reward_components,
    reconstruct_continuation_totals,
)


def component(
    value: float, *, candidates: int = 4, horizon: int = 3
) -> dict[str, np.ndarray]:
    """Build three aligned constant object-component traces."""
    return {
        "robot_object_penalty": np.full((candidates, horizon), value),
        "leg_object_penalty": np.full((candidates, horizon), value * 2),
        "surface_band_rew": np.full((candidates, horizon), value * 3),
    }


def test_local_component_replacement() -> None:
    """Old object terms are removed once and only new local terms are added."""
    recorded = np.array([20.0, 21.0, 22.0, 23.0])
    totals = reconstruct_continuation_totals(
        recorded, component(1.0), component(2.0), component(3.0)
    )
    np.testing.assert_array_equal(totals["base_total"], recorded - 6.0)
    np.testing.assert_array_equal(totals["grid_total"], recorded + 6.0)
    np.testing.assert_array_equal(totals["exact_total"], recorded + 12.0)


def test_cross_component_replacement() -> None:
    """Each component can change independently without leaking old terms."""
    old = component(0.0)
    grid = component(0.0)
    exact = component(0.0)
    old["robot_object_penalty"][:] = -2.0
    old["leg_object_penalty"][:] = -3.0
    old["surface_band_rew"][:] = 5.0
    grid["robot_object_penalty"][:] = -1.0
    grid["leg_object_penalty"][:] = -4.0
    grid["surface_band_rew"][:] = 9.0
    exact["robot_object_penalty"][:] = -0.5
    exact["leg_object_penalty"][:] = -1.5
    exact["surface_band_rew"][:] = 7.0
    totals = reconstruct_continuation_totals(np.arange(4.0), old, grid, exact)
    np.testing.assert_array_equal(totals["base_total"], np.arange(4.0))
    np.testing.assert_array_equal(totals["grid_total"], np.arange(4.0) + 4.0)
    np.testing.assert_array_equal(totals["exact_total"], np.arange(4.0) + 5.0)


def test_reward_uses_raw_grid_but_gate_uses_lower_bound() -> None:
    """epsilon_grid changes G only and never biases continuation reward."""
    distances = {
        "body": np.full((2, 2), 0.006),
        "hand": np.full((2, 2), 0.006),
        "leg": np.full((2, 2), 0.006),
    }
    config = {
        "robot_object_penalty_margin_m": 0.02,
        "robot_object_penalty_deep_threshold_m": 0.0,
        "robot_object_penalty_scale": 2.0,
        "leg_object_penalty_margin_m": 0.02,
        "leg_object_penalty_scale": 2.0,
        "surface_band_rew_scale": 1.5,
        "cem_safety_gate_min_sdf_m": 0.005,
        "cem_safety_gate_max_violation_pct": 0.0,
        "cem_safety_gate_hard_floor_m": float("nan"),
        "cem_hand_gate_min_sdf_m": 0.005,
        "cem_hand_gate_max_violation_pct": 0.0,
        "cem_hand_gate_hard_floor_m": float("nan"),
        "cem_leg_gate_min_sdf_m": 0.005,
        "cem_leg_gate_max_violation_pct": 0.0,
        "cem_leg_gate_hard_floor_m": float("nan"),
    }
    ones = np.ones((2, 2))
    raw_reward = object_reward_components(distances, config, ones, ones)
    lowered = conservative_grid_distances(distances, 0.002)
    lowered_reward = object_reward_components(lowered, config, ones, ones)
    assert not np.array_equal(
        raw_reward["surface_band_rew"], lowered_reward["surface_band_rew"]
    )
    assert all(
        axis["valid"].all() for axis in geometry_gates(distances, config).values()
    )
    assert not any(
        axis["valid"].any() for axis in geometry_gates(lowered, config).values()
    )


def test_rho_overlap_regret_inclusive_boundaries() -> None:
    """The preregistered overlap/regret/top-1% boundaries are inclusive."""
    candidates = 100
    exact = np.linspace(1.0, 0.0, candidates)
    grid = exact.copy()
    exact_selected = np.arange(10)
    grid_selected = np.concatenate((np.arange(9), np.array([10])))
    metrics = elite_fidelity_metrics(
        grid, exact, grid_selected, exact_selected, np.ones(candidates, dtype=bool)
    )
    assert metrics["topk_overlap_frac"] == 0.9
    assert metrics["grid_selected_exact_valid_frac"] == 1.0
    assert metrics["grid_selected0_exact_regret_frac"] == 0.0
    assert metrics["grid_selected0_exact_top1pct"]


def test_regret_and_validity_failures_are_visible() -> None:
    """A non-top candidate or one false-safe selected candidate cannot pass."""
    exact = np.linspace(1.0, 0.0, 100)
    grid = exact.copy()
    grid_selected = np.arange(2, 12)
    exact_selected = np.arange(10)
    valid = np.ones(100, dtype=bool)
    valid[2] = False
    metrics = elite_fidelity_metrics(grid, exact, grid_selected, exact_selected, valid)
    assert metrics["topk_overlap_frac"] < 0.9
    assert metrics["grid_selected_exact_valid_frac"] == 0.9
    assert metrics["grid_selected0_exact_regret_frac"] > 0.005
    assert not metrics["grid_selected0_exact_top1pct"]


def test_bucket003_capture_support_shape() -> None:
    """Candidate capture uses its closest/peak frame, not a horizon mean."""
    candidates, horizon = 1024, 48
    distance = np.full((candidates, horizon), 0.5)
    distance[:, 17] = np.linspace(0.0458, 0.0486, candidates)
    distances = {
        "body": np.full((candidates, horizon), 0.1),
        "hand": distance,
        "leg": np.full((candidates, horizon), 0.1),
    }
    config = {
        "robot_object_penalty_margin_m": 0.02,
        "robot_object_penalty_deep_threshold_m": 0.0,
        "robot_object_penalty_scale": 2.0,
        "leg_object_penalty_margin_m": 0.02,
        "leg_object_penalty_scale": 2.0,
        "surface_band_rew_scale": 1.5,
    }
    traces = object_reward_components(
        distances,
        config,
        np.ones((candidates, horizon)),
        np.ones((candidates, horizon)),
    )
    candidate_score = continuation_score(distance)
    assert float((candidate_score.max(axis=1) > 0.05).mean()) == 1.0
    assert float((candidate_score.mean(axis=1) > 0.05).mean()) == 0.0
    assert np.isfinite(traces["surface_band_rew"]).all()


def main() -> int:
    """Run all contracts without pytest collection dependencies."""
    tests = (
        test_local_component_replacement,
        test_cross_component_replacement,
        test_reward_uses_raw_grid_but_gate_uses_lower_bound,
        test_rho_overlap_regret_inclusive_boundaries,
        test_regret_and_validity_failures_are_visible,
        test_bucket003_capture_support_shape,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E187_DISTANCE_FIDELITY_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
