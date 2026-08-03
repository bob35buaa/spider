"""Shared exact/grid reward-fidelity primitives for canonical object distance."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.stats import spearmanr

CONTINUATION_PARAMETERS = {
    "far_weight": 0.25,
    "near_weight": 0.75,
    "far_scale_m": 0.050,
    "near_scale_m": 0.015,
    "smooth_delta_m": 0.001,
}
OBJECT_COMPONENT_KEYS = (
    "robot_object_penalty",
    "leg_object_penalty",
    "surface_band_rew",
)


def continuation_score(distance_m: np.ndarray) -> np.ndarray:
    """Evaluate the preregistered E187 score on signed distances."""
    distance = np.asarray(distance_m, dtype=np.float64)
    delta = CONTINUATION_PARAMETERS["smooth_delta_m"]
    smooth_abs = np.sqrt(np.square(distance) + delta**2) - delta
    far = np.exp(-smooth_abs / CONTINUATION_PARAMETERS["far_scale_m"])
    near = np.exp(-smooth_abs / CONTINUATION_PARAMETERS["near_scale_m"])
    return (
        CONTINUATION_PARAMETERS["far_weight"] * far
        + CONTINUATION_PARAMETERS["near_weight"] * near
    )


def object_reward_components(
    distances: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    temporal_gate: np.ndarray,
    temporal_decay: np.ndarray,
) -> dict[str, np.ndarray]:
    """Recompute body/leg penalties and E187 continuation surface reward."""
    required = {"body", "hand", "leg"}
    if set(distances) != required:
        raise ValueError(f"distance groups must be exactly {sorted(required)}")
    shapes = {np.asarray(value).shape for value in distances.values()}
    shapes.update(
        {
            np.asarray(temporal_gate).shape,
            np.asarray(temporal_decay).shape,
        }
    )
    if len(shapes) != 1:
        raise ValueError(f"reward trace shapes disagree: {sorted(shapes)}")

    body = np.asarray(distances["body"], dtype=np.float64)
    hand = np.asarray(distances["hand"], dtype=np.float64)
    leg_distance = np.asarray(distances["leg"], dtype=np.float64)
    robot_limit = float(config["robot_object_penalty_margin_m"]) - float(
        config["robot_object_penalty_deep_threshold_m"]
    )
    robot = -float(config["robot_object_penalty_scale"]) * np.maximum(
        robot_limit - body, 0.0
    )
    leg = -float(config["leg_object_penalty_scale"]) * np.maximum(
        float(config["leg_object_penalty_margin_m"]) - leg_distance, 0.0
    )
    surface = (
        float(config["surface_band_rew_scale"])
        * continuation_score(hand)
        * np.asarray(temporal_gate, dtype=np.float64)
        * np.asarray(temporal_decay, dtype=np.float64)
    )
    return {
        "robot_object_penalty": robot,
        "leg_object_penalty": leg,
        "surface_band_rew": surface,
    }


def candidate_component_total(components: Mapping[str, np.ndarray]) -> np.ndarray:
    """Reduce the three per-frame object components to candidate totals."""
    if set(components) != set(OBJECT_COMPONENT_KEYS):
        raise ValueError("object component keys disagree with the frozen contract")
    traces = [
        np.asarray(components[key], dtype=np.float64) for key in OBJECT_COMPONENT_KEYS
    ]
    if not traces or any(trace.ndim != 2 for trace in traces):
        raise ValueError("object component traces must have [candidate, horizon] shape")
    if len({trace.shape for trace in traces}) != 1:
        raise ValueError("object component trace shapes disagree")
    return np.sum(traces, axis=0).mean(axis=1)


def reconstruct_continuation_totals(
    recorded_old_total: np.ndarray,
    recorded_old_components: Mapping[str, np.ndarray],
    grid_new_components: Mapping[str, np.ndarray],
    exact_new_components: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Replace only old object components in one recorded candidate total."""
    recorded = np.asarray(recorded_old_total, dtype=np.float64)
    old = candidate_component_total(recorded_old_components)
    grid = candidate_component_total(grid_new_components)
    exact = candidate_component_total(exact_new_components)
    if recorded.ndim != 1 or not (
        recorded.shape == old.shape == grid.shape == exact.shape
    ):
        raise ValueError("candidate reward shapes disagree")
    base = recorded - old
    return {
        "base_total": base,
        "grid_total": base + grid,
        "exact_total": base + exact,
        "old_object_total": old,
        "grid_object_total": grid,
        "exact_object_total": exact,
    }


def gate_summary(
    values: np.ndarray,
    *,
    threshold: float,
    max_violation_pct: float,
    hard_floor: float,
) -> dict[str, np.ndarray]:
    """Mirror one optimizer geometry-gate reduction over horizon time."""
    distances = np.asarray(values, dtype=np.float64)
    if distances.ndim != 2:
        raise ValueError("gate distances must have [candidate, horizon] shape")
    violation = distances < threshold
    depth = np.maximum(threshold - distances, 0.0)
    floor = threshold if np.isnan(hard_floor) else hard_floor
    minimum = distances.min(axis=1)
    violation_pct = violation.mean(axis=1)
    return {
        "min_sdf": minimum,
        "violation_pct": violation_pct,
        "violation_depth_mean": depth.mean(axis=1),
        "valid": (minimum >= floor) & (violation_pct <= max_violation_pct),
    }


def geometry_gates(
    distances: Mapping[str, np.ndarray], config: Mapping[str, Any]
) -> dict[str, dict[str, np.ndarray]]:
    """Compute frozen body, hand, and leg optimizer gate summaries."""
    parameters = {
        "body": "cem_safety_gate",
        "hand": "cem_hand_gate",
        "leg": "cem_leg_gate",
    }
    return {
        key: gate_summary(
            distances[key],
            threshold=float(config[f"{prefix}_min_sdf_m"]),
            max_violation_pct=float(config[f"{prefix}_max_violation_pct"]),
            hard_floor=float(config[f"{prefix}_hard_floor_m"]),
        )
        for key, prefix in parameters.items()
    }


def conservative_grid_distances(
    grid_distances: Mapping[str, np.ndarray], epsilon_grid_m: float
) -> dict[str, np.ndarray]:
    """Apply epsilon only to G; reward callers must use raw grid distances."""
    if not np.isfinite(epsilon_grid_m) or epsilon_grid_m < 0.0:
        raise ValueError("epsilon_grid_m must be finite and non-negative")
    return {
        key: np.asarray(value, dtype=np.float64) - epsilon_grid_m
        for key, value in grid_distances.items()
    }


def combine_gates(
    geometry: Mapping[str, Mapping[str, np.ndarray]],
    posture_valid: np.ndarray,
    posture_violation: np.ndarray,
) -> dict[str, np.ndarray]:
    """Mirror optimizer max/min combination with the frozen posture gate."""
    axes = list(geometry.values())
    posture_ok = np.asarray(posture_valid, dtype=bool)
    posture_error = np.asarray(posture_violation, dtype=np.float64)
    if not axes or any(
        np.asarray(axis["valid"]).shape != posture_ok.shape for axis in axes
    ):
        raise ValueError("gate and posture candidate shapes disagree")
    if posture_error.shape != posture_ok.shape:
        raise ValueError("posture arrays disagree")
    return {
        "valid": np.logical_and.reduce([axis["valid"] for axis in axes]) & posture_ok,
        "min_sdf": np.minimum.reduce(
            [np.asarray(axis["min_sdf"]) for axis in axes] + [-posture_error]
        ),
        "violation_pct": np.maximum.reduce(
            [np.asarray(axis["violation_pct"]) for axis in axes] + [posture_error]
        ),
        "violation_depth_mean": np.maximum.reduce(
            [np.asarray(axis["violation_depth_mean"]) for axis in axes]
            + [posture_error]
        ),
    }


def select_elites(
    reward: np.ndarray,
    gate: Mapping[str, np.ndarray],
    posture_violation: np.ndarray,
    *,
    top_k: int,
    min_valid_frac: float,
    fallback_lambda: float,
) -> tuple[np.ndarray, bool]:
    """Mirror stable hard-gate elite selection and least-violation fallback."""
    values = np.asarray(reward, dtype=np.float64)
    valid = np.asarray(gate["valid"], dtype=bool)
    violation = np.asarray(posture_violation, dtype=np.float64)
    if values.ndim != 1 or not (values.shape == valid.shape == violation.shape):
        raise ValueError("selection arrays must be aligned 1-D candidates")
    if not 0 < top_k <= len(values):
        raise ValueError("top_k is outside candidate count")
    minimum_valid = max(1, int(np.ceil(min_valid_frac * len(values))))
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) >= minimum_valid:
        order = np.argsort(values[valid_indices], kind="stable")[::-1]
        return valid_indices[order[:top_k]], False
    fallback = values - fallback_lambda * violation
    return np.argsort(fallback, kind="stable")[::-1][:top_k], True


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Return finite Spearman correlation, including equal constant arrays."""
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    if lhs.shape != rhs.shape or lhs.ndim != 1:
        raise ValueError("rank arrays must be aligned 1-D candidates")
    if np.allclose(lhs, rhs, atol=1e-12, rtol=0.0):
        return 1.0
    value = float(spearmanr(lhs, rhs).statistic)
    return value if np.isfinite(value) else 0.0


def elite_fidelity_metrics(
    grid_total: np.ndarray,
    exact_total: np.ndarray,
    grid_selected: np.ndarray,
    exact_selected: np.ndarray,
    exact_valid: np.ndarray,
) -> dict[str, Any]:
    """Measure overlap, exact validity, regret, and exact percentile rank."""
    grid_reward = np.asarray(grid_total, dtype=np.float64)
    exact_reward = np.asarray(exact_total, dtype=np.float64)
    grid_indices = np.asarray(grid_selected, dtype=np.int64)
    exact_indices = np.asarray(exact_selected, dtype=np.int64)
    valid = np.asarray(exact_valid, dtype=bool)
    if grid_reward.shape != exact_reward.shape or exact_reward.shape != valid.shape:
        raise ValueError("fidelity candidate arrays disagree")
    if grid_indices.ndim != 1 or exact_indices.ndim != 1 or len(grid_indices) == 0:
        raise ValueError("elite arrays must be non-empty 1-D arrays")
    if len(grid_indices) != len(exact_indices):
        raise ValueError("grid/exact elite counts disagree")
    if not valid.any():
        return {
            "topk_overlap_frac": 0.0,
            "grid_selected_exact_valid_frac": 0.0,
            "grid_selected0_exact_regret_frac": float("inf"),
            "grid_selected0_exact_percentile": 0.0,
            "grid_selected0_exact_top1pct": False,
        }

    overlap = len(set(grid_indices.tolist()) & set(exact_indices.tolist()))
    best_exact = float(np.max(exact_reward[valid]))
    selected0_reward = float(exact_reward[grid_indices[0]])
    regret = max(best_exact - selected0_reward, 0.0) / max(abs(best_exact), 1e-12)
    valid_rewards = exact_reward[valid]
    percentile = float((valid_rewards <= selected0_reward).mean())
    return {
        "topk_overlap_frac": overlap / len(grid_indices),
        "grid_selected_exact_valid_frac": float(valid[grid_indices].mean()),
        "grid_selected0_exact_regret_frac": regret,
        "grid_selected0_exact_percentile": percentile,
        "grid_selected0_exact_top1pct": bool(percentile >= 0.99),
    }
