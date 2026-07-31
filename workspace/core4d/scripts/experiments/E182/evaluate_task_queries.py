#!/usr/bin/env python3
"""Task-query geometry primitives for E182 D_M and exact convex-union audits."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import open3d as o3d
import trimesh

BOOLEAN_ENGINE = "manifold"
SIGNED_DISTANCE_BACKEND = "open3d_raycasting_scene_nsamples5_negative_inside"


def build_exact_union_mesh(parts: Sequence[trimesh.Trimesh]) -> trimesh.Trimesh:
    """Boolean-union closed convex parts so internal faces never enter exact-C SDF."""
    if not parts:
        raise ValueError("exact union requires at least one convex part")
    checked: list[trimesh.Trimesh] = []
    for index, part in enumerate(parts):
        if not isinstance(part, trimesh.Trimesh):
            raise TypeError(f"part {index} is not a Trimesh")
        if part.is_empty or not part.is_watertight or not part.is_winding_consistent:
            raise ValueError(f"part {index} is not a closed consistently-wound mesh")
        if not np.isfinite(part.vertices).all() or float(part.volume) <= 0.0:
            raise ValueError(f"part {index} is non-finite or non-volumetric")
        checked.append(part)
    result = trimesh.boolean.union(
        checked,
        engine=BOOLEAN_ENGINE,
        check_volume=True,
    )
    if isinstance(result, list):
        if len(result) != 1:
            raise RuntimeError(f"boolean union returned {len(result)} meshes")
        result = result[0]
    if not isinstance(result, trimesh.Trimesh):
        raise RuntimeError("boolean union did not return a Trimesh")
    if (
        result.is_empty
        or not result.is_watertight
        or not result.is_winding_consistent
        or not np.isfinite(result.vertices).all()
        or float(result.volume) <= 0.0
    ):
        raise RuntimeError("boolean union is not a valid closed volume")
    return result


def _raycasting_scene(mesh: trimesh.Trimesh) -> o3d.t.geometry.RaycastingScene:
    """Create an Open3D scene without changing the authoritative mesh geometry."""
    if mesh.is_empty or not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError("signed-distance mesh must be closed and consistently wound")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)),
    )
    return scene


def mesh_signed_distance(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    """Return negative-inside signed distance for arbitrary leading point axes."""
    queries = np.asarray(points, dtype=np.float64)
    if queries.ndim < 2 or queries.shape[-1] != 3:
        raise ValueError(f"query points must end in axis 3, got shape={queries.shape}")
    flat = queries.reshape(-1, 3)
    if not np.isfinite(flat).all():
        raise ValueError("query points contain non-finite values")
    if len(flat) == 0:
        return np.empty(queries.shape[:-1], dtype=np.float64)
    signed = (
        _raycasting_scene(mesh)
        .compute_signed_distance(
            o3d.core.Tensor(flat.astype(np.float32)),
            nsamples=5,
        )
        .numpy()
        .astype(np.float64)
    )
    return signed.reshape(queries.shape[:-1])


def _fraction(numerator: int, denominator: int) -> float:
    """Return a defined zero fraction for an empty diagnostic subset."""
    return float(numerator / denominator) if denominator else 0.0


def compare_signed_distance_pair(
    candidate_sdf_m: np.ndarray,
    oracle_sdf_m: np.ndarray,
    *,
    radii_m: np.ndarray | float = 0.0,
    deep_margin_m: float = 0.02,
) -> dict[str, int | float | str | None]:
    """Compare candidate/oracle surface clearances with explicit finite accounting."""
    candidate = np.asarray(candidate_sdf_m, dtype=np.float64)
    oracle = np.asarray(oracle_sdf_m, dtype=np.float64)
    if candidate.shape != oracle.shape:
        raise ValueError(
            f"candidate/oracle shape mismatch: {candidate.shape} != {oracle.shape}"
        )
    if deep_margin_m < 0.0 or not np.isfinite(deep_margin_m):
        raise ValueError("deep margin must be finite and non-negative")
    try:
        radii = np.broadcast_to(np.asarray(radii_m, dtype=np.float64), candidate.shape)
    except ValueError as error:
        raise ValueError(
            f"radius shape is not broadcastable to query shape {candidate.shape}"
        ) from error

    candidate_clearance = (candidate - radii).reshape(-1)
    oracle_clearance = (oracle - radii).reshape(-1)
    radius_flat = radii.reshape(-1)
    finite = (
        np.isfinite(candidate_clearance)
        & np.isfinite(oracle_clearance)
        & np.isfinite(radius_flat)
    )
    query_count = int(candidate_clearance.size)
    finite_count = int(finite.sum())
    nonfinite_count = query_count - finite_count
    result: dict[str, int | float | str | None] = {
        "status": "PASS" if nonfinite_count == 0 else "NONFINITE",
        "query_count": query_count,
        "finite_count": finite_count,
        "nonfinite_count": nonfinite_count,
        "coverage_fraction": _fraction(finite_count, query_count),
    }
    if finite_count == 0:
        result.update(
            {
                "sign_disagreement_count": 0,
                "sign_disagreement_fraction": 0.0,
                "false_reject_count": 0,
                "false_reject_fraction": 0.0,
                "false_accept_count": 0,
                "false_accept_fraction": 0.0,
                "deep_query_count": 0,
                "deep_sign_mismatch_count": 0,
                "deep_sign_mismatch_fraction": 0.0,
                "absolute_error_p90_m": None,
                "absolute_error_max_m": None,
            }
        )
        return result

    candidate_finite = candidate_clearance[finite]
    oracle_finite = oracle_clearance[finite]
    candidate_inside = candidate_finite <= 0.0
    oracle_inside = oracle_finite <= 0.0
    mismatch = candidate_inside != oracle_inside
    false_reject = ~candidate_inside & oracle_inside
    false_accept = candidate_inside & ~oracle_inside
    deep = np.abs(oracle_finite) > deep_margin_m
    absolute_error = np.abs(candidate_finite - oracle_finite)
    sign_count = int(mismatch.sum())
    false_reject_count = int(false_reject.sum())
    false_accept_count = int(false_accept.sum())
    deep_count = int(deep.sum())
    deep_mismatch_count = int((mismatch & deep).sum())
    result.update(
        {
            "sign_disagreement_count": sign_count,
            "sign_disagreement_fraction": _fraction(sign_count, finite_count),
            "false_reject_count": false_reject_count,
            "false_reject_fraction": _fraction(false_reject_count, finite_count),
            "false_accept_count": false_accept_count,
            "false_accept_fraction": _fraction(false_accept_count, finite_count),
            "deep_query_count": deep_count,
            "deep_sign_mismatch_count": deep_mismatch_count,
            "deep_sign_mismatch_fraction": _fraction(
                deep_mismatch_count,
                deep_count,
            ),
            "absolute_error_p90_m": float(np.quantile(absolute_error, 0.90)),
            "absolute_error_max_m": float(absolute_error.max()),
        }
    )
    return result


def consumer_min_clearance(
    clearances_m: np.ndarray,
    point_consumer_mask: np.ndarray,
    consumer_names: list[str],
) -> dict[str, np.ndarray]:
    """Reduce point surface clearances to each authoritative R/G consumer group."""
    clearances = np.asarray(clearances_m, dtype=np.float64)
    mask = np.asarray(point_consumer_mask, dtype=bool)
    if clearances.ndim < 2:
        raise ValueError(
            "clearances require at least one leading axis and one point axis"
        )
    if mask.ndim != 2 or mask.shape[0] != clearances.shape[-1]:
        raise ValueError(
            f"consumer mask shape {mask.shape} does not match points {clearances.shape[-1]}"
        )
    if mask.shape[1] != len(consumer_names):
        raise ValueError("consumer name count does not match mask columns")
    reduced = {}
    for index, name in enumerate(consumer_names):
        selected = mask[:, index]
        if not selected.any():
            continue
        reduced[name] = clearances[..., selected].min(axis=-1)
    return reduced


def compute_gate_samples(
    per_pose_clearance_m: np.ndarray,
    *,
    min_sdf_m: float,
    max_violation_pct: float,
    hard_floor_m: float = float("nan"),
) -> dict[str, np.ndarray]:
    """Mirror CEM geometry-gate reduction for arrays shaped (samples, horizon)."""
    clearance = np.asarray(per_pose_clearance_m, dtype=np.float64)
    if clearance.ndim != 2:
        raise ValueError(f"gate clearance must be 2-D, got {clearance.shape}")
    if not np.isfinite(clearance).all():
        raise ValueError("gate clearance contains non-finite values")
    minimum = clearance.min(axis=1)
    violation_depth = np.maximum(min_sdf_m - clearance, 0.0)
    violation_pct = (violation_depth > 0.0).mean(axis=1)
    violation_depth_mean = violation_depth.mean(axis=1)
    floor = min_sdf_m if np.isnan(hard_floor_m) else hard_floor_m
    valid = (minimum >= floor) & (violation_pct <= max_violation_pct)
    return {
        "min_sdf_m": minimum,
        "violation_pct": violation_pct,
        "violation_depth_mean_m": violation_depth_mean,
        "valid_mask": valid,
    }


def compare_gate_validity(
    candidate_valid: np.ndarray,
    oracle_valid: np.ndarray,
) -> dict[str, int | float]:
    """Compare candidate geometry-gate decisions against D_M oracle decisions."""
    candidate = np.asarray(candidate_valid, dtype=bool)
    oracle = np.asarray(oracle_valid, dtype=bool)
    if candidate.shape != oracle.shape:
        raise ValueError(
            f"gate mask shape mismatch: {candidate.shape} != {oracle.shape}"
        )
    flip = candidate != oracle
    false_reject = ~candidate & oracle
    false_accept = candidate & ~oracle
    count = int(candidate.size)
    return {
        "sample_count": count,
        "mask_flip_count": int(flip.sum()),
        "mask_flip_fraction": _fraction(int(flip.sum()), count),
        "false_reject_count": int(false_reject.sum()),
        "false_reject_fraction": _fraction(int(false_reject.sum()), count),
        "false_accept_count": int(false_accept.sum()),
        "false_accept_fraction": _fraction(int(false_accept.sum()), count),
    }


def geometry_reward_components(
    consumer_clearance_m: dict[str, np.ndarray],
    config: object,
) -> dict[str, np.ndarray]:
    """Recompute ungated geometry-dependent R components from consumer clearances."""
    if not consumer_clearance_m:
        raise ValueError("at least one R consumer clearance is required")
    template = np.asarray(next(iter(consumer_clearance_m.values())), dtype=np.float64)
    zero = np.zeros_like(template)

    robot_sdf = np.asarray(
        consumer_clearance_m.get("R_robot_penalty", zero), dtype=np.float64
    )
    robot_limit = float(config.robot_object_penalty_margin_m) - float(
        config.robot_object_penalty_deep_threshold_m
    )
    robot = -float(config.robot_object_penalty_scale) * np.maximum(
        robot_limit - robot_sdf,
        0.0,
    )

    leg_sdf = np.asarray(
        consumer_clearance_m.get("R_leg_penalty", zero), dtype=np.float64
    )
    leg = -float(config.leg_object_penalty_scale) * np.maximum(
        float(config.leg_object_penalty_margin_m) - leg_sdf,
        0.0,
    )

    surface_sdf = np.asarray(
        consumer_clearance_m.get("R_surface_band", zero), dtype=np.float64
    )
    in_band = (surface_sdf >= float(config.surface_band_min_sdf_m)) & (
        surface_sdf <= float(config.surface_band_width_m)
    )
    sigma = max(float(config.surface_band_sigma), 1e-6)
    if config.surface_band_score_mode == "one_sided":
        raw_score = np.exp(-np.maximum(surface_sdf, 0.0) / sigma)
    elif config.surface_band_score_mode == "symmetric_abs":
        raw_score = np.exp(-np.abs(surface_sdf) / sigma)
    else:
        raise ValueError(
            f"unsupported surface band score mode: {config.surface_band_score_mode}"
        )
    surface_reward = float(config.surface_band_rew_scale) * np.where(
        in_band,
        raw_score,
        0.0,
    )
    penetration = np.maximum(
        -surface_sdf - float(config.surface_band_penetration_tol_m),
        0.0,
    )
    surface_penalty = -float(config.surface_band_penalty_scale) * penetration
    total = robot + leg + surface_reward + surface_penalty
    return {
        "robot_object_penalty": robot,
        "leg_object_penalty": leg,
        "surface_band_rew": surface_reward,
        "surface_band_penalty": surface_penalty,
        "total": total,
    }
