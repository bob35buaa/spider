#!/usr/bin/env python3
"""Task-query geometry primitives for E182 D_M and exact convex-union audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import trimesh
from build_prg_query_tape import load_case_context
from build_task_query_fixture import DEFAULT_OUTPUT_ROOT, EXPECTED_POINTS_PER_POSE
from e182_common import atomic_json, relative_to_repo, repo_path, sha256_file
from run_query_tape_replay import DEFAULT_RESULT_ROOT as DEFAULT_S1_ROOT
from run_query_tape_replay import load_dev_rows
from scipy.stats import spearmanr

BOOLEAN_ENGINE = "manifold"
SIGNED_DISTANCE_BACKEND = "open3d_raycasting_scene_nsamples5_negative_inside"
DEFAULT_FIXTURE = DEFAULT_OUTPUT_ROOT / "query_fixture_manifest.json"
PAIR_ERROR_BIN_WIDTH_M = 1e-4
PAIR_ERROR_HIST_MAX_M = 2.0
R_NORMALIZED_ERROR_BIN_WIDTH = 1e-3
R_NORMALIZED_ERROR_HIST_MAX = 10.0
EVALUATOR_SCHEMA_VERSION = 1


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


def _scene_signed_distance(
    scene: o3d.t.geometry.RaycastingScene, points: np.ndarray
) -> np.ndarray:
    """Query one already-built scene with negative-inside sign convention."""
    queries = np.asarray(points, dtype=np.float64)
    if queries.ndim < 2 or queries.shape[-1] != 3:
        raise ValueError(f"query points must end in axis 3, got shape={queries.shape}")
    flat = queries.reshape(-1, 3)
    if not np.isfinite(flat).all():
        raise ValueError("query points contain non-finite values")
    if len(flat) == 0:
        return np.empty(queries.shape[:-1], dtype=np.float64)
    signed = (
        scene.compute_signed_distance(
            o3d.core.Tensor(flat.astype(np.float32)),
            nsamples=5,
        )
        .numpy()
        .astype(np.float64)
    )
    return signed.reshape(queries.shape[:-1])


def mesh_signed_distance(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    """Return negative-inside signed distance for arbitrary leading point axes."""
    queries = np.asarray(points, dtype=np.float64)
    if queries.ndim < 2 or queries.shape[-1] != 3:
        raise ValueError(f"query points must end in axis 3, got shape={queries.shape}")
    return _scene_signed_distance(_raycasting_scene(mesh), queries)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write one deterministic uncompressed cache chunk atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def _candidate_asset_sha256(candidate: Mapping[str, Any]) -> str:
    """Recompute the E181 candidate identity from ordered part payloads."""
    payload = {
        "object_key": candidate["object_key"],
        "parameters": candidate["parameters"],
        "parts": [
            {
                "part_index": part["part_index"],
                "sha256": part["sha256"],
                "vertex_count": part["vertex_count"],
                "face_count": part["face_count"],
            }
            for part in candidate["parts"]
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_checked_mesh(path: Path, expected_sha256: str) -> trimesh.Trimesh:
    """Load one immutable closed mesh after exact source SHA verification."""
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise RuntimeError(f"mesh file/SHA mismatch: {path}")
    mesh = trimesh.load(path, force="mesh", process=False, maintain_order=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"mesh loader returned non-mesh: {path}")
    if mesh.is_empty or not mesh.is_watertight or not mesh.is_winding_consistent:
        raise RuntimeError(f"mesh is not a closed consistent volume: {path}")
    return mesh


def _load_candidate_and_oracle(
    fixture_candidate: Mapping[str, Any],
) -> tuple[dict[str, Any], trimesh.Trimesh, trimesh.Trimesh]:
    """Resolve candidate manifest, ordered exact union, and D_M oracle mesh."""
    manifest_path = repo_path(fixture_candidate["manifest"]["path"])
    if sha256_file(manifest_path) != fixture_candidate["manifest"]["sha256"]:
        raise RuntimeError("candidate manifest SHA changed after fixture freeze")
    candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity_fields = (
        "object_key",
        "candidate_id",
        "candidate_asset_sha256",
    )
    for field in identity_fields:
        if candidate[field] != fixture_candidate[field]:
            raise RuntimeError(f"candidate fixture identity changed: {field}")
    if _candidate_asset_sha256(candidate) != candidate["candidate_asset_sha256"]:
        raise RuntimeError("candidate ordered asset digest changed")
    parts = [
        _load_checked_mesh(repo_path(part["path"]), part["sha256"])
        for part in candidate["parts"]
    ]
    exact_union = build_exact_union_mesh(parts)
    oracle_entry = candidate["oracle_cleaned_mesh"]
    oracle = _load_checked_mesh(repo_path(oracle_entry["path"]), oracle_entry["sha256"])
    return candidate, exact_union, oracle


class StreamingHistogram:
    """Fixed-bin streaming histogram with conservative upper quantiles."""

    def __init__(self, *, bin_width: float, maximum: float) -> None:
        if bin_width <= 0.0 or maximum <= bin_width:
            raise ValueError("invalid streaming histogram range")
        self.bin_width = float(bin_width)
        self.maximum = float(maximum)
        self.bin_count = int(np.ceil(maximum / bin_width))
        self.counts = np.zeros(self.bin_count + 1, dtype=np.int64)
        self.count = 0
        self.maximum_observed = 0.0

    def update(self, values: np.ndarray) -> None:
        """Accumulate finite non-negative values; last bin is overflow."""
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if not np.isfinite(array).all() or np.any(array < 0.0):
            raise ValueError("histogram values must be finite and non-negative")
        if not len(array):
            return
        indices = np.floor(array / self.bin_width).astype(np.int64)
        indices = np.clip(indices, 0, self.bin_count)
        self.counts += np.bincount(indices, minlength=len(self.counts))
        self.count += len(array)
        self.maximum_observed = max(self.maximum_observed, float(array.max()))

    def conservative_quantile(self, quantile: float) -> float | None:
        """Return the occupied bin upper edge, hence never understate a quantile."""
        if not self.count:
            return None
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile must lie in [0,1]")
        rank = max(1, int(np.ceil(quantile * self.count)))
        index = int(np.searchsorted(np.cumsum(self.counts), rank, side="left"))
        if index >= self.bin_count:
            return self.maximum_observed
        return min((index + 1) * self.bin_width, self.maximum)

    def summary(self) -> dict[str, int | float | None]:
        """Return bounded provenance and the pre-registered p90 upper estimate."""
        return {
            "count": self.count,
            "bin_width": self.bin_width,
            "histogram_max": self.maximum,
            "overflow_count": int(self.counts[-1]),
            "p90_conservative_upper": self.conservative_quantile(0.90),
            "maximum_observed": self.maximum_observed if self.count else None,
        }


class PairMetricAccumulator:
    """Stream exact point-level D_M versus exact-C counts and SDF error."""

    def __init__(self, *, deep_margin_m: float = 0.02) -> None:
        self.deep_margin_m = float(deep_margin_m)
        self.query_count = 0
        self.finite_count = 0
        self.sign_disagreement_count = 0
        self.false_reject_count = 0
        self.false_accept_count = 0
        self.deep_query_count = 0
        self.deep_sign_mismatch_count = 0
        self.absolute_error = StreamingHistogram(
            bin_width=PAIR_ERROR_BIN_WIDTH_M,
            maximum=PAIR_ERROR_HIST_MAX_M,
        )

    def update(
        self,
        candidate_sdf_m: np.ndarray,
        oracle_sdf_m: np.ndarray,
        radii_m: np.ndarray | float,
    ) -> None:
        """Accumulate one query batch with radius-adjusted sign accounting."""
        candidate = np.asarray(candidate_sdf_m, dtype=np.float64)
        oracle = np.asarray(oracle_sdf_m, dtype=np.float64)
        if candidate.shape != oracle.shape:
            raise ValueError("streamed pair shapes do not match")
        radii = np.broadcast_to(np.asarray(radii_m, dtype=np.float64), candidate.shape)
        candidate_clearance = (candidate - radii).reshape(-1)
        oracle_clearance = (oracle - radii).reshape(-1)
        finite = (
            np.isfinite(candidate_clearance)
            & np.isfinite(oracle_clearance)
            & np.isfinite(radii.reshape(-1))
        )
        self.query_count += candidate_clearance.size
        self.finite_count += int(finite.sum())
        if not finite.any():
            return
        candidate_clearance = candidate_clearance[finite]
        oracle_clearance = oracle_clearance[finite]
        candidate_inside = candidate_clearance <= 0.0
        oracle_inside = oracle_clearance <= 0.0
        mismatch = candidate_inside != oracle_inside
        deep = np.abs(oracle_clearance) > self.deep_margin_m
        self.sign_disagreement_count += int(mismatch.sum())
        self.false_reject_count += int((~candidate_inside & oracle_inside).sum())
        self.false_accept_count += int((candidate_inside & ~oracle_inside).sum())
        self.deep_query_count += int(deep.sum())
        self.deep_sign_mismatch_count += int((mismatch & deep).sum())
        self.absolute_error.update(np.abs(candidate_clearance - oracle_clearance))

    def summary(self) -> dict[str, Any]:
        """Return launch-floor-ready aggregate point metrics."""
        nonfinite = self.query_count - self.finite_count
        error = self.absolute_error.summary()
        return {
            "status": "PASS" if nonfinite == 0 else "NONFINITE",
            "query_count": self.query_count,
            "finite_count": self.finite_count,
            "nonfinite_count": nonfinite,
            "coverage_fraction": _fraction(self.finite_count, self.query_count),
            "sign_disagreement_count": self.sign_disagreement_count,
            "sign_disagreement_fraction": _fraction(
                self.sign_disagreement_count, self.finite_count
            ),
            "false_reject_count": self.false_reject_count,
            "false_reject_fraction": _fraction(
                self.false_reject_count, self.finite_count
            ),
            "false_accept_count": self.false_accept_count,
            "false_accept_fraction": _fraction(
                self.false_accept_count, self.finite_count
            ),
            "deep_query_count": self.deep_query_count,
            "deep_sign_mismatch_count": self.deep_sign_mismatch_count,
            "deep_sign_mismatch_fraction": _fraction(
                self.deep_sign_mismatch_count, self.deep_query_count
            ),
            "absolute_error_p90_m": error["p90_conservative_upper"],
            "absolute_error_max_m": error["maximum_observed"],
            "absolute_error_histogram": error,
        }


def materialize_selected_query_points(
    chunk: Mapping[str, np.ndarray], point_indices: np.ndarray
) -> np.ndarray:
    """Materialize one frozen nested point subset from factored geom poses."""
    indices = np.asarray(point_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError(f"point indices must be 1-D, got {indices.shape}")
    offsets_all = np.asarray(chunk["point_offsets_geom_local"], dtype=np.float64)
    columns_all = np.asarray(chunk["point_geom_column"], dtype=np.int64)
    if offsets_all.ndim != 2 or offsets_all.shape[1] != 3:
        raise ValueError("point offsets must have shape (points,3)")
    if columns_all.shape != (len(offsets_all),):
        raise ValueError("point geom columns do not match point offsets")
    if len(indices) and (
        int(indices.min()) < 0 or int(indices.max()) >= len(offsets_all)
    ):
        raise ValueError("point index is outside the frozen point inventory")

    positions = np.asarray(chunk["geom_pos_object_local"], dtype=np.float32)
    matrices = np.asarray(chunk["geom_mat_object_local"], dtype=np.float32)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("geom positions must have shape (poses,geoms,3)")
    if matrices.shape != (*positions.shape[:2], 3, 3):
        raise ValueError("geom matrices do not match geom positions")
    columns = columns_all[indices]
    if len(columns) and int(columns.max()) >= positions.shape[1]:
        raise ValueError("point geom column is outside the chunk geom inventory")
    offsets = offsets_all[indices]
    return positions[:, columns] + np.einsum(
        "npij,pj->npi",
        matrices[:, columns],
        offsets,
    )


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


def combine_gate_summaries(
    summaries: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Mirror the optimizer's AND/min/max aggregation across enabled gates."""
    if not summaries:
        raise ValueError("at least one gate summary is required")
    required = {
        "min_sdf_m",
        "violation_pct",
        "violation_depth_mean_m",
        "valid_mask",
    }
    combined: dict[str, np.ndarray] | None = None
    for index, summary in enumerate(summaries):
        missing = required - set(summary)
        if missing:
            raise ValueError(f"gate summary {index} is missing {sorted(missing)}")
        current = {
            "min_sdf_m": np.asarray(summary["min_sdf_m"], dtype=np.float64),
            "violation_pct": np.asarray(summary["violation_pct"], dtype=np.float64),
            "violation_depth_mean_m": np.asarray(
                summary["violation_depth_mean_m"], dtype=np.float64
            ),
            "valid_mask": np.asarray(summary["valid_mask"], dtype=bool),
        }
        shape = current["valid_mask"].shape
        if any(value.shape != shape for value in current.values()):
            raise ValueError(f"gate summary {index} has inconsistent shapes")
        if not all(
            np.isfinite(current[name]).all()
            for name in (
                "min_sdf_m",
                "violation_pct",
                "violation_depth_mean_m",
            )
        ):
            raise ValueError(f"gate summary {index} contains non-finite values")
        if combined is None:
            combined = {name: value.copy() for name, value in current.items()}
            continue
        if combined["valid_mask"].shape != shape:
            raise ValueError("gate summary shapes do not match")
        combined["min_sdf_m"] = np.minimum(combined["min_sdf_m"], current["min_sdf_m"])
        combined["violation_pct"] = np.maximum(
            combined["violation_pct"], current["violation_pct"]
        )
        combined["violation_depth_mean_m"] = np.maximum(
            combined["violation_depth_mean_m"],
            current["violation_depth_mean_m"],
        )
        combined["valid_mask"] &= current["valid_mask"]
    if combined is None:
        raise AssertionError("unreachable empty gate summary")
    return combined


def _descending_indices(values: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Return deterministic descending score order with source index tie-break."""
    return candidates[np.lexsort((candidates, -values[candidates]))]


def select_shadow_elites(
    rewards: np.ndarray,
    valid_mask: np.ndarray,
    violation_pct: np.ndarray,
    violation_depth_mean_m: np.ndarray,
    *,
    top_k: int,
    min_valid_frac: float,
    fallback_score: np.ndarray | None = None,
) -> dict[str, np.ndarray | int | bool]:
    """Mirror CEM gated elite selection, including precomputed fallback scores."""
    reward = np.asarray(rewards, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    pct = np.asarray(violation_pct, dtype=np.float64)
    depth = np.asarray(violation_depth_mean_m, dtype=np.float64)
    if reward.ndim != 1:
        raise ValueError("shadow rewards must be one-dimensional")
    if any(value.shape != reward.shape for value in (valid, pct, depth)):
        raise ValueError("shadow gate arrays do not match rewards")
    if top_k < 1 or top_k > len(reward):
        raise ValueError("top_k is outside the sample count")
    if not 0.0 <= min_valid_frac <= 1.0:
        raise ValueError("min_valid_frac must lie in [0,1]")
    finite_reward = np.isfinite(reward)
    if not np.isfinite(pct).all() or not np.isfinite(depth).all():
        raise ValueError("shadow gate severity contains non-finite values")
    if finite_reward.any():
        reward_clean = np.where(finite_reward, reward, reward[finite_reward].min())
    else:
        reward_clean = np.full_like(reward, -1000.0)
    valid = valid & finite_reward
    valid_count = int(valid.sum())
    min_valid_count = max(1, int(np.ceil(min_valid_frac * len(reward))))
    fallback_used = valid_count < min_valid_count
    if fallback_used:
        if fallback_score is None:
            std = max(float(reward_clean.std()), 1e-6)
            normalized = (reward_clean - float(reward_clean.mean())) / std
            score = -depth - pct + 1e-3 * normalized
        else:
            score = np.asarray(fallback_score, dtype=np.float64)
            if score.shape != reward.shape:
                raise ValueError("fallback score does not match rewards")
        score = np.where(finite_reward & np.isfinite(score), score, -np.inf)
        candidates = np.arange(len(reward), dtype=np.int64)
        selected = _descending_indices(score, candidates)[:top_k]
    else:
        candidates = np.flatnonzero(valid)
        selected = _descending_indices(reward_clean, candidates)[:top_k]
    return {
        "selected_indices": selected.astype(np.int64),
        "fallback_used": fallback_used,
        "valid_count": valid_count,
        "min_valid_count": min_valid_count,
    }


def compare_pose_contacts(
    candidate_min_clearance_m: np.ndarray,
    oracle_min_clearance_m: np.ndarray,
) -> dict[str, int | float]:
    """Compare pose-level P contact from minimum robot-object clearance."""
    candidate = np.asarray(candidate_min_clearance_m, dtype=np.float64)
    oracle = np.asarray(oracle_min_clearance_m, dtype=np.float64)
    if candidate.shape != oracle.shape:
        raise ValueError("candidate/oracle pose clearance shapes do not match")
    if not np.isfinite(candidate).all() or not np.isfinite(oracle).all():
        raise ValueError("pose contact clearances contain non-finite values")
    candidate_contact = candidate.reshape(-1) <= 0.0
    oracle_contact = oracle.reshape(-1) <= 0.0
    true_positive = int((candidate_contact & oracle_contact).sum())
    phantom = int((candidate_contact & ~oracle_contact).sum())
    missed = int((~candidate_contact & oracle_contact).sum())
    true_negative = int((~candidate_contact & ~oracle_contact).sum())
    predicted = true_positive + phantom
    actual = true_positive + missed
    return {
        "pose_count": int(candidate_contact.size),
        "true_positive_count": true_positive,
        "true_negative_count": true_negative,
        "phantom_contact_count": phantom,
        "missed_contact_count": missed,
        "precision": _fraction(true_positive, predicted),
        "recall": _fraction(true_positive, actual),
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


def _load_frozen_fixture(path: Path) -> tuple[dict[str, Any], str]:
    """Load the pre-score dev3-only fixture and reject any isolation drift."""
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if fixture.get("status") != "FROZEN":
        raise RuntimeError("S2 query fixture is not FROZEN")
    if fixture.get("heldout_access") != "NOT_ACCESSED_DEV3_ONLY":
        raise RuntimeError("S2 fixture heldout isolation changed")
    if fixture.get("candidate_count") != 54 or len(fixture.get("cases", [])) != 3:
        raise RuntimeError("S2 fixture authority count changed")
    return fixture, sha256_file(path)


def _resolve_case_and_candidate(
    fixture: Mapping[str, Any], case_id: str, candidate_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve one dev case and same-object candidate from the frozen fixture."""
    cases = [case for case in fixture["cases"] if case["case_id"] == case_id]
    if len(cases) != 1:
        raise RuntimeError(f"case is outside frozen dev3: {case_id}")
    case = cases[0]
    candidates = [
        candidate
        for candidate in fixture["candidates"]
        if candidate["candidate_id"] == candidate_id
        and candidate["object_key"] == case["object_key"]
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"candidate is absent or belongs to another object: {candidate_id}"
        )
    return case, candidates[0]


def _load_case_indices(
    output_root: Path, case: Mapping[str, Any], tier: str
) -> tuple[np.ndarray, str]:
    """Load one frozen nested index tier and verify its immutable NPZ."""
    if tier not in ("screen", "finalist", "production_canary"):
        raise ValueError(f"unsupported query tier: {tier}")
    path = output_root / case["indices"]["relative_path"]
    if sha256_file(path) != case["indices"]["sha256"]:
        raise RuntimeError("frozen case index NPZ changed")
    key = {
        "screen": "screen_point_indices",
        "finalist": "finalist_point_indices",
        "production_canary": "full_point_indices",
    }[tier]
    with np.load(path, allow_pickle=False) as values:
        indices = np.asarray(values[key], dtype=np.int32)
    return indices, sha256_file(path)


def _prg_manifest(case: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Load one frozen factored tape manifest after fixture SHA verification."""
    path = repo_path(case["prg_manifest"]["path"])
    if sha256_file(path) != case["prg_manifest"]["sha256"]:
        raise RuntimeError("PRG manifest changed after fixture freeze")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE" or payload["case_id"] != case["case_id"]:
        raise RuntimeError("PRG manifest identity/status mismatch")
    return path, payload


def _tape_chunk_path(entry: Mapping[str, Any]) -> Path:
    """Resolve an immutable PRG chunk from its root-relative manifest path."""
    return DEFAULT_S1_ROOT / "prg_query_tape" / entry["relative_path"]


def _entry_point_indices(
    entry: Mapping[str, Any], full: np.ndarray, tier_indices: np.ndarray
) -> np.ndarray:
    """Use full points for static poses and the frozen tier for CEM poses."""
    return full if entry["source_family"] != "cem_on_a" else tier_indices


def _query_factored_chunk(
    scene: o3d.t.geometry.RaycastingScene,
    chunk: Mapping[str, np.ndarray],
    point_indices: np.ndarray,
    *,
    pose_batch_size: int = 256,
) -> np.ndarray:
    """Stream one factored chunk through one SDF scene in bounded pose batches."""
    pose_count = int(np.asarray(chunk["geom_pos_object_local"]).shape[0])
    result = np.empty((pose_count, len(point_indices)), dtype=np.float32)
    for start in range(0, pose_count, pose_batch_size):
        stop = min(start + pose_batch_size, pose_count)
        batch = {
            "geom_pos_object_local": np.asarray(chunk["geom_pos_object_local"])[
                start:stop
            ],
            "geom_mat_object_local": np.asarray(chunk["geom_mat_object_local"])[
                start:stop
            ],
            "point_offsets_geom_local": np.asarray(chunk["point_offsets_geom_local"]),
            "point_geom_column": np.asarray(chunk["point_geom_column"]),
        }
        points = materialize_selected_query_points(batch, point_indices)
        result[start:stop] = _scene_signed_distance(scene, points).astype(np.float32)
    return result


def _indices_digest(indices: np.ndarray) -> str:
    """Hash point-index order and dtype-independent integer values."""
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def build_oracle_cache(
    *,
    fixture_path: Path,
    case_id: str,
    candidate_id: str,
    tier: str,
    output_root: Path,
    max_cem_chunks: int | None = None,
) -> dict[str, Any]:
    """Stream D_M over one frozen case/tier without storing expanded query points."""
    fixture, fixture_sha = _load_frozen_fixture(fixture_path)
    case, fixture_candidate = _resolve_case_and_candidate(
        fixture, case_id, candidate_id
    )
    _, _, oracle = _load_candidate_and_oracle(fixture_candidate)
    oracle_scene = _raycasting_scene(oracle)
    fixture_root = fixture_path.parent
    tier_indices, indices_file_sha = _load_case_indices(fixture_root, case, tier)
    full_indices, _ = _load_case_indices(fixture_root, case, "production_canary")
    manifest_path, prg = _prg_manifest(case)
    entries = []
    cem_seen = 0
    started = time.perf_counter()
    for source in prg["chunks"]:
        if source["source_family"] == "cem_on_a":
            if max_cem_chunks is not None and cem_seen >= max_cem_chunks:
                continue
            cem_seen += 1
        point_indices = _entry_point_indices(source, full_indices, tier_indices)
        source_path = _tape_chunk_path(source)
        if not source_path.is_file() or sha256_file(source_path) != source["sha256"]:
            raise RuntimeError(f"invalid PRG source chunk: {source_path}")
        with np.load(source_path, allow_pickle=False) as values:
            signed = _query_factored_chunk(oracle_scene, values, point_indices)
        relative = (
            Path("chunks")
            / source["source_family"]
            / (f"chunk_{int(source['source_chunk_index']):06d}.npz")
        )
        cache_path = output_root / relative
        _atomic_npz(cache_path, signed_distance_m=signed)
        entries.append(
            {
                "source_family": source["source_family"],
                "source_chunk_index": int(source["source_chunk_index"]),
                "source_chunk": relative_to_repo(source_path),
                "source_chunk_sha256": source["sha256"],
                "source_raw_sha256": source["source_sha256"],
                "pose_count": int(signed.shape[0]),
                "point_count": int(signed.shape[1]),
                "query_count": int(signed.size),
                "point_indices_sha256": _indices_digest(point_indices),
                "cache_relative_path": relative.as_posix(),
                "cache_sha256": sha256_file(cache_path),
                "cache_size_bytes": cache_path.stat().st_size,
            }
        )
    payload = {
        "experiment_id": "E182",
        "stage": "S2_D_M_oracle_cache",
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "status": "COMPLETE",
        "selection_eligible": max_cem_chunks is None,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "case_id": case_id,
        "object_key": case["object_key"],
        "tier": tier,
        "max_cem_chunks": max_cem_chunks,
        "signed_distance_backend": SIGNED_DISTANCE_BACKEND,
        "fixture": {
            "path": relative_to_repo(fixture_path),
            "sha256": fixture_sha,
            "candidate_identity_sha256": fixture["candidate_identity_sha256"],
        },
        "indices_file_sha256": indices_file_sha,
        "prg_manifest": {
            "path": relative_to_repo(manifest_path),
            "sha256": case["prg_manifest"]["sha256"],
        },
        "oracle_mesh": {
            "path": fixture_candidate["manifest"]["path"],
            "cleaned_mesh_path": json.loads(
                repo_path(fixture_candidate["manifest"]["path"]).read_text(
                    encoding="utf-8"
                )
            )["oracle_cleaned_mesh"]["path"],
            "cleaned_mesh_sha256": json.loads(
                repo_path(fixture_candidate["manifest"]["path"]).read_text(
                    encoding="utf-8"
                )
            )["oracle_cleaned_mesh"]["sha256"],
        },
        "entry_count": len(entries),
        "query_count": sum(entry["query_count"] for entry in entries),
        "entries": entries,
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(output_root / "manifest.json", payload)
    return payload


def _raw_cem_path(case_id: str, chunk_index: int) -> Path:
    """Resolve the local frozen on_a recorder chunk for gate/reward provenance."""
    return (
        DEFAULT_S1_ROOT / "raw_chunks/on_a" / case_id / f"chunk_{chunk_index:06d}.npz"
    )


def _gate_summaries(
    consumer: Mapping[str, np.ndarray], config: object
) -> dict[str, dict[str, np.ndarray]]:
    """Compute all enabled geometry gates from per-pose consumer minima."""
    specs = {
        "body": (
            "G_safety",
            config.cem_safety_gate_min_sdf_m,
            config.cem_safety_gate_max_violation_pct,
            config.cem_safety_gate_hard_floor_m,
        ),
        "hand": (
            "G_hand",
            config.cem_hand_gate_min_sdf_m,
            config.cem_hand_gate_max_violation_pct,
            config.cem_hand_gate_hard_floor_m,
        ),
        "leg": (
            "G_leg",
            config.cem_leg_gate_min_sdf_m,
            config.cem_leg_gate_max_violation_pct,
            config.cem_leg_gate_hard_floor_m,
        ),
    }
    out = {}
    for name, (consumer_name, minimum, max_pct, floor) in specs.items():
        values = np.asarray(consumer[consumer_name], dtype=np.float64)
        if values.size % 48:
            raise RuntimeError(f"{consumer_name}: pose count is not divisible by 48")
        out[name] = compute_gate_samples(
            values.reshape(-1, 48),
            min_sdf_m=float(minimum),
            max_violation_pct=float(max_pct),
            hard_floor_m=float(floor),
        )
    return out


def _posture_gate_summary(raw: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convert frozen posture diagnostics into the optimizer combination schema."""
    violation = np.asarray(raw["sample_posture_violation"], dtype=np.float64)
    return {
        "min_sdf_m": -violation,
        "violation_pct": violation,
        "violation_depth_mean_m": violation,
        "valid_mask": np.asarray(raw["sample_posture_valid_mask"], dtype=bool),
    }


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Define identical constant ranks as one and other degenerate ranks as zero."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return 1.0 if np.allclose(left, right) else 0.0
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else 0.0


def _merge_gate_counts(target: dict[str, int], metrics: Mapping[str, Any]) -> None:
    """Add one exact gate-confusion dictionary into a running counter."""
    for key in (
        "sample_count",
        "mask_flip_count",
        "false_reject_count",
        "false_accept_count",
    ):
        target[key] = target.get(key, 0) + int(metrics[key])


def _gate_counter_summary(counts: Mapping[str, int]) -> dict[str, int | float]:
    """Finish an aggregate gate-confusion counter with exact fractions."""
    count = int(counts.get("sample_count", 0))
    return {
        **{key: int(value) for key, value in counts.items()},
        "mask_flip_fraction": _fraction(int(counts.get("mask_flip_count", 0)), count),
        "false_reject_fraction": _fraction(
            int(counts.get("false_reject_count", 0)), count
        ),
        "false_accept_fraction": _fraction(
            int(counts.get("false_accept_count", 0)), count
        ),
    }


def _merge_contact_counts(target: dict[str, int], metrics: Mapping[str, Any]) -> None:
    """Add pose-contact confusion counts without averaging per-chunk ratios."""
    for key in (
        "pose_count",
        "true_positive_count",
        "true_negative_count",
        "phantom_contact_count",
        "missed_contact_count",
    ):
        target[key] = target.get(key, 0) + int(metrics[key])


def _contact_counter_summary(counts: Mapping[str, int]) -> dict[str, int | float]:
    """Finish aggregate pose contact precision and recall."""
    tp = int(counts.get("true_positive_count", 0))
    phantom = int(counts.get("phantom_contact_count", 0))
    missed = int(counts.get("missed_contact_count", 0))
    return {
        **{key: int(value) for key, value in counts.items()},
        "precision": _fraction(tp, tp + phantom),
        "recall": _fraction(tp, tp + missed),
    }


def _selection_overlap(
    candidate_indices: np.ndarray, oracle_indices: np.ndarray
) -> dict[str, int | float | bool]:
    """Compare shadow top-k sets and the selected index0 decision."""
    candidate = np.asarray(candidate_indices, dtype=np.int64)
    oracle = np.asarray(oracle_indices, dtype=np.int64)
    intersection = len(set(candidate.tolist()) & set(oracle.tolist()))
    return {
        "top_k": int(len(candidate)),
        "overlap_count": intersection,
        "overlap_fraction": _fraction(intersection, len(oracle)),
        "exact_set_match": set(candidate.tolist()) == set(oracle.tolist()),
        "selected_index0_match": bool(candidate[0] == oracle[0]),
    }


def _launch_floor(
    point: Mapping[str, Any],
    reward: Mapping[str, Any],
    gate: Mapping[str, Any],
    contact: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the pre-registered catastrophic technical floor to one screen result."""
    checks = {
        "coverage_and_finite": point["coverage_fraction"] == 1.0
        and point["nonfinite_count"] == 0,
        "sign_disagreement": point["sign_disagreement_fraction"] <= 0.15,
        "deep_sign_mismatch": point["deep_sign_mismatch_fraction"] <= 0.05,
        "sdf_p90": point["absolute_error_p90_m"] is not None
        and point["absolute_error_p90_m"] <= 0.03,
        "r_normalized_error_p90": reward["normalized_error_p90"] is not None
        and reward["normalized_error_p90"] <= 0.30,
        "r_rank_spearman": reward["geometry_spearman_median"] >= 0.80,
        "g_mask_flip": gate["optimizer_combined"]["mask_flip_fraction"] <= 0.15,
        "g_false_reject": gate["optimizer_combined"]["false_reject_fraction"] <= 0.15,
        "g_false_accept": gate["optimizer_combined"]["false_accept_fraction"] <= 0.15,
        "g_fallback_increase": gate["candidate_fallback_fraction"]
        - gate["raw_fallback_fraction"]
        <= 0.20,
        "p_contact_precision": contact["precision"] >= 0.70,
        "p_contact_recall": contact["recall"] >= 0.70,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
    }


def evaluate_candidate_streaming(
    *,
    fixture_path: Path,
    case_id: str,
    candidate_id: str,
    tier: str,
    oracle_cache_root: Path,
    output_path: Path,
    max_cem_chunks: int | None = None,
) -> dict[str, Any]:
    """Evaluate one exact-C candidate against cached D_M on frozen real queries."""
    fixture, fixture_sha = _load_frozen_fixture(fixture_path)
    case, fixture_candidate = _resolve_case_and_candidate(
        fixture, case_id, candidate_id
    )
    candidate, exact_union, _ = _load_candidate_and_oracle(fixture_candidate)
    candidate_scene = _raycasting_scene(exact_union)
    fixture_root = fixture_path.parent
    tier_indices, indices_file_sha = _load_case_indices(fixture_root, case, tier)
    full_indices, _ = _load_case_indices(fixture_root, case, "production_canary")
    _, prg = _prg_manifest(case)
    context = load_case_context(
        next(row for row in load_dev_rows() if row["case_id"] == case_id)
    )
    config = context.config

    cache_path = oracle_cache_root / "manifest.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if (
        cache.get("status") != "COMPLETE"
        or cache["case_id"] != case_id
        or cache["tier"] != tier
        or cache["max_cem_chunks"] != max_cem_chunks
        or cache["fixture"]["sha256"] != fixture_sha
    ):
        raise RuntimeError("oracle cache contract does not match evaluation request")

    pair = PairMetricAccumulator()
    contact_counts: dict[str, int] = {}
    gate_counts = {
        name: {}
        for name in ("body", "hand", "leg", "geometry_combined", "optimizer_combined")
    }
    r_error = StreamingHistogram(
        bin_width=R_NORMALIZED_ERROR_BIN_WIDTH,
        maximum=R_NORMALIZED_ERROR_HIST_MAX,
    )
    geometry_spearman: list[float] = []
    shadow_spearman: list[float] = []
    selection_overlap: list[dict[str, Any]] = []
    candidate_fallback_count = 0
    oracle_fallback_count = 0
    raw_fallback_count = 0
    cem_chunk_count = 0
    query_count_by_family: dict[str, int] = {}
    started = time.perf_counter()

    source_lookup = {
        (entry["source_family"], int(entry["source_chunk_index"])): entry
        for entry in prg["chunks"]
    }
    for cache_entry in cache["entries"]:
        identity = (
            cache_entry["source_family"],
            int(cache_entry["source_chunk_index"]),
        )
        source = source_lookup[identity]
        source_path = _tape_chunk_path(source)
        if sha256_file(source_path) != source["sha256"]:
            raise RuntimeError("PRG chunk SHA changed during candidate evaluation")
        cache_chunk_path = oracle_cache_root / cache_entry["cache_relative_path"]
        if sha256_file(cache_chunk_path) != cache_entry["cache_sha256"]:
            raise RuntimeError("oracle cache chunk SHA mismatch")
        point_indices = _entry_point_indices(source, full_indices, tier_indices)
        if _indices_digest(point_indices) != cache_entry["point_indices_sha256"]:
            raise RuntimeError("oracle cache point-index identity mismatch")
        with (
            np.load(source_path, allow_pickle=False) as chunk,
            np.load(cache_chunk_path, allow_pickle=False) as oracle_values,
        ):
            oracle_sdf = np.asarray(
                oracle_values["signed_distance_m"], dtype=np.float32
            )
            candidate_sdf = _query_factored_chunk(candidate_scene, chunk, point_indices)
            if candidate_sdf.shape != oracle_sdf.shape:
                raise RuntimeError("candidate/oracle streamed query shape mismatch")
            radii = np.asarray(chunk["point_radius_m"], dtype=np.float64)[point_indices]
            pair.update(candidate_sdf, oracle_sdf, radii[None, :])
            family = source["source_family"]
            query_count_by_family[family] = query_count_by_family.get(family, 0) + int(
                candidate_sdf.size
            )
            candidate_clearance = candidate_sdf.astype(np.float64) - radii[None, :]
            oracle_clearance = oracle_sdf.astype(np.float64) - radii[None, :]
            point_consumer_mask = np.asarray(chunk["point_consumer_mask"], dtype=bool)[
                point_indices
            ]
            consumer_names = np.asarray(chunk["consumer_names"]).tolist()
            candidate_consumer = consumer_min_clearance(
                candidate_clearance,
                point_consumer_mask,
                consumer_names,
            )
            oracle_consumer = consumer_min_clearance(
                oracle_clearance,
                point_consumer_mask,
                consumer_names,
            )

            if family != "cem_on_a":
                geom_ids = np.asarray(chunk["point_geom_id"], dtype=np.int32)[
                    point_indices
                ]
                p_mask = np.isin(geom_ids, prg["consumer_geom_ids"]["P_collision"])
                if not p_mask.any():
                    raise RuntimeError(
                        "static full-point chunk lost all P collision geoms"
                    )
                metrics = compare_pose_contacts(
                    candidate_clearance[:, p_mask].min(axis=1),
                    oracle_clearance[:, p_mask].min(axis=1),
                )
                _merge_contact_counts(contact_counts, metrics)
                continue

            cem_chunk_count += 1
            raw_path = _raw_cem_path(case_id, int(source["source_chunk_index"]))
            if (
                not raw_path.is_file()
                or sha256_file(raw_path) != source["source_sha256"]
            ):
                raise RuntimeError("frozen raw CEM chunk/SHA mismatch")
            with np.load(raw_path, allow_pickle=False) as raw:
                candidate_r = geometry_reward_components(candidate_consumer, config)
                oracle_r = geometry_reward_components(oracle_consumer, config)
                normalized_error = np.abs(
                    candidate_r["total"] - oracle_r["total"]
                ) / np.maximum(1.0, np.abs(oracle_r["total"]))
                r_error.update(normalized_error)
                candidate_geometry_score = (
                    candidate_r["total"].reshape(-1, 48).mean(axis=1)
                )
                oracle_geometry_score = oracle_r["total"].reshape(-1, 48).mean(axis=1)
                geometry_spearman.append(
                    _spearman(candidate_geometry_score, oracle_geometry_score)
                )
                raw_rewards = np.asarray(raw["rewards"], dtype=np.float64)
                candidate_shadow_reward = raw_rewards + candidate_geometry_score
                oracle_shadow_reward = raw_rewards + oracle_geometry_score
                shadow_spearman.append(
                    _spearman(candidate_shadow_reward, oracle_shadow_reward)
                )

                candidate_gate = _gate_summaries(candidate_consumer, config)
                oracle_gate = _gate_summaries(oracle_consumer, config)
                for name in ("body", "hand", "leg"):
                    _merge_gate_counts(
                        gate_counts[name],
                        compare_gate_validity(
                            candidate_gate[name]["valid_mask"],
                            oracle_gate[name]["valid_mask"],
                        ),
                    )
                candidate_geometry = combine_gate_summaries(
                    [candidate_gate[name] for name in ("body", "hand", "leg")]
                )
                oracle_geometry = combine_gate_summaries(
                    [oracle_gate[name] for name in ("body", "hand", "leg")]
                )
                _merge_gate_counts(
                    gate_counts["geometry_combined"],
                    compare_gate_validity(
                        candidate_geometry["valid_mask"],
                        oracle_geometry["valid_mask"],
                    ),
                )
                posture = _posture_gate_summary(raw)
                candidate_combined = combine_gate_summaries(
                    [candidate_geometry, posture]
                )
                oracle_combined = combine_gate_summaries([oracle_geometry, posture])
                _merge_gate_counts(
                    gate_counts["optimizer_combined"],
                    compare_gate_validity(
                        candidate_combined["valid_mask"],
                        oracle_combined["valid_mask"],
                    ),
                )
                top_k = int(len(np.asarray(raw["selected_indices"])))
                min_valid_frac = max(
                    float(config.cem_safety_gate_min_valid_frac),
                    float(config.cem_leg_gate_min_valid_frac),
                    float(config.cem_posture_gate_min_valid_frac),
                )
                posture_violation = np.asarray(
                    raw["sample_posture_violation"], dtype=np.float64
                )
                candidate_selection = select_shadow_elites(
                    candidate_shadow_reward,
                    candidate_combined["valid_mask"],
                    candidate_combined["violation_pct"],
                    candidate_combined["violation_depth_mean_m"],
                    top_k=top_k,
                    min_valid_frac=min_valid_frac,
                    fallback_score=candidate_shadow_reward
                    - float(config.cem_posture_gate_fallback_lambda)
                    * posture_violation,
                )
                oracle_selection = select_shadow_elites(
                    oracle_shadow_reward,
                    oracle_combined["valid_mask"],
                    oracle_combined["violation_pct"],
                    oracle_combined["violation_depth_mean_m"],
                    top_k=top_k,
                    min_valid_frac=min_valid_frac,
                    fallback_score=oracle_shadow_reward
                    - float(config.cem_posture_gate_fallback_lambda)
                    * posture_violation,
                )
                candidate_fallback_count += int(candidate_selection["fallback_used"])
                oracle_fallback_count += int(oracle_selection["fallback_used"])
                raw_min_valid_count = max(
                    1, int(np.ceil(min_valid_frac * len(raw_rewards)))
                )
                raw_fallback_count += int(
                    int(np.asarray(raw["sample_gate_valid_mask"], dtype=bool).sum())
                    < raw_min_valid_count
                )
                selection_overlap.append(
                    _selection_overlap(
                        candidate_selection["selected_indices"],
                        oracle_selection["selected_indices"],
                    )
                )

    point_summary = pair.summary()
    contact_summary = _contact_counter_summary(contact_counts)
    r_hist = r_error.summary()
    reward_summary = {
        "score_contract": "RAW_E178_REWARD_PLUS_DERIVED_GEOMETRY_COMPONENT",
        "normalized_error_contract": "ABS_C_MINUS_M_DIV_MAX_1_ABS_M",
        "normalized_error_p90": r_hist["p90_conservative_upper"],
        "normalized_error_histogram": r_hist,
        "geometry_spearman_median": float(np.median(geometry_spearman)),
        "geometry_spearman_p10": float(np.quantile(geometry_spearman, 0.10)),
        "shadow_reward_spearman_median": float(np.median(shadow_spearman)),
        "shadow_reward_spearman_p10": float(np.quantile(shadow_spearman, 0.10)),
        "chunk_count": cem_chunk_count,
    }
    gate_summary = {
        name: _gate_counter_summary(values) for name, values in gate_counts.items()
    }
    gate_summary.update(
        {
            "chunk_count": cem_chunk_count,
            "candidate_fallback_count": candidate_fallback_count,
            "candidate_fallback_fraction": _fraction(
                candidate_fallback_count, cem_chunk_count
            ),
            "oracle_fallback_count": oracle_fallback_count,
            "oracle_fallback_fraction": _fraction(
                oracle_fallback_count, cem_chunk_count
            ),
            "raw_fallback_count": raw_fallback_count,
            "raw_fallback_fraction": _fraction(raw_fallback_count, cem_chunk_count),
            "selected_index0_mismatch_count": sum(
                not value["selected_index0_match"] for value in selection_overlap
            ),
            "topk_overlap_mean": float(
                np.mean([value["overlap_fraction"] for value in selection_overlap])
            ),
            "topk_exact_set_match_fraction": float(
                np.mean([value["exact_set_match"] for value in selection_overlap])
            ),
        }
    )
    floor = _launch_floor(point_summary, reward_summary, gate_summary, contact_summary)
    payload = {
        "experiment_id": "E182",
        "stage": "S2_exact_C_task_query_evaluation",
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "status": "COMPLETE",
        "selection_eligible": max_cem_chunks is None,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "case_id": case_id,
        "object_key": case["object_key"],
        "candidate_id": candidate_id,
        "candidate_asset_sha256": candidate["candidate_asset_sha256"],
        "actual_hulls": int(candidate["hull_count"]),
        "max_hulls": int(candidate["parameters"]["max_convex_hull"]),
        "tier": tier,
        "max_cem_chunks": max_cem_chunks,
        "fixture": {
            "path": relative_to_repo(fixture_path),
            "sha256": fixture_sha,
            "candidate_identity_sha256": fixture["candidate_identity_sha256"],
        },
        "indices_file_sha256": indices_file_sha,
        "oracle_cache": {
            "path": relative_to_repo(cache_path),
            "sha256": sha256_file(cache_path),
        },
        "query_count_by_family": query_count_by_family,
        "point_metrics": point_summary,
        "p_pose_contact": contact_summary,
        "r_shadow": reward_summary,
        "g_shadow": gate_summary,
        "launch_floor": floor,
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(output_path, payload)
    return payload


def run_bounded_real_test(
    *,
    fixture_path: Path = DEFAULT_FIXTURE,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Run the fixed bucket007/K8/one-chunk non-selection integration test."""
    case_id = "bucket007_20231020_055_p1"
    candidate_id = "t005_k08_v032"
    bounded_root = output_root / "bounded_real" / f"{case_id}__{candidate_id}"
    oracle_root = bounded_root / "oracle_cache"
    oracle = build_oracle_cache(
        fixture_path=fixture_path,
        case_id=case_id,
        candidate_id=candidate_id,
        tier="screen",
        output_root=oracle_root,
        max_cem_chunks=1,
    )
    result_path = bounded_root / "candidate_result.json"
    result = evaluate_candidate_streaming(
        fixture_path=fixture_path,
        case_id=case_id,
        candidate_id=candidate_id,
        tier="screen",
        oracle_cache_root=oracle_root,
        output_path=result_path,
        max_cem_chunks=1,
    )
    fixture, fixture_sha = _load_frozen_fixture(fixture_path)
    case, _ = _resolve_case_and_candidate(fixture, case_id, candidate_id)
    static_query_count = (
        case["source_pose_counts"]["reference"]
        + case["source_pose_counts"]["e178_final"]
    ) * EXPECTED_POINTS_PER_POSE
    expected_query_count = static_query_count + 64 * 48 * case["screen_point_count"]
    checks = {
        "fixture_sha_exact": result["fixture"]["sha256"] == fixture_sha,
        "oracle_query_count": oracle["query_count"] == expected_query_count,
        "candidate_query_count": result["point_metrics"]["query_count"]
        == expected_query_count,
        "one_cem_chunk": result["r_shadow"]["chunk_count"] == 1
        and result["g_shadow"]["chunk_count"] == 1,
        "finite": result["point_metrics"]["nonfinite_count"] == 0,
        "p_schema": result["p_pose_contact"]["pose_count"]
        == case["source_pose_counts"]["reference"]
        + case["source_pose_counts"]["e178_final"],
        "r_schema": result["r_shadow"]["normalized_error_p90"] is not None,
        "g_schema": result["g_shadow"]["optimizer_combined"]["sample_count"] == 64,
        "selection_forbidden": not result["selection_eligible"]
        and not oracle["selection_eligible"],
        "heldout_not_accessed": result["heldout_access"] == "NOT_ACCESSED_DEV3_ONLY",
    }
    payload = {
        "experiment_id": "E182",
        "stage": "S2_bounded_real_integration_test",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "selection_eligible": False,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "case_id": case_id,
        "candidate_id": candidate_id,
        "tier": "screen",
        "max_cem_chunks": 1,
        "expected_query_count": expected_query_count,
        "checks": checks,
        "oracle_manifest": {
            "path": relative_to_repo(oracle_root / "manifest.json"),
            "sha256": sha256_file(oracle_root / "manifest.json"),
        },
        "candidate_result": {
            "path": relative_to_repo(result_path),
            "sha256": sha256_file(result_path),
        },
    }
    atomic_json(bounded_root / "audit.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    """Parse bounded-real and reusable cache/evaluation commands."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("bounded-real", "build-oracle", "evaluate-candidate")
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--case-id")
    parser.add_argument("--candidate-id")
    parser.add_argument(
        "--tier", choices=("screen", "finalist", "production_canary"), default="screen"
    )
    parser.add_argument("--max-cem-chunks", type=int)
    parser.add_argument("--oracle-cache-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    """Execute one explicitly bounded S2 task-query action."""
    args = parse_args()
    if args.command == "bounded-real":
        payload = run_bounded_real_test(
            fixture_path=args.fixture, output_root=args.output_root
        )
        print(
            f"E182_S2_BOUNDED_REAL={payload['status']} "
            f"queries={payload['expected_query_count']}"
        )
        return 0 if payload["status"] == "PASS" else 1
    if not args.case_id or not args.candidate_id:
        raise RuntimeError("case-id and candidate-id are required")
    if args.command == "build-oracle":
        oracle_root = args.oracle_cache_root or (
            args.output_root / "oracle_cache" / args.tier / args.case_id
        )
        payload = build_oracle_cache(
            fixture_path=args.fixture,
            case_id=args.case_id,
            candidate_id=args.candidate_id,
            tier=args.tier,
            output_root=oracle_root,
            max_cem_chunks=args.max_cem_chunks,
        )
        print(
            f"E182_S2_ORACLE_CACHE={payload['status']} queries={payload['query_count']}"
        )
        return 0
    if args.oracle_cache_root is None or args.output is None:
        raise RuntimeError(
            "evaluate-candidate requires --oracle-cache-root and --output"
        )
    payload = evaluate_candidate_streaming(
        fixture_path=args.fixture,
        case_id=args.case_id,
        candidate_id=args.candidate_id,
        tier=args.tier,
        oracle_cache_root=args.oracle_cache_root,
        output_path=args.output,
        max_cem_chunks=args.max_cem_chunks,
    )
    print(
        f"E182_S2_CANDIDATE={payload['status']} "
        f"launch_floor={payload['launch_floor']['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
