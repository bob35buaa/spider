#!/usr/bin/env python3
"""Deterministic semantic 1/5-box bucket proxies for E177."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
E176_DIR = HERE.parent / "E176"
S2_TEMPLATES = (
    REPO
    / "workspace/core4d/scripts/data_construction_v3/stages/s2_templates"
)
sys.path.insert(0, str(E176_DIR))
sys.path.insert(0, str(S2_TEMPLATES))

from build_or_audit_templates import geom_box_xml  # noqa: E402
from lowgeom_proxy import (  # noqa: E402
    fidelity_metrics,
    point_to_proxy_surface_distance,
)


BODY_LAYERS_BY_OBJECT = {
    "bucket003": 5,
    "bucket007": 5,
}
BODY_XZ_SCALE_BY_OBJECT = {
    "bucket003": 0.94,
    "bucket007": 0.82,
}
EXPECTED_BOXES_BY_OBJECT = {
    "bucket003": 5,
    "bucket004": 1,
    "bucket007": 5,
}
SURFACE_SAMPLE_COUNT = 200_000
CROSS_QUANTILE = 0.005
LAYER_OVERLAP_M = 0.004


@dataclass(frozen=True)
class ProxyBox:
    center: np.ndarray
    half_size: np.ndarray
    label: str


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(mesh_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"expected Trimesh from {mesh_path}, got {type(loaded)}")
    if not np.isfinite(loaded.bounds).all() or float(loaded.extents.max()) <= 0:
        raise ValueError(f"invalid mesh bounds: {mesh_path}")
    return loaded


def _surface_points(mesh: trimesh.Trimesh) -> np.ndarray:
    points, _ = trimesh.sample.sample_surface(
        mesh,
        SURFACE_SAMPLE_COUNT,
        seed=0,
    )
    return np.asarray(points, dtype=np.float64)


def _robust_xz_bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 100:
        raise ValueError(f"too few points for robust bounds: {len(points)}")
    lower = np.quantile(
        points[:, (0, 2)],
        CROSS_QUANTILE,
        axis=0,
    )
    upper = np.quantile(
        points[:, (0, 2)],
        1.0 - CROSS_QUANTILE,
        axis=0,
    )
    if np.any(upper <= lower):
        raise ValueError(f"invalid robust XZ bounds: {lower=} {upper=}")
    return lower, upper


def _box(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    label: str,
) -> ProxyBox:
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("box bounds must be 3-vectors")
    if np.any(upper <= lower):
        raise ValueError(f"non-positive box {label}: {lower=} {upper=}")
    return ProxyBox(
        center=(lower + upper) / 2.0,
        half_size=(upper - lower) / 2.0,
        label=label,
    )


def _solid_aabb(mesh: trimesh.Trimesh) -> list[ProxyBox]:
    return [
        _box(
            mesh.bounds[0],
            mesh.bounds[1],
            label="solid_aabb",
        )
    ]


def _stepped_body(
    mesh: trimesh.Trimesh,
    object_key: str,
) -> tuple[list[ProxyBox], dict[str, Any]]:
    points = _surface_points(mesh)
    lower, upper = np.asarray(mesh.bounds, dtype=np.float64)
    body_layers = BODY_LAYERS_BY_OBJECT[object_key]
    body_xz_scale = BODY_XZ_SCALE_BY_OBJECT[object_key]
    edges = np.linspace(lower[1], upper[1], body_layers + 1)
    boxes: list[ProxyBox] = []
    layer_rows: list[dict[str, Any]] = []

    for index, (raw_lo, raw_hi) in enumerate(
        zip(edges[:-1], edges[1:], strict=True)
    ):
        select_lo = raw_lo - (LAYER_OVERLAP_M if index else 0.0)
        select_hi = raw_hi + (
            LAYER_OVERLAP_M if index < body_layers - 1 else 0.0
        )
        band = points[
            (points[:, 1] >= select_lo)
            & (points[:, 1] <= select_hi)
        ]
        xz_lo, xz_hi = _robust_xz_bounds(band)
        xz_center = (xz_lo + xz_hi) / 2.0
        xz_half = (xz_hi - xz_lo) / 2.0 * body_xz_scale
        xz_lo = xz_center - xz_half
        xz_hi = xz_center + xz_half
        y_lo = max(float(lower[1]), select_lo)
        y_hi = min(float(upper[1]), select_hi)
        box = _box(
            np.asarray([xz_lo[0], y_lo, xz_lo[1]]),
            np.asarray([xz_hi[0], y_hi, xz_hi[1]]),
            label=f"body_{index:03d}",
        )
        boxes.append(box)
        layer_rows.append(
            {
                "label": box.label,
                "axis": "y",
                "y_lower_m": y_lo,
                "y_upper_m": y_hi,
                "xz_lower_m": xz_lo.tolist(),
                "xz_upper_m": xz_hi.tolist(),
                "xz_inward_scale": body_xz_scale,
                "surface_samples": int(len(band)),
            }
        )

    return boxes, {
        "body_layers": body_layers,
        "body_xz_inward_scale": body_xz_scale,
        "frustum_axis": "local_y_positive",
        "cross_section_axes": "local_xz",
        "has_separate_lid": False,
        "layer_overlap_m": LAYER_OVERLAP_M,
        "layer_rows": layer_rows,
    }


def build_semantic_boxes(
    mesh_path: Path,
    object_key: str,
) -> tuple[list[ProxyBox], dict[str, Any]]:
    if object_key not in EXPECTED_BOXES_BY_OBJECT:
        raise KeyError(f"unsupported E177 object: {object_key}")
    mesh = load_mesh(mesh_path)
    if object_key == "bucket004":
        boxes = _solid_aabb(mesh)
        detail: dict[str, Any] = {
            "body_layers": 1,
            "body_xz_inward_scale": None,
            "frustum_axis": None,
            "cross_section_axes": None,
            "has_separate_lid": False,
            "layer_overlap_m": 0.0,
            "layer_rows": [],
        }
        policy = "user_authorized_solid_mesh_aabb"
    else:
        boxes, detail = _stepped_body(mesh, object_key)
        policy = "semantic_five_solid_body_steps_no_lid"

    expected = EXPECTED_BOXES_BY_OBJECT[object_key]
    if len(boxes) != expected:
        raise AssertionError(
            f"{object_key} count={len(boxes)} expected={expected}"
        )
    proxy_lower = np.min(
        [box.center - box.half_size for box in boxes],
        axis=0,
    )
    proxy_upper = np.max(
        [box.center + box.half_size for box in boxes],
        axis=0,
    )
    tolerance = 0.005
    if np.any(proxy_lower < mesh.bounds[0] - tolerance) or np.any(
        proxy_upper > mesh.bounds[1] + tolerance
    ):
        raise AssertionError(
            f"{object_key} proxy exceeds mesh AABB tolerance: "
            f"{proxy_lower=} {proxy_upper=} bounds={mesh.bounds}"
        )
    mesh_center = mesh.bounds.mean(axis=0)
    center_inside_count = sum(
        bool(np.all(np.abs(mesh_center - box.center) <= box.half_size))
        for box in boxes
    )
    if center_inside_count < 1:
        raise AssertionError(
            f"{object_key} solid proxy does not cover mesh center"
        )
    return boxes, {
        "object_key": object_key,
        "collision_policy": policy,
        "object_geom_count": len(boxes),
        "proxy_aabb_lower_m": proxy_lower.tolist(),
        "proxy_aabb_upper_m": proxy_upper.tolist(),
        "mesh_aabb_lower_m": mesh.bounds[0].tolist(),
        "mesh_aabb_upper_m": mesh.bounds[1].tolist(),
        "mesh_center_inside_count": center_inside_count,
        **detail,
    }


def proxy_xml(
    boxes: list[ProxyBox],
    *,
    rgba: str = "1 0.18 0.02 0.45",
) -> tuple[str, list[str]]:
    geoms: list[str] = []
    names: list[str] = []
    for index, box in enumerate(boxes):
        name = (
            "object_collision"
            if index == 0
            else f"object_collision_{box.label}"
        )
        names.append(name)
        geoms.append(
            geom_box_xml(
                name,
                box.center,
                box.half_size,
                rgba=rgba,
            )
        )
    return "\n".join(geoms), names


def _sample_union_surface(
    boxes: list[ProxyBox],
    *,
    samples_per_face: int,
) -> np.ndarray:
    rng = np.random.default_rng(0)
    samples: list[np.ndarray] = []
    for box_index, box in enumerate(boxes):
        for axis in range(3):
            other_axes = [value for value in range(3) if value != axis]
            for sign in (-1.0, 1.0):
                points = np.tile(box.center, (samples_per_face, 1))
                points[:, axis] += sign * box.half_size[axis]
                for other_axis in other_axes:
                    points[:, other_axis] += rng.uniform(
                        -box.half_size[other_axis],
                        box.half_size[other_axis],
                        samples_per_face,
                    )
                keep = np.ones(samples_per_face, dtype=bool)
                for other_index, other_box in enumerate(boxes):
                    if other_index == box_index:
                        continue
                    keep &= ~np.all(
                        np.abs(points - other_box.center)
                        < other_box.half_size - 1e-6,
                        axis=1,
                    )
                samples.append(points[keep])
    return np.concatenate(samples, axis=0)


def union_fidelity_metrics(
    mesh_path: Path,
    boxes: list[ProxyBox],
    *,
    mesh_sample_count: int = 8_000,
    proxy_samples_per_face: int = 64,
) -> dict[str, float]:
    mesh = load_mesh(mesh_path)
    mesh_points, _ = trimesh.sample.sample_surface(
        mesh,
        mesh_sample_count,
        seed=0,
    )
    mesh_to_proxy = point_to_proxy_surface_distance(mesh_points, boxes)
    proxy_points = _sample_union_surface(
        boxes,
        samples_per_face=proxy_samples_per_face,
    )
    _, proxy_to_mesh, _ = trimesh.proximity.closest_point_naive(
        mesh,
        proxy_points,
    )
    return {
        "union_mesh_to_proxy_p50_m": float(
            np.quantile(mesh_to_proxy, 0.50)
        ),
        "union_mesh_to_proxy_p90_m": float(
            np.quantile(mesh_to_proxy, 0.90)
        ),
        "union_mesh_to_proxy_p95_m": float(
            np.quantile(mesh_to_proxy, 0.95)
        ),
        "union_proxy_to_mesh_p50_m": float(
            np.quantile(proxy_to_mesh, 0.50)
        ),
        "union_proxy_to_mesh_p90_m": float(
            np.quantile(proxy_to_mesh, 0.90)
        ),
        "union_proxy_to_mesh_p95_m": float(
            np.quantile(proxy_to_mesh, 0.95)
        ),
        "union_proxy_surface_samples": int(len(proxy_points)),
    }


__all__ = [
    "EXPECTED_BOXES_BY_OBJECT",
    "ProxyBox",
    "build_semantic_boxes",
    "fidelity_metrics",
    "load_mesh",
    "point_to_proxy_surface_distance",
    "proxy_xml",
    "union_fidelity_metrics",
]
