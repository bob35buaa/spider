#!/usr/bin/env python3
"""Deterministic <=9-box surface-voxel proxies for E176."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


REPO = Path(__file__).resolve().parents[5]
S2_TEMPLATES = (
    REPO
    / "workspace/core4d/scripts/data_construction_v3/stages/s2_templates"
)
sys.path.insert(0, str(S2_TEMPLATES))
from build_or_audit_templates import (  # noqa: E402
    geom_box_xml,
    merge_occupied_voxels,
)


MAX_OBJECT_GEOMS = 9
TARGET_CELLS_BY_OBJECT = {
    "bucket003": 4,
    "bucket004": 4,
    "bucket007": 3,
    "bucket009": 4,
    "bucket010": 9,
    "desk007": 5,
}
EXPECTED_BOXES_BY_OBJECT = {
    "bucket003": 7,
    "bucket004": 6,
    "bucket007": 7,
    "bucket009": 6,
    "bucket010": 7,
    "desk007": 9,
}


@dataclass(frozen=True)
class ProxyBox:
    center: np.ndarray
    half_size: np.ndarray


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(mesh_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"expected Trimesh from {mesh_path}, got {type(loaded)}")
    if not np.isfinite(loaded.bounds).all() or float(loaded.extents.max()) <= 0:
        raise ValueError(f"invalid mesh bounds: {mesh_path}")
    return loaded


def build_lowgeom_boxes(
    mesh_path: Path,
    object_key: str,
) -> tuple[list[ProxyBox], dict[str, Any]]:
    if object_key not in TARGET_CELLS_BY_OBJECT:
        raise KeyError(f"missing frozen E176 target_cells for {object_key}")
    mesh = load_mesh(mesh_path)
    target_cells = TARGET_CELLS_BY_OBJECT[object_key]
    pitch_scalar = float(mesh.extents.max()) / float(target_cells)
    voxels = mesh.voxelized(pitch_scalar)
    pitch = np.asarray(voxels.pitch, dtype=np.float64)
    raw_boxes, left_unmerged = merge_occupied_voxels(
        voxels.matrix.astype(bool),
        pitch,
        np.asarray(voxels.transform, dtype=np.float64),
        max_boxes=MAX_OBJECT_GEOMS,
    )
    if left_unmerged:
        raise ValueError(
            f"{object_key} exceeds {MAX_OBJECT_GEOMS} boxes at "
            f"target_cells={target_cells}: left_unmerged={left_unmerged}"
        )
    expected = EXPECTED_BOXES_BY_OBJECT[object_key]
    if len(raw_boxes) != expected:
        raise AssertionError(
            f"{object_key} frozen box-count drift: "
            f"observed={len(raw_boxes)} expected={expected}"
        )

    boxes: list[ProxyBox] = []
    for center, half_size in raw_boxes:
        shrink = np.minimum(pitch * 0.12, half_size * 0.25)
        shrunk = np.maximum(half_size - shrink, pitch * 0.22)
        boxes.append(
            ProxyBox(
                center=np.asarray(center, dtype=np.float64),
                half_size=np.asarray(shrunk, dtype=np.float64),
            )
        )
    if not 1 <= len(boxes) <= MAX_OBJECT_GEOMS:
        raise AssertionError(f"invalid E176 box count: {len(boxes)}")

    mesh_center = mesh.bounds.mean(axis=0)
    center_inside_count = sum(
        bool(np.all(np.abs(mesh_center - box.center) <= box.half_size))
        for box in boxes
    )
    if center_inside_count:
        raise AssertionError(
            f"{object_key} coarse proxy fills mesh AABB center"
        )
    return boxes, {
        "object_key": object_key,
        "target_cells": target_cells,
        "object_geom_count": len(boxes),
        "voxel_pitch_m": float(pitch[0]),
        "occupied_voxels": int(voxels.matrix.sum()),
        "center_inside_count": center_inside_count,
    }


def proxy_xml(
    boxes: list[ProxyBox],
    *,
    rgba: str = "1 0.18 0.02 0.45",
) -> tuple[str, list[str]]:
    if not 1 <= len(boxes) <= MAX_OBJECT_GEOMS:
        raise ValueError(f"proxy box count must be 1..9, got {len(boxes)}")
    geoms: list[str] = []
    names: list[str] = []
    for index, box in enumerate(boxes):
        name = (
            "object_collision"
            if index == 0
            else f"object_collision_coarse_{index:03d}"
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


def point_to_proxy_surface_distance(
    points: np.ndarray,
    boxes: list[ProxyBox],
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    distance = np.full(points.shape[0], np.inf, dtype=np.float64)
    for box in boxes:
        q = np.abs(points - box.center) - box.half_size
        signed = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        signed += np.minimum(np.max(q, axis=1), 0.0)
        distance = np.minimum(distance, np.abs(signed))
    return distance


def _sample_proxy_surface(
    boxes: list[ProxyBox],
    *,
    samples_per_face: int,
    rng: np.random.Generator,
) -> np.ndarray:
    samples: list[np.ndarray] = []
    for box in boxes:
        for axis in range(3):
            other_axes = [value for value in range(3) if value != axis]
            for sign in (-1.0, 1.0):
                points = np.tile(
                    box.center,
                    (samples_per_face, 1),
                )
                points[:, axis] += sign * box.half_size[axis]
                for other in other_axes:
                    points[:, other] += rng.uniform(
                        -box.half_size[other],
                        box.half_size[other],
                        samples_per_face,
                    )
                samples.append(points)
    return np.concatenate(samples, axis=0)


def fidelity_metrics(
    mesh_path: Path,
    boxes: list[ProxyBox],
    *,
    mesh_sample_count: int = 8_000,
    proxy_samples_per_face: int = 64,
) -> dict[str, float]:
    mesh = load_mesh(mesh_path)
    state = np.random.get_state()
    try:
        np.random.seed(0)
        mesh_points, _ = trimesh.sample.sample_surface(
            mesh,
            mesh_sample_count,
        )
    finally:
        np.random.set_state(state)
    mesh_to_proxy = point_to_proxy_surface_distance(mesh_points, boxes)
    proxy_points = _sample_proxy_surface(
        boxes,
        samples_per_face=proxy_samples_per_face,
        rng=np.random.default_rng(0),
    )
    _, proxy_to_mesh, _ = trimesh.proximity.closest_point_naive(
        mesh,
        proxy_points,
    )
    return {
        "mesh_to_proxy_p50_m": float(np.quantile(mesh_to_proxy, 0.50)),
        "mesh_to_proxy_p90_m": float(np.quantile(mesh_to_proxy, 0.90)),
        "mesh_to_proxy_p95_m": float(np.quantile(mesh_to_proxy, 0.95)),
        "mesh_to_proxy_max_m": float(mesh_to_proxy.max()),
        "proxy_to_mesh_p50_m": float(np.quantile(proxy_to_mesh, 0.50)),
        "proxy_to_mesh_p90_m": float(np.quantile(proxy_to_mesh, 0.90)),
        "proxy_to_mesh_p95_m": float(np.quantile(proxy_to_mesh, 0.95)),
        "proxy_to_mesh_max_m": float(proxy_to_mesh.max()),
    }

