"""Shared production-grid and exact convex-union distance query primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import open3d as o3d
import torch
import trimesh
from scipy.spatial import ConvexHull

from spider.geometry.grid_sdf import sha256_file
from spider.simulators.mjwp_object_distance import (
    GridObjectDistanceRuntime,
    sample_robot_geoms,
)

BOOLEAN_ENGINE = "manifold"
DEFAULT_QUERY_CHUNK = 500_000


def repo_path(repo_root: Path, value: str | Path) -> Path:
    """Resolve a repository-relative authority path without following display aliases."""
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def load_frozen_parts(
    collider: Mapping[str, Any], repo_root: Path
) -> list[trimesh.Trimesh]:
    """Load and SHA-check the frozen ordered convex part meshes."""
    parts: list[trimesh.Trimesh] = []
    for index, part in enumerate(collider["ordered_parts"]):
        path = repo_path(repo_root, part["path"])
        if int(part["part_index"]) != index or sha256_file(path) != part["sha256"]:
            raise RuntimeError(f"frozen collider part changed: {path}")
        mesh = trimesh.load(path, force="mesh", process=False, maintain_order=True)
        if not isinstance(mesh, trimesh.Trimesh):
            raise RuntimeError(f"part loader returned non-mesh: {path}")
        if (
            mesh.is_empty
            or not mesh.is_convex
            or not mesh.is_watertight
            or not mesh.is_winding_consistent
            or not np.isfinite(mesh.vertices).all()
            or float(mesh.volume) <= 0.0
        ):
            raise RuntimeError(f"invalid frozen convex part: {path}")
        parts.append(mesh)
    if not parts:
        raise ValueError("exact convex union requires at least one part")
    return parts


def build_exact_union_mesh(parts: Sequence[trimesh.Trimesh]) -> trimesh.Trimesh:
    """Boolean-union closed parts so internal faces do not enter exact-C distance."""
    if not parts:
        raise ValueError("exact convex union requires at least one part")
    result = trimesh.boolean.union(
        list(parts),
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


def convex_union_halfspaces(
    parts: Sequence[trimesh.Trimesh],
) -> list[dict[str, np.ndarray]]:
    """Build deterministic analytic half-spaces for convex-union containment."""
    volumes: list[dict[str, np.ndarray]] = []
    for part in parts:
        hull = ConvexHull(np.asarray(part.vertices, dtype=np.float64))
        equations = np.asarray(hull.equations, dtype=np.float64)
        normalized = np.round(equations, decimals=12)
        _, unique_indices = np.unique(normalized, axis=0, return_index=True)
        equations = equations[np.sort(unique_indices)]
        volumes.append(
            {
                "minimum": np.asarray(part.bounds[0], dtype=np.float64),
                "maximum": np.asarray(part.bounds[1], dtype=np.float64),
                "equations": equations,
            }
        )
    return volumes


def convex_union_contains(
    volumes: Sequence[Mapping[str, np.ndarray]],
    points: np.ndarray,
    *,
    tolerance_m: float = 1e-9,
) -> np.ndarray:
    """Return exact convex-union occupancy from half-space inequalities."""
    queries = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    contains = np.zeros(len(queries), dtype=bool)
    for volume in volumes:
        active = ~contains
        active &= np.all(queries >= volume["minimum"] - tolerance_m, axis=1)
        active &= np.all(queries <= volume["maximum"] + tolerance_m, axis=1)
        indices = np.flatnonzero(active)
        if not len(indices):
            continue
        equations = volume["equations"]
        residual = queries[indices] @ equations[:, :3].T + equations[:, 3]
        contains[indices] = np.all(residual <= tolerance_m, axis=1)
    return contains


def build_exact_scene(mesh: trimesh.Trimesh) -> o3d.t.geometry.RaycastingScene:
    """Create the exact-C unsigned-distance scene."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)),
    )
    return scene


def exact_signed_distance(
    scene: o3d.t.geometry.RaycastingScene,
    volumes: Sequence[Mapping[str, np.ndarray]],
    points: np.ndarray,
    *,
    chunk_size: int = DEFAULT_QUERY_CHUNK,
) -> np.ndarray:
    """Query negative-inside exact-C distance for arbitrary leading point axes."""
    queries = np.asarray(points, dtype=np.float32)
    if queries.ndim < 2 or queries.shape[-1] != 3:
        raise ValueError(f"query points must end in axis 3, got {queries.shape}")
    flat = queries.reshape(-1, 3)
    result = np.empty(len(flat), dtype=np.float32)
    for start in range(0, len(flat), chunk_size):
        stop = min(start + chunk_size, len(flat))
        block = flat[start:stop]
        magnitude = scene.compute_distance(o3d.core.Tensor(block)).numpy()
        inside = convex_union_contains(volumes, block)
        result[start:stop] = np.where(inside, -magnitude, magnitude)
    if not np.isfinite(result).all():
        raise RuntimeError("exact-C distance contains non-finite values")
    return result.reshape(queries.shape[:-1])


def resolve_geom_ids(model: mujoco.MjModel, names: Sequence[str]) -> list[int]:
    """Resolve required robot geom names and fail on any missing name."""
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in names]
    if any(gid < 0 for gid in ids):
        raise ValueError(f"missing geom names: {list(names)}")
    return ids


def exact_per_geom(
    runtime: GridObjectDistanceRuntime,
    model: mujoco.MjModel,
    ordered_ids: list[int],
    geom_xpos: torch.Tensor,
    geom_xmat: torch.Tensor,
    body_xpos: torch.Tensor,
    body_xmat: torch.Tensor,
    exact_scene: o3d.t.geometry.RaycastingScene,
    volumes: Sequence[Mapping[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate exact-C and diagnostic production grid values per robot geom."""
    sampled = sample_robot_geoms(
        model, ordered_ids, geom_xpos=geom_xpos, geom_xmat=geom_xmat
    )
    exact_columns: list[np.ndarray] = []
    grid_columns: list[np.ndarray] = []
    for gid in ordered_ids:
        sample = sampled[gid]
        local_tensor = runtime.world_to_object(
            sample.points_world, body_xpos, body_xmat
        )
        local = local_tensor.numpy()
        exact_samples = exact_signed_distance(exact_scene, volumes, local)
        grid_samples = runtime.grid.query(local_tensor).numpy()
        exact_columns.append((exact_samples - sample.radius_m).min(axis=1))
        grid_columns.append((grid_samples - sample.radius_m).min(axis=1))
    return np.stack(exact_columns, axis=1), np.stack(grid_columns, axis=1)


def group_min(
    per_geom: np.ndarray, ordered_ids: Sequence[int], group: Sequence[int]
) -> np.ndarray:
    """Reduce per-geom distances over one configured geometry group."""
    columns = {gid: index for index, gid in enumerate(ordered_ids)}
    return np.asarray(per_geom)[:, [columns[gid] for gid in group]].min(axis=1)
