#!/usr/bin/env python3
"""Bake E186 canonical object-local grid-SDFs from the frozen hull unions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import torch
import trimesh
from scipy.spatial import ConvexHull

REPO_ROOT = Path(__file__).resolve().parents[5]
E182_ROOT = REPO_ROOT / "workspace/core4d/scripts/experiments/E182"
if str(E182_ROOT) not in sys.path:
    sys.path.insert(0, str(E182_ROOT))

from evaluate_task_queries import build_exact_union_mesh  # noqa: E402

from spider.geometry.grid_sdf import CanonicalGridSDF, sha256_file  # noqa: E402

COLLIDER_LOCK = (
    REPO_ROOT / "workspace/core4d/results/E186/s0_environment/collider_lock.json"
)
EXPECTED_COLLIDER_LOCK_SHA256 = (
    "6a20df7c2b47d0d5a5ea104bdd18ab129753db7c3a14e5f82aa0ce93a1a66065"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "workspace/core4d/results/E186/s1_canonical_grid_sdf_v4"
)
DEFAULT_VOXEL_SIZE_M = 0.005
DEFAULT_VOXEL_SIZE_BY_OBJECT = {
    "bucket003": 0.005,
    "bucket004": 0.0025,
    "bucket007": 0.005,
}
MAX_PRODUCTION_QUERY_RADIUS_M = 0.09
MAX_ACTIVE_REWARD_SUPPORT_M = 0.02
MIN_MINKOWSKI_SUPPORT_MARGIN_M = (
    MAX_PRODUCTION_QUERY_RADIUS_M + MAX_ACTIVE_REWARD_SUPPORT_M
)
DEFAULT_MARGIN_M = 0.12
DEFAULT_VALIDATION_POINTS = 1_000_000
DEFAULT_QUERY_CHUNK = 500_000


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _display_path(path: Path) -> str:
    absolute = path.absolute()
    try:
        return absolute.relative_to(REPO_ROOT.absolute()).as_posix()
    except ValueError:
        return str(absolute)


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError(f"immutable grid artifact mismatch: {path}")
        return
    _atomic_bytes(path, payload)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def load_collider_lock(path: Path = COLLIDER_LOCK) -> dict[str, Any]:
    """Load the exact Gate0 collider lock and recheck every ordered part."""
    if sha256_file(path) != EXPECTED_COLLIDER_LOCK_SHA256:
        raise RuntimeError("E186 collider lock SHA changed")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("status") != "FROZEN" or lock.get("collider_count") != 3:
        raise RuntimeError("invalid E186 collider lock")
    for object_key, collider in lock["objects"].items():
        if collider["object_key"] != object_key:
            raise RuntimeError("collider lock object mismatch")
        parts = collider["ordered_parts"]
        if _canonical_digest(parts) != collider["ordered_parts_sha256"]:
            raise RuntimeError(f"{object_key}: ordered part digest changed")
        for index, part in enumerate(parts):
            part_path = _repo_path(part["path"])
            if part["part_index"] != index or sha256_file(part_path) != part["sha256"]:
                raise RuntimeError(f"{object_key}: frozen part changed: {part_path}")
    return lock


def _load_parts(collider: dict[str, Any]) -> list[trimesh.Trimesh]:
    """Load the frozen ordered convex part meshes."""
    parts: list[trimesh.Trimesh] = []
    for part in collider["ordered_parts"]:
        mesh = trimesh.load(
            _repo_path(part["path"]),
            force="mesh",
            process=False,
            maintain_order=True,
        )
        if not isinstance(mesh, trimesh.Trimesh):
            raise RuntimeError("part loader did not return Trimesh")
        if not mesh.is_convex:
            raise RuntimeError("frozen CoACD part is not convex")
        parts.append(mesh)
    return parts


def _load_union(collider: dict[str, Any]) -> trimesh.Trimesh:
    parts = _load_parts(collider)
    return build_exact_union_mesh(parts)


def _convex_union_halfspaces(
    parts: list[trimesh.Trimesh],
) -> list[dict[str, np.ndarray]]:
    """Build deterministic analytic half-spaces for a convex-part union."""
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


def _convex_union_contains(
    volumes: list[dict[str, np.ndarray]],
    points: np.ndarray,
    *,
    tolerance_m: float = 1e-9,
) -> np.ndarray:
    """Return exact union occupancy from convex half-space inequalities."""
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


def _scene(mesh: trimesh.Trimesh) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)),
    )
    return scene


def _signed_distance(
    scene: o3d.t.geometry.RaycastingScene,
    volumes: list[dict[str, np.ndarray]],
    points: np.ndarray,
    *,
    chunk_size: int = DEFAULT_QUERY_CHUNK,
) -> np.ndarray:
    """Query robust negative-inside distance for the watertight exact union.

    Open3D supplies only the unsigned nearest-triangle magnitude. Its ray-parity
    sign is not reliable for these boolean-union meshes. Since C is a union of
    convex CoACD parts, containment is evaluated analytically from part
    half-spaces instead.
    """
    flat = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    result = np.empty(len(flat), dtype=np.float32)
    for start in range(0, len(flat), chunk_size):
        stop = min(start + chunk_size, len(flat))
        block = flat[start:stop]
        magnitude = scene.compute_distance(o3d.core.Tensor(block)).numpy()
        inside = _convex_union_contains(volumes, block)
        result[start:stop] = np.where(inside, -magnitude, magnitude)
    if not np.isfinite(result).all():
        raise RuntimeError("exact-C signed distance contains non-finite values")
    return result.reshape(np.asarray(points).shape[:-1])


def grid_geometry(
    bounds: np.ndarray,
    voxel_size_m: float,
    margin_m: float,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Return voxel-aligned padded origin and grid shape."""
    if voxel_size_m <= 0.0 or margin_m <= 0.0:
        raise ValueError("voxel size and margin must be positive")
    minimum = np.floor((bounds[0] - margin_m) / voxel_size_m) * voxel_size_m
    maximum = np.ceil((bounds[1] + margin_m) / voxel_size_m) * voxel_size_m
    shape_array = np.rint((maximum - minimum) / voxel_size_m).astype(np.int64) + 1
    shape = tuple(int(value) for value in shape_array)
    if min(shape) < 2:
        raise RuntimeError("degenerate grid shape")
    return minimum.astype(np.float64), shape


def bake_values(
    scene: o3d.t.geometry.RaycastingScene,
    volumes: list[dict[str, np.ndarray]],
    origin: np.ndarray,
    shape: tuple[int, int, int],
    voxel_size_m: float,
) -> np.ndarray:
    """Evaluate exact-C at every canonical grid vertex."""
    values = np.empty(shape, dtype=np.float32)
    y_index, z_index = np.meshgrid(
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing="ij",
    )
    yz = np.stack((y_index.reshape(-1), z_index.reshape(-1)), axis=1)
    for x_index in range(shape[0]):
        points = np.empty((len(yz), 3), dtype=np.float32)
        points[:, 0] = origin[0] + x_index * voxel_size_m
        points[:, 1] = origin[1] + yz[:, 0] * voxel_size_m
        points[:, 2] = origin[2] + yz[:, 1] * voxel_size_m
        values[x_index] = _signed_distance(scene, volumes, points).reshape(
            shape[1], shape[2]
        )
    return values


def _npy_bytes(values: np.ndarray) -> bytes:
    import io

    stream = io.BytesIO()
    np.save(stream, values, allow_pickle=False)
    return stream.getvalue()


def _percentile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile, method="higher"))


def _validation_points(
    bounds: np.ndarray,
    union: trimesh.Trimesh,
    *,
    count: int,
    seed: int,
    voxel_size_m: float,
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    uniform_count = max(count - 6 * len(union.vertices), 1)
    uniform = rng.uniform(bounds[0], bounds[1], size=(uniform_count, 3))
    vertices = np.asarray(union.vertices, dtype=np.float64)
    center = np.asarray(union.centroid, dtype=np.float64)
    directions = vertices - center
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(norms, 1e-12)
    offsets = []
    strata: dict[str, tuple[int, int]] = {"uniform": (0, len(uniform))}
    start = len(uniform)
    for multiple in (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0):
        block = vertices + directions * (multiple * voxel_size_m)
        offsets.append(block)
        stop = start + len(block)
        strata[f"vertex_radial_{multiple:+.0f}h"] = (start, stop)
        start = stop
    return np.concatenate((uniform, *offsets), axis=0), strata


def validate_grid(
    manifest_path: Path,
    union: trimesh.Trimesh,
    volumes: list[dict[str, np.ndarray]],
    scene: o3d.t.geometry.RaycastingScene,
    *,
    validation_points: int,
) -> dict[str, Any]:
    """Measure exact-C interpolation error, sign, surface, and CPU/CUDA parity."""
    backend = CanonicalGridSDF.load(manifest_path, allow_pending=True)
    voxel = backend.manifest.voxel_size_m
    origin = np.asarray(backend.manifest.origin_object_m)
    maximum = origin + voxel * (np.asarray(backend.manifest.shape) - 1)
    safe_minimum = origin + 1e-6
    safe_maximum = maximum - 1e-6
    points, strata = _validation_points(
        np.stack((safe_minimum, safe_maximum)),
        union,
        count=validation_points,
        seed=int(
            hashlib.sha256(backend.manifest.object_key.encode()).hexdigest()[:8], 16
        ),
        voxel_size_m=voxel,
    )
    exact = _signed_distance(scene, volumes, points)
    query_points = torch.from_numpy(points.astype(np.float32))
    interpolated = backend.query(query_points).numpy().astype(np.float64)
    error = np.abs(interpolated - exact.astype(np.float64))
    far = np.abs(exact) > 2.0 * voxel
    sign_disagreement = (interpolated <= 0.0) != (exact <= 0.0)
    surface_start, surface_stop = strata["vertex_radial_+0h"]
    vertex_indices = np.arange(surface_start, surface_stop, dtype=np.int64)
    surface_error = np.abs(interpolated[vertex_indices])
    object_bounds = np.asarray(union.bounds, dtype=np.float64)
    actual_padding = np.concatenate(
        (object_bounds[0] - origin, maximum - object_bounds[1])
    )
    minimum_padding = float(actual_padding.min())
    cuda = {
        "available": torch.cuda.is_available(),
        "checked": False,
        "max_abs_error_m": None,
        "status": "NOT_AVAILABLE",
    }
    if torch.cuda.is_available():
        parity_count = min(100_000, len(points))
        cuda_values = backend.query(query_points[:parity_count].cuda()).cpu().numpy()
        parity_error = np.abs(cuda_values - interpolated[:parity_count])
        cuda = {
            "available": True,
            "checked": True,
            "max_abs_error_m": float(parity_error.max(initial=0.0)),
            "status": "PASS" if parity_error.max(initial=0.0) <= 1e-5 else "FAIL",
        }
    epsilon = float(error.max(initial=0.0))
    maximum_error_index = int(np.argmax(error))
    gates = {
        "finite": bool(np.isfinite(interpolated).all()),
        "sign_disagreement_outside_2h_zero": bool(
            int((sign_disagreement & far).sum()) == 0
        ),
        "absolute_error_p99_le_h": bool(_percentile(error, 0.99) <= voxel),
        "surface_error_p99_le_voxel_diagonal": bool(
            _percentile(surface_error, 0.99) <= np.sqrt(3.0) * voxel
        ),
        "cpu_cuda_max_le_1e5": bool(cuda["status"] in {"PASS", "NOT_AVAILABLE"}),
        "minkowski_reward_support_covered": bool(
            minimum_padding >= MIN_MINKOWSKI_SUPPORT_MARGIN_M
        ),
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "validation_point_count": len(points),
        "strata": {key: stop - start for key, (start, stop) in strata.items()},
        "epsilon_grid_m": epsilon,
        "absolute_error_p50_m": _percentile(error, 0.50),
        "absolute_error_p90_m": _percentile(error, 0.90),
        "absolute_error_p99_m": _percentile(error, 0.99),
        "absolute_error_max_m": epsilon,
        "absolute_error_max_point_object_m": points[maximum_error_index].tolist(),
        "absolute_error_max_exact_m": float(exact[maximum_error_index]),
        "absolute_error_max_grid_m": float(interpolated[maximum_error_index]),
        "sign_disagreement_outside_2h_count": int((sign_disagreement & far).sum()),
        "surface_error_p99_m": _percentile(surface_error, 0.99),
        "minkowski_support_coverage": {
            "maximum_production_query_radius_m": MAX_PRODUCTION_QUERY_RADIUS_M,
            "maximum_active_reward_support_m": MAX_ACTIVE_REWARD_SUPPORT_M,
            "required_padding_m": MIN_MINKOWSKI_SUPPORT_MARGIN_M,
            "minimum_actual_padding_m": minimum_padding,
        },
        "cpu_cuda_parity": cuda,
        "gates": gates,
    }


def bake_object(
    object_key: str,
    collider: dict[str, Any],
    output_root: Path,
    *,
    voxel_size_m: float,
    margin_m: float,
    validation_points: int,
) -> dict[str, Any]:
    """Bake, validate, and freeze one object grid."""
    object_root = output_root / object_key
    manifest_path = object_root / "manifest.json"
    pending_payload: dict[str, Any] | None = None
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity_matches = (
            payload.get("object_key") == object_key
            and payload.get("source", {}).get("candidate_asset_sha256")
            == collider["candidate_asset_sha256"]
            and payload.get("source", {}).get("ordered_parts_sha256")
            == collider["ordered_parts_sha256"]
        )
        if not identity_matches:
            raise RuntimeError(f"immutable grid manifest mismatch: {manifest_path}")
        if payload.get("status") == "GRID_FROZEN":
            if payload.get("validation", {}).get("status") != "PASS":
                raise RuntimeError(f"frozen grid did not pass: {manifest_path}")
            CanonicalGridSDF.load(
                manifest_path,
                expected_candidate_asset_sha256=collider["candidate_asset_sha256"],
            )
            return {
                "object_key": object_key,
                "status": "PASS",
                "manifest": {
                    "path": _display_path(manifest_path),
                    "sha256": sha256_file(manifest_path),
                },
                "grid": payload["grid"],
                "validation": payload["validation"],
            }
        if payload.get("status") != "GRID_PENDING_VALIDATION":
            raise RuntimeError(f"non-resumable grid status: {manifest_path}")
        pending_payload = payload
    start = time.perf_counter()
    parts = _load_parts(collider)
    union = build_exact_union_mesh(parts)
    volumes = _convex_union_halfspaces(parts)
    scene = _scene(union)
    bounds = np.asarray(union.bounds, dtype=np.float64)
    origin, shape = grid_geometry(bounds, voxel_size_m, margin_m)
    if pending_payload is None:
        values = bake_values(scene, volumes, origin, shape, voxel_size_m)
    else:
        pending_grid = pending_payload["grid"]
        if (
            tuple(pending_grid["shape"]) != shape
            or float(pending_grid["voxel_size_m"]) != voxel_size_m
            or not np.allclose(
                pending_grid["origin_object_m"], origin, atol=0.0, rtol=0.0
            )
        ):
            raise RuntimeError(f"pending grid geometry changed: {manifest_path}")
        pending_path = _repo_path(pending_grid["path"])
        if sha256_file(pending_path) != pending_grid["sha256"]:
            raise RuntimeError(f"pending grid payload changed: {pending_path}")
        values = np.load(pending_path, allow_pickle=False)
    grid_path = object_root / "grid.npy"
    _write_immutable(grid_path, _npy_bytes(values))
    provisional = {
        "schema": "spider_canonical_grid_sdf_v1",
        "experiment_id": "E186",
        "stage": "S1_CANONICAL_GRID_SDF",
        "status": "GRID_PENDING_VALIDATION",
        "object_key": object_key,
        "source": {
            "collider_lock": _display_path(COLLIDER_LOCK),
            "collider_lock_sha256": EXPECTED_COLLIDER_LOCK_SHA256,
            "candidate_key": collider["candidate_key"],
            "candidate_asset_sha256": collider["candidate_asset_sha256"],
            "ordered_parts_sha256": collider["ordered_parts_sha256"],
            "ordered_part_sha256": [
                part["sha256"] for part in collider["ordered_parts"]
            ],
        },
        "sign_convention": "negative_inside",
        "signed_distance_backend": (
            "open3d_unsigned_union_surface_plus_convex_part_halfspaces_v3"
        ),
        "outside_rule": "object_aabb_distance_lower_bound",
        "object_aabb_m": {"min": bounds[0].tolist(), "max": bounds[1].tolist()},
        "grid": {
            "path": _display_path(grid_path),
            "sha256": sha256_file(grid_path),
            "dtype": "float32",
            "shape": list(shape),
            "origin_object_m": origin.tolist(),
            "voxel_size_m": voxel_size_m,
            "margin_m": margin_m,
            "size_bytes": grid_path.stat().st_size,
            "interpolation": "trilinear",
        },
        "validation": {"epsilon_grid_m": 0.0, "status": "PENDING"},
        "builder": {
            "path": _display_path(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    _atomic_bytes(manifest_path, _json_bytes(provisional))
    validation = validate_grid(
        manifest_path,
        union,
        volumes,
        scene,
        validation_points=validation_points,
    )
    provisional["validation"] = validation
    provisional["build_wall_seconds"] = time.perf_counter() - start
    provisional["status"] = (
        "GRID_FROZEN" if validation["status"] == "PASS" else "GRID_REJECTED"
    )
    _atomic_bytes(manifest_path, _json_bytes(provisional))
    if validation["status"] != "PASS":
        raise RuntimeError(f"{object_key}: canonical grid-SDF validation failed")
    return {
        "object_key": object_key,
        "status": "PASS",
        "manifest": {
            "path": _display_path(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "grid": provisional["grid"],
        "validation": validation,
    }


def bake_all(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    voxel_size_m: float | None = None,
    margin_m: float = DEFAULT_MARGIN_M,
    validation_points: int = DEFAULT_VALIDATION_POINTS,
) -> dict[str, Any]:
    """Bake all three frozen object grids and write an aggregate."""
    lock = load_collider_lock()
    objects = [
        bake_object(
            object_key,
            lock["objects"][object_key],
            output_root,
            voxel_size_m=(
                voxel_size_m
                if voxel_size_m is not None
                else DEFAULT_VOXEL_SIZE_BY_OBJECT[object_key]
            ),
            margin_m=margin_m,
            validation_points=validation_points,
        )
        for object_key in sorted(lock["objects"])
    ]
    aggregate = {
        "experiment_id": "E186",
        "stage": "S1_CANONICAL_GRID_SDF",
        "status": "PASS" if all(row["status"] == "PASS" for row in objects) else "FAIL",
        "object_count": len(objects),
        "voxel_size_by_object_m": {
            object_key: (
                voxel_size_m
                if voxel_size_m is not None
                else DEFAULT_VOXEL_SIZE_BY_OBJECT[object_key]
            )
            for object_key in sorted(lock["objects"])
        },
        "objects": objects,
    }
    aggregate_path = output_root / "aggregate.json"
    _write_immutable(aggregate_path, _json_bytes(aggregate))
    return aggregate


def parse_args() -> argparse.Namespace:
    """Parse the canonical grid builder command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--voxel-size-m",
        type=float,
        default=None,
        help="Override all objects; default uses the frozen object resolution map.",
    )
    parser.add_argument("--margin-m", type=float, default=DEFAULT_MARGIN_M)
    parser.add_argument(
        "--validation-points", type=int, default=DEFAULT_VALIDATION_POINTS
    )
    return parser.parse_args()


def main() -> int:
    """Bake the three E186 canonical grids."""
    args = parse_args()
    result = bake_all(
        args.output_root,
        voxel_size_m=args.voxel_size_m,
        margin_m=args.margin_m,
        validation_points=args.validation_points,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
