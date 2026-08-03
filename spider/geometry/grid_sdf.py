"""Object-local canonical grid-SDF loading and trilinear queries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class GridSDFManifest:
    """Validated immutable metadata for one canonical object-local SDF grid."""

    path: Path
    object_key: str
    source_candidate_asset_sha256: str
    source_ordered_parts_sha256: str
    grid_path: Path
    grid_sha256: str
    origin_object_m: tuple[float, float, float]
    voxel_size_m: float
    shape: tuple[int, int, int]
    object_aabb_min_m: tuple[float, float, float]
    object_aabb_max_m: tuple[float, float, float]
    epsilon_grid_m: float
    outside_rule: str

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_candidate_asset_sha256: str = "",
        allow_pending: bool = False,
    ) -> GridSDFManifest:
        """Load a manifest and fail closed on schema, identity, or payload mismatch."""
        manifest_path = Path(path).resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema") != "spider_canonical_grid_sdf_v1":
            raise RuntimeError(f"unsupported grid-SDF schema: {manifest_path}")
        allowed_statuses = {"GRID_FROZEN"}
        if allow_pending:
            allowed_statuses.add("GRID_PENDING_VALIDATION")
        if payload.get("status") not in allowed_statuses:
            raise RuntimeError(f"grid-SDF is not frozen: {manifest_path}")
        candidate_sha = str(payload["source"]["candidate_asset_sha256"])
        if (
            expected_candidate_asset_sha256
            and candidate_sha != expected_candidate_asset_sha256
        ):
            raise RuntimeError("grid-SDF candidate asset SHA does not match config")
        grid_entry = payload["grid"]
        grid_path = Path(grid_entry["path"])
        if not grid_path.is_absolute():
            repository = Path(__file__).resolve().parents[2]
            grid_path = repository / grid_path
        grid_path = grid_path.resolve()
        if not grid_path.is_file() or sha256_file(grid_path) != grid_entry["sha256"]:
            raise RuntimeError(f"grid-SDF payload missing or SHA mismatch: {grid_path}")
        shape = tuple(int(value) for value in grid_entry["shape"])
        if len(shape) != 3 or min(shape) < 2:
            raise RuntimeError(f"invalid grid-SDF shape: {shape}")
        voxel = float(grid_entry["voxel_size_m"])
        epsilon = float(payload["validation"]["epsilon_grid_m"])
        if voxel <= 0.0 or epsilon < 0.0:
            raise RuntimeError("invalid grid-SDF voxel/error bound")
        if grid_entry.get("dtype") != "float32":
            raise RuntimeError("canonical grid-SDF authority must be float32")
        if payload.get("sign_convention") != "negative_inside":
            raise RuntimeError("unsupported grid-SDF sign convention")
        outside_rule = str(payload.get("outside_rule"))
        if outside_rule != "object_aabb_distance_lower_bound":
            raise RuntimeError(f"unsupported grid-SDF outside rule: {outside_rule}")
        return cls(
            path=manifest_path,
            object_key=str(payload["object_key"]),
            source_candidate_asset_sha256=candidate_sha,
            source_ordered_parts_sha256=str(payload["source"]["ordered_parts_sha256"]),
            grid_path=grid_path,
            grid_sha256=str(grid_entry["sha256"]),
            origin_object_m=tuple(
                float(value) for value in grid_entry["origin_object_m"]
            ),
            voxel_size_m=voxel,
            shape=shape,
            object_aabb_min_m=tuple(
                float(value) for value in payload["object_aabb_m"]["min"]
            ),
            object_aabb_max_m=tuple(
                float(value) for value in payload["object_aabb_m"]["max"]
            ),
            epsilon_grid_m=epsilon,
            outside_rule=outside_rule,
        )


class CanonicalGridSDF:
    """Torch trilinear queries over an immutable object-local float32 SDF."""

    def __init__(self, manifest: GridSDFManifest, values: np.ndarray) -> None:
        array = np.asarray(values)
        if array.dtype != np.float32 or array.shape != manifest.shape:
            raise RuntimeError(
                f"grid payload contract mismatch: {array.dtype}/{array.shape} "
                f"!= float32/{manifest.shape}"
            )
        if not np.isfinite(array).all():
            raise RuntimeError("grid payload contains non-finite values")
        self.manifest = manifest
        self._values_cpu = np.ascontiguousarray(array)
        self._tensor_cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_candidate_asset_sha256: str = "",
        allow_pending: bool = False,
    ) -> CanonicalGridSDF:
        """Load and validate one canonical grid-SDF authority."""
        manifest = GridSDFManifest.load(
            path,
            expected_candidate_asset_sha256=expected_candidate_asset_sha256,
            allow_pending=allow_pending,
        )
        values = np.load(manifest.grid_path, allow_pickle=False)
        return cls(manifest, values)

    def _tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (str(device), dtype)
        if key not in self._tensor_cache:
            self._tensor_cache[key] = torch.from_numpy(self._values_cpu).to(
                device=device,
                dtype=dtype,
            )
        return self._tensor_cache[key]

    def in_bounds_mask(self, points_object_m: torch.Tensor) -> torch.Tensor:
        """Return whether every query point is covered by trilinear grid cells."""
        if points_object_m.ndim < 1 or points_object_m.shape[-1] != 3:
            raise ValueError("grid-SDF points must end in axis 3")
        if not points_object_m.is_floating_point():
            raise TypeError("grid-SDF points must be floating point")
        origin = torch.tensor(
            self.manifest.origin_object_m,
            device=points_object_m.device,
            dtype=points_object_m.dtype,
        )
        shape = torch.tensor(
            self.manifest.shape,
            device=points_object_m.device,
            dtype=points_object_m.dtype,
        )
        coordinates = (points_object_m - origin) / self.manifest.voxel_size_m
        return ((coordinates >= 0.0) & (coordinates <= shape - 1.0)).all(dim=-1)

    def query(self, points_object_m: torch.Tensor) -> torch.Tensor:
        """Return signed distance for arbitrary leading point axes.

        Points outside the padded grid use distance to the frozen object AABB.
        Because the object is contained by that AABB, this is a conservative
        non-negative lower bound and cannot create a collision false-safe accept.
        """
        if points_object_m.ndim < 1 or points_object_m.shape[-1] != 3:
            raise ValueError("grid-SDF points must end in axis 3")
        if not points_object_m.is_floating_point():
            raise TypeError("grid-SDF points must be floating point")
        original_shape = points_object_m.shape[:-1]
        points = points_object_m.reshape(-1, 3)
        device = points.device
        dtype = points.dtype
        origin = torch.tensor(
            self.manifest.origin_object_m,
            device=device,
            dtype=dtype,
        )
        coordinates = (points - origin) / self.manifest.voxel_size_m
        in_bounds = self.in_bounds_mask(points)
        result = torch.empty(len(points), device=device, dtype=dtype)
        if bool(in_bounds.any()):
            selected = coordinates[in_bounds]
            maximum_floor = torch.tensor(
                [value - 2 for value in self.manifest.shape],
                device=device,
                dtype=torch.long,
            )
            lower = torch.floor(selected).to(torch.long)
            lower = torch.minimum(torch.clamp(lower, min=0), maximum_floor)
            fraction = selected - lower.to(dtype)
            values = self._tensor(device, dtype).reshape(-1)
            nx, ny, nz = self.manifest.shape

            def gather(dx: int, dy: int, dz: int) -> torch.Tensor:
                index = (
                    (lower[:, 0] + dx) * ny * nz
                    + (lower[:, 1] + dy) * nz
                    + lower[:, 2]
                    + dz
                )
                return values[index]

            x, y, z = fraction.unbind(dim=1)
            c00 = gather(0, 0, 0) * (1 - x) + gather(1, 0, 0) * x
            c01 = gather(0, 0, 1) * (1 - x) + gather(1, 0, 1) * x
            c10 = gather(0, 1, 0) * (1 - x) + gather(1, 1, 0) * x
            c11 = gather(0, 1, 1) * (1 - x) + gather(1, 1, 1) * x
            c0 = c00 * (1 - y) + c10 * y
            c1 = c01 * (1 - y) + c11 * y
            result[in_bounds] = c0 * (1 - z) + c1 * z
        if bool((~in_bounds).any()):
            outside_points = points[~in_bounds]
            aabb_min = torch.tensor(
                self.manifest.object_aabb_min_m,
                device=device,
                dtype=dtype,
            )
            aabb_max = torch.tensor(
                self.manifest.object_aabb_max_m,
                device=device,
                dtype=dtype,
            )
            delta = torch.maximum(
                torch.maximum(aabb_min - outside_points, outside_points - aabb_max),
                torch.zeros((), device=device, dtype=dtype),
            )
            result[~in_bounds] = torch.linalg.vector_norm(delta, dim=1)
        return result.reshape(original_shape)

    def conservative_query(self, points_object_m: torch.Tensor) -> torch.Tensor:
        """Return the grid query reduced by the frozen interpolation error bound."""
        return self.query(points_object_m) - self.manifest.epsilon_grid_m

    def metadata(self) -> dict[str, Any]:
        """Return runtime provenance suitable for effective configs."""
        return {
            "manifest": str(self.manifest.path),
            "object_key": self.manifest.object_key,
            "candidate_asset_sha256": self.manifest.source_candidate_asset_sha256,
            "grid_sha256": self.manifest.grid_sha256,
            "voxel_size_m": self.manifest.voxel_size_m,
            "epsilon_grid_m": self.manifest.epsilon_grid_m,
            "outside_rule": self.manifest.outside_rule,
        }
