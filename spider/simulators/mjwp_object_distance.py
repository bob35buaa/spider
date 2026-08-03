"""Production object-distance queries for the MJWarp reward and gate paths.

The legacy box implementation remains in :mod:`spider.simulators.mjwp`.  This
module owns the canonical object-local grid-SDF backend used by E186.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from spider.geometry.grid_sdf import CanonicalGridSDF

MESH_SDF_SAMPLE_COUNT = 800


@dataclass(frozen=True)
class SampledRobotGeom:
    """World-space query points and the post-query radial adjustment."""

    points_world: torch.Tensor
    radius_m: float


def sample_robot_geoms(
    model: mujoco.MjModel,
    geom_ids: list[int],
    *,
    geom_xpos: torch.Tensor,
    geom_xmat: torch.Tensor,
) -> dict[int, SampledRobotGeom]:
    """Sample robot geoms using the exact production sphere/capsule/mesh rule."""
    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    capsule_type = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    sampled: dict[int, SampledRobotGeom] = {}
    for gid in geom_ids:
        if int(model.geom_type[gid]) == mesh_type:
            mesh_id = int(model.geom_dataid[gid])
            vertex_start = int(model.mesh_vertadr[mesh_id])
            vertex_count = int(model.mesh_vertnum[mesh_id])
            vertices_np = model.mesh_vert[
                vertex_start : vertex_start + vertex_count
            ].reshape(-1, 3)
            if vertex_count > MESH_SDF_SAMPLE_COUNT:
                indices = np.linspace(
                    0, vertex_count - 1, MESH_SDF_SAMPLE_COUNT
                ).astype(int)
                vertices_np = vertices_np[indices]
            vertices = torch.tensor(
                vertices_np,
                device=geom_xpos.device,
                dtype=geom_xpos.dtype,
            )
            points_world = geom_xpos[:, gid, None, :] + torch.einsum(
                "nij,sj->nsi", geom_xmat[:, gid], vertices
            )
            sampled[gid] = SampledRobotGeom(points_world, 0.0)
            continue

        radius = float(model.geom_size[gid, 0])
        half_length = (
            float(model.geom_size[gid, 1])
            if int(model.geom_type[gid]) == capsule_type
            else 0.0
        )
        offsets = torch.tensor(
            [-half_length, 0.0, half_length],
            device=geom_xpos.device,
            dtype=geom_xpos.dtype,
        )
        axis = geom_xmat[:, gid, :, 2]
        points_world = geom_xpos[:, gid, None, :] + axis[:, None, :] * offsets.view(
            1, 3, 1
        )
        sampled[gid] = SampledRobotGeom(points_world, radius)
    return sampled


@dataclass(frozen=True)
class GridObjectDistanceRuntime:
    """Immutable grid authority plus the object body used for local queries."""

    grid: CanonicalGridSDF
    object_body_id: int

    @classmethod
    def load(
        cls,
        manifest_path: str,
        *,
        expected_candidate_asset_sha256: str,
        expected_error_bound_m: float,
        object_body_id: int,
    ) -> GridObjectDistanceRuntime:
        """Load and validate the frozen manifest against the runtime config."""
        if not manifest_path:
            raise ValueError("grid_sdf backend requires object_distance_manifest")
        if not expected_candidate_asset_sha256:
            raise ValueError(
                "grid_sdf backend requires object_distance_expected_asset_sha256"
            )
        if expected_error_bound_m <= 0.0:
            raise ValueError(
                "grid_sdf backend requires a positive object_distance_error_bound_m"
            )
        if object_body_id < 0:
            raise ValueError("grid_sdf backend requires a named object body")
        grid = CanonicalGridSDF.load(
            manifest_path,
            expected_candidate_asset_sha256=expected_candidate_asset_sha256,
        )
        actual = float(grid.manifest.epsilon_grid_m)
        if abs(actual - float(expected_error_bound_m)) > 1e-12:
            raise RuntimeError(
                "grid-SDF error bound does not match config: "
                f"manifest={actual:.17g}, config={expected_error_bound_m:.17g}"
            )
        return cls(grid=grid, object_body_id=object_body_id)

    @property
    def epsilon_grid_m(self) -> float:
        """Return the frozen conservative interpolation-error budget."""
        return float(self.grid.manifest.epsilon_grid_m)

    def metadata(self) -> dict[str, object]:
        """Return immutable runtime provenance for effective-config logs."""
        metadata = self.grid.metadata()
        metadata["object_body_id"] = self.object_body_id
        return metadata

    def world_to_object(
        self,
        points_world: torch.Tensor,
        body_xpos: torch.Tensor,
        body_xmat: torch.Tensor,
    ) -> torch.Tensor:
        """Transform arbitrary leading point axes into the object body frame."""
        object_position = body_xpos[:, self.object_body_id]
        object_rotation = body_xmat[:, self.object_body_id]
        delta = points_world - object_position.view(
            object_position.shape[0], *([1] * (points_world.ndim - 2)), 3
        )
        return torch.einsum("nji,n...j->n...i", object_rotation, delta)

    def per_geom_sdf(
        self,
        model: mujoco.MjModel,
        geom_ids: list[int],
        *,
        geom_xpos: torch.Tensor,
        geom_xmat: torch.Tensor,
        body_xpos: torch.Tensor,
        body_xmat: torch.Tensor,
    ) -> torch.Tensor:
        """Return nominal ``D_C`` independently for every requested robot geom."""
        if not geom_ids:
            return torch.empty(
                (geom_xpos.shape[0], 0),
                device=geom_xpos.device,
                dtype=geom_xpos.dtype,
            )

        sampled = sample_robot_geoms(
            model,
            geom_ids,
            geom_xpos=geom_xpos,
            geom_xmat=geom_xmat,
        )
        columns = []
        for gid in geom_ids:
            sample = sampled[gid]
            points_object = self.world_to_object(
                sample.points_world, body_xpos, body_xmat
            )
            columns.append(
                self.grid.query(points_object).min(dim=1).values - sample.radius_m
            )
        return torch.stack(columns, dim=1)

    def group_cache(
        self,
        model: mujoco.MjModel,
        geom_groups: list[list[int]],
        *,
        geom_xpos: torch.Tensor,
        geom_xmat: torch.Tensor,
        body_xpos: torch.Tensor,
        body_xmat: torch.Tensor,
    ) -> dict[tuple[int, ...], torch.Tensor]:
        """Evaluate unique groups from one nominal per-geom grid query."""
        group_keys = list(
            dict.fromkeys(
                tuple(int(gid) for gid in group) for group in geom_groups if group
            )
        )
        if not group_keys:
            return {}
        ordered_geom_ids = list(dict.fromkeys(gid for key in group_keys for gid in key))
        per_geom = self.per_geom_sdf(
            model,
            ordered_geom_ids,
            geom_xpos=geom_xpos,
            geom_xmat=geom_xmat,
            body_xpos=body_xpos,
            body_xmat=body_xmat,
        )
        column_by_geom = {gid: index for index, gid in enumerate(ordered_geom_ids)}
        return {
            key: per_geom[:, [column_by_geom[gid] for gid in key]].min(dim=1).values
            for key in group_keys
        }
