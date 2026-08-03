#!/usr/bin/env python3
"""Deterministic production tests for the E186 MJWarp grid-SDF backend."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from spider.config import Config, resolve_object_collision_geom_ids
from spider.geometry.grid_sdf import CanonicalGridSDF, GridSDFManifest
from spider.simulators import mjwp
from spider.simulators.mjwp_object_distance import GridObjectDistanceRuntime

MODEL_XML = """
<mujoco>
  <asset>
    <mesh name="probe_tetra"
          vertex="0.20 0 0  0.30 0 0  0.25 0.10 0  0.25 0 0.10"
          face="0 1 2  0 1 3  0 2 3  1 2 3"/>
    <mesh name="object_tetra"
          vertex="0 0 0  0.5 0 0  0 0.5 0  0 0 0.5"
          face="0 1 2  0 1 3  0 2 3  1 2 3"/>
  </asset>
  <worldbody>
    <body name="robot">
      <geom name="probe_sphere" type="sphere" size="0.10"/>
      <geom name="probe_capsule" type="capsule" size="0.05 0.20"/>
      <geom name="probe_mesh" type="mesh" mesh="probe_tetra"/>
    </body>
    <body name="object">
      <geom name="object_collision" type="mesh" mesh="object_tetra"/>
      <geom name="object_collision_001" type="mesh" mesh="object_tetra"/>
    </body>
  </worldbody>
</mujoco>
"""


def geom_id(model: mujoco.MjModel, name: str) -> int:
    """Resolve a required geometry name."""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def linear_x_grid() -> CanonicalGridSDF:
    """Build a grid whose exact trilinear field is the local x coordinate."""
    origin = (-1.0, -1.0, -1.0)
    voxel = 0.25
    shape = (9, 9, 9)
    x = origin[0] + np.arange(shape[0], dtype=np.float32) * voxel
    values = np.broadcast_to(x[:, None, None], shape).copy()
    manifest = GridSDFManifest(
        path=Path("/unused/manifest.json"),
        object_key="test",
        source_candidate_asset_sha256="a" * 64,
        source_ordered_parts_sha256="b" * 64,
        grid_path=Path("/unused/grid.npy"),
        grid_sha256="c" * 64,
        origin_object_m=origin,
        voxel_size_m=voxel,
        shape=shape,
        object_aabb_min_m=(-1.0, -1.0, -1.0),
        object_aabb_max_m=(1.0, 1.0, 1.0),
        epsilon_grid_m=0.02,
        outside_rule="object_aabb_distance_lower_bound",
    )
    return CanonicalGridSDF(manifest, values)


def main() -> int:
    """Run deterministic CPU production-backend checks."""
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    sphere = geom_id(model, "probe_sphere")
    capsule = geom_id(model, "probe_capsule")
    probe_mesh = geom_id(model, "probe_mesh")
    object_ids = resolve_object_collision_geom_ids(model, "compound")
    assert len(object_ids) == 2
    try:
        resolve_object_collision_geom_ids(model, "union")
    except ValueError as exc:
        assert "requires box geoms" in str(exc)
    else:
        raise AssertionError("legacy union must still reject non-box geoms")

    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    runtime = GridObjectDistanceRuntime(
        grid=linear_x_grid(), object_body_id=object_body_id
    )
    config = Config()
    config.device = "cpu"
    config.object_distance_backend = "grid_sdf"
    config._object_distance_grid_runtime = runtime
    env = SimpleNamespace(model_cpu=model)

    worlds = 2
    geom_xpos = torch.zeros((worlds, model.ngeom, 3), dtype=torch.float64)
    geom_xmat = torch.eye(3, dtype=torch.float64).repeat(worlds, model.ngeom, 1, 1)
    body_xpos = torch.zeros((worlds, model.nbody, 3), dtype=torch.float64)
    body_xmat = torch.eye(3, dtype=torch.float64).repeat(worlds, model.nbody, 1, 1)
    object_positions = [
        torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([1.2, -0.7, 0.4], dtype=torch.float64),
    ]
    object_rotations = [
        torch.eye(3, dtype=torch.float64),
        torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        ),
    ]
    for world, (position, rotation) in enumerate(
        zip(object_positions, object_rotations, strict=True)
    ):
        body_xpos[world, object_body_id] = position
        body_xmat[world, object_body_id] = rotation
        geom_xpos[world, sphere] = position + rotation @ torch.tensor(
            [0.4, 0.0, 0.0], dtype=torch.float64
        )
        geom_xpos[world, capsule] = position + rotation @ torch.tensor(
            [0.5, 0.0, 0.0], dtype=torch.float64
        )
        geom_xmat[world, capsule, :, 2] = rotation @ torch.tensor(
            [1.0, 0.0, 0.0], dtype=torch.float64
        )
        geom_xpos[world, probe_mesh] = position
        geom_xmat[world, probe_mesh] = rotation

    geom_ids = [sphere, capsule, probe_mesh]
    per_geom = runtime.per_geom_sdf(
        model,
        geom_ids,
        geom_xpos=geom_xpos,
        geom_xmat=geom_xmat,
        body_xpos=body_xpos,
        body_xmat=body_xmat,
    )
    mesh_id = int(model.geom_dataid[probe_mesh])
    vertex_start = int(model.mesh_vertadr[mesh_id])
    vertex_count = int(model.mesh_vertnum[mesh_id])
    expected_mesh = float(
        model.mesh_vert[vertex_start : vertex_start + vertex_count, 0].min()
    )
    expected = torch.tensor([0.30, 0.25, expected_mesh], dtype=torch.float64).repeat(
        worlds, 1
    )
    torch.testing.assert_close(per_geom, expected, atol=1e-7, rtol=0.0)

    groups = [[sphere], [capsule], [sphere, capsule], [probe_mesh]]
    cache = runtime.group_cache(
        model,
        groups,
        geom_xpos=geom_xpos,
        geom_xmat=geom_xmat,
        body_xpos=body_xpos,
        body_xmat=body_xmat,
    )
    torch.testing.assert_close(
        cache[(sphere, capsule)],
        torch.full((worlds,), 0.25, dtype=torch.float64),
        atol=1e-7,
        rtol=0.0,
    )
    nominal = mjwp._cached_object_distance_sdf_min(
        cache,
        config,
        env,
        [sphere, capsule],
        object_ids,
        geom_xpos=geom_xpos,
        geom_xmat=geom_xmat,
        body_xpos=body_xpos,
        body_xmat=body_xmat,
    )
    conservative = mjwp._cached_object_distance_sdf_min(
        cache,
        config,
        env,
        [sphere, capsule],
        object_ids,
        geom_xpos=geom_xpos,
        geom_xmat=geom_xmat,
        body_xpos=body_xpos,
        body_xmat=body_xmat,
        conservative=True,
    )
    torch.testing.assert_close(
        conservative,
        nominal - 0.02,
        atol=0.0,
        rtol=0.0,
    )
    print("E186 production grid-SDF backend tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
