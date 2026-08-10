# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Simulator for sampling with MuJoCo Warp (mjwarp).

This module provides a minimal MJWP backend that matches the sampling API used by
the generic optimizer pipeline. It intentionally keeps the implementation simple
and robust (no DR groups here; see legacy script for advanced features).
"""

from __future__ import annotations

from dataclasses import dataclass

import loguru
import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

# NOTE: this is a hacky solution to make sure domain randomization works for contact margin. Otherwise, it will create a surrogate memory for all worlds and we cannot override each individual world's contact parameters.
# mjwarp._src.io.MAX_WORLDS = 1024
from spider.config import (
    Config,
)
from spider.config import (
    resolve_object_collision_geom_ids as _resolve_object_collision_geom_ids,
)
from spider.math import quat_sub
from spider.rewards.surface_distance import (
    surface_distance_score,
    surface_distance_support_mask,
)
from spider.simulators.mjwp_object_distance import GridObjectDistanceRuntime

MESH_SDF_SAMPLE_COUNT = 800

# Initialize Warp once per process
try:
    wp.init()
except RuntimeError:
    # Already initialized
    pass


@dataclass
class MJWPEnv:
    model_cpu: mujoco.MjModel
    data_cpu: mujoco.MjData
    # Unified data sink always reflecting last step's state
    model_wp: mjwarp.Model
    data_wp: mjwarp.Data
    data_wp_prev: mjwarp.Data
    graph: wp.ScopedCapture.Graph
    # Device alias used for Warp allocations/launches (e.g., "cuda:1" or "cpu")
    device: str
    num_worlds: int


def _compile_step(
    model_wp: mjwarp.Model, data_wp: mjwarp.Data, decimation: int = 1
) -> wp.ScopedCapture.Graph:
    """Warm up and capture a CUDA graph that runs `decimation` × mjwarp.step."""

    def _step_once():
        for _ in range(decimation):
            mjwarp.step(model_wp, data_wp)

    # Capture
    with wp.ScopedCapture() as capture:
        _step_once()
    wp.synchronize()
    return capture.graph


def _geom_box_sdf_min(
    config: Config,
    env: MJWPEnv,
    geom_ids: list[int],
    object_geom_id: int,
    geom_xpos: torch.Tensor | None = None,
    geom_xmat: torch.Tensor | None = None,
    half_ext: torch.Tensor | None = None,
) -> torch.Tensor:
    """Minimum adjusted SDF from selected robot geoms to object_collision box."""
    if geom_xpos is None:
        geom_xpos = wp.to_torch(env.data_wp.geom_xpos)
    if geom_xmat is None:
        geom_xmat = wp.to_torch(env.data_wp.geom_xmat).reshape(
            geom_xpos.shape[0], geom_xpos.shape[1], 3, 3
        )
    if half_ext is None:
        half_ext = torch.tensor(
            config.hand_approach_obj_half_extents,
            device=config.device,
            dtype=geom_xpos.dtype,
        )
    if not geom_ids:
        return torch.full(
            (geom_xpos.shape[0],),
            float("inf"),
            device=config.device,
            dtype=geom_xpos.dtype,
        )

    obj_pos = geom_xpos[:, object_geom_id]
    obj_mat = geom_xmat[:, object_geom_id]
    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    mesh_ids = [
        gid for gid in geom_ids if int(env.model_cpu.geom_type[gid]) == mesh_type
    ]
    primitive_ids = [gid for gid in geom_ids if gid not in mesh_ids]
    candidates: list[torch.Tensor] = []

    if primitive_ids:
        centers = geom_xpos[:, primitive_ids]
        mats = geom_xmat[:, primitive_ids]
        axes = mats[:, :, :, 2]
        radii = torch.tensor(
            [float(env.model_cpu.geom_size[gid, 0]) for gid in primitive_ids],
            device=config.device,
            dtype=geom_xpos.dtype,
        )
        half_lens = torch.tensor(
            [
                (
                    float(env.model_cpu.geom_size[gid, 1])
                    if int(env.model_cpu.geom_type[gid])
                    == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
                    else 0.0
                )
                for gid in primitive_ids
            ],
            device=config.device,
            dtype=geom_xpos.dtype,
        )
        samples = torch.stack(
            (-half_lens, torch.zeros_like(half_lens), half_lens), dim=1
        )
        points = centers.unsqueeze(2) + axes.unsqueeze(2) * samples.view(1, -1, 3, 1)
        delta = points - obj_pos[:, None, None, :]
        local = torch.einsum("nji,nkpj->nkpi", obj_mat, delta)
        q = torch.abs(local) - half_ext.view(1, 1, 1, 3)
        outside = torch.clamp(q, min=0.0).norm(dim=-1)
        inside = torch.clamp(q.max(dim=-1).values, max=0.0)
        sdf_points = outside + inside
        sdf_geom = sdf_points.min(dim=2).values - radii.view(1, -1)
        candidates.append(sdf_geom.min(dim=1).values)

    for gid in mesh_ids:
        mesh_id = int(env.model_cpu.geom_dataid[gid])
        v0 = int(env.model_cpu.mesh_vertadr[mesh_id])
        nv = int(env.model_cpu.mesh_vertnum[mesh_id])
        verts_np = env.model_cpu.mesh_vert[v0 : v0 + nv].reshape(-1, 3)
        if nv > MESH_SDF_SAMPLE_COUNT:
            idx = np.linspace(0, nv - 1, MESH_SDF_SAMPLE_COUNT).astype(int)
            verts_np = verts_np[idx]
        verts = torch.tensor(verts_np, device=config.device, dtype=geom_xpos.dtype)
        center = geom_xpos[:, gid]
        mat = geom_xmat[:, gid]
        points = center[:, None, :] + torch.einsum("nij,sj->nsi", mat, verts)
        delta = points - obj_pos[:, None, :]
        local = torch.einsum("nji,nsj->nsi", obj_mat, delta)
        q = torch.abs(local) - half_ext.view(1, 1, 3)
        outside = torch.clamp(q, min=0.0).norm(dim=-1)
        inside = torch.clamp(q.max(dim=-1).values, max=0.0)
        candidates.append((outside + inside).min(dim=1).values)

    return torch.stack(candidates, dim=1).min(dim=1).values


def _uses_single_box_sdf(env: MJWPEnv, object_geom_ids: list[int]) -> bool:
    """Whether this object has the historical single-box SDF contract."""
    return (
        len(object_geom_ids) == 1
        and int(env.model_cpu.geom_type[object_geom_ids[0]])
        == int(mujoco.mjtGeom.mjGEOM_BOX)
    )


def _geom_box_union_sdf_min(
    config: Config,
    env: MJWPEnv,
    geom_ids: list[int],
    object_geom_ids: list[int],
    geom_xpos: torch.Tensor | None = None,
    geom_xmat: torch.Tensor | None = None,
) -> torch.Tensor:
    """Minimum adjusted SDF from robot geoms to a union of object boxes.

    Object boxes are processed in chunks so surface-voxel proxies do not issue
    one Python/Torch kernel sequence per box.  The result remains the exact
    minimum of the legacy per-box SDF values.
    """
    if geom_xpos is None:
        geom_xpos = wp.to_torch(env.data_wp.geom_xpos)
    if geom_xmat is None:
        geom_xmat = wp.to_torch(env.data_wp.geom_xmat).reshape(
            geom_xpos.shape[0], geom_xpos.shape[1], 3, 3
        )
    if not geom_ids or not object_geom_ids:
        return torch.full(
            (geom_xpos.shape[0],),
            float("inf"),
            device=config.device,
            dtype=geom_xpos.dtype,
        )

    box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
    non_box = [
        gid for gid in object_geom_ids if int(env.model_cpu.geom_type[gid]) != box_type
    ]
    if non_box:
        names = [
            mujoco.mj_id2name(env.model_cpu, mujoco.mjtObj.mjOBJ_GEOM, gid) or str(gid)
            for gid in non_box
        ]
        raise ValueError(
            f"Object SDF union currently supports box geoms only; got {','.join(names)}"
        )

    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    mesh_ids = [
        gid for gid in geom_ids if int(env.model_cpu.geom_type[gid]) == mesh_type
    ]
    primitive_ids = [gid for gid in geom_ids if gid not in mesh_ids]
    candidates: list[torch.Tensor] = []
    object_chunk_size = 16

    primitive_points: torch.Tensor | None = None
    primitive_radii: torch.Tensor | None = None
    if primitive_ids:
        centers = geom_xpos[:, primitive_ids]
        mats = geom_xmat[:, primitive_ids]
        axes = mats[:, :, :, 2]
        primitive_radii = torch.tensor(
            [float(env.model_cpu.geom_size[gid, 0]) for gid in primitive_ids],
            device=config.device,
            dtype=geom_xpos.dtype,
        )
        half_lens = torch.tensor(
            [
                (
                    float(env.model_cpu.geom_size[gid, 1])
                    if int(env.model_cpu.geom_type[gid])
                    == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
                    else 0.0
                )
                for gid in primitive_ids
            ],
            device=config.device,
            dtype=geom_xpos.dtype,
        )
        samples = torch.stack(
            (-half_lens, torch.zeros_like(half_lens), half_lens), dim=1
        )
        primitive_points = centers.unsqueeze(2) + axes.unsqueeze(2) * samples.view(
            1, -1, 3, 1
        )

    mesh_points: list[torch.Tensor] = []
    for gid in mesh_ids:
        mesh_id = int(env.model_cpu.geom_dataid[gid])
        v0 = int(env.model_cpu.mesh_vertadr[mesh_id])
        nv = int(env.model_cpu.mesh_vertnum[mesh_id])
        verts_np = env.model_cpu.mesh_vert[v0 : v0 + nv].reshape(-1, 3)
        if nv > MESH_SDF_SAMPLE_COUNT:
            idx = np.linspace(0, nv - 1, MESH_SDF_SAMPLE_COUNT).astype(int)
            verts_np = verts_np[idx]
        verts = torch.tensor(verts_np, device=config.device, dtype=geom_xpos.dtype)
        center = geom_xpos[:, gid]
        mat = geom_xmat[:, gid]
        mesh_points.append(center[:, None, :] + torch.einsum("nij,sj->nsi", mat, verts))

    for start in range(0, len(object_geom_ids), object_chunk_size):
        chunk_ids = object_geom_ids[start : start + object_chunk_size]
        obj_pos = geom_xpos[:, chunk_ids]
        obj_mat = geom_xmat[:, chunk_ids]
        half_ext = torch.tensor(
            env.model_cpu.geom_size[chunk_ids, :3],
            device=config.device,
            dtype=geom_xpos.dtype,
        )

        if primitive_points is not None and primitive_radii is not None:
            delta = primitive_points[:, None, :, :, :] - obj_pos[:, :, None, None, :]
            local = torch.einsum("nbji,nbrpj->nbrpi", obj_mat, delta)
            q = torch.abs(local) - half_ext.view(1, len(chunk_ids), 1, 1, 3)
            outside = torch.clamp(q, min=0.0).norm(dim=-1)
            inside = torch.clamp(q.max(dim=-1).values, max=0.0)
            sdf_geom = (outside + inside).min(dim=3).values
            sdf_geom = sdf_geom - primitive_radii.view(1, 1, -1)
            candidates.append(sdf_geom.amin(dim=(1, 2)))

        for points in mesh_points:
            delta = points[:, None, :, :] - obj_pos[:, :, None, :]
            local = torch.einsum("nbji,nbsj->nbsi", obj_mat, delta)
            q = torch.abs(local) - half_ext.view(1, len(chunk_ids), 1, 3)
            outside = torch.clamp(q, min=0.0).norm(dim=-1)
            inside = torch.clamp(q.max(dim=-1).values, max=0.0)
            candidates.append((outside + inside).amin(dim=(1, 2)))

    return torch.stack(candidates, dim=1).min(dim=1).values


def _cached_geom_box_union_sdf_min(
    cache: dict[tuple[int, ...], torch.Tensor],
    config: Config,
    env: MJWPEnv,
    geom_ids: list[int],
    object_geom_ids: list[int],
    geom_xpos: torch.Tensor | None = None,
    geom_xmat: torch.Tensor | None = None,
) -> torch.Tensor:
    """Memoize an exact union-SDF query within one reward tick only."""
    key = tuple(int(gid) for gid in geom_ids)
    if key not in cache:
        cache[key] = _geom_box_union_sdf_min(
            config,
            env,
            geom_ids,
            object_geom_ids,
            geom_xpos=geom_xpos,
            geom_xmat=geom_xmat,
        )
    return cache[key]


def _geom_box_union_sdf_per_geom(
    config: Config,
    env: MJWPEnv,
    geom_ids: list[int],
    object_geom_ids: list[int],
    geom_xpos: torch.Tensor | None = None,
    geom_xmat: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return exact object-union SDF independently for every robot geom.

    The output has shape ``(num_worlds, len(geom_ids))``. Primitive robot geoms
    share one batched object-box calculation; mesh robot geoms retain the same
    vertex sampling and reductions as :func:`_geom_box_union_sdf_min`.
    """
    if geom_xpos is None:
        geom_xpos = wp.to_torch(env.data_wp.geom_xpos)
    if geom_xmat is None:
        geom_xmat = wp.to_torch(env.data_wp.geom_xmat).reshape(
            geom_xpos.shape[0], geom_xpos.shape[1], 3, 3
        )
    if not geom_ids or not object_geom_ids:
        return torch.full(
            (geom_xpos.shape[0], len(geom_ids)),
            float("inf"),
            device=config.device,
            dtype=geom_xpos.dtype,
        )

    box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
    non_box = [
        gid for gid in object_geom_ids if int(env.model_cpu.geom_type[gid]) != box_type
    ]
    if non_box:
        names = [
            mujoco.mj_id2name(env.model_cpu, mujoco.mjtObj.mjOBJ_GEOM, gid) or str(gid)
            for gid in non_box
        ]
        raise ValueError(
            f"Object SDF union currently supports box geoms only; got {','.join(names)}"
        )

    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    mesh_ids = [
        gid for gid in geom_ids if int(env.model_cpu.geom_type[gid]) == mesh_type
    ]
    primitive_ids = [gid for gid in geom_ids if gid not in mesh_ids]
    object_chunk_size = 16

    primitive_points: torch.Tensor | None = None
    primitive_radii: torch.Tensor | None = None
    if primitive_ids:
        centers = geom_xpos[:, primitive_ids]
        mats = geom_xmat[:, primitive_ids]
        axes = mats[:, :, :, 2]
        primitive_radii = torch.tensor(
            [float(env.model_cpu.geom_size[gid, 0]) for gid in primitive_ids],
            device=config.device,
            dtype=geom_xpos.dtype,
        )
        half_lens = torch.tensor(
            [
                (
                    float(env.model_cpu.geom_size[gid, 1])
                    if int(env.model_cpu.geom_type[gid])
                    == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
                    else 0.0
                )
                for gid in primitive_ids
            ],
            device=config.device,
            dtype=geom_xpos.dtype,
        )
        samples = torch.stack(
            (-half_lens, torch.zeros_like(half_lens), half_lens), dim=1
        )
        primitive_points = centers.unsqueeze(2) + axes.unsqueeze(2) * samples.view(
            1, -1, 3, 1
        )

    mesh_points: dict[int, torch.Tensor] = {}
    for gid in mesh_ids:
        mesh_id = int(env.model_cpu.geom_dataid[gid])
        v0 = int(env.model_cpu.mesh_vertadr[mesh_id])
        nv = int(env.model_cpu.mesh_vertnum[mesh_id])
        verts_np = env.model_cpu.mesh_vert[v0 : v0 + nv].reshape(-1, 3)
        if nv > MESH_SDF_SAMPLE_COUNT:
            idx = np.linspace(0, nv - 1, MESH_SDF_SAMPLE_COUNT).astype(int)
            verts_np = verts_np[idx]
        verts = torch.tensor(verts_np, device=config.device, dtype=geom_xpos.dtype)
        center = geom_xpos[:, gid]
        mat = geom_xmat[:, gid]
        mesh_points[gid] = center[:, None, :] + torch.einsum("nij,sj->nsi", mat, verts)

    primitive_chunks: list[torch.Tensor] = []
    mesh_chunks: dict[int, list[torch.Tensor]] = {gid: [] for gid in mesh_ids}
    for start in range(0, len(object_geom_ids), object_chunk_size):
        chunk_ids = object_geom_ids[start : start + object_chunk_size]
        obj_pos = geom_xpos[:, chunk_ids]
        obj_mat = geom_xmat[:, chunk_ids]
        half_ext = torch.tensor(
            env.model_cpu.geom_size[chunk_ids, :3],
            device=config.device,
            dtype=geom_xpos.dtype,
        )

        if primitive_points is not None and primitive_radii is not None:
            delta = primitive_points[:, None, :, :, :] - obj_pos[:, :, None, None, :]
            local = torch.einsum("nbji,nbrpj->nbrpi", obj_mat, delta)
            q = torch.abs(local) - half_ext.view(1, len(chunk_ids), 1, 1, 3)
            outside = torch.clamp(q, min=0.0).norm(dim=-1)
            inside = torch.clamp(q.max(dim=-1).values, max=0.0)
            sdf_geom = (outside + inside).min(dim=3).values
            sdf_geom = sdf_geom - primitive_radii.view(1, 1, -1)
            primitive_chunks.append(sdf_geom.amin(dim=1))

        for gid, points in mesh_points.items():
            delta = points[:, None, :, :] - obj_pos[:, :, None, :]
            local = torch.einsum("nbji,nbsj->nbsi", obj_mat, delta)
            q = torch.abs(local) - half_ext.view(1, len(chunk_ids), 1, 3)
            outside = torch.clamp(q, min=0.0).norm(dim=-1)
            inside = torch.clamp(q.max(dim=-1).values, max=0.0)
            mesh_chunks[gid].append((outside + inside).amin(dim=(1, 2)))

    if len(primitive_chunks) == 1:
        primitive_sdf = primitive_chunks[0]
    elif primitive_chunks:
        primitive_sdf = torch.stack(primitive_chunks, dim=2).min(dim=2).values
    else:
        primitive_sdf = None
    primitive_columns = {gid: index for index, gid in enumerate(primitive_ids)}
    mesh_sdf = {
        gid: (
            values[0]
            if len(values) == 1
            else torch.stack(values, dim=1).min(dim=1).values
        )
        for gid, values in mesh_chunks.items()
    }
    columns = [
        (mesh_sdf[gid] if gid in mesh_sdf else primitive_sdf[:, primitive_columns[gid]])
        for gid in geom_ids
    ]
    return torch.stack(columns, dim=1)


def _batched_geom_box_union_sdf_cache(
    config: Config,
    env: MJWPEnv,
    geom_groups: list[list[int]],
    object_geom_ids: list[int],
    geom_xpos: torch.Tensor | None = None,
    geom_xmat: torch.Tensor | None = None,
) -> dict[tuple[int, ...], torch.Tensor]:
    """Evaluate all requested robot-geom groups from one per-geom SDF batch."""
    group_keys = list(
        dict.fromkeys(
            tuple(int(gid) for gid in group) for group in geom_groups if group
        )
    )
    if not group_keys:
        return {}
    ordered_geom_ids = list(dict.fromkeys(gid for key in group_keys for gid in key))
    per_geom_sdf = _geom_box_union_sdf_per_geom(
        config,
        env,
        ordered_geom_ids,
        object_geom_ids,
        geom_xpos=geom_xpos,
        geom_xmat=geom_xmat,
    )
    column_by_geom_id = {gid: index for index, gid in enumerate(ordered_geom_ids)}
    return {
        key: per_geom_sdf[
            :,
            [column_by_geom_id[gid] for gid in key],
        ]
        .min(dim=1)
        .values
        for key in group_keys
    }


def _require_grid_object_distance_runtime(
    config: Config,
) -> GridObjectDistanceRuntime:
    runtime = getattr(config, "_object_distance_grid_runtime", None)
    if not isinstance(runtime, GridObjectDistanceRuntime):
        raise RuntimeError(
            "grid_sdf object-distance backend was not resolved before runtime"
        )
    return runtime


def _object_distance_group_cache(
    config: Config,
    env: MJWPEnv,
    geom_groups: list[list[int]],
    object_geom_ids: list[int],
    *,
    geom_xpos: torch.Tensor,
    geom_xmat: torch.Tensor,
    body_xpos: torch.Tensor | None = None,
    body_xmat: torch.Tensor | None = None,
) -> dict[tuple[int, ...], torch.Tensor]:
    """Build one-tick nominal distance cache for the configured backend."""
    if config.object_distance_backend == "legacy_box":
        return _batched_geom_box_union_sdf_cache(
            config,
            env,
            geom_groups,
            object_geom_ids,
            geom_xpos=geom_xpos,
            geom_xmat=geom_xmat,
        )
    if config.object_distance_backend != "grid_sdf":
        raise ValueError(
            f"Unsupported object_distance_backend={config.object_distance_backend!r}"
        )
    runtime = _require_grid_object_distance_runtime(config)
    if body_xpos is None:
        body_xpos = wp.to_torch(env.data_wp.xpos)
    if body_xmat is None:
        body_xmat = wp.to_torch(env.data_wp.xmat).reshape(body_xpos.shape[0], -1, 3, 3)
    return runtime.group_cache(
        env.model_cpu,
        geom_groups,
        geom_xpos=geom_xpos,
        geom_xmat=geom_xmat,
        body_xpos=body_xpos,
        body_xmat=body_xmat,
    )


def _cached_object_distance_sdf_min(
    cache: dict[tuple[int, ...], torch.Tensor],
    config: Config,
    env: MJWPEnv,
    geom_ids: list[int],
    object_geom_ids: list[int],
    *,
    geom_xpos: torch.Tensor,
    geom_xmat: torch.Tensor,
    body_xpos: torch.Tensor | None = None,
    body_xmat: torch.Tensor | None = None,
    conservative: bool = False,
) -> torch.Tensor:
    """Query nominal reward distance or conservative hard-gate distance."""
    if config.object_distance_backend == "legacy_box":
        return _cached_geom_box_union_sdf_min(
            cache,
            config,
            env,
            geom_ids,
            object_geom_ids,
            geom_xpos=geom_xpos,
            geom_xmat=geom_xmat,
        )
    key = tuple(int(gid) for gid in geom_ids)
    if key not in cache:
        cache.update(
            _object_distance_group_cache(
                config,
                env,
                [geom_ids],
                object_geom_ids,
                geom_xpos=geom_xpos,
                geom_xmat=geom_xmat,
                body_xpos=body_xpos,
                body_xmat=body_xmat,
            )
        )
    value = cache[key]
    if conservative:
        value = value - _require_grid_object_distance_runtime(config).epsilon_grid_m
    return value


def _object_sdf_geom_groups_for_tick(config: Config) -> list[list[int]]:
    """Collect every robot-geom group that this reward tick can query."""
    groups: list[list[int]] = []

    def add(enabled: bool, geom_ids: list[int]) -> None:
        if enabled and geom_ids:
            groups.append(list(geom_ids))

    add(
        config.robot_object_penalty_scale > 0.0,
        config.robot_object_penalty_geom_ids,
    )
    add(
        config.leg_object_penalty_scale > 0.0,
        config.leg_object_penalty_geom_ids,
    )
    add(
        config.hand_object_deep_penalty_scale > 0.0,
        config.hand_object_deep_penalty_geom_ids,
    )
    add(config.hand_support_rew_scale > 0.0, config.hand_support_geom_ids)

    surface_band_enabled = (
        config.surface_band_rew_scale > 0.0 or config.surface_band_penalty_scale > 0.0
    ) and bool(config.surface_band_geom_ids)
    if surface_band_enabled and config.surface_band_bimanual_required:
        add(True, config.surface_band_left_geom_ids)
        add(True, config.surface_band_right_geom_ids)
    else:
        add(surface_band_enabled, config.surface_band_geom_ids)

    add(
        config.nonhand_support_penalty_scale > 0.0,
        config.nonhand_support_penalty_geom_ids,
    )
    add(config.cem_safety_gate_enabled, config.cem_safety_gate_geom_ids)
    add(config.cem_hand_gate_enabled, config.cem_hand_gate_geom_ids)
    add(config.cem_leg_gate_enabled, config.cem_leg_gate_geom_ids)
    add(
        config.carry_corridor_rew_scale > 0.0,
        config.carry_corridor_leg_geom_ids,
    )
    return groups


def _sample_gate_from_ref_mask(
    mask,
    num_samples: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert scalar/per-EEF/per-sample contact mask to one gate per sample."""
    if not torch.is_tensor(mask):
        mask = torch.tensor(mask, device=device, dtype=dtype)
    else:
        mask = mask.to(device=device, dtype=dtype)
    if mask.ndim == 0:
        return mask.view(1).expand(num_samples)
    if mask.ndim == 1:
        if mask.shape[0] == num_samples:
            return mask
        return mask.max().view(1).expand(num_samples)
    if mask.ndim == 2:
        if mask.shape[0] == num_samples:
            return mask.max(dim=1).values
        if mask.shape[1] == num_samples:
            return mask.max(dim=0).values
        return mask[0].max().view(1).expand(num_samples)
    return mask.reshape(-1)[0].view(1).expand(num_samples)


def _per_eef_mask_from_ref_mask(
    mask,
    num_samples: int,
    num_eef: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert scalar/per-EEF/per-sample masks to (num_samples, num_eef)."""
    if not torch.is_tensor(mask):
        mask = torch.tensor(mask, device=device, dtype=dtype)
    else:
        mask = mask.to(device=device, dtype=dtype)
    if mask.ndim == 0:
        return mask.view(1, 1).expand(num_samples, num_eef)
    if mask.ndim == 1:
        if mask.shape[0] == num_eef:
            return mask.unsqueeze(0).expand(num_samples, num_eef)
        if mask.shape[0] == num_samples:
            return mask.unsqueeze(1).expand(num_samples, num_eef)
        if mask.shape[0] == 1:
            return mask.view(1, 1).expand(num_samples, num_eef)
        return mask[0].view(1, 1).expand(num_samples, num_eef)
    if mask.ndim == 2:
        if mask.shape == (num_samples, num_eef):
            return mask
        if mask.shape[1] == num_eef:
            return mask[0].unsqueeze(0).expand(num_samples, num_eef)
        if mask.shape[0] == num_samples and mask.shape[1] == 1:
            return mask.expand(num_samples, num_eef)
        if mask.shape[1] == num_samples:
            return mask.max(dim=0).values.unsqueeze(1).expand(num_samples, num_eef)
        return mask.reshape(-1)[0].view(1, 1).expand(num_samples, num_eef)
    return mask.reshape(-1)[0].view(1, 1).expand(num_samples, num_eef)


# TODO: define update environment parameter kernel functions, combine them compile step, also add parameter to be modified into MJWPEnv

# --
# Key functions
# --


def setup_mj_model(config: Config) -> mujoco.MjModel:
    model_cpu = mujoco.MjModel.from_xml_path(config.model_path)
    # Path Y: physics_dt for Holosoma alignment (decimation handled in graph capture)
    if config.physics_dt > 0:
        model_cpu.opt.timestep = float(config.physics_dt)
    else:
        model_cpu.opt.timestep = float(config.sim_dt)
    if config.embodiment_type in ["left", "right", "bimanual"]:
        # setup for hand
        model_cpu.opt.iterations = 20
        model_cpu.opt.ls_iterations = 50
        model_cpu.opt.o_solref = [0.02, 1.0]
        model_cpu.opt.o_solimp = [
            0.0,
            0.95,
            0.03,
            0.5,
            2,
        ]  # softer contact for sim2real
        model_cpu.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    elif config.embodiment_type in [
        "humanoid",
        "humanoid_object",
        "dual_humanoid_object",
    ]:
        # setup for humanoid
        model_cpu.opt.iterations = 5
        model_cpu.opt.ls_iterations = 10
        model_cpu.opt.o_solref = [0.02, 1.0]
        model_cpu.opt.o_solimp = [
            0.9,
            0.95,
            0.001,
            0.5,
            2,
        ]  # softer contact for sim2real
        model_cpu.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # Path Y: override PD gains with Holosoma values
    if getattr(config, "apply_holosoma_pd", False):
        from spider.mujoco_utils import apply_holosoma_g1_pd

        n = apply_holosoma_g1_pd(model_cpu, verbose=False)
        loguru.logger.info(f"Applied Holosoma G1 PD to {n} actuators")
    # HDMI R013: add dof_damping to wrist joints (critically damped)
    if getattr(config, "apply_wrist_dof_damping", False):
        for ji in range(model_cpu.njnt):
            jname = mujoco.mj_id2name(model_cpu, mujoco.mjtObj.mjOBJ_JOINT, ji)
            if jname and "wrist" in jname:
                model_cpu.dof_damping[model_cpu.jnt_dofadr[ji]] = (
                    config.wrist_dof_damping
                )
        loguru.logger.info(f"Applied wrist dof_damping={config.wrist_dof_damping}")
    return model_cpu


def setup_env(config: Config, ref_data: tuple[torch.Tensor, ...]) -> MJWPEnv:
    """Setup and reset the environment backed by MJWP.
    Returns an MJWPEnv with captured graph.
    """
    qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref = ref_data
    qpos_init = qpos_ref[0]

    # CPU model/data
    model_cpu = setup_mj_model(config)
    data_cpu = mujoco.MjData(model_cpu)
    # Seed initial state
    arrs = (qpos_init, qvel_ref[0], ctrl_ref[0])
    data_cpu.qpos[:] = arrs[0].detach().cpu().numpy()
    data_cpu.qvel[:] = arrs[1].detach().cpu().numpy()
    data_cpu.ctrl[:] = arrs[2].detach().cpu().numpy()
    mujoco.mj_step(model_cpu, data_cpu)

    # Move to Warp (batched worlds)
    # Set Warp default device to match config to ensure kernels/modules load on it
    wp.set_device(str(config.device))
    # Build default model/data/graph on the configured device
    dev = str(config.device)
    with wp.ScopedDevice(dev):
        default_model_wp = mjwarp.put_model(model_cpu)
        # pair_margin_override_np = (
        #     np.zeros((int(config.num_samples), model_cpu.npair)).astype(np.float32)
        #     + 0.01
        # )
        # pair_margin_override_wp = wp.from_numpy(
        #     pair_margin_override_np, dtype=wp.float32, device=dev
        # )
        # default_model_wp.pair_margin = pair_margin_override_wp

        default_data_wp = mjwarp.put_data(
            model_cpu,
            data_cpu,
            nworld=int(config.num_samples),
            nconmax=int(config.nconmax_per_env),
            njmax=int(config.njmax_per_env),
        )
        data_wp_prev = mjwarp.put_data(
            model_cpu,
            data_cpu,
            nworld=int(config.num_samples),
            nconmax=int(config.nconmax_per_env),
            njmax=int(config.njmax_per_env),
        )
        default_graph = _compile_step(
            default_model_wp, default_data_wp, decimation=config.sim_decimation
        )

    # Initialize env; default active is main
    env = MJWPEnv(
        model_cpu=model_cpu,
        data_cpu=data_cpu,
        model_wp=default_model_wp,
        data_wp=default_data_wp,
        data_wp_prev=data_wp_prev,
        graph=default_graph,
        device=dev,
        num_worlds=int(config.num_samples),
    )

    # Load mocap partner trajectory if configured
    if config.mocap_partner_trajectory:
        _load_mocap_partner(config, env)
    if config.support_proxy_enabled:
        _load_support_proxy(config, env, qpos_ref, qvel_ref)

    return env


def _weight_diff_qpos(config: Config) -> torch.Tensor:
    w = torch.ones(config.nv, device=config.device)
    if config.embodiment_type == "bimanual":
        half_dof = int(config.nu // 2)
        w[:3] = config.base_pos_rew_scale
        w[3:6] = config.base_rot_rew_scale
        w[6:half_dof] = config.joint_rew_scale
        w[half_dof : half_dof + 3] = config.base_pos_rew_scale
        w[half_dof + 3 : half_dof + 6] = config.base_rot_rew_scale
        w[half_dof + 6 : config.nu] = config.joint_rew_scale
        # object: weights live in nv-space (6-dim per freejoint), regardless of nq_obj
        w[-12:-9] = config.pos_rew_scale
        w[-9:-6] = config.rot_rew_scale
        w[-6:-3] = config.pos_rew_scale
        w[-3:] = config.rot_rew_scale
    elif config.embodiment_type in ["right", "left"]:
        w[:3] = config.base_pos_rew_scale
        w[3:6] = config.base_rot_rew_scale
        w[6 : config.nu] = config.joint_rew_scale
        w[-6:-3] = config.pos_rew_scale
        w[-3:] = config.rot_rew_scale
    elif config.embodiment_type in ["humanoid"]:  # humanoid robot
        # robot pos and rot
        w[:3] = config.pos_rew_scale
        w[3:6] = config.rot_rew_scale
        # robot joint
        w[6:] = config.joint_rew_scale
    elif config.embodiment_type in ["humanoid_object"]:
        # robot pos and rot
        w[:3] = config.base_pos_rew_scale
        w[3:6] = config.base_rot_rew_scale
        # robot joint
        w[6:-6] = config.joint_rew_scale
        # object pos and rot
        w[-6:-3] = config.pos_rew_scale
        w[-3:] = config.rot_rew_scale
    elif config.embodiment_type == "dual_humanoid_object":
        # Two robots + one shared object
        # nv layout: robot1_base(3)+rot(3)+joints(29) + robot2_base(3)+rot(3)+joints(29) + obj_pos(3)+rot(3)
        nv_robot = (config.nv - 6) // 2  # 35 per robot
        # robot1
        w[:3] = config.base_pos_rew_scale
        w[3:6] = config.base_rot_rew_scale
        w[6:nv_robot] = config.joint_rew_scale
        # robot2
        w[nv_robot : nv_robot + 3] = config.base_pos_rew_scale
        w[nv_robot + 3 : nv_robot + 6] = config.base_rot_rew_scale
        w[nv_robot + 6 : 2 * nv_robot] = config.joint_rew_scale
        # object
        w[-6:-3] = config.pos_rew_scale
        w[-3:] = config.rot_rew_scale
    else:
        raise ValueError(f"Invalid embodiment_type: {config.embodiment_type}")
    return w


def _diff_qpos(
    config: Config, qpos_sim: torch.Tensor, qpos_ref: torch.Tensor
) -> torch.Tensor:
    """Compute the difference between qpos_sim and qpos_ref
    TODO: replace with mujoco built-in function, not sure how to call warp internal function yet.
    """
    batch_size = qpos_sim.shape[0]
    qpos_diff = torch.zeros((batch_size, config.nv), device=config.device)
    if config.embodiment_type == "bimanual":
        if config.nq_obj == 12:
            qpos_diff[:, :-12] = qpos_sim[:, :-12] - qpos_ref[:, :-12]
            qpos_diff[:, -12:-9] = qpos_sim[:, -12:-9] - qpos_ref[:, -12:-9]
            qpos_diff[:, -9:-6] = qpos_sim[:, -9:-6] - qpos_ref[:, -9:-6]
            qpos_diff[:, -6:-3] = qpos_sim[:, -6:-3] - qpos_ref[:, -6:-3]
            qpos_diff[:, -3:] = qpos_sim[:, -3:] - qpos_ref[:, -3:]
            return qpos_diff
        # joint
        qpos_diff[:, :-12] = qpos_sim[:, :-14] - qpos_ref[:, :-14]
        # position
        qpos_diff[:, -12:-9] = qpos_sim[:, -14:-11] - qpos_ref[:, -14:-11]
        qpos_diff[:, -6:-3] = qpos_sim[:, -7:-4] - qpos_ref[:, -7:-4]
        # rotation
        qpos_diff[:, -9:-6] = quat_sub(qpos_sim[:, -11:-7], qpos_ref[:, -11:-7])
        qpos_diff[:, -3:] = quat_sub(qpos_sim[:, -4:], qpos_ref[:, -4:])
    elif config.embodiment_type in ["right", "left"]:
        if config.nq_obj == 6:
            qpos_diff[:, :-6] = qpos_sim[:, :-6] - qpos_ref[:, :-6]
            qpos_diff[:, -6:-3] = qpos_sim[:, -6:-3] - qpos_ref[:, -6:-3]
            qpos_diff[:, -3:] = qpos_sim[:, -3:] - qpos_ref[:, -3:]
            return qpos_diff
        # joint
        qpos_diff[:, :-6] = qpos_sim[:, :-7] - qpos_ref[:, :-7]
        # position
        qpos_diff[:, -6:-3] = qpos_sim[:, -7:-4] - qpos_ref[:, -7:-4]
        # rotation
        qpos_diff[:, -3:] = quat_sub(qpos_sim[:, -4:], qpos_ref[:, -4:])
    elif config.embodiment_type in ["humanoid"]:
        # joint
        qpos_diff[:, 6:] = qpos_sim[:, 7:] - qpos_ref[:, 7:]
        # position
        qpos_diff[:, :3] = qpos_sim[:, :3] - qpos_ref[:, :3]
        # rotation
        qpos_diff[:, 3:6] = quat_sub(qpos_sim[:, 3:7], qpos_ref[:, 3:7])
    elif config.embodiment_type in ["humanoid_object"]:
        nq_obj = config.nq_obj  # 7 (freejoint) or 6 (contact_guidance)
        qpos_humanoid = qpos_sim[:, :-nq_obj]
        qpos_object = qpos_sim[:, -nq_obj:]
        qpos_ref_humanoid = qpos_ref[:, :-nq_obj]
        qpos_ref_object = qpos_ref[:, -nq_obj:]
        # position
        qpos_diff[:, :3] = qpos_humanoid[:, :3] - qpos_ref_humanoid[:, :3]
        # rotation
        qpos_diff[:, 3:6] = quat_sub(qpos_humanoid[:, 3:7], qpos_ref_humanoid[:, 3:7])
        # joint
        qpos_diff[:, 6:-6] = qpos_humanoid[:, 7:] - qpos_ref_humanoid[:, 7:]
        # object
        if nq_obj == 7:
            # freejoint: pos(3) + quat(4)
            qpos_diff[:, -6:-3] = qpos_object[:, :3] - qpos_ref_object[:, :3]
            qpos_diff[:, -3:] = quat_sub(qpos_object[:, 3:7], qpos_ref_object[:, 3:7])
        else:
            # contact_guidance: pos(3) + rpy(3), all direct subtraction
            qpos_diff[:, -6:-3] = qpos_object[:, :3] - qpos_ref_object[:, :3]
            qpos_diff[:, -3:] = qpos_object[:, 3:6] - qpos_ref_object[:, 3:6]
    elif config.embodiment_type == "dual_humanoid_object":
        nq_obj = config.nq_obj  # 7 (freejoint) or 6 (contact_guidance)
        nq_robot = (config.nq - nq_obj) // 2  # 36 per robot
        nv_robot = (config.nv - 6) // 2  # 35 per robot
        # robot1: nq[0:nq_robot], robot2: nq[nq_robot:2*nq_robot], obj: nq[-nq_obj:]
        r1 = qpos_sim[:, :nq_robot]
        r2 = qpos_sim[:, nq_robot : 2 * nq_robot]
        obj = qpos_sim[:, -nq_obj:]
        r1_ref = qpos_ref[:, :nq_robot]
        r2_ref = qpos_ref[:, nq_robot : 2 * nq_robot]
        obj_ref = qpos_ref[:, -nq_obj:]
        # robot1: base pos/rot/joints
        qpos_diff[:, :3] = r1[:, :3] - r1_ref[:, :3]
        qpos_diff[:, 3:6] = quat_sub(r1[:, 3:7], r1_ref[:, 3:7])
        qpos_diff[:, 6:nv_robot] = r1[:, 7:] - r1_ref[:, 7:]
        # robot2: base pos/rot/joints
        qpos_diff[:, nv_robot : nv_robot + 3] = r2[:, :3] - r2_ref[:, :3]
        qpos_diff[:, nv_robot + 3 : nv_robot + 6] = quat_sub(r2[:, 3:7], r2_ref[:, 3:7])
        qpos_diff[:, nv_robot + 6 : 2 * nv_robot] = r2[:, 7:] - r2_ref[:, 7:]
        # object
        if nq_obj == 7:
            qpos_diff[:, -6:-3] = obj[:, :3] - obj_ref[:, :3]
            qpos_diff[:, -3:] = quat_sub(obj[:, 3:7], obj_ref[:, 3:7])
        else:
            qpos_diff[:, -6:-3] = obj[:, :3] - obj_ref[:, :3]
            qpos_diff[:, -3:] = obj[:, 3:6] - obj_ref[:, 3:6]
    else:
        raise ValueError(f"Invalid embodiment_type: {config.embodiment_type}")
    return qpos_diff


# ---------------------------------------------------------------------------
# E035: Local-frame tracking helpers (ported from HDMI)
# ---------------------------------------------------------------------------


def _lf_yaw_quat(q: torch.Tensor) -> torch.Tensor:
    """Extract yaw-only rotation from quaternion. q: (..., 4) wxyz."""
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack(
        [
            torch.cos(yaw / 2),
            torch.zeros_like(yaw),
            torch.zeros_like(yaw),
            torch.sin(yaw / 2),
        ],
        dim=-1,
    )


def _lf_quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate. q: (..., 4) wxyz."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _lf_quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product. q1, q2: (..., 4) wxyz."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _lf_quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q. q: (..., 4) wxyz, v: (..., 3)."""
    t = 2.0 * torch.cross(q[..., 1:], v, dim=-1)
    return v + q[..., :1] * t + torch.cross(q[..., 1:], t, dim=-1)


def _lf_quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by inverse of quaternion q."""
    return _lf_quat_apply(_lf_quat_conjugate(q), v)


def _clamp_vector_norm(vec: torch.Tensor, max_norm: float) -> torch.Tensor:
    if max_norm <= 0:
        return vec
    norm = vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.where(norm > max_norm, vec / norm * max_norm, vec)


def _clear_object_wrench_once(
    env: MJWPEnv, obj_body_id: int, xfrc_applied: torch.Tensor
):
    """Clear persistent object xfrc once per step before accumulating helpers."""
    if not getattr(env, "_object_wrench_cleared_this_step", False):
        xfrc_applied[:, obj_body_id, :6] = 0.0
        env._object_wrench_cleared_this_step = True


def _lf_axis_angle_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to axis-angle. q: (..., 4) wxyz -> (..., 3)."""
    sin_half = torch.norm(q[..., 1:], dim=-1, keepdim=True).clamp(min=1e-8)
    cos_half = q[..., :1]
    angle = 2.0 * torch.atan2(sin_half, cos_half)
    axis = q[..., 1:] / sin_half
    return axis * angle


def _local_pos_tracking(
    xpos_batch: torch.Tensor,
    xquat_batch: torch.Tensor,
    body_ids: list[int],
    root_id: int,
    ref_body_pos: torch.Tensor,
    ref_root_pos: torch.Tensor,
    ref_root_quat: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Position tracking in root-yaw-relative frame. Returns (N,).
    ref_body_pos: (B, 3), ref_root_pos: (3,), ref_root_quat: (4,)
    """
    N = xpos_batch.shape[0]
    body_pos = xpos_batch[:, body_ids, :]  # (N, B, 3)
    root_pos = xpos_batch[:, root_id, :]  # (N, 3)
    root_quat = xquat_batch[:, root_id, :]  # (N, 4)
    B = len(body_ids)

    root_pos_xy = root_pos.clone()
    root_pos_xy[..., 2] = 0.0
    root_quat_yaw = _lf_yaw_quat(root_quat)  # (N, 4)

    ref_root_xy = ref_root_pos.clone()
    ref_root_xy[2] = 0.0
    ref_root_quat_yaw = _lf_yaw_quat(ref_root_quat.unsqueeze(0)).squeeze(0)  # (4,)

    # Expand for batch and body dims
    rp = root_pos_xy.unsqueeze(1).expand(-1, B, -1)  # (N, B, 3)
    rq = root_quat_yaw.unsqueeze(1).expand(-1, B, -1)  # (N, B, 4)
    ref_rp = ref_root_xy.unsqueeze(0).unsqueeze(0).expand(N, B, -1)  # (N, B, 3)
    ref_rq = ref_root_quat_yaw.unsqueeze(0).unsqueeze(0).expand(N, B, -1)  # (N, B, 4)

    body_local = _lf_quat_apply_inverse(rq, body_pos - rp)  # (N, B, 3)
    ref_body_pos_exp = ref_body_pos.unsqueeze(0).expand(N, -1, -1)  # (N, B, 3)
    ref_local = _lf_quat_apply_inverse(ref_rq, ref_body_pos_exp - ref_rp)  # (N, B, 3)

    error = (ref_local - body_local).norm(dim=-1).clamp_min(0.0)  # (N, B)
    return torch.exp(-error.mean(dim=1) / sigma)


def _local_ori_tracking(
    xquat_batch: torch.Tensor,
    body_ids: list[int],
    root_id: int,
    ref_body_quat: torch.Tensor,
    ref_root_quat: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Orientation tracking in root-yaw-relative frame. Returns (N,).
    ref_body_quat: (B, 4), ref_root_quat: (4,)
    """
    N = xquat_batch.shape[0]
    B = len(body_ids)
    body_quat = xquat_batch[:, body_ids, :]  # (N, B, 4)
    root_quat = xquat_batch[:, root_id, :]  # (N, 4)

    root_yaw = _lf_yaw_quat(root_quat)  # (N, 4)
    ref_root_yaw = _lf_yaw_quat(ref_root_quat.unsqueeze(0)).squeeze(0)  # (4,)

    rq = root_yaw.unsqueeze(1).expand(-1, B, -1)  # (N, B, 4)
    ref_rq = ref_root_yaw.unsqueeze(0).unsqueeze(0).expand(N, B, -1)  # (N, B, 4)

    body_local = _lf_quat_mul(_lf_quat_conjugate(rq), body_quat)  # (N, B, 4)
    ref_body_quat_exp = ref_body_quat.unsqueeze(0).expand(N, -1, -1)  # (N, B, 4)
    ref_local = _lf_quat_mul(_lf_quat_conjugate(ref_rq), ref_body_quat_exp)  # (N, B, 4)

    diff = _lf_quat_mul(_lf_quat_conjugate(ref_local), body_local)  # (N, B, 4)
    error = _lf_axis_angle_from_quat(diff).norm(dim=-1).clamp_min(0.0)  # (N, B)
    return torch.exp(-error.mean(dim=1) / sigma)


def get_reward(
    config: Config,
    env: MJWPEnv,
    ref: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Non-terminal step reward for MJWP batched worlds.
    ref is a tuple: (qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref,
                     body_xpos_ref) where body_xpos_ref is (K, 3) per timestep
    Returns (N,)

    TODO: move reward computation to task-specific module
    """
    # Unpack with backward compatibility (5-tuple legacy, 6-tuple E018, 7-tuple E034, 8-tuple E035, 9-tuple E040)
    approach_mask_val = 1.0
    body_xquat_ref = None
    contact_target_dynamic = None
    if len(ref) == 5:
        qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref = ref
        body_xpos_ref = None
    elif len(ref) == 6:
        qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref, body_xpos_ref = ref
    elif len(ref) == 7:
        (
            qpos_ref,
            qvel_ref,
            ctrl_ref,
            contact_ref,
            contact_pos_ref,
            body_xpos_ref,
            approach_mask_val,
        ) = ref
    elif len(ref) == 8:
        (
            qpos_ref,
            qvel_ref,
            ctrl_ref,
            contact_ref,
            contact_pos_ref,
            body_xpos_ref,
            approach_mask_val,
            body_xquat_ref,
        ) = ref
    else:
        (
            qpos_ref,
            qvel_ref,
            ctrl_ref,
            contact_ref,
            contact_pos_ref,
            body_xpos_ref,
            approach_mask_val,
            body_xquat_ref,
            contact_target_dynamic,
        ) = ref
    qpos_sim = wp.to_torch(env.data_wp.qpos)
    qvel_sim = wp.to_torch(env.data_wp.qvel)
    N = qpos_sim.shape[0]

    # weighted qpos tracking
    qpos_diff = _diff_qpos(config, qpos_sim, qpos_ref.unsqueeze(0).repeat(N, 1))
    qpos_weight = _weight_diff_qpos(config)
    delta_qpos = qpos_diff * qpos_weight
    qpos_dist = torch.norm(delta_qpos, p=2, dim=1)
    qvel_dist = torch.norm(qvel_sim - qvel_ref, p=2, dim=1)

    qpos_rew = (
        config.qpos_reward_scale * torch.exp(-qpos_dist / config.qpos_reward_sigma)
        if config.use_bounded_qpos_reward
        else -qpos_dist * 1.0
    )
    qvel_rew = -config.vel_rew_scale * qvel_dist * 1.0

    # E035: local-frame body tracking (replaces qpos_rew when enabled)
    local_frame_rew = torch.zeros(N, device=config.device)
    if (
        config.use_local_frame_reward
        and body_xpos_ref is not None
        and body_xquat_ref is not None
    ):
        xpos_sim = wp.to_torch(env.data_wp.xpos)  # (N, nbody, 3)
        xquat_sim = wp.to_torch(env.data_wp.xquat)  # (N, nbody, 4) wxyz

        root_id = 1  # pelvis
        ref_root_pos = body_xpos_ref[root_id]  # (3,)
        ref_root_quat = body_xquat_ref[root_id]  # (4,)

        upper_ids = config.local_frame_upper_ids
        lower_ids = config.local_frame_lower_ids

        upper_pos_rew = _local_pos_tracking(
            xpos_sim,
            xquat_sim,
            upper_ids,
            root_id,
            body_xpos_ref[upper_ids],
            ref_root_pos,
            ref_root_quat,
            config.local_frame_pos_sigma,
        )
        upper_ori_rew = _local_ori_tracking(
            xquat_sim,
            upper_ids,
            root_id,
            body_xquat_ref[upper_ids],
            ref_root_quat,
            config.local_frame_ori_sigma,
        )

        # E044: extra weight for wrist bodies
        if config.local_frame_wrist_weight != 1.0 and config.local_frame_wrist_ids:
            wrist_ids = [
                wid for wid in config.local_frame_wrist_ids if wid in upper_ids
            ]
            if wrist_ids:
                wrist_pos_extra = _local_pos_tracking(
                    xpos_sim,
                    xquat_sim,
                    wrist_ids,
                    root_id,
                    body_xpos_ref[wrist_ids],
                    ref_root_pos,
                    ref_root_quat,
                    config.local_frame_pos_sigma,
                )
                upper_pos_rew = (
                    upper_pos_rew
                    + (config.local_frame_wrist_weight - 1.0) * wrist_pos_extra
                )
        lower_pos_rew = _local_pos_tracking(
            xpos_sim,
            xquat_sim,
            lower_ids,
            root_id,
            body_xpos_ref[lower_ids],
            ref_root_pos,
            ref_root_quat,
            config.local_frame_pos_sigma,
        )
        # E166: extra weight for ankle bodies, mirroring the wrist hook above.
        if config.local_frame_ankle_weight != 1.0 and config.local_frame_ankle_ids:
            ankle_ids = [
                aid for aid in config.local_frame_ankle_ids if aid in lower_ids
            ]
            if ankle_ids:
                ankle_pos_extra = _local_pos_tracking(
                    xpos_sim,
                    xquat_sim,
                    ankle_ids,
                    root_id,
                    body_xpos_ref[ankle_ids],
                    ref_root_pos,
                    ref_root_quat,
                    config.local_frame_pos_sigma,
                )
                lower_pos_rew = (
                    lower_pos_rew
                    + (config.local_frame_ankle_weight - 1.0) * ankle_pos_extra
                )
        lower_ori_rew = _local_ori_tracking(
            xquat_sim,
            lower_ids,
            root_id,
            body_xquat_ref[lower_ids],
            ref_root_quat,
            config.local_frame_ori_sigma,
        )

        # Root global tracking
        root_pos_err = (xpos_sim[:, root_id] - ref_root_pos.unsqueeze(0)).norm(dim=-1)
        root_pos_rew = torch.exp(-root_pos_err / config.local_frame_root_sigma)

        root_quat_sim = xquat_sim[:, root_id]  # (N, 4)
        root_quat_ref = ref_root_quat.unsqueeze(0).expand(N, -1)
        root_diff = _lf_quat_mul(_lf_quat_conjugate(root_quat_ref), root_quat_sim)
        root_ori_err = _lf_axis_angle_from_quat(root_diff).norm(dim=-1)
        root_ori_rew = torch.exp(-root_ori_err / config.local_frame_root_sigma)

        # Joint tracking from qpos (joint angles only, not base)
        if config.embodiment_type == "humanoid_object":
            nq_obj = max(int(config.nq_obj), 0)
            obj_start = -nq_obj if nq_obj > 0 else None
            jt_sim = qpos_sim[:, 7:obj_start]
            jt_ref = qpos_ref[7:obj_start]
            jt_err = (jt_sim - jt_ref.unsqueeze(0)).abs().mean(dim=1)
        else:
            jt_err = torch.zeros(N, device=config.device)
        joint_rew = torch.exp(-jt_err / config.local_frame_joint_sigma)

        W = config.local_frame_w_track
        local_frame_rew = W * (
            upper_pos_rew
            + upper_ori_rew
            + lower_pos_rew
            + lower_ori_rew
            + root_pos_rew
            + root_ori_rew
            + joint_rew
        )  # max = W * 7

        # Replace qpos_rew with local_frame_rew
        qpos_rew = local_frame_rew

    # contact reward
    if config.contact_rew_scale > 0.0 and len(config.contact_site_ids) > 0:
        site_xpos_torch = wp.to_torch(env.data_wp.site_xpos)
        contact_pos = site_xpos_torch[:, config.contact_site_ids]
        contact_dist = torch.norm(contact_pos - contact_pos_ref, p=2, dim=-1)
        contact_dist_masked = contact_dist * contact_ref.unsqueeze(0)
        contact_rew = -contact_dist_masked.sum(dim=1)
    else:
        contact_rew = 0.0

    # E018: task-space body world-position tracking (DynaRetarget Table II)
    task_body_rew = torch.zeros(N, device=config.device)
    if (
        config.task_body_rew_scale > 0.0
        and config.task_body_ids
        and body_xpos_ref is not None
        and body_xpos_ref.shape[0] > 0
    ):
        xpos_sim = wp.to_torch(env.data_wp.xpos)  # (N, nbody, 3)
        body_pos_sim = xpos_sim[:, config.task_body_ids]  # (N, K, 3)
        task_body_xpos_ref = body_xpos_ref
        if body_xpos_ref.shape[0] != len(config.task_body_ids) and body_xpos_ref.shape[
            0
        ] > max(config.task_body_ids):
            task_body_xpos_ref = body_xpos_ref[config.task_body_ids]
        body_weights = torch.tensor(
            config.task_body_weights, device=config.device, dtype=body_pos_sim.dtype
        )  # (K,)
        # body_xpos_ref is (K, 3) for this timestep
        err = ((body_pos_sim - task_body_xpos_ref.unsqueeze(0)) ** 2).sum(dim=-1)
        task_body_rew = -config.task_body_rew_scale * (err * body_weights).sum(dim=1)

    # E018: separate object position/orientation tracking with high weight
    task_obj_rew = torch.zeros(N, device=config.device)
    if (
        config.task_obj_pos_rew_scale > 0.0 or config.task_obj_rot_rew_scale > 0.0
    ) and config.embodiment_type in [
        "humanoid_object",
        "dual_humanoid_object",
        "bimanual",
        "right",
        "left",
    ]:
        nq_obj = config.nq_obj
        # E065: switch between unbounded -L2 (default) and saturating exp form.
        # exp form: scale * exp(-||err||/sigma) ∈ [0, scale], aligned with HDMI.
        use_exp = config.task_obj_use_exp
        if nq_obj == 7:
            obj_pos_sim = qpos_sim[:, -7:-4]
            obj_pos_ref = qpos_ref[-7:-4].unsqueeze(0)
            if use_exp:
                pos_err_norm = (obj_pos_sim - obj_pos_ref).norm(dim=-1)
                task_obj_rew = task_obj_rew + config.task_obj_pos_rew_scale * torch.exp(
                    -pos_err_norm / config.task_obj_pos_sigma
                )
            else:
                pos_err = ((obj_pos_sim - obj_pos_ref) ** 2).sum(dim=-1)
                task_obj_rew = task_obj_rew - config.task_obj_pos_rew_scale * pos_err
            if config.task_obj_rot_rew_scale > 0.0:
                obj_quat_sim = qpos_sim[:, -4:]
                obj_quat_ref = qpos_ref[-4:].unsqueeze(0).repeat(N, 1)
                if use_exp:
                    rot_err_norm = quat_sub(obj_quat_sim, obj_quat_ref).norm(dim=-1)
                    task_obj_rew = (
                        task_obj_rew
                        + config.task_obj_rot_rew_scale
                        * torch.exp(-rot_err_norm / config.task_obj_rot_sigma)
                    )
                else:
                    rot_err = (quat_sub(obj_quat_sim, obj_quat_ref) ** 2).sum(dim=-1)
                    task_obj_rew = (
                        task_obj_rew - config.task_obj_rot_rew_scale * rot_err
                    )
        elif nq_obj == 6:
            obj_pos_sim = qpos_sim[:, -6:-3]
            obj_pos_ref = qpos_ref[-6:-3].unsqueeze(0)
            if use_exp:
                pos_err_norm = (obj_pos_sim - obj_pos_ref).norm(dim=-1)
                task_obj_rew = task_obj_rew + config.task_obj_pos_rew_scale * torch.exp(
                    -pos_err_norm / config.task_obj_pos_sigma
                )
            else:
                pos_err = ((obj_pos_sim - obj_pos_ref) ** 2).sum(dim=-1)
                task_obj_rew = task_obj_rew - config.task_obj_pos_rew_scale * pos_err
            if config.task_obj_rot_rew_scale > 0.0:
                obj_euler_sim = qpos_sim[:, -3:]
                obj_euler_ref = qpos_ref[-3:].unsqueeze(0)
                if use_exp:
                    rot_err_norm = (obj_euler_sim - obj_euler_ref).norm(dim=-1)
                    task_obj_rew = (
                        task_obj_rew
                        + config.task_obj_rot_rew_scale
                        * torch.exp(-rot_err_norm / config.task_obj_rot_sigma)
                    )
                else:
                    rot_err = ((obj_euler_sim - obj_euler_ref) ** 2).sum(dim=-1)
                    task_obj_rew = (
                        task_obj_rew - config.task_obj_rot_rew_scale * rot_err
                    )

    # E018: interaction reward (Harmanoid Eq.15) — match relative offsets
    # between pairs of bodies in task_body_ids
    interact_rew = torch.zeros(N, device=config.device)
    if (
        config.interact_rew_scale > 0.0
        and config.interact_pairs
        and config.task_body_ids
        and body_xpos_ref is not None
        and body_xpos_ref.shape[0] > 0
    ):
        xpos_sim = wp.to_torch(env.data_wp.xpos)
        body_pos_sim = xpos_sim[:, config.task_body_ids]  # (N, K, 3)
        pair_err_total = torch.zeros(N, device=config.device)
        for ia, ib in config.interact_pairs:
            delta_sim = body_pos_sim[:, ia] - body_pos_sim[:, ib]  # (N, 3)
            delta_ref = body_xpos_ref[ia] - body_xpos_ref[ib]  # (3,)
            pair_err_total = pair_err_total + (
                (delta_sim - delta_ref.unsqueeze(0)) ** 2
            ).sum(dim=-1)
        interact_rew = config.interact_rew_scale * torch.exp(
            -config.interact_sigma * pair_err_total
        )

    # E025: hand approach reward — exp decay of hand-to-object-surface distance
    hand_approach_rew = torch.zeros(N, device=config.device)
    if config.hand_approach_rew_scale > 0.0 and config.hand_approach_body_ids:
        obj_body_id = mujoco.mj_name2id(
            env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "object"
        )
        if obj_body_id != -1 and config.hand_approach_obj_half_extents:
            xpos_sim = wp.to_torch(env.data_wp.xpos)  # (N, nbody, 3)
            hand_pos = xpos_sim[:, config.hand_approach_body_ids]  # (N, K_hand, 3)
            obj_pos = xpos_sim[:, obj_body_id : obj_body_id + 1]  # (N, 1, 3)
            # Surface distance: clamp(|delta| - half_ext, min=0) then norm
            half_ext = torch.tensor(
                config.hand_approach_obj_half_extents,
                device=config.device,
                dtype=hand_pos.dtype,
            )
            delta = torch.abs(hand_pos - obj_pos)  # (N, K_hand, 3)
            surface_dist = torch.clamp(delta - half_ext, min=0.0)  # (N, K_hand, 3)
            # Min distance over hands (reward best hand)
            dist_per_hand = surface_dist.norm(dim=-1)  # (N, K_hand)
            min_dist = dist_per_hand.min(dim=1).values  # (N,)
            hand_approach_rew = (
                approach_mask_val
                * config.hand_approach_rew_scale
                * torch.exp(-config.hand_approach_sigma * min_dist)
            )

    # E037: contact mask-gated reward — HDMI-style proximity with mask gate
    contact_mask_rew = torch.zeros(N, device=config.device)
    if config.contact_mask_rew_scale > 0.0 and config.hand_approach_body_ids:
        obj_body_id = mujoco.mj_name2id(
            env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "object"
        )
        if obj_body_id != -1 and config.hand_approach_obj_half_extents:
            xpos_sim = wp.to_torch(env.data_wp.xpos)  # (N, nbody, 3)
            hand_pos = xpos_sim[:, config.hand_approach_body_ids]  # (N, K, 3)
            obj_pos = xpos_sim[:, obj_body_id : obj_body_id + 1]  # (N, 1, 3)
            half_ext = torch.tensor(
                config.hand_approach_obj_half_extents,
                device=config.device,
                dtype=hand_pos.dtype,
            )
            delta = torch.abs(hand_pos - obj_pos)
            surface_dist = torch.clamp(delta - half_ext, min=0.0)
            dist_per_hand = surface_dist.norm(dim=-1)  # (N, K)
            min_dist = dist_per_hand.min(dim=1).values  # (N,)
            # mask=1 → exp proximity reward; mask=0 → baseline (CEM ignores)
            gain = config.contact_mask_rew_scale
            mask = approach_mask_val  # scalar or (N,) from ref[6], per-timestep
            proximity = gain * torch.exp(-min_dist / config.contact_mask_rew_sigma)
            baseline = config.contact_mask_rew_baseline
            contact_mask_rew = mask * proximity + (1.0 - mask) * baseline

    # E039: HDMI-aligned contact — predefined target points + per-EEF + mask gate
    # E040: dynamic per-frame target support
    contact_hdmi_rew = torch.zeros(N, device=config.device)
    contact_hdmi_left_score = torch.zeros(N, device=config.device)
    contact_hdmi_right_score = torch.zeros(N, device=config.device)
    contact_hdmi_bimanual_gate = torch.zeros(N, device=config.device)
    contact_hdmi_bimanual_score = torch.zeros(N, device=config.device)
    if config.contact_hdmi_gain > 0.0 and (
        config.contact_hdmi_target_left or contact_target_dynamic is not None
    ):
        obj_body_id = mujoco.mj_name2id(
            env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "object"
        )
        if obj_body_id != -1 and config.hand_approach_body_ids:
            xpos_sim = wp.to_torch(env.data_wp.xpos)  # (N, nbody, 3)
            xquat_sim = wp.to_torch(env.data_wp.xquat)  # (N, nbody, 4) wxyz
            obj_pos = xpos_sim[:, obj_body_id]  # (N, 3)
            obj_quat = xquat_sim[:, obj_body_id]  # (N, 4) wxyz

            eef_bids = config.hand_approach_body_ids  # [left_wrist, right_wrist]
            eef_offset = torch.tensor(
                config.contact_hdmi_eef_offset,
                device=config.device,
                dtype=obj_pos.dtype,
            )

            # E041: palm normal vectors for orientation reward
            palm_normals = None
            if config.contact_hdmi_ori_weight > 0.0:
                palm_normals = [
                    torch.tensor(
                        config.contact_hdmi_palm_normal_left,
                        device=config.device,
                        dtype=obj_pos.dtype,
                    ),
                    torch.tensor(
                        config.contact_hdmi_palm_normal_right,
                        device=config.device,
                        dtype=obj_pos.dtype,
                    ),
                ]

            # E040: choose between dynamic per-frame targets and fixed targets
            if contact_target_dynamic is not None:
                # contact_target_dynamic is (n_eef, 3) for this timestep (already indexed by t)
                targets = [contact_target_dynamic[ei] for ei in range(len(eef_bids))]
            else:
                targets = [
                    torch.tensor(
                        config.contact_hdmi_target_left,
                        device=config.device,
                        dtype=obj_pos.dtype,
                    ),
                    torch.tensor(
                        config.contact_hdmi_target_right,
                        device=config.device,
                        dtype=obj_pos.dtype,
                    ),
                ]

            per_eef_rew = []
            for ei, (bid, target_off) in enumerate(zip(eef_bids, targets)):
                # Target in world = obj_pos + quat_apply(obj_quat, target_offset)
                target_world = obj_pos + _lf_quat_apply(
                    obj_quat, target_off.unsqueeze(0).expand(N, -1)
                )
                # EEF contact point = eef_pos + quat_apply(eef_quat, eef_offset)
                eef_pos = xpos_sim[:, bid]  # (N, 3)
                eef_quat = xquat_sim[:, bid]  # (N, 4)
                contact_point = eef_pos + _lf_quat_apply(
                    eef_quat, eef_offset.unsqueeze(0).expand(N, -1)
                )
                # Distance and exp reward
                dist = (target_world - contact_point).norm(dim=-1)  # (N,)
                pos_rew = torch.exp(-dist / config.contact_hdmi_sigma)

                # E041: orientation gating — palm must face toward target
                if palm_normals is not None:
                    palm_local = palm_normals[ei]  # (3,)
                    palm_world = _lf_quat_apply(
                        eef_quat, palm_local.unsqueeze(0).expand(N, -1)
                    )  # (N, 3)
                    dir_to_target = target_world - contact_point  # (N, 3)
                    dir_norm = dir_to_target.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    dir_to_target = dir_to_target / dir_norm  # (N, 3) normalized
                    dot = (palm_world * dir_to_target).sum(dim=-1)  # (N,) in [-1, 1]
                    ori_rew = torch.clamp(dot, min=0.0)  # (N,) in [0, 1]

                    ori_mode = config.contact_hdmi_ori_mode
                    if ori_mode == "multiply":
                        # Strict: only reward when both close AND palm faces target
                        pos_rew = pos_rew * ori_rew
                    elif ori_mode == "additive":
                        # Softer: weighted combination
                        w = config.contact_hdmi_ori_weight
                        pos_rew = (1.0 - w) * pos_rew + w * ori_rew
                    elif ori_mode == "near_field":
                        # Only enforce orientation when already close (dist < 0.15m)
                        near_mask = (dist < 0.15).float()
                        pos_rew = pos_rew * (1.0 - near_mask + near_mask * ori_rew)

                per_eef_rew.append(pos_rew)

            # Stack per-EEF rewards: (N, 2)
            rew_stack = torch.stack(per_eef_rew, dim=1)
            mask = approach_mask_val
            if not torch.is_tensor(mask):
                mask = torch.tensor(mask, device=config.device, dtype=rew_stack.dtype)
            else:
                mask = mask.to(device=config.device, dtype=rew_stack.dtype)
            if mask.ndim == 0:
                mask_eef = mask.view(1, 1).expand_as(rew_stack)
            elif mask.ndim == 1:
                if mask.shape[0] == rew_stack.shape[1]:
                    # Current timestep per-EEF mask, e.g. (2,).
                    mask_eef = mask.unsqueeze(0).expand_as(rew_stack)
                elif mask.shape[0] == rew_stack.shape[0]:
                    # Per-sample scalar mask, legacy behavior.
                    mask_eef = mask.unsqueeze(1).expand_as(rew_stack)
                elif mask.shape[0] == 1:
                    mask_eef = mask.view(1, 1).expand_as(rew_stack)
                else:
                    # Horizon-shaped masks can appear in tracing; use the first timestep
                    # for this reward call to preserve previous scalar-time behavior.
                    mask_eef = mask[0].view(1, 1).expand_as(rew_stack)
            elif mask.ndim == 2:
                if mask.shape == rew_stack.shape:
                    mask_eef = mask
                elif mask.shape[1] == rew_stack.shape[1]:
                    # Horizon x EEF mask: use current timestep.
                    mask_eef = mask[0].unsqueeze(0).expand_as(rew_stack)
                elif mask.shape[0] == rew_stack.shape[0] and mask.shape[1] == 1:
                    mask_eef = mask.expand_as(rew_stack)
                else:
                    raise ValueError(
                        f"Unsupported contact_hdmi mask shape {tuple(mask.shape)} for reward {tuple(rew_stack.shape)}"
                    )
            else:
                raise ValueError(f"Unsupported contact_hdmi mask ndim {mask.ndim}")
            gain = config.contact_hdmi_gain
            # HDMI formula: mask=1 → gain*pos_rew, mask=0 → 1.0
            if config.contact_hdmi_bimanual_required:
                if rew_stack.shape[1] != 2:
                    raise ValueError(
                        "contact_hdmi_bimanual_required expects exactly two EEF rewards"
                    )
                if config.contact_hdmi_bimanual_score_reduce != "min":
                    raise ValueError(
                        "Unsupported contact_hdmi_bimanual_score_reduce="
                        f"{config.contact_hdmi_bimanual_score_reduce!r}"
                    )
                active = mask_eef > 0.0
                active_any = active.any(dim=1)
                active_both = active.all(dim=1)
                contact_hdmi_left_score = rew_stack[:, 0]
                contact_hdmi_right_score = rew_stack[:, 1]
                contact_hdmi_bimanual_gate = active_both.to(rew_stack.dtype)
                contact_hdmi_bimanual_score = torch.minimum(
                    contact_hdmi_left_score, contact_hdmi_right_score
                )
                contact_hdmi_rew = torch.where(
                    active_any,
                    torch.where(
                        active_both,
                        gain * contact_hdmi_bimanual_score,
                        torch.zeros_like(contact_hdmi_bimanual_score),
                    ),
                    torch.ones_like(contact_hdmi_bimanual_score),
                )
            else:
                contact_hdmi_rew = (
                    rew_stack * mask_eef * gain + (1.0 - mask_eef)
                ).mean(dim=1)

    # E074A: robot control trust-region guard.
    ctrl_ref_guard_rew = torch.zeros(N, device=config.device)
    if config.ctrl_ref_guard_scale > 0.0:
        ctrl_sim = wp.to_torch(env.data_wp.ctrl)
        ctrl_dim = min(ctrl_sim.shape[1], ctrl_ref.shape[0])
        guard_dim = ctrl_dim
        if config.ctrl_ref_guard_robot_only:
            obj_dims = (
                int(config.object_action_dims)
                if config.object_action_dims > 0
                else (6 if config.contact_guidance and ctrl_dim > 29 else 0)
            )
            guard_dim = max(ctrl_dim - obj_dims, 0)
        if guard_dim > 0:
            diff = ctrl_sim[:, :guard_dim] - ctrl_ref[:guard_dim].unsqueeze(0)
            sigma = max(float(config.ctrl_ref_guard_sigma), 1e-6)
            abs_scaled = torch.abs(diff) / sigma
            huber = torch.where(
                abs_scaled <= 1.0,
                0.5 * abs_scaled * abs_scaled,
                abs_scaled - 0.5,
            )
            time_arr = wp.to_torch(env.data_wp.time)
            gate = (
                (time_arr >= config.ctrl_ref_guard_start_eval_time)
                & (time_arr <= config.ctrl_ref_guard_end_eval_time)
            ).to(ctrl_sim.dtype)
            ctrl_ref_guard_rew = -config.ctrl_ref_guard_scale * huber.mean(dim=1) * gate

    # E074C: maintain near-field hand/object contact during reference hold phase.
    hold_contact_rew = torch.zeros(N, device=config.device)
    if (
        config.hold_contact_rew_scale > 0.0
        and config.hand_approach_body_ids
        and config.hand_approach_obj_half_extents
    ):
        obj_body_id = mujoco.mj_name2id(
            env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "object"
        )
        if obj_body_id != -1:
            xpos_sim = wp.to_torch(env.data_wp.xpos)
            xquat_sim = wp.to_torch(env.data_wp.xquat)
            obj_pos = xpos_sim[:, obj_body_id]
            obj_quat = xquat_sim[:, obj_body_id]
            hand_pos = xpos_sim[:, config.hand_approach_body_ids]
            local = _lf_quat_apply_inverse(
                obj_quat.unsqueeze(1).expand(-1, hand_pos.shape[1], -1),
                hand_pos - obj_pos.unsqueeze(1),
            )
            half_ext = torch.tensor(
                config.hand_approach_obj_half_extents,
                device=config.device,
                dtype=hand_pos.dtype,
            )
            clamped = torch.clamp(local, -half_ext, half_ext)
            dist = (local - clamped).norm(dim=-1)
            min_dist = dist.min(dim=1).values
            time_arr = wp.to_torch(env.data_wp.time)
            time_gate = (
                (time_arr >= config.hold_contact_start_eval_time)
                & (time_arr <= config.hold_contact_end_eval_time)
            ).to(hand_pos.dtype)
            if config.hold_contact_require_ref_contact:
                ref_gate = (
                    approach_mask_val
                    if torch.is_tensor(approach_mask_val)
                    else torch.tensor(
                        approach_mask_val, device=config.device, dtype=hand_pos.dtype
                    )
                )
                ref_gate = ref_gate.to(device=config.device, dtype=hand_pos.dtype)
                if ref_gate.ndim == 1 and ref_gate.shape[0] == hand_pos.shape[1]:
                    ref_gate = ref_gate.max().expand_as(time_gate)
                elif ref_gate.ndim == 2 and ref_gate.shape[1] == hand_pos.shape[1]:
                    ref_gate = ref_gate[0].max().expand_as(time_gate)
                elif ref_gate.ndim > 1:
                    ref_gate = ref_gate.reshape(-1)[0].expand_as(time_gate)
            else:
                ref_gate = torch.ones_like(time_gate)
            sigma = max(float(config.hold_contact_sigma), 1e-6)
            hold_contact_rew = (
                config.hold_contact_rew_scale
                * torch.exp(-min_dist / sigma)
                * time_gate
                * ref_gate
            )

    robot_object_penalty = torch.zeros(N, device=config.device)
    leg_object_penalty = torch.zeros(N, device=config.device)
    leg_object_penalty_gate = torch.ones(N, device=config.device)
    hand_floor_penalty = torch.zeros(N, device=config.device)
    hand_object_deep_penalty = torch.zeros(N, device=config.device)
    object_lift_rew = torch.zeros(N, device=config.device)
    object_floor_penalty = torch.zeros(N, device=config.device)
    cem_gate_min_sdf = torch.zeros(N, device=config.device)
    cem_gate_violation = torch.zeros(N, device=config.device)
    cem_gate_violation_depth = torch.zeros(N, device=config.device)
    cem_body_gate_min_sdf = torch.zeros(N, device=config.device)
    cem_body_gate_violation = torch.zeros(N, device=config.device)
    cem_body_gate_violation_depth = torch.zeros(N, device=config.device)
    cem_hand_gate_min_sdf = torch.zeros(N, device=config.device)
    cem_hand_gate_violation = torch.zeros(N, device=config.device)
    cem_hand_gate_violation_depth = torch.zeros(N, device=config.device)
    cem_leg_gate_min_sdf = torch.zeros(N, device=config.device)
    cem_leg_gate_violation = torch.zeros(N, device=config.device)
    cem_leg_gate_violation_depth = torch.zeros(N, device=config.device)
    object_clearance_rew = torch.zeros(N, device=config.device)
    object_clearance_penalty = torch.zeros(N, device=config.device)
    object_clearance_m = torch.zeros(N, device=config.device)
    carry_corridor_rew = torch.zeros(N, device=config.device)
    carry_corridor_gate = torch.ones(N, device=config.device)
    carry_corridor_hand_score = torch.ones(N, device=config.device)
    carry_corridor_clearance_score = torch.ones(N, device=config.device)
    carry_corridor_pelvis_score = torch.ones(N, device=config.device)
    carry_corridor_rot_score = torch.ones(N, device=config.device)
    carry_corridor_leg_score = torch.ones(N, device=config.device)
    hand_support_rew = torch.zeros(N, device=config.device)
    hand_support_gate = torch.ones(N, device=config.device)
    hand_support_sdf = torch.zeros(N, device=config.device)
    hand_support_score = torch.ones(N, device=config.device)
    surface_band_rew = torch.zeros(N, device=config.device)
    surface_band_penalty = torch.zeros(N, device=config.device)
    surface_band_gate = torch.ones(N, device=config.device)
    surface_band_sdf = torch.zeros(N, device=config.device)
    surface_band_score = torch.zeros(N, device=config.device)
    surface_band_penetration = torch.zeros(N, device=config.device)
    surface_band_decay_factor = torch.ones(N, device=config.device)
    surface_band_left_sdf = torch.zeros(N, device=config.device)
    surface_band_right_sdf = torch.zeros(N, device=config.device)
    surface_band_left_score = torch.zeros(N, device=config.device)
    surface_band_right_score = torch.zeros(N, device=config.device)
    surface_band_bimanual_gate = torch.zeros(N, device=config.device)
    surface_band_bimanual_score = torch.zeros(N, device=config.device)
    cem_posture_z_err = torch.zeros(N, device=config.device)
    cem_posture_z_drop = torch.zeros(N, device=config.device)
    cem_peak_margin_ee_body_err = torch.zeros(N, device=config.device)
    cem_peak_margin_anchor_pos_err = torch.zeros(N, device=config.device)
    cem_peak_margin_anchor_ori_err = torch.zeros(N, device=config.device)
    e166_aux_info: dict[str, torch.Tensor] = {}
    nonhand_support_penalty = torch.zeros(N, device=config.device)
    nonhand_support_gate = torch.ones(N, device=config.device)
    nonhand_support_sdf = torch.zeros(N, device=config.device)
    nonhand_support_violation = torch.zeros(N, device=config.device)
    terminal_carry_gate_penalty = torch.zeros(N, device=config.device)
    terminal_carry_gate_violation = torch.zeros(N, device=config.device)
    terminal_carry_gate_valid = torch.ones(N, device=config.device)
    terminal_carry_gate_pelvis_z = torch.zeros(N, device=config.device)
    terminal_carry_gate_obj_rot_err = torch.zeros(N, device=config.device)
    terminal_carry_gate_nonhand_sdf = torch.zeros(N, device=config.device)
    terminal_carry_gate_hand_near_frac = torch.zeros(N, device=config.device)
    if (
        (
            config.robot_object_penalty_scale > 0.0
            and config.robot_object_penalty_geom_ids
        )
        or (
            config.leg_object_penalty_scale > 0.0 and config.leg_object_penalty_geom_ids
        )
        or (
            config.hand_object_deep_penalty_scale > 0.0
            and config.hand_object_deep_penalty_geom_ids
        )
        or (
            config.object_lift_rew_scale > 0.0
            or config.object_floor_penalty_scale > 0.0
        )
        or (config.cem_safety_gate_enabled and config.cem_safety_gate_geom_ids)
        or (config.cem_hand_gate_enabled and config.cem_hand_gate_geom_ids)
        or (config.cem_leg_gate_enabled and config.cem_leg_gate_geom_ids)
        or (
            config.object_clearance_rew_scale > 0.0
            or config.object_clearance_penalty_scale > 0.0
        )
        or config.carry_corridor_rew_scale > 0.0
        or (config.hand_support_rew_scale > 0.0 and config.hand_support_geom_ids)
        or (
            config.nonhand_support_penalty_scale > 0.0
            and config.nonhand_support_penalty_geom_ids
        )
    ) and config.hand_approach_obj_half_extents:
        object_geom_ids = list(config.object_collision_geom_ids)
        if not object_geom_ids:
            object_geom_ids = _resolve_object_collision_geom_ids(
                env.model_cpu, config.object_collision_sdf_mode
            )
        object_geom_id = object_geom_ids[0] if object_geom_ids else -1
        if object_geom_id != -1:
            geom_xpos = wp.to_torch(env.data_wp.geom_xpos)
            geom_xmat = wp.to_torch(env.data_wp.geom_xmat).reshape(
                geom_xpos.shape[0], geom_xpos.shape[1], 3, 3
            )
            use_single_box_sdf = _uses_single_box_sdf(env, object_geom_ids)
            body_xpos = None
            body_xmat = None
            if config.object_distance_backend == "grid_sdf":
                body_xpos = wp.to_torch(env.data_wp.xpos)
                body_xmat = wp.to_torch(env.data_wp.xmat).reshape(
                    body_xpos.shape[0], -1, 3, 3
                )
            if config.object_collision_sdf_batch_groups:
                object_sdf_cache = _object_distance_group_cache(
                    config,
                    env,
                    _object_sdf_geom_groups_for_tick(config),
                    object_geom_ids,
                    geom_xpos=geom_xpos,
                    geom_xmat=geom_xmat,
                    body_xpos=body_xpos,
                    body_xmat=body_xmat,
                )
            else:
                object_sdf_cache = {}

            def geom_object_sdf_min(
                geom_ids: list[int], *, conservative: bool = False
            ) -> torch.Tensor:
                if use_single_box_sdf:
                    # Preserve E172/E173 single-box reward and gate numerics.
                    # Compound box proxies and all non-box objects stay on the
                    # cache-backed object-distance implementation below.
                    return _geom_box_sdf_min(
                        config,
                        env,
                        geom_ids,
                        object_geom_id,
                        geom_xpos=geom_xpos,
                        geom_xmat=geom_xmat,
                    )
                return _cached_object_distance_sdf_min(
                    object_sdf_cache,
                    config,
                    env,
                    geom_ids,
                    object_geom_ids,
                    geom_xpos=geom_xpos,
                    geom_xmat=geom_xmat,
                    body_xpos=body_xpos,
                    body_xmat=body_xmat,
                    conservative=conservative,
                )

            def support_gate(
                source: str,
                start_eval_time: float,
                end_eval_time: float,
                dtype: torch.dtype,
            ) -> torch.Tensor:
                if source == "always":
                    return torch.ones(N, device=config.device, dtype=dtype)
                valid_sources = {
                    "contact_mask",
                    "contact_mask_strict_current",
                    "time_window",
                    "contact_mask_time_window",
                }
                if source not in valid_sources:
                    raise ValueError(f"Unsupported support gate source={source!r}")
                gates = []
                if source == "contact_mask_strict_current":
                    ref_gate = (
                        approach_mask_val
                        if torch.is_tensor(approach_mask_val)
                        else torch.tensor(
                            approach_mask_val, device=config.device, dtype=dtype
                        )
                    )
                    ref_gate = ref_gate.to(device=config.device, dtype=dtype)
                    if ref_gate.ndim == 0:
                        gates.append(ref_gate.view(1).expand(N))
                    elif ref_gate.ndim == 1:
                        gates.append(ref_gate[0].view(1).expand(N))
                    elif ref_gate.ndim == 2:
                        gates.append(ref_gate[0].max().view(1).expand(N))
                    else:
                        gates.append(ref_gate.reshape(-1)[0].view(1).expand(N))
                if source in {"contact_mask", "contact_mask_time_window"}:
                    gates.append(
                        _sample_gate_from_ref_mask(
                            approach_mask_val,
                            N,
                            config.device,
                            dtype,
                        )
                    )
                if source in {"time_window", "contact_mask_time_window"}:
                    time_arr = wp.to_torch(env.data_wp.time)
                    gates.append(
                        (
                            (time_arr >= start_eval_time) & (time_arr <= end_eval_time)
                        ).to(dtype)
                    )
                gate = torch.ones(N, device=config.device, dtype=dtype)
                for g in gates:
                    gate = gate * g
                return gate

            if (
                config.robot_object_penalty_scale > 0.0
                and config.robot_object_penalty_geom_ids
            ):
                robot_sdf = geom_object_sdf_min(config.robot_object_penalty_geom_ids)
                deep_limit = (
                    config.robot_object_penalty_margin_m
                    - config.robot_object_penalty_deep_threshold_m
                )
                robot_hinge = torch.clamp(deep_limit - robot_sdf, min=0.0)
                robot_object_penalty = -config.robot_object_penalty_scale * robot_hinge
            if (
                config.leg_object_penalty_scale > 0.0
                and config.leg_object_penalty_geom_ids
            ):
                leg_sdf = geom_object_sdf_min(config.leg_object_penalty_geom_ids)
                leg_hinge = torch.clamp(
                    config.leg_object_penalty_margin_m - leg_sdf, min=0.0
                )
                gate_source = config.leg_object_penalty_gate_source
                leg_object_penalty_gate = torch.ones_like(leg_hinge)
                if gate_source != "always":
                    valid_sources = {
                        "contact_mask",
                        "time_window",
                        "contact_mask_time_window",
                        "hand_target",
                        "contact_mask_and_hand_target",
                    }
                    if gate_source not in valid_sources:
                        raise ValueError(
                            f"Unsupported leg_object_penalty_gate_source={gate_source!r}"
                        )
                    gates = []
                    if gate_source in {
                        "contact_mask",
                        "contact_mask_time_window",
                        "contact_mask_and_hand_target",
                    }:
                        gates.append(
                            _sample_gate_from_ref_mask(
                                approach_mask_val,
                                N,
                                config.device,
                                leg_hinge.dtype,
                            )
                        )
                    if gate_source in {"time_window", "contact_mask_time_window"}:
                        time_arr = wp.to_torch(env.data_wp.time)
                        gates.append(
                            (
                                (time_arr >= config.leg_object_penalty_start_eval_time)
                                & (time_arr <= config.leg_object_penalty_end_eval_time)
                            ).to(leg_hinge.dtype)
                        )
                    if gate_source in {"hand_target", "contact_mask_and_hand_target"}:
                        obj_body_id = mujoco.mj_name2id(
                            env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "object"
                        )
                        if obj_body_id == -1 or not config.hand_approach_body_ids:
                            raise ValueError(
                                "hand_target leg-object gate requires object body and hand_approach_body_ids"
                            )
                        if contact_target_dynamic is None and not (
                            config.contact_hdmi_target_left
                            and config.contact_hdmi_target_right
                        ):
                            raise ValueError(
                                "hand_target leg-object gate requires dynamic or fixed contact_hdmi targets"
                            )
                        xpos_sim = wp.to_torch(env.data_wp.xpos)
                        xquat_sim = wp.to_torch(env.data_wp.xquat)
                        body_obj_pos = xpos_sim[:, obj_body_id]
                        body_obj_quat = xquat_sim[:, obj_body_id]
                        eef_offset = torch.tensor(
                            config.contact_hdmi_eef_offset,
                            device=config.device,
                            dtype=leg_hinge.dtype,
                        )
                        if contact_target_dynamic is not None:
                            targets = [
                                contact_target_dynamic[ei].to(
                                    device=config.device, dtype=leg_hinge.dtype
                                )
                                for ei in range(len(config.hand_approach_body_ids))
                            ]
                        else:
                            targets = [
                                torch.tensor(
                                    config.contact_hdmi_target_left,
                                    device=config.device,
                                    dtype=leg_hinge.dtype,
                                ),
                                torch.tensor(
                                    config.contact_hdmi_target_right,
                                    device=config.device,
                                    dtype=leg_hinge.dtype,
                                ),
                            ]
                        dist_per_eef = []
                        for bid, target_off in zip(
                            config.hand_approach_body_ids, targets
                        ):
                            target_world = body_obj_pos + _lf_quat_apply(
                                body_obj_quat, target_off.unsqueeze(0).expand(N, -1)
                            )
                            eef_pos = xpos_sim[:, bid]
                            eef_quat = xquat_sim[:, bid]
                            contact_point = eef_pos + _lf_quat_apply(
                                eef_quat, eef_offset.unsqueeze(0).expand(N, -1)
                            )
                            dist_per_eef.append(
                                (target_world - contact_point).norm(dim=-1)
                            )
                        dist_stack = torch.stack(dist_per_eef, dim=1)
                        hand_success = (
                            dist_stack
                            <= config.leg_object_penalty_hand_target_threshold_m
                        ).to(leg_hinge.dtype)
                        if gate_source == "contact_mask_and_hand_target":
                            hand_success = hand_success * _per_eef_mask_from_ref_mask(
                                approach_mask_val,
                                N,
                                dist_stack.shape[1],
                                config.device,
                                leg_hinge.dtype,
                            )
                        gates.append(hand_success.max(dim=1).values)
                    for gate in gates:
                        leg_object_penalty_gate = leg_object_penalty_gate * gate
                leg_object_penalty = (
                    -config.leg_object_penalty_scale
                    * leg_hinge
                    * leg_object_penalty_gate
                )
            if (
                config.hand_object_deep_penalty_scale > 0.0
                and config.hand_object_deep_penalty_geom_ids
            ):
                hand_sdf = geom_object_sdf_min(config.hand_object_deep_penalty_geom_ids)
                deep_hinge = torch.clamp(
                    -config.hand_object_deep_penalty_threshold_m - hand_sdf,
                    min=0.0,
                )
                hand_object_deep_penalty = (
                    -config.hand_object_deep_penalty_scale * deep_hinge
                )
            if config.hand_support_rew_scale > 0.0 and config.hand_support_geom_ids:
                hand_support_sdf = geom_object_sdf_min(config.hand_support_geom_ids)
                hand_err = torch.clamp(
                    torch.abs(hand_support_sdf) - config.hand_support_margin_m,
                    min=0.0,
                )
                hand_support_score = torch.exp(
                    -hand_err / max(float(config.hand_support_sigma), 1e-6)
                )
                hand_support_gate = support_gate(
                    config.hand_support_gate_source,
                    config.hand_support_start_eval_time,
                    config.hand_support_end_eval_time,
                    hand_support_score.dtype,
                )
                hand_support_rew = (
                    config.hand_support_rew_scale
                    * hand_support_score
                    * hand_support_gate
                )
                # E155-D: neutral baseline when gate=0 (like contact_hdmi_rew)
                if config.hand_support_neutral_baseline > 0.0:
                    hand_support_rew = (
                        hand_support_rew
                        + config.hand_support_neutral_baseline
                        * (1.0 - hand_support_gate)
                    )
                # E155-C: tail decay (last decay_frac of episode)
                if config.hand_support_decay_frac > 0.0:
                    time_arr = wp.to_torch(env.data_wp.time)
                    total_time = float(config.max_sim_steps) * config.sim_dt
                    decay_start = total_time * (1.0 - config.hand_support_decay_frac)
                    if total_time > decay_start:
                        decay_factor = torch.clamp(
                            (total_time - time_arr) / (total_time - decay_start),
                            0.0,
                            1.0,
                        )
                        hand_support_rew = hand_support_rew * decay_factor
            if (
                config.surface_band_rew_scale > 0.0
                or config.surface_band_penalty_scale > 0.0
            ) and config.surface_band_geom_ids:
                band_width = float(config.surface_band_width_m)
                band_min_sdf = float(config.surface_band_min_sdf_m)
                sigma = float(config.surface_band_sigma)

                def surface_score_raw(sdf: torch.Tensor) -> torch.Tensor:
                    return surface_distance_score(
                        sdf,
                        mode=config.surface_band_score_mode,
                        sigma_m=sigma,
                        continuation_far_weight=float(
                            config.surface_band_continuation_far_weight
                        ),
                        continuation_near_weight=float(
                            config.surface_band_continuation_near_weight
                        ),
                        continuation_far_scale_m=float(
                            config.surface_band_continuation_far_scale_m
                        ),
                        continuation_near_scale_m=float(
                            config.surface_band_continuation_near_scale_m
                        ),
                        continuation_smooth_delta_m=float(
                            config.surface_band_continuation_smooth_delta_m
                        ),
                    )

                def surface_support_mask(sdf: torch.Tensor) -> torch.Tensor:
                    return surface_distance_support_mask(
                        sdf,
                        mode=config.surface_band_score_mode,
                        band_min_sdf_m=band_min_sdf,
                        band_width_m=band_width,
                    )

                if config.surface_band_bimanual_required:
                    if (
                        not config.surface_band_left_geom_ids
                        or not config.surface_band_right_geom_ids
                    ):
                        raise ValueError(
                            "surface_band_bimanual_required requires left/right surface-band geoms"
                        )
                    if config.surface_band_bimanual_score_reduce != "min":
                        raise ValueError(
                            "Unsupported surface_band_bimanual_score_reduce="
                            f"{config.surface_band_bimanual_score_reduce!r}"
                        )
                    surface_band_left_sdf = geom_object_sdf_min(
                        config.surface_band_left_geom_ids
                    )
                    surface_band_right_sdf = geom_object_sdf_min(
                        config.surface_band_right_geom_ids
                    )
                    left_in_band = surface_support_mask(surface_band_left_sdf)
                    right_in_band = surface_support_mask(surface_band_right_sdf)
                    both_in_band = left_in_band & right_in_band
                    surface_band_left_score = surface_score_raw(surface_band_left_sdf)
                    surface_band_right_score = surface_score_raw(surface_band_right_sdf)
                    surface_band_bimanual_gate = both_in_band.to(
                        surface_band_left_score.dtype
                    )
                    surface_band_bimanual_score = torch.minimum(
                        surface_band_left_score, surface_band_right_score
                    )
                    surface_band_score = torch.where(
                        both_in_band,
                        surface_band_bimanual_score,
                        torch.zeros_like(surface_band_bimanual_score),
                    )
                    # Report the bimanual bottleneck distance; penetration uses
                    # the deepest hand below so one over-penetrating hand is visible.
                    surface_band_sdf = torch.maximum(
                        surface_band_left_sdf, surface_band_right_sdf
                    )
                    surface_band_penetration_sdf = torch.minimum(
                        surface_band_left_sdf, surface_band_right_sdf
                    )
                else:
                    surface_band_sdf = geom_object_sdf_min(config.surface_band_geom_ids)
                    in_band = surface_support_mask(surface_band_sdf)
                    surface_band_score_raw = surface_score_raw(surface_band_sdf)
                    surface_band_score = torch.where(
                        in_band,
                        surface_band_score_raw,
                        torch.zeros_like(surface_band_sdf),
                    )
                    surface_band_penetration_sdf = surface_band_sdf
                surface_band_gate = support_gate(
                    config.surface_band_gate_source,
                    config.surface_band_start_eval_time,
                    config.surface_band_end_eval_time,
                    surface_band_score.dtype,
                )
                surface_band_rew = (
                    config.surface_band_rew_scale
                    * surface_band_score
                    * surface_band_gate
                )
                surface_band_penetration = torch.clamp(
                    -surface_band_penetration_sdf
                    - config.surface_band_penetration_tol_m,
                    min=0.0,
                )
                surface_band_penalty = (
                    -config.surface_band_penalty_scale
                    * surface_band_penetration
                    * surface_band_gate
                )
                if config.surface_band_decay_frac > 0.0:
                    time_arr = wp.to_torch(env.data_wp.time)
                    total_time = float(config.max_sim_steps) * config.sim_dt
                    decay_frac = min(
                        max(float(config.surface_band_decay_frac), 0.0), 1.0
                    )
                    decay_start = total_time * (1.0 - decay_frac)
                    if total_time > decay_start:
                        surface_band_decay_factor = torch.clamp(
                            (total_time - time_arr) / (total_time - decay_start),
                            0.0,
                            1.0,
                        )
                        surface_band_rew = surface_band_rew * surface_band_decay_factor
                        surface_band_penalty = (
                            surface_band_penalty * surface_band_decay_factor
                        )
            if (
                config.nonhand_support_penalty_scale > 0.0
                and config.nonhand_support_penalty_geom_ids
            ):
                nonhand_support_sdf = geom_object_sdf_min(
                    config.nonhand_support_penalty_geom_ids
                )
                nonhand_support_violation = torch.clamp(
                    config.nonhand_support_penalty_margin_m - nonhand_support_sdf,
                    min=0.0,
                )
                nonhand_support_gate = support_gate(
                    config.nonhand_support_penalty_gate_source,
                    config.nonhand_support_penalty_start_eval_time,
                    config.nonhand_support_penalty_end_eval_time,
                    nonhand_support_violation.dtype,
                )
                nonhand_support_penalty = (
                    -config.nonhand_support_penalty_scale
                    * nonhand_support_violation
                    * nonhand_support_gate
                )
            if config.cem_safety_gate_enabled and config.cem_safety_gate_geom_ids:
                cem_body_gate_min_sdf = geom_object_sdf_min(
                    config.cem_safety_gate_geom_ids,
                    conservative=True,
                )
                cem_body_gate_violation_depth = torch.clamp(
                    config.cem_safety_gate_min_sdf_m - cem_body_gate_min_sdf,
                    min=0.0,
                )
                cem_body_gate_violation = (cem_body_gate_violation_depth > 0.0).to(
                    geom_xpos.dtype
                )
                cem_gate_min_sdf = cem_body_gate_min_sdf
                cem_gate_violation_depth = cem_body_gate_violation_depth
                cem_gate_violation = cem_body_gate_violation
            if config.cem_hand_gate_enabled and config.cem_hand_gate_geom_ids:
                cem_hand_gate_min_sdf = geom_object_sdf_min(
                    config.cem_hand_gate_geom_ids,
                    conservative=True,
                )
                cem_hand_gate_violation_depth = torch.clamp(
                    config.cem_hand_gate_min_sdf_m - cem_hand_gate_min_sdf,
                    min=0.0,
                )
                cem_hand_gate_violation = (cem_hand_gate_violation_depth > 0.0).to(
                    geom_xpos.dtype
                )
                if config.cem_safety_gate_enabled and config.cem_safety_gate_geom_ids:
                    cem_gate_min_sdf = torch.minimum(
                        cem_gate_min_sdf, cem_hand_gate_min_sdf
                    )
                else:
                    cem_gate_min_sdf = cem_hand_gate_min_sdf
                cem_gate_violation_depth = torch.maximum(
                    cem_gate_violation_depth, cem_hand_gate_violation_depth
                )
                cem_gate_violation = torch.maximum(
                    cem_gate_violation, cem_hand_gate_violation
                )
            if config.cem_leg_gate_enabled and config.cem_leg_gate_geom_ids:
                cem_leg_gate_min_sdf = geom_object_sdf_min(
                    config.cem_leg_gate_geom_ids,
                    conservative=True,
                )
                cem_leg_gate_violation_depth = torch.clamp(
                    config.cem_leg_gate_min_sdf_m - cem_leg_gate_min_sdf,
                    min=0.0,
                )
                cem_leg_gate_violation = (cem_leg_gate_violation_depth > 0.0).to(
                    cem_leg_gate_min_sdf.dtype
                )
                if config.cem_safety_gate_enabled or config.cem_hand_gate_enabled:
                    cem_gate_min_sdf = torch.minimum(
                        cem_gate_min_sdf, cem_leg_gate_min_sdf
                    )
                else:
                    cem_gate_min_sdf = cem_leg_gate_min_sdf
                cem_gate_violation_depth = torch.maximum(
                    cem_gate_violation_depth, cem_leg_gate_violation_depth
                )
                cem_gate_violation = torch.maximum(
                    cem_gate_violation, cem_leg_gate_violation
                )
            if (
                config.object_lift_rew_scale > 0.0
                or config.object_floor_penalty_scale > 0.0
            ):
                obj_half_z = float(config.hand_approach_obj_half_extents[2])
                obj_bottom = geom_xpos[:, object_geom_id, 2] - obj_half_z
                if config.nq_obj == 7:
                    ref_obj_z = qpos_ref[-5]
                else:
                    ref_obj_z = qpos_ref[-4]
                ref_bottom = ref_obj_z - obj_half_z
                if config.object_lift_rew_scale > 0.0:
                    sigma = max(float(config.object_lift_sigma), 1e-6)
                    lift_err = torch.abs(obj_bottom - ref_bottom)
                    object_lift_rew = config.object_lift_rew_scale * torch.exp(
                        -lift_err / sigma
                    )
                if config.object_floor_penalty_scale > 0.0:
                    min_bottom = ref_bottom - config.object_floor_margin_m
                    floor_hinge = torch.clamp(min_bottom - obj_bottom, min=0.0)
                    object_floor_penalty = (
                        -config.object_floor_penalty_scale * floor_hinge
                    )
            if (
                config.object_clearance_rew_scale > 0.0
                or config.object_clearance_penalty_scale > 0.0
            ):
                obj_half_z = float(config.hand_approach_obj_half_extents[2])
                obj_bottom = geom_xpos[:, object_geom_id, 2] - obj_half_z
                object_clearance_m = obj_bottom - config.object_clearance_floor_z
                source = config.object_clearance_gate_source
                if source == "always":
                    window_gate = torch.ones_like(object_clearance_m)
                elif source == "time_window":
                    time_arr = wp.to_torch(env.data_wp.time)
                    window_gate = (
                        (time_arr >= config.object_clearance_start_eval_time)
                        & (time_arr <= config.object_clearance_end_eval_time)
                    ).to(object_clearance_m.dtype)
                elif source == "contact_mask":
                    window_gate = _sample_gate_from_ref_mask(
                        approach_mask_val,
                        N,
                        config.device,
                        object_clearance_m.dtype,
                    )
                else:
                    raise ValueError(
                        f"Unsupported object_clearance_gate_source={source!r}"
                    )
                if config.object_clearance_rew_scale > 0.0:
                    sigma = max(float(config.object_clearance_sigma), 1e-6)
                    target_mid = 0.5 * (
                        config.object_clearance_min_m + config.object_clearance_max_m
                    )
                    clearance_err = torch.abs(object_clearance_m - target_mid)
                    object_clearance_rew = (
                        config.object_clearance_rew_scale
                        * torch.exp(-clearance_err / sigma)
                        * window_gate
                    )
                if config.object_clearance_penalty_scale > 0.0:
                    below = torch.clamp(
                        config.object_clearance_min_m - object_clearance_m,
                        min=0.0,
                    )
                    above = torch.clamp(
                        object_clearance_m - config.object_clearance_max_m,
                        min=0.0,
                    )
                    object_clearance_penalty = (
                        -config.object_clearance_penalty_scale
                        * (below + config.object_clearance_above_weight * above)
                        * window_gate
                    )
            if config.carry_corridor_rew_scale > 0.0:
                source = config.carry_corridor_gate_source
                if source == "contact_mask":
                    carry_corridor_gate = _sample_gate_from_ref_mask(
                        approach_mask_val,
                        N,
                        config.device,
                        geom_xpos.dtype,
                    )
                elif source == "time_window":
                    time_arr = wp.to_torch(env.data_wp.time)
                    carry_corridor_gate = (
                        (time_arr >= config.carry_corridor_start_eval_time)
                        & (time_arr <= config.carry_corridor_end_eval_time)
                    ).to(geom_xpos.dtype)
                elif source == "contact_mask_time_window":
                    time_arr = wp.to_torch(env.data_wp.time)
                    mask_gate = _sample_gate_from_ref_mask(
                        approach_mask_val,
                        N,
                        config.device,
                        geom_xpos.dtype,
                    )
                    time_gate = (
                        (time_arr >= config.carry_corridor_start_eval_time)
                        & (time_arr <= config.carry_corridor_end_eval_time)
                    ).to(geom_xpos.dtype)
                    carry_corridor_gate = mask_gate * time_gate
                else:
                    raise ValueError(
                        f"Unsupported carry_corridor_gate_source={source!r}"
                    )

                obj_body_id = mujoco.mj_name2id(
                    env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "object"
                )
                if obj_body_id == -1 or not config.hand_approach_body_ids:
                    raise ValueError(
                        "carry_corridor requires object body and hand_approach_body_ids"
                    )
                if contact_target_dynamic is None and not (
                    config.contact_hdmi_target_left and config.contact_hdmi_target_right
                ):
                    raise ValueError(
                        "carry_corridor requires dynamic or fixed contact_hdmi targets"
                    )

                xpos_sim = wp.to_torch(env.data_wp.xpos)
                xquat_sim = wp.to_torch(env.data_wp.xquat)
                body_obj_pos = xpos_sim[:, obj_body_id]
                body_obj_quat = xquat_sim[:, obj_body_id]
                eef_offset = torch.tensor(
                    config.contact_hdmi_eef_offset,
                    device=config.device,
                    dtype=geom_xpos.dtype,
                )
                if contact_target_dynamic is not None:
                    targets = [
                        contact_target_dynamic[ei].to(
                            device=config.device, dtype=geom_xpos.dtype
                        )
                        for ei in range(len(config.hand_approach_body_ids))
                    ]
                else:
                    targets = [
                        torch.tensor(
                            config.contact_hdmi_target_left,
                            device=config.device,
                            dtype=geom_xpos.dtype,
                        ),
                        torch.tensor(
                            config.contact_hdmi_target_right,
                            device=config.device,
                            dtype=geom_xpos.dtype,
                        ),
                    ]
                dist_per_eef = []
                for bid, target_off in zip(config.hand_approach_body_ids, targets):
                    target_world = body_obj_pos + _lf_quat_apply(
                        body_obj_quat, target_off.unsqueeze(0).expand(N, -1)
                    )
                    eef_pos = xpos_sim[:, bid]
                    eef_quat = xquat_sim[:, bid]
                    contact_point = eef_pos + _lf_quat_apply(
                        eef_quat, eef_offset.unsqueeze(0).expand(N, -1)
                    )
                    dist_per_eef.append((target_world - contact_point).norm(dim=-1))
                min_hand_dist = torch.stack(dist_per_eef, dim=1).min(dim=1).values
                hand_excess = torch.clamp(
                    min_hand_dist - config.carry_corridor_hand_target_threshold_m,
                    min=0.0,
                )
                carry_corridor_hand_score = torch.exp(
                    -hand_excess / max(float(config.carry_corridor_hand_sigma), 1e-6)
                )

                obj_half_z = float(config.hand_approach_obj_half_extents[2])
                obj_bottom = geom_xpos[:, object_geom_id, 2] - obj_half_z
                clearance_m = obj_bottom - config.object_clearance_floor_z
                clearance_below = torch.clamp(
                    config.carry_corridor_clearance_min_m - clearance_m,
                    min=0.0,
                )
                clearance_above = torch.clamp(
                    clearance_m - config.carry_corridor_clearance_max_m,
                    min=0.0,
                )
                clearance_err = clearance_below + clearance_above
                carry_corridor_clearance_score = torch.exp(
                    -clearance_err
                    / max(float(config.carry_corridor_clearance_sigma), 1e-6)
                )

                pelvis_z = xpos_sim[:, 1, 2]
                pelvis_err = torch.clamp(
                    config.carry_corridor_pelvis_min_m - pelvis_z,
                    min=0.0,
                )
                carry_corridor_pelvis_score = torch.exp(
                    -pelvis_err / max(float(config.carry_corridor_pelvis_sigma), 1e-6)
                )

                if config.nq_obj == 7:
                    obj_quat_sim = qpos_sim[:, -4:]
                    obj_quat_ref = qpos_ref[-4:].unsqueeze(0).expand(N, -1)
                    rot_err = quat_sub(obj_quat_sim, obj_quat_ref).norm(dim=-1)
                else:
                    rot_err = (qpos_sim[:, -3:] - qpos_ref[-3:].unsqueeze(0)).norm(
                        dim=-1
                    )
                carry_corridor_rot_score = torch.exp(
                    -rot_err / max(float(config.carry_corridor_rot_sigma), 1e-6)
                )

                if config.carry_corridor_leg_geom_ids:
                    corridor_leg_sdf = geom_object_sdf_min(
                        config.carry_corridor_leg_geom_ids
                    )
                    leg_excess = torch.clamp(
                        config.carry_corridor_leg_margin_m - corridor_leg_sdf,
                        min=0.0,
                    )
                    carry_corridor_leg_score = torch.exp(
                        -leg_excess / max(float(config.carry_corridor_leg_sigma), 1e-6)
                    )

                carry_corridor_rew = (
                    config.carry_corridor_rew_scale
                    * carry_corridor_gate
                    * carry_corridor_hand_score
                    * carry_corridor_clearance_score
                    * carry_corridor_pelvis_score
                    * carry_corridor_rot_score
                    * carry_corridor_leg_score
                )

    if config.hand_floor_penalty_scale > 0.0 and config.hand_floor_penalty_geom_ids:
        geom_xpos = wp.to_torch(env.data_wp.geom_xpos)
        radii = torch.tensor(
            [
                float(env.model_cpu.geom_size[gid, 0])
                for gid in config.hand_floor_penalty_geom_ids
            ],
            device=config.device,
            dtype=geom_xpos.dtype,
        )
        centers_z = geom_xpos[:, config.hand_floor_penalty_geom_ids, 2]
        clearance = centers_z - radii.view(1, -1)
        hinge = torch.clamp(config.hand_floor_penalty_margin_m - clearance, min=0.0)
        hand_floor_penalty = -config.hand_floor_penalty_scale * hinge.sum(dim=1)

    reward = (
        qpos_rew
        + qvel_rew
        + contact_rew
        + task_body_rew
        + task_obj_rew
        + interact_rew
        + hand_approach_rew
        + contact_mask_rew
        + contact_hdmi_rew
        + ctrl_ref_guard_rew
        + hold_contact_rew
        + robot_object_penalty
        + leg_object_penalty
        + hand_floor_penalty
        + hand_object_deep_penalty
        + object_lift_rew
        + object_floor_penalty
        + object_clearance_rew
        + object_clearance_penalty
        + carry_corridor_rew
        + hand_support_rew
        + surface_band_rew
        + surface_band_penalty
        + nonhand_support_penalty
    )

    # E034: stability penalty — penalize when pelvis z drops below threshold
    stability_penalty = torch.zeros(N, device=config.device)
    if config.stability_penalty_scale > 0.0:
        xpos_sim = wp.to_torch(env.data_wp.xpos)  # (N, nbody, 3)
        pelvis_z = xpos_sim[:, 1, 2]  # body 1 is typically pelvis/torso
        below = torch.clamp(config.stability_penalty_threshold - pelvis_z, min=0.0)
        stability_penalty = -config.stability_penalty_scale * below
        reward = reward + stability_penalty
    if config.cem_posture_gate_enabled and qpos_sim.shape[1] >= 3:
        sim_root_z = qpos_sim[:, 2]
        ref_root_z = qpos_ref[2]
        cem_posture_z_err = torch.abs(sim_root_z - ref_root_z)
        cem_posture_z_drop = ref_root_z - sim_root_z
    if (
        config.cem_peak_margin_enabled
        and config.cem_peak_margin_ee_body_ids
        and config.cem_peak_margin_anchor_body_id >= 0
        and body_xpos_ref is not None
        and body_xpos_ref.shape[0] > config.cem_peak_margin_anchor_body_id
        and body_xpos_ref.shape[0] > max(config.cem_peak_margin_ee_body_ids)
    ):
        xpos_sim = wp.to_torch(env.data_wp.xpos)
        anchor_id = int(config.cem_peak_margin_anchor_body_id)
        ee_ids = config.cem_peak_margin_ee_body_ids
        ref_anchor_pos = body_xpos_ref[anchor_id].to(
            device=config.device, dtype=xpos_sim.dtype
        )
        ref_ee_pos = body_xpos_ref[ee_ids].to(
            device=config.device, dtype=xpos_sim.dtype
        )
        anchor_pos = xpos_sim[:, anchor_id]
        ee_pos = xpos_sim[:, ee_ids]
        cem_peak_margin_anchor_pos_err = (
            anchor_pos - ref_anchor_pos.unsqueeze(0)
        ).norm(dim=-1)
        if body_xquat_ref is not None and body_xquat_ref.shape[0] > anchor_id:
            xquat_sim = wp.to_torch(env.data_wp.xquat)
            anchor_yaw = _lf_yaw_quat(xquat_sim[:, anchor_id])
            ref_anchor_quat = body_xquat_ref[anchor_id].to(
                device=config.device, dtype=xquat_sim.dtype
            )
            ref_anchor_yaw = _lf_yaw_quat(ref_anchor_quat.unsqueeze(0)).squeeze(0)
            sim_local = _lf_quat_apply_inverse(
                anchor_yaw.unsqueeze(1).expand(-1, len(ee_ids), -1),
                ee_pos - anchor_pos.unsqueeze(1),
            )
            ref_local = _lf_quat_apply_inverse(
                ref_anchor_yaw.unsqueeze(0).unsqueeze(0).expand(N, len(ee_ids), -1),
                ref_ee_pos.unsqueeze(0).expand(N, -1, -1)
                - ref_anchor_pos.unsqueeze(0).unsqueeze(0),
            )
            ee_err = (ref_local - sim_local).norm(dim=-1)
            anchor_quat = xquat_sim[:, anchor_id]
            ref_anchor_quat_batch = ref_anchor_quat.unsqueeze(0).expand(N, -1)
            anchor_diff = _lf_quat_mul(
                _lf_quat_conjugate(ref_anchor_quat_batch), anchor_quat
            )
            cem_peak_margin_anchor_ori_err = _lf_axis_angle_from_quat(anchor_diff).norm(
                dim=-1
            )
        else:
            ee_err = (ee_pos - ref_ee_pos.unsqueeze(0)).norm(dim=-1)
        cem_peak_margin_ee_body_err = ee_err.max(dim=1).values

    if config.cem_smooth_enabled and config.cem_smooth_body_ids:
        xpos_sim = wp.to_torch(env.data_wp.xpos)
        smooth_ids = [
            bid for bid in config.cem_smooth_body_ids if bid < xpos_sim.shape[1]
        ]
        if smooth_ids:
            e166_aux_info["cem_smooth_body_pos"] = xpos_sim[:, smooth_ids]

    if (
        config.e167_body_z_enabled
        and config.e167_body_z_ids
        and body_xpos_ref is not None
    ):
        xpos_sim = wp.to_torch(env.data_wp.xpos)
        body_ids = [
            bid
            for bid in config.e167_body_z_ids
            if bid < xpos_sim.shape[1] and bid < body_xpos_ref.shape[0]
        ]
        if body_ids:
            body_pos = xpos_sim[:, body_ids]
            body_ref = body_xpos_ref[body_ids].to(
                device=config.device, dtype=body_pos.dtype
            )
            e166_aux_info["e167_body_z_pos"] = body_pos
            e166_aux_info["e167_body_z_ref_pos"] = body_ref.unsqueeze(0).expand(
                N, -1, -1
            )

    if (
        config.e167_ground_z_enabled
        and config.e167_ground_z_ids
        and body_xpos_ref is not None
    ):
        xpos_sim = wp.to_torch(env.data_wp.xpos)
        ground_ids = [
            bid
            for bid in config.e167_ground_z_ids
            if bid < xpos_sim.shape[1] and bid < body_xpos_ref.shape[0]
        ]
        if ground_ids:
            ground_pos = xpos_sim[:, ground_ids]
            ground_ref = body_xpos_ref[ground_ids].to(
                device=config.device, dtype=ground_pos.dtype
            )
            e166_aux_info["e167_ground_z_pos"] = ground_pos
            e166_aux_info["e167_ground_z_ref_pos"] = ground_ref.unsqueeze(0).expand(
                N, -1, -1
            )

    if (
        (config.foot_slip_enabled or config.foot_ground_enabled)
        and config.local_frame_ankle_ids
        and body_xpos_ref is not None
    ):
        xpos_sim = wp.to_torch(env.data_wp.xpos)
        foot_ids = [
            bid
            for bid in config.local_frame_ankle_ids
            if bid < xpos_sim.shape[1] and bid < body_xpos_ref.shape[0]
        ]
        if foot_ids:
            foot_pos = xpos_sim[:, foot_ids]
            foot_ref = body_xpos_ref[foot_ids].to(
                device=config.device, dtype=foot_pos.dtype
            )
            e166_aux_info["foot_body_pos"] = foot_pos
            e166_aux_info["foot_body_ref_pos"] = foot_ref.unsqueeze(0).expand(N, -1, -1)

    info = {
        "qpos_dist": qpos_dist,
        "qvel_dist": qvel_dist,
        "qpos_rew": qpos_rew,
        "qvel_rew": qvel_rew,
        "task_body_rew": task_body_rew,
        "task_obj_rew": task_obj_rew,
        "interact_rew": interact_rew,
        "hand_approach_rew": hand_approach_rew,
        "contact_hdmi_rew": contact_hdmi_rew,
        "contact_hdmi_left_score": contact_hdmi_left_score,
        "contact_hdmi_right_score": contact_hdmi_right_score,
        "contact_hdmi_bimanual_gate": contact_hdmi_bimanual_gate,
        "contact_hdmi_bimanual_score": contact_hdmi_bimanual_score,
        "ctrl_ref_guard_rew": ctrl_ref_guard_rew,
        "hold_contact_rew": hold_contact_rew,
        "robot_object_penalty": robot_object_penalty,
        "leg_object_penalty": leg_object_penalty,
        "leg_object_penalty_gate": leg_object_penalty_gate,
        "hand_floor_penalty": hand_floor_penalty,
        "hand_object_deep_penalty": hand_object_deep_penalty,
        "object_lift_rew": object_lift_rew,
        "object_floor_penalty": object_floor_penalty,
        "cem_gate_min_sdf": cem_gate_min_sdf,
        "cem_gate_violation": cem_gate_violation,
        "cem_gate_violation_depth": cem_gate_violation_depth,
        "cem_body_gate_min_sdf": cem_body_gate_min_sdf,
        "cem_body_gate_violation": cem_body_gate_violation,
        "cem_body_gate_violation_depth": cem_body_gate_violation_depth,
        "cem_hand_gate_min_sdf": cem_hand_gate_min_sdf,
        "cem_hand_gate_violation": cem_hand_gate_violation,
        "cem_hand_gate_violation_depth": cem_hand_gate_violation_depth,
        "cem_leg_gate_min_sdf": cem_leg_gate_min_sdf,
        "cem_leg_gate_violation": cem_leg_gate_violation,
        "cem_leg_gate_violation_depth": cem_leg_gate_violation_depth,
        "object_clearance_rew": object_clearance_rew,
        "object_clearance_penalty": object_clearance_penalty,
        "object_clearance_m": object_clearance_m,
        "carry_corridor_rew": carry_corridor_rew,
        "carry_corridor_gate": carry_corridor_gate,
        "carry_corridor_hand_score": carry_corridor_hand_score,
        "carry_corridor_clearance_score": carry_corridor_clearance_score,
        "carry_corridor_pelvis_score": carry_corridor_pelvis_score,
        "carry_corridor_rot_score": carry_corridor_rot_score,
        "carry_corridor_leg_score": carry_corridor_leg_score,
        "hand_support_rew": hand_support_rew,
        "hand_support_gate": hand_support_gate,
        "hand_support_sdf": hand_support_sdf,
        "hand_support_score": hand_support_score,
        "surface_band_rew": surface_band_rew,
        "surface_band_penalty": surface_band_penalty,
        "surface_band_gate": surface_band_gate,
        "surface_band_decay_factor": surface_band_decay_factor,
        "surface_band_sdf": surface_band_sdf,
        "surface_band_score": surface_band_score,
        "surface_band_penetration": surface_band_penetration,
        "surface_band_left_sdf": surface_band_left_sdf,
        "surface_band_right_sdf": surface_band_right_sdf,
        "surface_band_left_score": surface_band_left_score,
        "surface_band_right_score": surface_band_right_score,
        "surface_band_bimanual_gate": surface_band_bimanual_gate,
        "surface_band_bimanual_score": surface_band_bimanual_score,
        "cem_posture_z_err": cem_posture_z_err,
        "cem_posture_z_drop": cem_posture_z_drop,
        "cem_peak_margin_ee_body_err": cem_peak_margin_ee_body_err,
        "cem_peak_margin_anchor_pos_err": cem_peak_margin_anchor_pos_err,
        "cem_peak_margin_anchor_ori_err": cem_peak_margin_anchor_ori_err,
        "nonhand_support_penalty": nonhand_support_penalty,
        "nonhand_support_gate": nonhand_support_gate,
        "nonhand_support_sdf": nonhand_support_sdf,
        "nonhand_support_violation": nonhand_support_violation,
        "terminal_carry_gate_penalty": terminal_carry_gate_penalty,
        "terminal_carry_gate_violation": terminal_carry_gate_violation,
        "terminal_carry_gate_valid": terminal_carry_gate_valid,
        "terminal_carry_gate_pelvis_z": terminal_carry_gate_pelvis_z,
        "terminal_carry_gate_obj_rot_err": terminal_carry_gate_obj_rot_err,
        "terminal_carry_gate_nonhand_sdf": terminal_carry_gate_nonhand_sdf,
        "terminal_carry_gate_hand_near_frac": terminal_carry_gate_hand_near_frac,
        **e166_aux_info,
    }
    # Do not serialize PRG diagnostics for methods where the corresponding
    # E170 feature is inactive.  These tensors are initialized above so the
    # reward implementation can remain branch-safe, but exposing their
    # zero/default values in every run makes a no-PRG artifact falsely look
    # as though lower-body PRG participated in optimization.
    if not (
        config.leg_object_penalty_scale > 0.0 and config.leg_object_penalty_geom_ids
    ):
        info.pop("leg_object_penalty", None)
        info.pop("leg_object_penalty_gate", None)
    if not (config.cem_leg_gate_enabled and config.cem_leg_gate_geom_ids):
        info.pop("cem_leg_gate_min_sdf", None)
        info.pop("cem_leg_gate_violation", None)
        info.pop("cem_leg_gate_violation_depth", None)
    return reward, info


def _terminal_carry_gate(
    config: Config,
    env: MJWPEnv,
    qpos_ref: torch.Tensor,
) -> dict[str, torch.Tensor]:
    qpos_sim = wp.to_torch(env.data_wp.qpos)
    geom_xpos = wp.to_torch(env.data_wp.geom_xpos)
    N = qpos_sim.shape[0]
    dtype = qpos_sim.dtype
    zeros = torch.zeros(N, device=config.device, dtype=dtype)
    ones = torch.ones(N, device=config.device, dtype=dtype)
    out = {
        "penalty": zeros,
        "violation": zeros,
        "valid": ones,
        "pelvis_z": zeros,
        "obj_rot_err": zeros,
        "nonhand_sdf": zeros,
        "hand_near_frac": zeros,
    }
    if not config.terminal_carry_gate_enabled:
        return out

    if config.terminal_carry_gate_mode not in {"soft", "hard", "hard_soft"}:
        raise ValueError(
            f"Unsupported terminal_carry_gate_mode={config.terminal_carry_gate_mode!r}"
        )

    pelvis_z = wp.to_torch(env.data_wp.xpos)[:, 1, 2]
    pelvis_violation = torch.clamp(
        config.terminal_carry_gate_pelvis_min_m - pelvis_z,
        min=0.0,
    )

    if config.nq_obj == 7:
        obj_rot_err = quat_sub(
            qpos_sim[:, -4:],
            qpos_ref[-4:].unsqueeze(0).expand(N, -1),
        ).norm(dim=-1)
    else:
        obj_rot_err = (qpos_sim[:, -3:] - qpos_ref[-3:].unsqueeze(0)).norm(dim=-1)
    rot_violation = torch.clamp(
        obj_rot_err - config.terminal_carry_gate_obj_rot_max_rad,
        min=0.0,
    )

    object_geom_ids = list(config.object_collision_geom_ids)
    if not object_geom_ids:
        object_geom_ids = _resolve_object_collision_geom_ids(
            env.model_cpu, config.object_collision_sdf_mode
        )
    object_geom_id = object_geom_ids[0] if object_geom_ids else -1
    nonhand_sdf = torch.full((N,), float("inf"), device=config.device, dtype=dtype)
    hand_near_frac = zeros
    nonhand_violation = zeros
    hand_violation = zeros
    if object_geom_id != -1 and config.hand_approach_obj_half_extents:
        geom_xmat = wp.to_torch(env.data_wp.geom_xmat).reshape(
            geom_xpos.shape[0], geom_xpos.shape[1], 3, 3
        )
        use_single_box_sdf = _uses_single_box_sdf(env, object_geom_ids)
        body_xpos = None
        body_xmat = None
        if config.object_distance_backend == "grid_sdf":
            body_xpos = wp.to_torch(env.data_wp.xpos)
            body_xmat = wp.to_torch(env.data_wp.xmat).reshape(
                body_xpos.shape[0], -1, 3, 3
            )
        object_sdf_cache: dict[tuple[int, ...], torch.Tensor] = {}

        def terminal_object_sdf(geom_ids: list[int]) -> torch.Tensor:
            if use_single_box_sdf:
                return _geom_box_sdf_min(
                    config,
                    env,
                    geom_ids,
                    object_geom_id,
                    geom_xpos=geom_xpos,
                    geom_xmat=geom_xmat,
                )
            return _cached_object_distance_sdf_min(
                object_sdf_cache,
                config,
                env,
                geom_ids,
                object_geom_ids,
                geom_xpos=geom_xpos,
                geom_xmat=geom_xmat,
                body_xpos=body_xpos,
                body_xmat=body_xmat,
                conservative=True,
            )

        if config.nonhand_support_penalty_geom_ids:
            nonhand_sdf = terminal_object_sdf(config.nonhand_support_penalty_geom_ids)
            nonhand_violation = torch.clamp(
                config.terminal_carry_gate_nonhand_margin_m - nonhand_sdf,
                min=0.0,
            )
        if config.hand_support_geom_ids:
            hand_sdfs = []
            for gid in config.hand_support_geom_ids:
                hand_sdfs.append(terminal_object_sdf([gid]))
            if hand_sdfs:
                hand_sdf_stack = torch.stack(hand_sdfs, dim=1)
                hand_near = (
                    torch.abs(hand_sdf_stack)
                    <= config.terminal_carry_gate_hand_near_margin_m
                ).to(dtype)
                hand_near_frac = hand_near.mean(dim=1)
                hand_violation = torch.clamp(
                    config.terminal_carry_gate_hand_min_near_frac - hand_near_frac,
                    min=0.0,
                )

    violation = pelvis_violation + rot_violation + nonhand_violation + hand_violation
    valid = (violation <= 0.0).to(dtype)
    penalty = -config.terminal_carry_gate_soft_scale * violation
    return {
        "penalty": penalty,
        "violation": violation,
        "valid": valid,
        "pelvis_z": pelvis_z,
        "obj_rot_err": obj_rot_err,
        "nonhand_sdf": nonhand_sdf,
        "hand_near_frac": hand_near_frac,
    }


def get_terminal_reward(
    config: Config,
    env: MJWPEnv,
    ref_slice: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Terminal reward focusing on object tracking."""
    # return config.terminal_rew_scale * get_reward(config, env, ref_slice)
    # qpos_ref, qvel_ref, ctrl_ref, contact_ref, _ = ref_slice
    # qpos_sim = wp.to_torch(env.data_wp.qpos)
    # qpos_weight = torch.zeros(qpos_sim.shape[1], device=config.device)
    # if config.embodiment_type == "bimanual":
    #     qpos_weight[-14:-11] = config.pos_rew_scale
    #     qpos_weight[-11:-7] = config.rot_rew_scale
    #     qpos_weight[-7:-4] = config.pos_rew_scale
    #     qpos_weight[-4:] = config.rot_rew_scale
    # elif config.embodiment_type in ["right", "left"]:
    #     qpos_weight[-7:-4] = config.pos_rew_scale
    #     qpos_weight[-4:] = config.rot_rew_scale
    # elif config.embodiment_type in ["CMU", "DanceDB"]:
    #     qpos_weight[:3] = config.pos_rew_scale
    #     qpos_weight[3:7] = config.rot_rew_scale
    # else:
    #     raise ValueError(f"Invalid embodiment_type: {config.embodiment_type}")
    # delta_qpos = (qpos_sim - qpos_ref) * qpos_weight
    # cost_object = config.terminal_rew_scale * torch.sum(delta_qpos**2, dim=1)

    rew, info = get_reward(config, env, ref_slice)
    terminal_rew = config.terminal_rew_scale * rew
    if config.terminal_carry_gate_enabled:
        gate = _terminal_carry_gate(config, env, ref_slice[0])
        mode = config.terminal_carry_gate_mode
        if mode in {"soft", "hard_soft"}:
            terminal_rew = terminal_rew + gate["penalty"]
        info["terminal_carry_gate_penalty"] = gate["penalty"]
        info["terminal_carry_gate_violation"] = gate["violation"]
        info["terminal_carry_gate_valid"] = gate["valid"]
        info["terminal_carry_gate_pelvis_z"] = gate["pelvis_z"]
        info["terminal_carry_gate_obj_rot_err"] = gate["obj_rot_err"]
        info["terminal_carry_gate_nonhand_sdf"] = gate["nonhand_sdf"]
        info["terminal_carry_gate_hand_near_frac"] = gate["hand_near_frac"]
        if mode in {"hard", "hard_soft"}:
            min_sdf_from_terminal = config.cem_safety_gate_min_sdf_m - gate["violation"]
            info["cem_gate_min_sdf"] = torch.minimum(
                info["cem_gate_min_sdf"], min_sdf_from_terminal
            )
            info["cem_gate_violation_depth"] = torch.maximum(
                info["cem_gate_violation_depth"], gate["violation"]
            )
            info["cem_gate_violation"] = torch.maximum(
                info["cem_gate_violation"],
                (gate["violation"] > 0.0).to(info["cem_gate_violation"].dtype),
            )
            info["cem_body_gate_min_sdf"] = torch.minimum(
                info["cem_body_gate_min_sdf"], min_sdf_from_terminal
            )
            info["cem_body_gate_violation_depth"] = torch.maximum(
                info["cem_body_gate_violation_depth"], gate["violation"]
            )
            info["cem_body_gate_violation"] = torch.maximum(
                info["cem_body_gate_violation"],
                (gate["violation"] > 0.0).to(info["cem_body_gate_violation"].dtype),
            )
    return terminal_rew, info


def get_terminate(
    config: Config, env: MJWPEnv, ref_slice: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    # compute object position and orientation error, compare to thereshold
    qpos_sim = wp.to_torch(env.data_wp.qpos)
    # Tolerate both legacy 5-tuple and E018 6-tuple (with body_xpos_ref).
    qpos_ref = ref_slice[0]
    qvel_ref = ref_slice[1]
    ctrl_ref = ref_slice[2]
    contact_ref = ref_slice[3]
    _contact_pos_ref = ref_slice[4]
    if config.embodiment_type == "bimanual":
        if config.nq_obj == 12:
            right_obj_pos = qpos_sim[:, -12:-9]
            right_obj_pos_ref = qpos_ref[-12:-9].unsqueeze(0)
            right_obj_pos_error = torch.norm(
                right_obj_pos - right_obj_pos_ref, p=2, dim=1
            )
            right_obj_rot = qpos_sim[:, -9:-6]
            right_obj_rot_ref = qpos_ref[-9:-6].unsqueeze(0)
            right_obj_rot_error = torch.norm(
                right_obj_rot - right_obj_rot_ref, p=2, dim=1
            )
            left_obj_pos = qpos_sim[:, -6:-3]
            left_obj_pos_ref = qpos_ref[-6:-3].unsqueeze(0)
            left_obj_pos_error = torch.norm(left_obj_pos - left_obj_pos_ref, p=2, dim=1)
            left_obj_rot = qpos_sim[:, -3:]
            left_obj_rot_ref = qpos_ref[-3:].unsqueeze(0)
            left_obj_rot_error = torch.norm(left_obj_rot - left_obj_rot_ref, p=2, dim=1)
            if torch.all(right_obj_pos_ref.abs() < 1e-4):
                right_obj_pos_error *= 0.0
                right_obj_rot_error *= 0.0
            if torch.all(left_obj_pos_ref.abs() < 1e-4):
                left_obj_pos_error *= 0.0
                left_obj_rot_error *= 0.0
            terminate = (
                (left_obj_pos_error > config.object_pos_threshold)
                | (right_obj_pos_error > config.object_pos_threshold)
                | (left_obj_rot_error > config.object_rot_threshold)
                | (right_obj_rot_error > config.object_rot_threshold)
            )
            return terminate
        left_obj_pos = qpos_sim[:, -14:-11]
        left_obj_pos_ref = qpos_ref[-14:-11].unsqueeze(0)
        left_obj_pos_error = torch.norm(left_obj_pos - left_obj_pos_ref, p=2, dim=1)
        left_obj_quat = qpos_sim[:, -11:-7]
        left_obj_quat_ref = qpos_ref[-11:-7].unsqueeze(0)
        left_obj_quat_error = torch.norm(
            quat_sub(left_obj_quat, left_obj_quat_ref.repeat(qpos_sim.shape[0], 1)),
            p=2,
            dim=1,
        )
        right_obj_pos = qpos_sim[:, -7:-4]
        right_obj_pos_ref = qpos_ref[-7:-4].unsqueeze(0)
        right_obj_pos_error = torch.norm(right_obj_pos - right_obj_pos_ref, p=2, dim=1)
        right_obj_quat = qpos_sim[:, -4:]
        right_obj_quat_ref = qpos_ref[-4:].unsqueeze(0)
        right_obj_quat_error = torch.norm(
            quat_sub(right_obj_quat, right_obj_quat_ref.repeat(qpos_sim.shape[0], 1)),
            p=2,
            dim=1,
        )
        # special case: only have left object
        if torch.all(right_obj_pos_ref.abs() < 1e-4):
            right_obj_pos_error *= 0.0
            right_obj_quat_error *= 0.0
        # special case: only have right object
        if torch.all(left_obj_pos_ref.abs() < 1e-4):
            left_obj_pos_error *= 0.0
            left_obj_quat_error *= 0.0
        terminate = (
            (left_obj_pos_error > config.object_pos_threshold)
            | (right_obj_pos_error > config.object_pos_threshold)
            | (left_obj_quat_error > config.object_rot_threshold)
            | (right_obj_quat_error > config.object_rot_threshold)
        )
    elif config.embodiment_type in ["right", "left"]:
        if config.nq_obj == 6:
            obj_pos = qpos_sim[:, -6:-3]
            obj_pos_ref = qpos_ref[-6:-3].unsqueeze(0)
            obj_pos_error = torch.norm(obj_pos - obj_pos_ref, p=2, dim=1)
            obj_rot = qpos_sim[:, -3:]
            obj_rot_ref = qpos_ref[-3:].unsqueeze(0)
            obj_rot_error = torch.norm(obj_rot - obj_rot_ref, p=2, dim=1)
            terminate = (obj_pos_error > config.object_pos_threshold) | (
                obj_rot_error > config.object_rot_threshold
            )
            return terminate
        obj_pos = qpos_sim[:, -7:-4]
        obj_pos_ref = qpos_ref[-7:-4].unsqueeze(0)
        obj_pos_error = torch.norm(obj_pos - obj_pos_ref, p=2, dim=1)
        obj_quat = qpos_sim[:, -4:]
        obj_quat_ref = qpos_ref[-4:].unsqueeze(0)
        obj_quat_error = torch.norm(
            quat_sub(obj_quat, obj_quat_ref.repeat(qpos_sim.shape[0], 1)), p=2, dim=1
        )
        terminate = (obj_pos_error > config.object_pos_threshold) | (
            obj_quat_error > config.object_rot_threshold
        )
    elif config.embodiment_type in ["humanoid", "humanoid_object"]:
        base_pos = qpos_sim[:, :3]
        base_pos_ref = qpos_ref[:3].unsqueeze(0)
        base_pos_error = torch.norm(base_pos - base_pos_ref, p=2, dim=1)
        base_quat = qpos_sim[:, 3:7]
        base_quat_ref = qpos_ref[3:7].unsqueeze(0)
        base_quat_error = torch.norm(
            quat_sub(base_quat, base_quat_ref.repeat(qpos_sim.shape[0], 1)), p=2, dim=1
        )
        terminate = (base_pos_error > config.base_pos_threshold) | (
            base_quat_error > config.base_rot_threshold
        )
    elif config.embodiment_type == "dual_humanoid_object":
        nq_robot = (config.nq - config.nq_obj) // 2  # 36 per robot
        N = qpos_sim.shape[0]
        # robot1 base
        r1_pos_err = torch.norm(qpos_sim[:, :3] - qpos_ref[:3].unsqueeze(0), p=2, dim=1)
        r1_rot_err = torch.norm(
            quat_sub(qpos_sim[:, 3:7], qpos_ref[3:7].unsqueeze(0).expand(N, -1)),
            p=2,
            dim=1,
        )
        # robot2 base
        r2_pos_err = torch.norm(
            qpos_sim[:, nq_robot : nq_robot + 3]
            - qpos_ref[nq_robot : nq_robot + 3].unsqueeze(0),
            p=2,
            dim=1,
        )
        r2_rot_err = torch.norm(
            quat_sub(
                qpos_sim[:, nq_robot + 3 : nq_robot + 7],
                qpos_ref[nq_robot + 3 : nq_robot + 7].unsqueeze(0).expand(N, -1),
            ),
            p=2,
            dim=1,
        )
        terminate = (
            (r1_pos_err > config.base_pos_threshold)
            | (r1_rot_err > config.base_rot_threshold)
            | (r2_pos_err > config.base_pos_threshold)
            | (r2_rot_err > config.base_rot_threshold)
        )
    else:
        raise ValueError(f"Invalid embodiment_type: {config.embodiment_type}")
    return terminate


def get_qpos(config: Config, env: MJWPEnv) -> torch.Tensor:
    return wp.to_torch(env.data_wp.qpos)


def get_geometry_state(config: Config, env: MJWPEnv) -> dict[str, torch.Tensor]:
    """Return the derived transforms consumed by object-distance rewards."""
    del config
    geom_xpos = wp.to_torch(env.data_wp.geom_xpos)
    body_xpos = wp.to_torch(env.data_wp.xpos)
    return {
        "geom_xpos": geom_xpos,
        "geom_xmat": wp.to_torch(env.data_wp.geom_xmat).reshape(
            geom_xpos.shape[0], geom_xpos.shape[1], 3, 3
        ),
        "body_xpos": body_xpos,
        "body_xmat": wp.to_torch(env.data_wp.xmat).reshape(
            body_xpos.shape[0], body_xpos.shape[1], 3, 3
        ),
    }


def set_qpos(config: Config, env: MJWPEnv, qpos: torch.Tensor):
    qpos = qpos.to(config.device)
    if qpos.dim() == 1:
        qpos = qpos.unsqueeze(0).repeat(env.num_worlds, 1)
    wp.copy(env.data_wp.qpos, wp.from_torch(qpos))
    # reset velocities/time as well for consistency
    zero_qvel = torch.zeros((env.num_worlds, env.model_cpu.nv), device=config.device)
    wp.copy(env.data_wp.qvel, wp.from_torch(zero_qvel))
    wp.copy(
        env.data_wp.time,
        wp.from_torch(
            torch.zeros(env.num_worlds, dtype=torch.float32, device=config.device)
        ),
    )


def get_qvel(config: Config, env: MJWPEnv) -> torch.Tensor:
    return wp.to_torch(env.data_wp.qvel)


def compute_contact_point_delta(
    contact_mask_step: torch.Tensor,
    contact_pos_ref_step: torch.Tensor,
    site_xpos: torch.Tensor,
    hand_contact_site_ids: list[int | None],
    contact_indices: list[int],
) -> torch.Tensor | None:
    """Compute mean contact position delta for a hand (current - reference).

    Args:
        contact_mask_step: (N_contact,) mask for active contacts.
        contact_pos_ref_step: (N_contact, 3) reference contact positions.
        site_xpos: (N_site, 3) current site positions for the active world.
        hand_contact_site_ids: list mapping contact indices to site ids (None if missing).
        contact_indices: indices for the hand contacts to aggregate.
    """
    current_positions = []
    reference_positions = []
    for idx in contact_indices:
        if idx >= len(hand_contact_site_ids) or idx >= contact_pos_ref_step.shape[0]:
            continue
        sid = hand_contact_site_ids[idx]
        if sid is None or contact_mask_step[idx] <= 0.5:
            continue
        current_positions.append(site_xpos[sid])
        reference_positions.append(contact_pos_ref_step[idx])

    if not current_positions:
        return None

    current_mean = torch.stack(current_positions, dim=0).mean(dim=0)
    reference_mean = torch.stack(reference_positions, dim=0).mean(dim=0)
    return current_mean - reference_mean


def get_trace(config: Config, env: MJWPEnv) -> torch.Tensor:
    """Return per-world trace points used for visualization. Minimal default returns
    an empty trace set of shape (N, 0, 3) when not configured.
    """
    site_xpos = wp.to_torch(env.data_wp.site_xpos)  # (N, nsite, 3)
    return site_xpos[:, config.trace_site_ids, :]


def save_state(env: MJWPEnv):
    """Clone the essential set of Warp arrays to restore later.
    Includes core state variables and key derived quantities.
    """
    _copy_state(env.data_wp, env.data_wp_prev)
    return env
    # qpos = wp.clone(env.data_wp.qpos)
    # qvel = wp.clone(env.data_wp.qvel)
    # qacc = wp.clone(env.data_wp.qacc)
    # time_arr = wp.clone(env.data_wp.time)
    # ctrl = wp.clone(env.data_wp.ctrl) if hasattr(env.data_wp, "ctrl") else None
    # act = wp.clone(env.data_wp.act) if hasattr(env.data_wp, "act") else None
    # act_dot = wp.clone(env.data_wp.act_dot) if hasattr(env.data_wp, "act_dot") else None
    # site_xpos = wp.clone(env.data_wp.site_xpos)
    # site_xmat = wp.clone(env.data_wp.site_xmat)
    # mocap_pos = (
    #     wp.clone(env.data_wp.mocap_pos) if hasattr(env.data_wp, "mocap_pos") else None
    # )
    # mocap_quat = (
    #     wp.clone(env.data_wp.mocap_quat) if hasattr(env.data_wp, "mocap_quat") else None
    # )
    # energy = wp.clone(env.data_wp.energy) if hasattr(env.data_wp, "energy") else None
    # return (
    #     qpos,
    #     qvel,
    #     qacc,
    #     time_arr,
    #     ctrl,
    #     act,
    #     act_dot,
    #     site_xpos,
    #     site_xmat,
    #     mocap_pos,
    #     mocap_quat,
    #     energy,
    # )


def load_state(env: MJWPEnv, state):
    _copy_state(env.data_wp_prev, env.data_wp)
    return env
    # (
    #     qpos,
    #     qvel,
    #     qacc,
    #     time_arr,
    #     ctrl,
    #     act,
    #     act_dot,
    #     site_xpos,
    #     site_xmat,
    #     mocap_pos,
    #     mocap_quat,
    #     energy,
    # ) = state
    # wp.copy(env.data_wp.qpos, qpos)
    # wp.copy(env.data_wp.qvel, qvel)
    # wp.copy(env.data_wp.qacc, qacc)
    # wp.copy(env.data_wp.time, time_arr)
    # if ctrl is not None and hasattr(env.data_wp, "ctrl"):
    #     wp.copy(env.data_wp.ctrl, ctrl)
    # if act is not None and hasattr(env.data_wp, "act"):
    #     wp.copy(env.data_wp.act, act)
    # if act_dot is not None and hasattr(env.data_wp, "act_dot"):
    #     wp.copy(env.data_wp.act_dot, act_dot)
    # if mocap_pos is not None and hasattr(env.data_wp, "mocap_pos"):
    #     wp.copy(env.data_wp.mocap_pos, mocap_pos)
    # if mocap_quat is not None and hasattr(env.data_wp, "mocap_quat"):
    #     wp.copy(env.data_wp.mocap_quat, mocap_quat)
    # if energy is not None and hasattr(env.data_wp, "energy"):
    #     wp.copy(env.data_wp.energy, energy)
    # if site_xpos is not None and hasattr(env.data_wp, "site_xpos"):
    #     wp.copy(env.data_wp.site_xpos, site_xpos)
    # if site_xmat is not None and hasattr(env.data_wp, "site_xmat"):
    #     wp.copy(env.data_wp.site_xmat, site_xmat)
    # return env


def apply_perturbation(config: Config, env: MJWPEnv):
    # get object id
    right_obj_id = mujoco.mj_name2id(
        env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "right_object"
    )
    left_obj_id = mujoco.mj_name2id(
        env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "left_object"
    )
    xfrc_applied = wp.to_torch(env.data_wp.xfrc_applied)
    if right_obj_id != -1:
        xfrc_applied[:, right_obj_id, :3] = config.perturb_force
        xfrc_applied[:, right_obj_id, 3:] = config.perturb_torque
    if left_obj_id != -1:
        xfrc_applied[:, left_obj_id, :3] = config.perturb_force
        xfrc_applied[:, left_obj_id, 3:] = config.perturb_torque
    wp.copy(env.data_wp.xfrc_applied, wp.from_torch(xfrc_applied))
    return env


def _apply_object_pd_override(config: Config, env: MJWPEnv):
    """E027b: Override object actuator ctrl to PD-track ref trajectory.

    For scene_act.xml with 6 object position actuators (3 slide + 3 hinge),
    sets ctrl = ref_target so the actuator's built-in PD drives the object.
    Called after CEM ctrl is written, effectively overriding CEM's object dims.
    Adds gravity compensation offset to z-target (mg/kp) for zero steady-state error.
    """
    time_arr = wp.to_torch(env.data_wp.time)
    t = time_arr[0].item()
    dt = 1.0 / 30.0
    T = env.object_pd_ref_pos.shape[0]
    idx = min(int(t / dt), T - 1)

    # Get ref pos (3) and euler (3) directly
    ref_pos = env.object_pd_ref_pos[idx]  # (3,)
    ref_euler = env.object_pd_ref_euler[idx]  # (3,) xyz euler

    # Gravity compensation offset for z: target += mg/kp
    grav_comp = env.object_mass * 9.81 / config.object_pd_kp_pos

    # Object actuator target = [pos_x, pos_y, pos_z + grav_comp, rot_x, rot_y, rot_z]
    obj_target = torch.tensor(
        [
            ref_pos[0].item(),
            ref_pos[1].item(),
            ref_pos[2].item() + grav_comp,
            ref_euler[0].item(),
            ref_euler[1].item(),
            ref_euler[2].item(),
        ],
        dtype=torch.float32,
        device=config.device,
    )

    # Write to ctrl for object actuator channels (last 6 of nu)
    ctrl = wp.to_torch(env.data_wp.ctrl)
    obj_act_start = ctrl.shape[1] - 6
    ctrl[:, obj_act_start:] = obj_target.unsqueeze(0)
    wp.copy(env.data_wp.ctrl, wp.from_torch(ctrl))


def _object_kinematic_override_enabled(config: Config) -> bool:
    return bool(config.object_kinematic_override) or config.partner_force_spring_kp < 0


def _apply_object_kinematic_override(config: Config, env: MJWPEnv):
    """E013: Write a true-freejoint object state from the reference trajectory."""
    if not hasattr(env, "object_kinematic_ref_qpos"):
        return
    if config.nq_obj != 7:
        raise ValueError(
            "object_kinematic_override requires nq_obj=7 true-freejoint object."
        )

    obj_body_id = mujoco.mj_name2id(env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "object")
    if obj_body_id == -1:
        return
    obj_jnt_id = env.model_cpu.body_jntadr[obj_body_id]
    if obj_jnt_id < 0:
        return
    obj_qadr = int(env.model_cpu.jnt_qposadr[obj_jnt_id])
    obj_vadr = int(env.model_cpu.jnt_dofadr[obj_jnt_id])

    time_arr = wp.to_torch(env.data_wp.time)
    t = float(time_arr[0].item())
    dt = float(getattr(env, "object_kinematic_ref_dt", config.sim_dt))
    T = int(env.object_kinematic_ref_qpos.shape[0])
    idx = min(max(int(t / max(dt, 1e-8) + 1e-6), 0), T - 1)

    qpos = wp.to_torch(env.data_wp.qpos)
    qpos[:, obj_qadr : obj_qadr + 7] = env.object_kinematic_ref_qpos[idx].unsqueeze(0)
    wp.copy(env.data_wp.qpos, wp.from_torch(qpos))

    if bool(config.object_kinematic_set_qvel) and hasattr(
        env, "object_kinematic_ref_qvel"
    ):
        qvel = wp.to_torch(env.data_wp.qvel)
        qvel[:, obj_vadr : obj_vadr + 6] = env.object_kinematic_ref_qvel[idx].unsqueeze(
            0
        )
        wp.copy(env.data_wp.qvel, wp.from_torch(qvel))


def _apply_partner_force(config: Config, env: MJWPEnv):
    """Apply external force on the object body to simulate partner support.

    Models the human partner holding one side of the object, providing:
    1. Gravity compensation: upward force = partner_force_scale * object_weight
    2. (Optional) Spring: pull toward reference position with partner_force_spring_kp
    3. (Optional) Multi-point spring: track object-local support points and
       synthesize a net wrench from their distributed forces.

    This enables single-robot retargeting of cooperative carrying tasks.
    """
    obj_body_id = mujoco.mj_name2id(env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "object")
    if obj_body_id == -1:
        return

    xfrc_applied = wp.to_torch(env.data_wp.xfrc_applied)
    _clear_object_wrench_once(env, obj_body_id, xfrc_applied)

    def _clamp_norm(vec: torch.Tensor, max_norm: float) -> torch.Tensor:
        if max_norm <= 0:
            return vec
        norm = vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.where(norm > max_norm, vec / norm * max_norm, vec)

    # Gravity compensation: upward force on object.
    obj_mass = env.model_cpu.body_mass[obj_body_id]
    gravity_z = -env.model_cpu.opt.gravity[2]  # positive (9.81)
    upward_force = config.partner_force_scale * obj_mass * gravity_z

    points_local_raw = list(config.partner_force_points_local or [])
    points_local: list[list[float]] = []
    for point in points_local_raw:
        point_list = list(point)
        if len(point_list) != 3:
            raise ValueError(
                "partner_force_points_local entries must be [x, y, z] triples."
            )
        points_local.append([float(v) for v in point_list])
    multi_point_mode = len(points_local) > 0
    point_local = list(config.partner_force_point_local or [])
    point_mode = len(point_local) == 3 and not multi_point_mode
    has_spring = config.partner_force_spring_kp > 0 and hasattr(
        env, "partner_force_ref_pos"
    )
    has_rot_spring = config.partner_force_spring_kp_rot > 0 and hasattr(
        env, "partner_force_ref_quat"
    )
    last_force = torch.zeros(
        (env.num_worlds, 3), device=config.device, dtype=torch.float32
    )
    last_torque = torch.zeros_like(last_force)

    if (
        not multi_point_mode
        and not point_mode
        and not has_spring
        and not has_rot_spring
    ):
        xfrc_applied[:, obj_body_id, 2] = upward_force
        last_force[:, 2] = upward_force
        env.partner_force_last_force = last_force.detach()
        env.partner_force_last_torque = last_torque.detach()
        wp.copy(env.data_wp.xfrc_applied, wp.from_torch(xfrc_applied))
        return

    qpos = wp.to_torch(env.data_wp.qpos)
    qvel = wp.to_torch(env.data_wp.qvel)
    obj_jnt_id = env.model_cpu.body_jntadr[obj_body_id]
    obj_qadr = env.model_cpu.jnt_qposadr[obj_jnt_id]
    obj_vadr = env.model_cpu.jnt_dofadr[obj_jnt_id]
    obj_pos_sim = qpos[:, obj_qadr : obj_qadr + 3]  # (N, 3)
    obj_vel_sim = qvel[:, obj_vadr : obj_vadr + 3]  # (N, 3)
    obj_quat_sim = qpos[:, obj_qadr + 3 : obj_qadr + 7]  # (N, 4) wxyz
    obj_angvel_sim = qvel[:, obj_vadr + 3 : obj_vadr + 6]  # (N, 3)
    quat_norm = obj_quat_sim.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    obj_quat_sim = obj_quat_sim / quat_norm

    time_arr = wp.to_torch(env.data_wp.time)
    t = time_arr[0].item()
    dt = float(
        getattr(
            env,
            "partner_force_ref_dt",
            config.partner_force_ref_dt
            if config.partner_force_ref_dt > 0
            else config.ref_dt,
        )
    )
    T = (
        env.partner_force_ref_pos.shape[0]
        if hasattr(env, "partner_force_ref_pos")
        else 1
    )
    idx = min(int(t / dt), T - 1)
    ramp = min(t / 0.5, 1.0)

    force = torch.zeros_like(obj_pos_sim)
    force[:, 2] = upward_force

    if has_spring:
        kp = config.partner_force_spring_kp * ramp
        if config.partner_force_spring_kd < 0:
            kd = 2.0 * (obj_mass * config.partner_force_spring_kp) ** 0.5
        else:
            kd = config.partner_force_spring_kd
        ref_pos = env.partner_force_ref_pos[idx]  # (3,) on GPU

        if multi_point_mode:
            points = torch.tensor(
                points_local,
                dtype=torch.float32,
                device=obj_pos_sim.device,
            )
            num_points = points.shape[0]
            local = points.unsqueeze(0).expand(obj_pos_sim.shape[0], -1, -1)

            q_expand = (
                obj_quat_sim.unsqueeze(1).expand(-1, num_points, -1).reshape(-1, 4)
            )
            r_world = _lf_quat_apply(
                q_expand,
                local.reshape(-1, 3),
            ).reshape(obj_pos_sim.shape[0], num_points, 3)
            point_pos_sim = obj_pos_sim.unsqueeze(1) + r_world
            point_vel_sim = obj_vel_sim.unsqueeze(1) + torch.cross(
                obj_angvel_sim.unsqueeze(1).expand(-1, num_points, -1),
                r_world,
                dim=-1,
            )

            ref_quat = env.partner_force_ref_quat[idx]
            ref_q_expand = (
                ref_quat.view(1, 1, 4)
                .expand(obj_pos_sim.shape[0], num_points, -1)
                .reshape(-1, 4)
            )
            ref_r_world = _lf_quat_apply(
                ref_q_expand,
                local.reshape(-1, 3),
            ).reshape(obj_pos_sim.shape[0], num_points, 3)
            ref_point = ref_pos.view(1, 1, 3) + ref_r_world

            point_force = (kp / num_points) * (ref_point - point_pos_sim) - (
                kd / num_points
            ) * point_vel_sim
            point_force[:, :, 2] += upward_force / num_points
            point_force = torch.nan_to_num(point_force, nan=0.0)
            force = point_force.sum(dim=1)
            force = _clamp_norm(force, config.partner_force_force_clamp)
            torque = torch.cross(r_world, point_force, dim=-1).sum(dim=1)
            torque = _clamp_norm(
                torch.nan_to_num(torque, nan=0.0),
                config.partner_force_torque_clamp,
            )
            xfrc_applied[:, obj_body_id, :3] += force
            xfrc_applied[:, obj_body_id, 3:6] += torque
            last_torque = last_torque + torque
        elif point_mode:
            local = torch.tensor(
                point_local,
                dtype=torch.float32,
                device=obj_pos_sim.device,
            ).unsqueeze(0)
            local = local.expand(obj_pos_sim.shape[0], -1)
            r_world = _lf_quat_apply(obj_quat_sim, local)
            point_pos_sim = obj_pos_sim + r_world
            point_vel_sim = obj_vel_sim + torch.cross(obj_angvel_sim, r_world, dim=-1)
            ref_quat = env.partner_force_ref_quat[idx]
            ref_r_world = _lf_quat_apply(
                ref_quat.unsqueeze(0).expand(obj_pos_sim.shape[0], -1),
                local,
            )
            ref_point = ref_pos.unsqueeze(0) + ref_r_world
            force = force + kp * (ref_point - point_pos_sim) - kd * point_vel_sim
            force = _clamp_norm(
                torch.nan_to_num(force, nan=0.0), config.partner_force_force_clamp
            )
            torque = torch.cross(r_world, force, dim=-1)
            torque = _clamp_norm(
                torch.nan_to_num(torque, nan=0.0),
                config.partner_force_torque_clamp,
            )
            xfrc_applied[:, obj_body_id, :3] += force
            xfrc_applied[:, obj_body_id, 3:6] += torque
            last_torque = last_torque + torque
        else:
            force = force + kp * (ref_pos.unsqueeze(0) - obj_pos_sim) - kd * obj_vel_sim
            force = _clamp_norm(
                torch.nan_to_num(force, nan=0.0), config.partner_force_force_clamp
            )
            xfrc_applied[:, obj_body_id, :3] += force
    else:
        force = _clamp_norm(
            torch.nan_to_num(force, nan=0.0), config.partner_force_force_clamp
        )
        xfrc_applied[:, obj_body_id, :3] += force
    last_force = last_force + force

    # E030: Orientation control via xfrc_applied torque. E005 keeps this disabled;
    # support-site torque comes from r x F instead of an orientation PD loop.
    if has_rot_spring:
        from spider.math import quat_sub

        ref_quat = env.partner_force_ref_quat[idx]  # (4,) wxyz on GPU
        aa_err = quat_sub(
            ref_quat.unsqueeze(0).expand(obj_quat_sim.shape[0], -1),
            obj_quat_sim,
        )
        aa_err = torch.nan_to_num(aa_err, nan=0.0)

        aa_mag = aa_err.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        clamp_val = config.partner_force_rot_clamp
        aa_err = torch.where(
            aa_mag > clamp_val,
            aa_err / aa_mag * clamp_val,
            aa_err,
        )

        obj_angvel_sim = torch.nan_to_num(obj_angvel_sim, nan=0.0).clamp(-10.0, 10.0)
        kp_rot = config.partner_force_spring_kp_rot * ramp
        if config.partner_force_spring_kd_rot < 0:
            avg_inertia = float(np.mean(env.model_cpu.body_inertia[obj_body_id]))
            kd_rot = 2.0 * (avg_inertia * config.partner_force_spring_kp_rot) ** 0.5
        else:
            kd_rot = config.partner_force_spring_kd_rot
        torque = kp_rot * aa_err - kd_rot * obj_angvel_sim
        torque = _clamp_norm(torch.nan_to_num(torque, nan=0.0), 5.0)
        xfrc_applied[:, obj_body_id, 3:6] += torque
        last_torque = last_torque + torque

    env.partner_force_last_force = last_force.detach()
    env.partner_force_last_torque = last_torque.detach()
    wp.copy(env.data_wp.xfrc_applied, wp.from_torch(xfrc_applied))


def get_partner_force_state(config: Config, env: MJWPEnv) -> dict[str, np.ndarray]:
    """Return first-world partner force diagnostics for trajectory saving."""
    if (
        config.partner_force_scale <= 0
        and config.partner_force_spring_kp <= 0
        and config.partner_force_spring_kp_rot <= 0
    ):
        return {}
    if not hasattr(env, "partner_force_last_force"):
        return {}

    def _first(tensor: torch.Tensor) -> np.ndarray:
        return tensor[0].detach().cpu().numpy().astype(np.float32)

    return {
        "partner_force_force": _first(env.partner_force_last_force),
        "partner_force_torque": _first(env.partner_force_last_torque),
    }


def _support_proxy_point_local(config: Config) -> list[float]:
    point_local = list(config.support_proxy_point_local or [])
    if len(point_local) != 3:
        raise ValueError(
            "support_proxy_enabled requires support_proxy_point_local=[x,y,z]."
        )
    return [float(v) for v in point_local]


def _load_support_proxy(
    config: Config,
    env: MJWPEnv,
    qpos_ref: torch.Tensor,
    qvel_ref: torch.Tensor | None = None,
):
    """Precompute a kinematic support-body proxy trajectory from object ref.

    The proxy is intentionally not a MuJoCo freejoint body in E006. It is an
    independent controller target whose connector wrench is applied to the true
    freejoint object at an object-local support site.
    """
    valid_modes = {"wrench", "mocap_pad", "wrench_pad", "dynamic_weld"}
    if config.support_proxy_mode not in valid_modes:
        raise ValueError(
            f"Unknown support_proxy_mode={config.support_proxy_mode!r}; "
            f"expected one of {sorted(valid_modes)}."
        )
    valid_quat_modes = {"identity", "object_ref"}
    if config.support_proxy_mocap_quat_mode not in valid_quat_modes:
        raise ValueError(
            f"Unknown support_proxy_mocap_quat_mode="
            f"{config.support_proxy_mocap_quat_mode!r}; "
            f"expected one of {sorted(valid_quat_modes)}."
        )
    if config.nq_obj != 7 or config.contact_guidance:
        raise ValueError(
            "support_proxy_enabled currently requires true-freejoint object "
            f"(contact_guidance=false, nq_obj=7), got nq_obj={config.nq_obj}."
        )

    point_local = torch.tensor(
        _support_proxy_point_local(config),
        dtype=torch.float32,
        device=config.device,
    )
    qpos_ref_t = qpos_ref.to(config.device).to(torch.float32)
    obj_pos_ref = qpos_ref_t[:, -7:-4]
    obj_quat_ref = qpos_ref_t[:, -4:]
    obj_quat_ref = obj_quat_ref / obj_quat_ref.norm(dim=-1, keepdim=True).clamp(
        min=1e-8
    )
    T = obj_pos_ref.shape[0]
    support_ref = obj_pos_ref + _lf_quat_apply(
        obj_quat_ref, point_local.unsqueeze(0).expand(T, -1)
    )

    # qpos_ref passed into this function has already been interpolated to
    # config.sim_dt by spider.io.load_data(). Index proxy_ref on that time base
    # unless a legacy experiment explicitly pins support_proxy_ref_dt.
    dt = (
        float(config.support_proxy_ref_dt)
        if config.support_proxy_ref_dt > 0
        else float(config.sim_dt)
    )
    proxy_pos = torch.empty_like(support_ref)
    proxy_pos[0] = support_ref[0]

    max_xy_step = (
        float(config.support_proxy_max_xy_speed) * dt
        if config.support_proxy_max_xy_speed > 0
        else 0.0
    )
    xy_scale = float(config.support_proxy_xy_velocity_scale)
    height_tau = float(config.support_proxy_height_tau)
    height_alpha = 1.0 if height_tau <= 0 else dt / (height_tau + dt)

    for i in range(1, T):
        delta_xy = (support_ref[i, :2] - support_ref[i - 1, :2]) * xy_scale
        if max_xy_step > 0:
            delta_norm = delta_xy.norm().clamp(min=1e-8)
            if bool(delta_norm > max_xy_step):
                delta_xy = delta_xy / delta_norm * max_xy_step
        proxy_pos[i, :2] = proxy_pos[i - 1, :2] + delta_xy
        proxy_pos[i, 2] = proxy_pos[i - 1, 2] + height_alpha * (
            support_ref[i, 2] - proxy_pos[i - 1, 2]
        )

    proxy_vel = torch.zeros_like(proxy_pos)
    if T > 1:
        proxy_vel[1:] = (proxy_pos[1:] - proxy_pos[:-1]) / max(dt, 1e-8)
        proxy_vel[0] = proxy_vel[1]

    env.support_proxy_point_local = point_local
    env.support_proxy_ref_pos = proxy_pos.detach()
    env.support_proxy_ref_vel = proxy_vel.detach()
    env.support_proxy_ref_quat = obj_quat_ref.detach()
    env.support_proxy_ref_dt = dt
    if config.support_proxy_mode == "dynamic_weld":
        _load_dynamic_support_ref(config, env, qpos_ref_t, qvel_ref, dt)
    env.support_proxy_last_force = torch.zeros(
        (env.num_worlds, 3), device=config.device, dtype=torch.float32
    )
    env.support_proxy_last_torque = torch.zeros(
        (env.num_worlds, 3), device=config.device, dtype=torch.float32
    )
    env.support_proxy_last_pos = proxy_pos[:1].repeat(env.num_worlds, 1).detach()
    env.support_proxy_last_vel = proxy_vel[:1].repeat(env.num_worlds, 1).detach()
    env.support_proxy_last_support_point_pos = env.support_proxy_last_pos.clone()
    env.support_proxy_last_support_point_vel = torch.zeros_like(
        env.support_proxy_last_support_point_pos
    )
    env.support_proxy_last_idx = 0
    loguru.logger.info(
        "support proxy: ref_pos={}, point_local={}, dt={}, kp={}, "
        "xy_vel_scale={}, max_xy_speed={}, height_tau={}",
        tuple(proxy_pos.shape),
        _support_proxy_point_local(config),
        dt,
        config.support_proxy_connector_kp,
        config.support_proxy_xy_velocity_scale,
        config.support_proxy_max_xy_speed,
        config.support_proxy_height_tau,
    )


def _dynamic_support_joint_addrs(config: Config, env: MJWPEnv) -> tuple[int, int]:
    if hasattr(env, "_support_dynamic_qadr") and hasattr(env, "_support_dynamic_dadr"):
        return int(env._support_dynamic_qadr), int(env._support_dynamic_dadr)

    body_id = mujoco.mj_name2id(
        env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, config.support_dynamic_body_name
    )
    if body_id == -1:
        raise ValueError(
            f"support dynamic body {config.support_dynamic_body_name!r} not found."
        )
    if env.model_cpu.body_mocapid[body_id] >= 0:
        raise ValueError(
            f"support dynamic body {config.support_dynamic_body_name!r} must not be mocap."
        )

    jadr = int(env.model_cpu.body_jntadr[body_id])
    jnum = int(env.model_cpu.body_jntnum[body_id])
    if jadr < 0 or jnum != 6:
        raise ValueError(
            f"{config.support_dynamic_body_name} requires exactly 6 scalar joints, "
            f"got {jnum}."
        )

    qaddrs = [int(env.model_cpu.jnt_qposadr[jadr + i]) for i in range(jnum)]
    daddrs = [int(env.model_cpu.jnt_dofadr[jadr + i]) for i in range(jnum)]
    if qaddrs != list(range(qaddrs[0], qaddrs[0] + 6)):
        raise ValueError(f"support dynamic qpos addresses are not contiguous: {qaddrs}")
    if daddrs != list(range(daddrs[0], daddrs[0] + 6)):
        raise ValueError(f"support dynamic dof addresses are not contiguous: {daddrs}")

    env._support_dynamic_body_id = body_id
    env._support_dynamic_qadr = qaddrs[0]
    env._support_dynamic_dadr = daddrs[0]
    return qaddrs[0], daddrs[0]


def _load_dynamic_support_ref(
    config: Config,
    env: MJWPEnv,
    qpos_ref: torch.Tensor,
    qvel_ref: torch.Tensor | None,
    dt: float,
) -> None:
    qadr, dadr = _dynamic_support_joint_addrs(config, env)
    if qpos_ref.shape[1] < qadr + 6:
        raise ValueError(
            "dynamic support reference qpos is too short: "
            f"{qpos_ref.shape[1]} < {qadr + 6}."
        )
    env.support_dynamic_ref_qpos = qpos_ref[:, qadr : qadr + 6].detach()

    if qvel_ref is not None:
        qvel_ref_t = qvel_ref.to(config.device).to(torch.float32)
        if qvel_ref_t.shape[1] >= dadr + 6:
            env.support_dynamic_ref_qvel = qvel_ref_t[:, dadr : dadr + 6].detach()
        else:
            env.support_dynamic_ref_qvel = torch.zeros_like(
                env.support_dynamic_ref_qpos
            )
    else:
        ref_vel = torch.zeros_like(env.support_dynamic_ref_qpos)
        if env.support_dynamic_ref_qpos.shape[0] > 1:
            ref_vel[1:] = (
                env.support_dynamic_ref_qpos[1:] - env.support_dynamic_ref_qpos[:-1]
            ) / max(dt, 1e-8)
            ref_vel[0] = ref_vel[1]
        env.support_dynamic_ref_qvel = ref_vel.detach()

    body_id = int(getattr(env, "_support_dynamic_body_id"))
    avg_inertia = float(np.mean(env.model_cpu.body_inertia[body_id]))
    pos_kp = max(float(config.support_dynamic_pos_kp), 0.0)
    rot_kp = max(float(config.support_dynamic_rot_kp), 0.0)
    env.support_dynamic_pos_kd = (
        2.0 * (max(float(config.support_dynamic_mass), 1e-6) * pos_kp) ** 0.5
        if config.support_dynamic_pos_kd < 0
        else float(config.support_dynamic_pos_kd)
    )
    env.support_dynamic_rot_kd = (
        2.0 * (max(avg_inertia, 1e-8) * rot_kp) ** 0.5
        if config.support_dynamic_rot_kd < 0
        else float(config.support_dynamic_rot_kd)
    )
    loguru.logger.info(
        "dynamic support: qadr={}, dadr={}, ref_qpos={}, pos_kp={}, "
        "pos_kd={}, rot_kp={}, rot_kd={}, mass={}, inertia={}",
        qadr,
        dadr,
        tuple(env.support_dynamic_ref_qpos.shape),
        config.support_dynamic_pos_kp,
        env.support_dynamic_pos_kd,
        config.support_dynamic_rot_kp,
        env.support_dynamic_rot_kd,
        config.support_dynamic_mass,
        avg_inertia,
    )


def _wrap_angle_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + torch.pi, 2.0 * torch.pi) - torch.pi


def _apply_dynamic_support_pd(config: Config, env: MJWPEnv):
    """Drive a dynamic 6-DoF support body with generalized PD forces."""
    if not hasattr(env, "support_dynamic_ref_qpos"):
        return

    qadr, dadr = _dynamic_support_joint_addrs(config, env)
    qpos = wp.to_torch(env.data_wp.qpos)
    qvel = wp.to_torch(env.data_wp.qvel)
    qfrc_applied = wp.to_torch(env.data_wp.qfrc_applied)
    qfrc_applied[:, dadr : dadr + 6] = 0.0

    time_arr = wp.to_torch(env.data_wp.time)
    t = time_arr[0].item()
    dt = float(getattr(env, "support_proxy_ref_dt", config.sim_dt))
    T = env.support_dynamic_ref_qpos.shape[0]
    idx = min(int(t / dt), T - 1)

    target_qpos = (
        env.support_dynamic_ref_qpos[idx].unsqueeze(0).expand(env.num_worlds, -1)
    )
    target_qvel = (
        env.support_dynamic_ref_qvel[idx].unsqueeze(0).expand(env.num_worlds, -1)
    )
    cur_qpos = qpos[:, qadr : qadr + 6]
    cur_qvel = qvel[:, dadr : dadr + 6]

    pos_err = target_qpos[:, :3] - cur_qpos[:, :3]
    rot_err = _wrap_angle_pi(target_qpos[:, 3:6] - cur_qpos[:, 3:6])
    pos_force = float(config.support_dynamic_pos_kp) * pos_err + float(
        env.support_dynamic_pos_kd
    ) * (target_qvel[:, :3] - cur_qvel[:, :3])
    rot_torque = float(config.support_dynamic_rot_kp) * rot_err + float(
        env.support_dynamic_rot_kd
    ) * (target_qvel[:, 3:6] - cur_qvel[:, 3:6])
    pos_force = _clamp_vector_norm(
        torch.nan_to_num(pos_force, nan=0.0), config.support_dynamic_force_clamp
    )
    rot_torque = _clamp_vector_norm(
        torch.nan_to_num(rot_torque, nan=0.0), config.support_dynamic_torque_clamp
    )
    qfrc_applied[:, dadr : dadr + 3] = pos_force
    qfrc_applied[:, dadr + 3 : dadr + 6] = rot_torque
    wp.copy(env.data_wp.qfrc_applied, wp.from_torch(qfrc_applied))

    obj_body_id = mujoco.mj_name2id(env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "object")
    if obj_body_id == -1:
        return
    obj_jnt_id = env.model_cpu.body_jntadr[obj_body_id]
    obj_qadr = env.model_cpu.jnt_qposadr[obj_jnt_id]
    obj_vadr = env.model_cpu.jnt_dofadr[obj_jnt_id]
    obj_pos_sim = qpos[:, obj_qadr : obj_qadr + 3]
    obj_vel_sim = qvel[:, obj_vadr : obj_vadr + 3]
    obj_quat_sim = qpos[:, obj_qadr + 3 : obj_qadr + 7]
    obj_angvel_sim = qvel[:, obj_vadr + 3 : obj_vadr + 6]
    obj_quat_sim = obj_quat_sim / obj_quat_sim.norm(dim=-1, keepdim=True).clamp(
        min=1e-8
    )
    local = env.support_proxy_point_local.unsqueeze(0).expand(env.num_worlds, -1)
    r_world = _lf_quat_apply(obj_quat_sim, local)
    support_point_pos = obj_pos_sim + r_world
    support_point_vel = obj_vel_sim + torch.cross(obj_angvel_sim, r_world, dim=-1)

    env.support_proxy_last_force = pos_force.detach()
    env.support_proxy_last_torque = rot_torque.detach()
    env.support_proxy_last_pos = cur_qpos[:, :3].detach()
    env.support_proxy_last_vel = cur_qvel[:, :3].detach()
    env.support_proxy_last_support_point_pos = support_point_pos.detach()
    env.support_proxy_last_support_point_vel = support_point_vel.detach()
    env.support_proxy_last_idx = idx


def _apply_support_proxy_force(config: Config, env: MJWPEnv):
    """Apply a connector wrench from the virtual support proxy to the object."""
    if not hasattr(env, "support_proxy_ref_pos"):
        return

    obj_body_id = mujoco.mj_name2id(env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "object")
    if obj_body_id == -1:
        return

    xfrc_applied = wp.to_torch(env.data_wp.xfrc_applied)
    _clear_object_wrench_once(env, obj_body_id, xfrc_applied)

    qpos = wp.to_torch(env.data_wp.qpos)
    qvel = wp.to_torch(env.data_wp.qvel)
    obj_jnt_id = env.model_cpu.body_jntadr[obj_body_id]
    obj_qadr = env.model_cpu.jnt_qposadr[obj_jnt_id]
    obj_vadr = env.model_cpu.jnt_dofadr[obj_jnt_id]
    obj_pos_sim = qpos[:, obj_qadr : obj_qadr + 3]
    obj_vel_sim = qvel[:, obj_vadr : obj_vadr + 3]
    obj_quat_sim = qpos[:, obj_qadr + 3 : obj_qadr + 7]
    obj_angvel_sim = qvel[:, obj_vadr + 3 : obj_vadr + 6]
    obj_quat_sim = obj_quat_sim / obj_quat_sim.norm(dim=-1, keepdim=True).clamp(
        min=1e-8
    )

    N = obj_pos_sim.shape[0]
    local = env.support_proxy_point_local.unsqueeze(0).expand(N, -1)
    r_world = _lf_quat_apply(obj_quat_sim, local)
    support_point_pos = obj_pos_sim + r_world
    support_point_vel = obj_vel_sim + torch.cross(obj_angvel_sim, r_world, dim=-1)

    time_arr = wp.to_torch(env.data_wp.time)
    t = time_arr[0].item()
    dt = float(getattr(env, "support_proxy_ref_dt", config.sim_dt))
    T = env.support_proxy_ref_pos.shape[0]
    idx = min(int(t / dt), T - 1)
    proxy_pos = env.support_proxy_ref_pos[idx].unsqueeze(0).expand(N, -1)
    proxy_vel = env.support_proxy_ref_vel[idx].unsqueeze(0).expand(N, -1)

    obj_mass = env.model_cpu.body_mass[obj_body_id]
    gravity_z = -env.model_cpu.opt.gravity[2]
    gravity_force = torch.zeros_like(support_point_pos)
    gravity_force[:, 2] = config.support_proxy_gravity_scale * obj_mass * gravity_z

    ramp = min(t / 0.5, 1.0)
    kp = float(config.support_proxy_connector_kp) * ramp
    if config.support_proxy_connector_kd < 0:
        kd = (
            2.0 * (obj_mass * max(float(config.support_proxy_connector_kp), 0.0)) ** 0.5
        )
    else:
        kd = float(config.support_proxy_connector_kd)
    kd *= ramp

    force = (
        gravity_force
        + kp * (proxy_pos - support_point_pos)
        + kd * (proxy_vel - support_point_vel)
    )
    force = _clamp_vector_norm(
        torch.nan_to_num(force, nan=0.0), config.support_proxy_force_clamp
    )
    torque = torch.cross(r_world, force, dim=-1)
    torque = _clamp_vector_norm(
        torch.nan_to_num(torque, nan=0.0), config.support_proxy_torque_clamp
    )

    xfrc_applied[:, obj_body_id, :3] += force
    xfrc_applied[:, obj_body_id, 3:6] += torque
    env.support_proxy_last_force = force.detach()
    env.support_proxy_last_torque = torque.detach()
    env.support_proxy_last_pos = proxy_pos.detach()
    env.support_proxy_last_vel = proxy_vel.detach()
    env.support_proxy_last_support_point_pos = support_point_pos.detach()
    env.support_proxy_last_support_point_vel = support_point_vel.detach()
    env.support_proxy_last_idx = idx
    wp.copy(env.data_wp.xfrc_applied, wp.from_torch(xfrc_applied))


def _update_support_proxy_mocap_pad(config: Config, env: MJWPEnv):
    """Move a mocap contact pad to the support proxy target."""
    if not hasattr(env, "support_proxy_ref_pos"):
        return

    if not hasattr(env, "_support_proxy_mocap_id"):
        body_id = mujoco.mj_name2id(
            env.model_cpu,
            mujoco.mjtObj.mjOBJ_BODY,
            config.support_proxy_mocap_body_name,
        )
        if body_id == -1:
            env._support_proxy_mocap_id = -1
            loguru.logger.warning(
                "support proxy mocap body '{}' not found; mocap_pad disabled.",
                config.support_proxy_mocap_body_name,
            )
            return
        env._support_proxy_mocap_id = env.model_cpu.body_mocapid[body_id]
        if env._support_proxy_mocap_id < 0:
            loguru.logger.warning(
                "support proxy body '{}' is not a mocap body; mocap_pad disabled.",
                config.support_proxy_mocap_body_name,
            )
            return

    if env._support_proxy_mocap_id < 0:
        return

    time_arr = wp.to_torch(env.data_wp.time)
    t = time_arr[0].item()
    dt = float(getattr(env, "support_proxy_ref_dt", config.sim_dt))
    T = env.support_proxy_ref_pos.shape[0]
    idx = min(int(t / dt), T - 1)
    proxy_pos = env.support_proxy_ref_pos[idx]
    proxy_vel = env.support_proxy_ref_vel[idx]

    mocap_pos_all = wp.to_torch(env.data_wp.mocap_pos)
    mocap_quat_all = wp.to_torch(env.data_wp.mocap_quat)
    mid = env._support_proxy_mocap_id
    N = mocap_pos_all.shape[0]
    mocap_pos_all[:, mid] = proxy_pos.unsqueeze(0).expand(N, -1)
    if config.support_proxy_mocap_quat_mode == "object_ref" and hasattr(
        env, "support_proxy_ref_quat"
    ):
        mocap_quat = env.support_proxy_ref_quat[idx]
    else:
        mocap_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=proxy_pos.device, dtype=proxy_pos.dtype
        )
    mocap_quat_all[:, mid] = mocap_quat.unsqueeze(0).expand(N, -1)

    obj_body_id = mujoco.mj_name2id(env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "object")
    if obj_body_id == -1:
        return
    if config.support_proxy_mode == "mocap_pad":
        xfrc_applied = wp.to_torch(env.data_wp.xfrc_applied)
        _clear_object_wrench_once(env, obj_body_id, xfrc_applied)
        wp.copy(env.data_wp.xfrc_applied, wp.from_torch(xfrc_applied))

    qpos = wp.to_torch(env.data_wp.qpos)
    qvel = wp.to_torch(env.data_wp.qvel)
    obj_jnt_id = env.model_cpu.body_jntadr[obj_body_id]
    obj_qadr = env.model_cpu.jnt_qposadr[obj_jnt_id]
    obj_vadr = env.model_cpu.jnt_dofadr[obj_jnt_id]
    obj_pos_sim = qpos[:, obj_qadr : obj_qadr + 3]
    obj_vel_sim = qvel[:, obj_vadr : obj_vadr + 3]
    obj_quat_sim = qpos[:, obj_qadr + 3 : obj_qadr + 7]
    obj_angvel_sim = qvel[:, obj_vadr + 3 : obj_vadr + 6]
    obj_quat_sim = obj_quat_sim / obj_quat_sim.norm(dim=-1, keepdim=True).clamp(
        min=1e-8
    )
    local = env.support_proxy_point_local.unsqueeze(0).expand(N, -1)
    r_world = _lf_quat_apply(obj_quat_sim, local)
    support_point_pos = obj_pos_sim + r_world
    support_point_vel = obj_vel_sim + torch.cross(obj_angvel_sim, r_world, dim=-1)

    env.support_proxy_last_force = torch.zeros_like(support_point_pos)
    env.support_proxy_last_torque = torch.zeros_like(support_point_pos)
    env.support_proxy_last_pos = proxy_pos.unsqueeze(0).expand(N, -1).detach()
    env.support_proxy_last_vel = proxy_vel.unsqueeze(0).expand(N, -1).detach()
    env.support_proxy_last_support_point_pos = support_point_pos.detach()
    env.support_proxy_last_support_point_vel = support_point_vel.detach()
    env.support_proxy_last_idx = idx


def get_support_proxy_state(config: Config, env: MJWPEnv) -> dict[str, np.ndarray]:
    """Return first-world support proxy diagnostics for trajectory saving."""
    if not config.support_proxy_enabled or not hasattr(env, "support_proxy_last_force"):
        return {}

    def _first(tensor: torch.Tensor) -> np.ndarray:
        return tensor[0].detach().cpu().numpy().astype(np.float32)

    return {
        "support_proxy_force": _first(env.support_proxy_last_force),
        "support_proxy_torque": _first(env.support_proxy_last_torque),
        "support_proxy_pos": _first(env.support_proxy_last_pos),
        "support_proxy_vel": _first(env.support_proxy_last_vel),
        "support_point_pos": _first(env.support_proxy_last_support_point_pos),
        "support_point_vel": _first(env.support_proxy_last_support_point_vel),
        "support_proxy_ref_idx": np.asarray(
            int(getattr(env, "support_proxy_last_idx", -1)), dtype=np.int32
        ),
    }


def _update_object_weld_target(config: Config, env: MJWPEnv):
    """Update the mocap 'object_target' body to track the reference trajectory.

    Used with scene_weld.xml: a soft weld equality constraint pulls the freejoint
    object toward this mocap body. MuJoCo's solver handles pos+orient coupling.
    """
    if not hasattr(env, "_weld_mocap_id"):
        # Find the mocap body index for "object_target"
        body_id = mujoco.mj_name2id(
            env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "object_target"
        )
        if body_id == -1:
            env._weld_mocap_id = -1
            return
        # mocap body index (0-based among mocap bodies)
        # In MuJoCo, body_mocapid maps body_id → mocap_id
        env._weld_mocap_id = env.model_cpu.body_mocapid[body_id]

    if env._weld_mocap_id < 0:
        return

    if not hasattr(env, "partner_force_ref_pos"):
        return

    # Get reference for current time
    time_arr = wp.to_torch(env.data_wp.time)
    t = time_arr[0].item()
    dt = float(
        getattr(
            env,
            "partner_force_ref_dt",
            config.partner_force_ref_dt
            if config.partner_force_ref_dt > 0
            else config.ref_dt,
        )
    )
    T = env.partner_force_ref_pos.shape[0]
    idx = min(int(t / dt), T - 1)

    ref_pos = env.partner_force_ref_pos[idx]  # (3,)
    ref_quat = env.partner_force_ref_quat[idx]  # (4,) wxyz

    # Write to mocap body (shared-memory in-place)
    mocap_pos_all = wp.to_torch(env.data_wp.mocap_pos)  # (N, nmocap, 3)
    mocap_quat_all = wp.to_torch(env.data_wp.mocap_quat)  # (N, nmocap, 4)
    mid = env._weld_mocap_id
    N = mocap_pos_all.shape[0]
    mocap_pos_all[:, mid] = ref_pos.unsqueeze(0).expand(N, -1)
    mocap_quat_all[:, mid] = ref_quat.unsqueeze(0).expand(N, -1)


def _update_mocap_partner(env: MJWPEnv):
    """Update mocap body positions from partner trajectory based on current sim time.

    Uses wp.to_torch for shared-memory in-place writes (wp.copy fails after
    CUDA graph capture because data_wp.mocap_pos.ptr becomes None).
    """
    # Get current time from data
    time_arr = wp.to_torch(env.data_wp.time)  # (N,)
    t = time_arr[0].item()  # all worlds share same time

    # Map sim time to trajectory frame index
    dt = env.mocap_partner_dt
    T = env.mocap_partner_pos.shape[0]
    idx = min(int(t / dt), T - 1)

    # Get position and quaternion for this frame: (2, 3) and (2, 4)
    pos = env.mocap_partner_pos[idx]  # (2, 3) on GPU
    quat = env.mocap_partner_quat[idx]  # (2, 4) on GPU

    # Write via shared-memory torch view (in-place, no wp.copy needed)
    mocap_pos_all = wp.to_torch(env.data_wp.mocap_pos)  # (N, nmocap, 3)
    if mocap_pos_all.shape[1] >= 2:
        mocap_quat_all = wp.to_torch(env.data_wp.mocap_quat)  # (N, nmocap, 4)
        N = mocap_pos_all.shape[0]
        mocap_pos_all[:, :2] = pos.unsqueeze(0).expand(N, -1, -1)
        mocap_quat_all[:, :2] = quat.unsqueeze(0).expand(N, -1, -1)


def _load_mocap_partner(config: Config, env: MJWPEnv):
    """Load partner trajectory data and attach to env for runtime updates."""
    import os

    path = config.mocap_partner_trajectory
    if not os.path.isabs(path):
        # Resolve relative to data directory (same dir as trajectory_kinematic.npz)
        data_dir = os.path.dirname(config.data_path)
        path = os.path.join(data_dir, path)

    data = np.load(path)
    partner_pos = data["partner_pos"]  # (T, 2, 3)
    partner_quat = data["partner_quat"]  # (T, 2, 4) wxyz format

    # Store as GPU tensors on env
    env.mocap_partner_pos = torch.from_numpy(partner_pos).float().to(config.device)
    env.mocap_partner_quat = torch.from_numpy(partner_quat).float().to(config.device)
    env.mocap_partner_dt = config.ref_dt  # partner trajectory is at ref framerate

    loguru.logger.info(
        f"Loaded mocap partner trajectory: {partner_pos.shape[0]} frames @ {1 / config.ref_dt:.0f}fps"
    )


def step_env(config: Config, env: MJWPEnv, ctrl_mujoco: torch.Tensor):
    """Step all worlds with provided MuJoCo-format controls of shape (N, nu)."""
    if ctrl_mujoco.dim() == 1:
        ctrl_mujoco = ctrl_mujoco.unsqueeze(0).repeat(env.num_worlds, 1)
    # Ensure we operate on the correct CUDA context/device
    with wp.ScopedDevice(env.device):
        env._object_wrench_cleared_this_step = False
        # apply perturbation
        env = apply_perturbation(config, env)
        # E024: apply partner force on object (simulating human partner support)
        if (
            config.partner_force_scale > 0
            or config.partner_force_spring_kp > 0
            or config.partner_force_spring_kp_rot > 0
        ):
            _apply_partner_force(config, env)
        # E006/E010: support proxy can act via direct wrench, mocap contact pad, or both.
        if config.support_proxy_enabled and config.support_proxy_mode in {
            "wrench",
            "wrench_pad",
        }:
            _apply_support_proxy_force(config, env)
        if config.support_proxy_enabled and config.support_proxy_mode in {
            "mocap_pad",
            "wrench_pad",
        }:
            _update_support_proxy_mocap_pad(config, env)
        if config.support_proxy_enabled and config.support_proxy_mode == "dynamic_weld":
            _apply_dynamic_support_pd(config, env)
        # E030: update weld target mocap body (for scene_weld.xml)
        if config.scene_name == "scene_weld":
            _update_object_weld_target(config, env)
        # step control
        wp.copy(env.data_wp.ctrl, wp.from_torch(ctrl_mujoco.to(torch.float32)))
        # E027b: object PD override — set object actuator ctrl to track ref
        if config.object_pd_override and hasattr(env, "object_pd_ref_pos"):
            _apply_object_pd_override(config, env)
        if _object_kinematic_override_enabled(config):
            _apply_object_kinematic_override(config, env)
        # Update partner mocap positions within rollout (E013: intra-rollout update)
        if (
            config.mocap_partner_intra_step
            and hasattr(env, "mocap_partner_pos")
            and env.mocap_partner_pos is not None
        ):
            _update_mocap_partner(env)
        wp.capture_launch(env.graph)
        if _object_kinematic_override_enabled(config):
            _apply_object_kinematic_override(config, env)


def save_env_params(config: Config, env: MJWPEnv):
    """Save the current simulation parameters."""
    # Only record which group is active; parameters are embedded in separate models
    # TODO: explicitly read pair_margin and xy_offset from env.data_wp
    # currently we choose this solution since pair_margin has a huge virtual dimension,
    # convert it to torch would lead to OOM
    pair_margin = 0.0
    xy_offset = 0.0
    return {"pair_margin": pair_margin, "xy_offset": xy_offset}


def load_env_params(config: Config, env: MJWPEnv, env_param: dict):
    """Load the simulation parameters.

    Parameters to be updated:
    - pair_margin
    - xy_offset of the object
    """
    # update model parameters (pair_margin)
    if "pair_margin" in env_param:
        pair_margin_single_np = np.full(
            shape=(config.npair,), fill_value=env_param["pair_margin"], dtype=np.float32
        )

        # 2. Copy this small array to the GPU
        pair_margin_override_wp = wp.from_numpy(
            pair_margin_single_np, dtype=wp.float32, device=config.device
        )

        # 3. Apply the stride trick to broadcast it
        # This makes Warp treat the single instance as if it were num_samples copies
        # without allocating any new memory.
        pair_margin_override_wp.strides = (0,) + pair_margin_override_wp.strides
        pair_margin_override_wp.shape = (
            config.num_samples,
        ) + pair_margin_override_wp.shape
        pair_margin_override_wp.ndim += 1
        wp.copy(env.model_wp.pair_margin, pair_margin_override_wp)

    # update object position (NOTE: currently, xy_offset is only one scalar, which means we only update in the diagonal direction)
    if "xy_offset" in env_param:
        qpos_override_th = wp.to_torch(env.data_wp.qpos)
        # TODO: make object pos detection automatic
        if config.embodiment_type == "bimanual":
            qpos_override_th[:, -14:-12] = (
                qpos_override_th[:, -14:-12] + env_param["xy_offset"]
            )
            qpos_override_th[:, -12:-10] = (
                qpos_override_th[:, -12:-10] + env_param["xy_offset"]
            )
        elif config.embodiment_type in ["right", "left"]:
            qpos_override_th[:, -7:-5] = (
                qpos_override_th[:, -7:-5] + env_param["xy_offset"]
            )
        elif config.embodiment_type in ["humanoid_object", "dual_humanoid_object"]:
            nq_obj = config.nq_obj  # 7 (freejoint) or 6 (contact_guidance)
            qpos_override_th[:, -nq_obj : -nq_obj + 2] = (
                qpos_override_th[:, -nq_obj : -nq_obj + 2] + env_param["xy_offset"]
            )

        wp.copy(env.data_wp.qpos, wp.from_torch(qpos_override_th))

    # update object actuator gains
    if "kp" in env_param or "kd" in env_param:
        actuator_ids = config.object_actuator_ids
        if not actuator_ids:
            loguru.logger.warning(
                "Object actuator ids are empty; skipping kp/kd updates."
            )
        else:
            kp = env_param.get("kp")
            kd = env_param.get("kd")
            if kp is None or kd is None:
                loguru.logger.warning(
                    "Both kp and kd are required to update actuator gains; skipping."
                )
            else:
                kp_np = np.asarray(kp, dtype=np.float32)
                kd_np = np.asarray(kd, dtype=np.float32)
                if kp_np.ndim == 0:
                    kp_np = np.full((len(actuator_ids),), kp_np, dtype=np.float32)
                if kd_np.ndim == 0:
                    kd_np = np.full((len(actuator_ids),), kd_np, dtype=np.float32)
                if kp_np.shape[0] != len(actuator_ids) or kd_np.shape[0] != len(
                    actuator_ids
                ):
                    raise ValueError(
                        "kp/kd size mismatch for object actuators: "
                        f"kp={kp_np.shape}, kd={kd_np.shape}, "
                        f"expected={len(actuator_ids)}"
                    )

                # Update CPU model (used for viewer and as source of truth)
                env.model_cpu.actuator_gainprm[actuator_ids, 0] = kp_np
                env.model_cpu.actuator_biasprm[actuator_ids, 1] = -kd_np

                # Propagate to MJWarp model if available
                if hasattr(env.model_wp, "actuator_gainprm") and hasattr(
                    env.model_wp, "actuator_biasprm"
                ):
                    gain_full = np.array(
                        env.model_cpu.actuator_gainprm, dtype=np.float32
                    )
                    bias_full = np.array(
                        env.model_cpu.actuator_biasprm, dtype=np.float32
                    )
                    wp.copy(
                        env.model_wp.actuator_gainprm,
                        wp.from_numpy(
                            gain_full, dtype=wp.float32, device=config.device
                        ),
                    )
                    wp.copy(
                        env.model_wp.actuator_biasprm,
                        wp.from_numpy(
                            bias_full, dtype=wp.float32, device=config.device
                        ),
                    )
                else:
                    loguru.logger.warning(
                        "MJWarp model has no actuator_gainprm/biasprm; updated CPU model only."
                    )

    return env


def _broadcast_state(data_wp, num_worlds: int):
    """Broadcast state from first world/env to all worlds/envs.

    This is a generic function that can be used by both MJWP and HDMI simulators.

    Args:
        data_wp: MuJoCo Warp data object (mjwarp.Data or wrapped version)
        num_worlds: Number of parallel worlds/environments
    """
    # Core state variables - always try these first
    qpos0 = wp.to_torch(data_wp.qpos)[:1]
    qvel0 = wp.to_torch(data_wp.qvel)[:1]
    time0 = wp.to_torch(data_wp.time)[:1]
    ctrl0 = wp.to_torch(data_wp.ctrl)[:1]

    # Handle time specially as it might be 1D
    if time0.dim() == 1:
        time_repeated = time0.repeat(num_worlds)
    else:
        time_repeated = time0.repeat(num_worlds, 1)

    wp.copy(data_wp.qpos, wp.from_torch(qpos0.repeat(num_worlds, 1)))
    wp.copy(data_wp.qvel, wp.from_torch(qvel0.repeat(num_worlds, 1)))
    wp.copy(data_wp.time, wp.from_torch(time_repeated))
    wp.copy(data_wp.ctrl, wp.from_torch(ctrl0.repeat(num_worlds, 1)))

    # Additional core state variables
    qacc0 = wp.to_torch(data_wp.qacc)[:1]
    wp.copy(data_wp.qacc, wp.from_torch(qacc0.repeat(num_worlds, 1)))

    act0 = wp.to_torch(data_wp.act)[:1]
    wp.copy(data_wp.act, wp.from_torch(act0.repeat(num_worlds, 1)))

    act_dot0 = wp.to_torch(data_wp.act_dot)[:1]
    wp.copy(data_wp.act_dot, wp.from_torch(act_dot0.repeat(num_worlds, 1)))

    # Forces and applied forces
    qfrc_applied0 = wp.to_torch(data_wp.qfrc_applied)[:1]
    wp.copy(data_wp.qfrc_applied, wp.from_torch(qfrc_applied0.repeat(num_worlds, 1)))

    xfrc_applied0 = wp.to_torch(data_wp.xfrc_applied)[:1]
    wp.copy(data_wp.xfrc_applied, wp.from_torch(xfrc_applied0.repeat(num_worlds, 1, 1)))

    # Mocap data
    mocap_pos0 = wp.to_torch(data_wp.mocap_pos)[:1]
    wp.copy(data_wp.mocap_pos, wp.from_torch(mocap_pos0.repeat(num_worlds, 1, 1)))

    mocap_quat0 = wp.to_torch(data_wp.mocap_quat)[:1]
    wp.copy(data_wp.mocap_quat, wp.from_torch(mocap_quat0.repeat(num_worlds, 1, 1)))

    # Spatial transformations
    xpos0 = wp.to_torch(data_wp.xpos)[:1]
    wp.copy(data_wp.xpos, wp.from_torch(xpos0.repeat(num_worlds, 1, 1)))

    xquat0 = wp.to_torch(data_wp.xquat)[:1]
    wp.copy(data_wp.xquat, wp.from_torch(xquat0.repeat(num_worlds, 1, 1)))

    xmat0 = wp.to_torch(data_wp.xmat)[:1]
    wp.copy(data_wp.xmat, wp.from_torch(xmat0.repeat(num_worlds, 1, 1, 1)))

    # Geometry positions
    geom_xpos0 = wp.to_torch(data_wp.geom_xpos)[:1]
    wp.copy(data_wp.geom_xpos, wp.from_torch(geom_xpos0.repeat(num_worlds, 1, 1)))

    geom_xmat0 = wp.to_torch(data_wp.geom_xmat)[:1]
    wp.copy(data_wp.geom_xmat, wp.from_torch(geom_xmat0.repeat(num_worlds, 1, 1, 1)))

    # Site positions
    site_xpos0 = wp.to_torch(data_wp.site_xpos)[:1]
    wp.copy(data_wp.site_xpos, wp.from_torch(site_xpos0.repeat(num_worlds, 1, 1)))


def sync_env(config: Config, env: MJWPEnv, mj_data: mujoco.MjData):
    """Broadcast the state from first env to all envs

    This function synchronizes states from the first environment to all environments.
    Uses safe copying with buffer size validation to avoid mismatches.
    """
    # Update mocap partner positions before broadcasting
    if hasattr(env, "mocap_partner_pos") and env.mocap_partner_pos is not None:
        t = mj_data.time
        dt = env.mocap_partner_dt
        T = env.mocap_partner_pos.shape[0]
        idx = min(int(t / dt), T - 1)
        pos = env.mocap_partner_pos[idx]  # (2, 3) GPU tensor
        quat = env.mocap_partner_quat[idx]  # (2, 4) GPU tensor
        # Write to world 0 of data_wp, _broadcast_state will copy to all worlds
        mocap_pos_all = wp.to_torch(env.data_wp.mocap_pos)  # (N, nmocap, 3)
        if mocap_pos_all.shape[1] > 0:
            mocap_quat_all = wp.to_torch(env.data_wp.mocap_quat)  # (N, nmocap, 4)
            mocap_pos_all[0] = pos
            mocap_quat_all[0] = quat

    _broadcast_state(env.data_wp, env.num_worlds)


def sync_env_mujoco(config: Config, env: MJWPEnv, mj_data: mujoco.MjData):
    """Sync state from mj_data to env.data_wp"""
    # Define field mappings with their data and target shapes
    fields = [
        # Core state variables
        ("qpos", mj_data.qpos, (env.data_wp.nworld, -1)),
        ("qvel", mj_data.qvel, (env.data_wp.nworld, -1)),
        ("qacc", mj_data.qacc, (env.data_wp.nworld, -1)),
        ("time", np.array([mj_data.time], dtype=np.float32), (env.data_wp.nworld, 1)),
        ("ctrl", mj_data.ctrl, (env.data_wp.nworld, -1)),
        ("act", mj_data.act, (env.data_wp.nworld, -1)),
        ("act_dot", mj_data.act_dot, (env.data_wp.nworld, -1)),
        ("qacc_warmstart", mj_data.qacc_warmstart, (env.data_wp.nworld, -1)),
        # Forces
        ("qfrc_applied", mj_data.qfrc_applied, (env.data_wp.nworld, -1)),
        ("xfrc_applied", mj_data.xfrc_applied, (env.data_wp.nworld, -1, -1)),
        # Energy (2D: kinetic + potential)
        ("energy", mj_data.energy, (env.data_wp.nworld, 2)),
        # Mocap data
        ("mocap_pos", mj_data.mocap_pos, (env.data_wp.nworld, -1, 3)),
        ("mocap_quat", mj_data.mocap_quat, (env.data_wp.nworld, -1, 4)),
        # Spatial transformations
        ("xpos", mj_data.xpos, (env.data_wp.nworld, -1, 3)),
        ("xquat", mj_data.xquat, (env.data_wp.nworld, -1, 4)),
        ("xmat", mj_data.xmat, (env.data_wp.nworld, -1, 9)),
        ("xipos", mj_data.xipos, (env.data_wp.nworld, -1, 3)),
        ("ximat", mj_data.ximat, (env.data_wp.nworld, -1, 9)),
        # Geometry positions
        ("geom_xpos", mj_data.geom_xpos, (env.data_wp.nworld, -1, 3)),
        ("geom_xmat", mj_data.geom_xmat, (env.data_wp.nworld, -1, 9)),
        ("site_xpos", mj_data.site_xpos, (env.data_wp.nworld, -1, 3)),
        ("site_xmat", mj_data.site_xmat, (env.data_wp.nworld, -1, 9)),
        # Body dynamics (spatial vectors)
        ("cacc", mj_data.cacc, (env.data_wp.nworld, -1, 6)),
        ("cfrc_int", mj_data.cfrc_int, (env.data_wp.nworld, -1, 6)),
        ("cfrc_ext", mj_data.cfrc_ext, (env.data_wp.nworld, -1, 6)),
        # Sensor data
        ("sensordata", mj_data.sensordata, (env.data_wp.nworld, -1)),
        # Actuator data
        ("actuator_length", mj_data.actuator_length, (env.data_wp.nworld, -1)),
        ("actuator_velocity", mj_data.actuator_velocity, (env.data_wp.nworld, -1)),
        ("actuator_force", mj_data.actuator_force, (env.data_wp.nworld, -1)),
        # Tendon data
        ("ten_length", mj_data.ten_length, (env.data_wp.nworld, -1)),
        ("ten_velocity", mj_data.ten_velocity, (env.data_wp.nworld, -1)),
    ]

    # Contact struct fields - these need special handling
    contact_fields = [
        ("dist", "contact.dist"),
        ("pos", "contact.pos"),
        ("frame", "contact.frame"),
        ("includemargin", "contact.includemargin"),
        ("friction", "contact.friction"),
        ("solref", "contact.solref"),
        ("solreffriction", "contact.solreffriction"),
        ("solimp", "contact.solimp"),
        ("dim", "contact.dim"),
        ("geom", "contact.geom"),
        ("efc_address", "contact.efc_address"),
        ("worldid", "contact.worldid"),
    ]

    # Constraint (efc) fields - these are direct fields on mj_data, not nested in a struct
    efc_fields = [
        ("efc_type", "efc.type"),
        ("efc_id", "efc.id"),
        ("efc_J", "efc.J"),
        ("efc_pos", "efc.pos"),
        ("efc_margin", "efc.margin"),
        ("efc_D", "efc.D"),
        ("efc_vel", "efc.vel"),
        ("efc_aref", "efc.aref"),
        ("efc_frictionloss", "efc.frictionloss"),
        ("efc_force", "efc.force"),
    ]

    # Copy data to all environments
    for field_name, source_data, target_shape in fields:
        # Skip if field doesn't exist in either source or destination
        if not hasattr(mj_data, field_name) or not hasattr(env.data_wp, field_name):
            continue

        source_data_np = np.array(source_data, dtype=np.float32)
        tensor = torch.from_numpy(source_data_np).to(config.device)

        # Handle scalar time field
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)

        # Reshape tensor to match target shape
        if len(target_shape) == 2:
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0).repeat(target_shape[0], 1)
            elif tensor.dim() == 2:
                tensor = tensor.unsqueeze(0).repeat(target_shape[0], 1, 1).squeeze(-1)
        elif len(target_shape) == 3:
            if tensor.dim() == 1:
                # For 1D data that needs to be 3D
                tensor = (
                    tensor.unsqueeze(0)
                    .unsqueeze(-1)
                    .repeat(target_shape[0], 1, target_shape[2])
                )
            elif tensor.dim() == 2:
                tensor = tensor.unsqueeze(0).repeat(target_shape[0], 1, 1)
            elif tensor.dim() == 3:
                tensor = tensor.unsqueeze(0).repeat(target_shape[0], 1, 1, 1).squeeze(1)
        elif len(target_shape) == 4:
            tensor = tensor.unsqueeze(0).repeat(target_shape[0], 1, 1, 1)

        wp.copy(getattr(env.data_wp, field_name), wp.from_torch(tensor))

    # Handle contact struct fields
    for mj_field, wp_field in contact_fields:
        if hasattr(mj_data.contact, mj_field):
            source_data = getattr(mj_data.contact, mj_field)
            source_data_np = np.array(source_data, dtype=np.float32)
            tensor = torch.from_numpy(source_data_np).to(config.device)

            # Handle scalar fields
            if tensor.dim() == 0:
                tensor = tensor.unsqueeze(0)

            # Reshape for batched environments
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0).repeat(env.data_wp.nworld, 1)
            elif tensor.dim() == 2:
                tensor = tensor.unsqueeze(0).repeat(env.data_wp.nworld, 1, 1)
            elif tensor.dim() == 3:
                tensor = tensor.unsqueeze(0).repeat(env.data_wp.nworld, 1, 1, 1)

            # Get the destination field using nested attribute access
            dst_obj = env.data_wp
            for attr in wp_field.split("."):
                dst_obj = getattr(dst_obj, attr)
            wp.copy(dst_obj, wp.from_torch(tensor))

    # Handle efc fields - these are direct fields on mj_data
    for mj_field, wp_field in efc_fields:
        if hasattr(mj_data, mj_field):
            source_data = getattr(mj_data, mj_field)
            source_data_np = np.array(source_data, dtype=np.float32)
            tensor = torch.from_numpy(source_data_np).to(config.device)

            # Handle scalar fields
            if tensor.dim() == 0:
                tensor = tensor.unsqueeze(0)

            # Reshape for batched environments
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0).repeat(env.data_wp.nworld, 1)
            elif tensor.dim() == 2:
                tensor = tensor.unsqueeze(0).repeat(env.data_wp.nworld, 1, 1)
            elif tensor.dim() == 3:
                tensor = tensor.unsqueeze(0).repeat(env.data_wp.nworld, 1, 1, 1)

            # Get the destination field using nested attribute access
            dst_obj = env.data_wp
            for attr in wp_field.split("."):
                dst_obj = getattr(dst_obj, attr)
            wp.copy(dst_obj, wp.from_torch(tensor))

    return env


def copy_sample_state(
    config: Config, env: MJWPEnv, src_indices: torch.Tensor, dst_indices: torch.Tensor
):
    """Copy simulation state from source samples to destination samples.

    Args:
        config: Config
        env: MJWPEnv environment
        src_indices: Tensor of shape (n,) containing source sample indices
        dst_indices: Tensor of shape (n,) containing destination sample indices
    """
    # Convert to numpy for indexing
    src_idx = src_indices.cpu().numpy()
    dst_idx = dst_indices.cpu().numpy()

    # Get all state data as torch tensors
    qpos = wp.to_torch(env.data_wp.qpos)
    qvel = wp.to_torch(env.data_wp.qvel)
    qacc = wp.to_torch(env.data_wp.qacc)
    time_arr = wp.to_torch(env.data_wp.time)
    ctrl = wp.to_torch(env.data_wp.ctrl)
    act = wp.to_torch(env.data_wp.act)
    act_dot = wp.to_torch(env.data_wp.act_dot)
    qacc_warmstart = wp.to_torch(env.data_wp.qacc_warmstart)
    qfrc_applied = wp.to_torch(env.data_wp.qfrc_applied)
    xfrc_applied = wp.to_torch(env.data_wp.xfrc_applied)
    energy = wp.to_torch(env.data_wp.energy)
    mocap_pos = wp.to_torch(env.data_wp.mocap_pos)
    mocap_quat = wp.to_torch(env.data_wp.mocap_quat)
    xpos = wp.to_torch(env.data_wp.xpos)
    xquat = wp.to_torch(env.data_wp.xquat)
    xmat = wp.to_torch(env.data_wp.xmat)
    xipos = wp.to_torch(env.data_wp.xipos)
    ximat = wp.to_torch(env.data_wp.ximat)
    geom_xpos = wp.to_torch(env.data_wp.geom_xpos)
    geom_xmat = wp.to_torch(env.data_wp.geom_xmat)
    site_xpos = wp.to_torch(env.data_wp.site_xpos)
    site_xmat = wp.to_torch(env.data_wp.site_xmat)
    cacc = wp.to_torch(env.data_wp.cacc)
    cfrc_int = wp.to_torch(env.data_wp.cfrc_int)
    cfrc_ext = wp.to_torch(env.data_wp.cfrc_ext)
    sensordata = wp.to_torch(env.data_wp.sensordata)
    actuator_length = wp.to_torch(env.data_wp.actuator_length)
    actuator_velocity = wp.to_torch(env.data_wp.actuator_velocity)
    actuator_force = wp.to_torch(env.data_wp.actuator_force)
    ten_length = wp.to_torch(env.data_wp.ten_length)
    ten_velocity = wp.to_torch(env.data_wp.ten_velocity)

    # Copy from src to dst
    qpos[dst_idx] = qpos[src_idx]
    qvel[dst_idx] = qvel[src_idx]
    qacc[dst_idx] = qacc[src_idx]
    time_arr[dst_idx] = time_arr[src_idx]
    ctrl[dst_idx] = ctrl[src_idx]
    act[dst_idx] = act[src_idx]
    act_dot[dst_idx] = act_dot[src_idx]
    qacc_warmstart[dst_idx] = qacc_warmstart[src_idx]
    qfrc_applied[dst_idx] = qfrc_applied[src_idx]
    xfrc_applied[dst_idx] = xfrc_applied[src_idx]
    energy[dst_idx] = energy[src_idx]
    mocap_pos[dst_idx] = mocap_pos[src_idx]
    mocap_quat[dst_idx] = mocap_quat[src_idx]
    xpos[dst_idx] = xpos[src_idx]
    xquat[dst_idx] = xquat[src_idx]
    xmat[dst_idx] = xmat[src_idx]
    xipos[dst_idx] = xipos[src_idx]
    ximat[dst_idx] = ximat[src_idx]
    geom_xpos[dst_idx] = geom_xpos[src_idx]
    geom_xmat[dst_idx] = geom_xmat[src_idx]
    site_xpos[dst_idx] = site_xpos[src_idx]
    site_xmat[dst_idx] = site_xmat[src_idx]
    cacc[dst_idx] = cacc[src_idx]
    cfrc_int[dst_idx] = cfrc_int[src_idx]
    cfrc_ext[dst_idx] = cfrc_ext[src_idx]
    sensordata[dst_idx] = sensordata[src_idx]
    actuator_length[dst_idx] = actuator_length[src_idx]
    actuator_velocity[dst_idx] = actuator_velocity[src_idx]
    actuator_force[dst_idx] = actuator_force[src_idx]
    ten_length[dst_idx] = ten_length[src_idx]
    ten_velocity[dst_idx] = ten_velocity[src_idx]

    # Copy back to warp arrays
    wp.copy(env.data_wp.qpos, wp.from_torch(qpos))
    wp.copy(env.data_wp.qvel, wp.from_torch(qvel))
    wp.copy(env.data_wp.qacc, wp.from_torch(qacc))
    wp.copy(env.data_wp.time, wp.from_torch(time_arr))
    wp.copy(env.data_wp.ctrl, wp.from_torch(ctrl))
    wp.copy(env.data_wp.act, wp.from_torch(act))
    wp.copy(env.data_wp.act_dot, wp.from_torch(act_dot))
    wp.copy(env.data_wp.qacc_warmstart, wp.from_torch(qacc_warmstart))
    wp.copy(env.data_wp.qfrc_applied, wp.from_torch(qfrc_applied))
    wp.copy(env.data_wp.xfrc_applied, wp.from_torch(xfrc_applied))
    wp.copy(env.data_wp.energy, wp.from_torch(energy))
    wp.copy(env.data_wp.mocap_pos, wp.from_torch(mocap_pos))
    wp.copy(env.data_wp.mocap_quat, wp.from_torch(mocap_quat))
    wp.copy(env.data_wp.xpos, wp.from_torch(xpos))
    wp.copy(env.data_wp.xquat, wp.from_torch(xquat))
    wp.copy(env.data_wp.xmat, wp.from_torch(xmat))
    wp.copy(env.data_wp.xipos, wp.from_torch(xipos))
    wp.copy(env.data_wp.ximat, wp.from_torch(ximat))
    wp.copy(env.data_wp.geom_xpos, wp.from_torch(geom_xpos))
    wp.copy(env.data_wp.geom_xmat, wp.from_torch(geom_xmat))
    wp.copy(env.data_wp.site_xpos, wp.from_torch(site_xpos))
    wp.copy(env.data_wp.site_xmat, wp.from_torch(site_xmat))
    wp.copy(env.data_wp.cacc, wp.from_torch(cacc))
    wp.copy(env.data_wp.cfrc_int, wp.from_torch(cfrc_int))
    wp.copy(env.data_wp.cfrc_ext, wp.from_torch(cfrc_ext))
    wp.copy(env.data_wp.sensordata, wp.from_torch(sensordata))
    wp.copy(env.data_wp.actuator_length, wp.from_torch(actuator_length))
    wp.copy(env.data_wp.actuator_velocity, wp.from_torch(actuator_velocity))
    wp.copy(env.data_wp.actuator_force, wp.from_torch(actuator_force))
    wp.copy(env.data_wp.ten_length, wp.from_torch(ten_length))
    wp.copy(env.data_wp.ten_velocity, wp.from_torch(ten_velocity))


def _copy_state(src: mjwarp.Data, dst: mjwarp.Data):
    """Copy the state from src to dst

    TODO: this function is a temporary solution for domain randomization. A better way should be defining a new warp kernel to update simulation parameter accordingly.

    Args:
        src: mjwarp.Data
            the source data to be copied from
        dst: mjwarp.Data
            the destination data to be copied to
    """
    # Core state variables
    wp.copy(dst.qpos, src.qpos)
    wp.copy(dst.qvel, src.qvel)
    wp.copy(dst.qacc, src.qacc)
    wp.copy(dst.time, src.time)
    wp.copy(dst.ctrl, src.ctrl)
    wp.copy(dst.act, src.act)
    wp.copy(dst.act_dot, src.act_dot)
    wp.copy(dst.qacc_warmstart, src.qacc_warmstart)

    # Forces and applied forces
    wp.copy(dst.qfrc_applied, src.qfrc_applied)
    wp.copy(dst.xfrc_applied, src.xfrc_applied)

    # Energy tracking
    wp.copy(dst.energy, src.energy)

    # Mocap data
    wp.copy(dst.mocap_pos, src.mocap_pos)
    wp.copy(dst.mocap_quat, src.mocap_quat)

    # Spatial transformations
    wp.copy(dst.xpos, src.xpos)
    wp.copy(dst.xquat, src.xquat)
    wp.copy(dst.xmat, src.xmat)
    wp.copy(dst.xipos, src.xipos)
    wp.copy(dst.ximat, src.ximat)

    # Geometry positions
    wp.copy(dst.geom_xpos, src.geom_xpos)
    wp.copy(dst.geom_xmat, src.geom_xmat)
    wp.copy(dst.site_xpos, src.site_xpos)
    wp.copy(dst.site_xmat, src.site_xmat)

    # Camera and lighting (if present)
    if hasattr(src, "cam_xpos") and hasattr(dst, "cam_xpos"):
        wp.copy(dst.cam_xpos, src.cam_xpos)
        wp.copy(dst.cam_xmat, src.cam_xmat)
    if hasattr(src, "light_xpos") and hasattr(dst, "light_xpos"):
        wp.copy(dst.light_xpos, src.light_xpos)
        wp.copy(dst.light_xdir, src.light_xdir)

    # Body dynamics
    wp.copy(dst.cacc, src.cacc)
    wp.copy(dst.cfrc_int, src.cfrc_int)
    wp.copy(dst.cfrc_ext, src.cfrc_ext)

    # Sensor data
    wp.copy(dst.sensordata, src.sensordata)

    # Actuator data
    wp.copy(dst.actuator_length, src.actuator_length)
    wp.copy(dst.actuator_velocity, src.actuator_velocity)
    wp.copy(dst.actuator_force, src.actuator_force)

    # Tendon data
    wp.copy(dst.ten_length, src.ten_length)
    wp.copy(dst.ten_velocity, src.ten_velocity)

    # Contact struct - copy all fields
    wp.copy(dst.contact.dist, src.contact.dist)
    wp.copy(dst.contact.pos, src.contact.pos)
    wp.copy(dst.contact.frame, src.contact.frame)
    wp.copy(dst.contact.includemargin, src.contact.includemargin)
    wp.copy(dst.contact.friction, src.contact.friction)
    wp.copy(dst.contact.solref, src.contact.solref)
    wp.copy(dst.contact.solreffriction, src.contact.solreffriction)
    wp.copy(dst.contact.solimp, src.contact.solimp)
    wp.copy(dst.contact.dim, src.contact.dim)
    wp.copy(dst.contact.geom, src.contact.geom)
    wp.copy(dst.contact.efc_address, src.contact.efc_address)
    wp.copy(dst.contact.worldid, src.contact.worldid)

    # Constraint (efc) struct - copy all fields
    wp.copy(dst.efc.type, src.efc.type)
    wp.copy(dst.efc.id, src.efc.id)
    wp.copy(dst.efc.J, src.efc.J)
    wp.copy(dst.efc.pos, src.efc.pos)
    wp.copy(dst.efc.margin, src.efc.margin)
    wp.copy(dst.efc.D, src.efc.D)
    wp.copy(dst.efc.vel, src.efc.vel)
    wp.copy(dst.efc.aref, src.efc.aref)
    wp.copy(dst.efc.frictionloss, src.efc.frictionloss)
    wp.copy(dst.efc.force, src.efc.force)
    # Note: The workspace fields (Jaref, Ma, grad, etc.) are typically not needed for state transfer
    # as they are recomputed during solving, but include if needed:
    # wp.copy(dst.efc.Jaref, src.efc.Jaref)
    # wp.copy(dst.efc.Ma, src.efc.Ma)
    # wp.copy(dst.efc.grad, src.efc.grad)
    # ... (other workspace fields)
    #
    return dst
