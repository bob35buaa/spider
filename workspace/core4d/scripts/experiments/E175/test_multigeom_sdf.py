#!/usr/bin/env python3
"""Deterministic CPU regression checks for E175 multi-geom object SDF."""

from __future__ import annotations

from types import SimpleNamespace

import mujoco
import torch

from spider.config import Config
from spider.simulators.mjwp import (
    _geom_box_sdf_min,
    _geom_box_union_sdf_min,
    _resolve_object_collision_geom_ids,
)


MODEL_XML = """
<mujoco>
  <worldbody>
    <body name="robot">
      <geom name="probe" type="sphere" size="0.1"/>
    </body>
    <body name="object">
      <geom name="object_collision" type="box" size="1 1 1"/>
      <geom name="object_collision_side" type="box" size="0.25 1 0.5"/>
    </body>
  </worldbody>
</mujoco>
"""


def many_box_model_xml(count: int = 33) -> str:
    boxes = "\n".join(
        (
            f'<geom name="object_collision{"_" + str(index) if index else ""}" '
            f'type="box" size="{0.05 + 0.001 * index:.6f} 0.08 0.10"/>'
        )
        for index in range(count)
    )
    return f"""
<mujoco>
  <worldbody>
    <body name="robot">
      <geom name="probe_sphere" type="sphere" size="0.07"/>
      <geom name="probe_capsule" type="capsule" size="0.04 0.11"/>
    </body>
    <body name="object">
      {boxes}
    </body>
  </worldbody>
</mujoco>
"""


def main() -> int:
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    probe = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "probe")
    primary = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision"
    )
    side = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision_side"
    )

    assert _resolve_object_collision_geom_ids(model, "primary") == [primary]
    assert _resolve_object_collision_geom_ids(model, "union") == [primary, side]
    try:
        _resolve_object_collision_geom_ids(model, "invalid")
    except ValueError as exc:
        assert "object_collision_sdf_mode" in str(exc)
    else:
        raise AssertionError("invalid SDF mode must fail")

    config = Config()
    config.device = "cpu"
    env = SimpleNamespace(model_cpu=model)
    geom_xpos = torch.zeros((2, model.ngeom, 3), dtype=torch.float64)
    geom_xmat = torch.eye(3, dtype=torch.float64).repeat(2, model.ngeom, 1, 1)

    # World 0: the secondary box is translated near the probe.
    geom_xpos[0, probe] = torch.tensor([2.5, 0.0, 0.0])
    geom_xpos[0, primary] = torch.tensor([0.0, 0.0, 0.0])
    geom_xpos[0, side] = torch.tensor([3.0, 0.0, 0.0])

    # World 1: rotate the secondary box by 90 degrees around z. Its long local
    # y axis becomes world x, making it the nearest proxy component.
    geom_xpos[1, probe] = torch.tensor([3.8, 0.0, 0.0])
    geom_xpos[1, primary] = torch.tensor([0.0, 0.0, 0.0])
    geom_xpos[1, side] = torch.tensor([3.0, 0.0, 0.0])
    geom_xmat[1, side] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )

    primary_sdf = _geom_box_sdf_min(
        config,
        env,
        [probe],
        primary,
        geom_xpos=geom_xpos,
        geom_xmat=geom_xmat,
        half_ext=torch.tensor(model.geom_size[primary], dtype=torch.float64),
    )
    side_sdf = _geom_box_sdf_min(
        config,
        env,
        [probe],
        side,
        geom_xpos=geom_xpos,
        geom_xmat=geom_xmat,
        half_ext=torch.tensor(model.geom_size[side], dtype=torch.float64),
    )
    union_sdf = _geom_box_union_sdf_min(
        config,
        env,
        [probe],
        [primary, side],
        geom_xpos=geom_xpos,
        geom_xmat=geom_xmat,
    )

    torch.testing.assert_close(
        union_sdf, torch.minimum(primary_sdf, side_sdf), atol=1e-7, rtol=0.0
    )
    torch.testing.assert_close(
        union_sdf,
        torch.tensor([0.15, -0.30], dtype=torch.float64),
        atol=1e-7,
        rtol=0.0,
    )

    legacy = _geom_box_union_sdf_min(
        config,
        env,
        [probe],
        [primary],
        geom_xpos=geom_xpos,
        geom_xmat=geom_xmat,
    )
    torch.testing.assert_close(legacy, primary_sdf, atol=1e-7, rtol=0.0)

    # Cross the 16-box chunk boundary and compare the vectorized production
    # implementation against the exact scalar per-box reference.
    many = mujoco.MjModel.from_xml_string(many_box_model_xml())
    object_ids = _resolve_object_collision_geom_ids(many, "union")
    robot_ids = [
        mujoco.mj_name2id(
            many, mujoco.mjtObj.mjOBJ_GEOM, "probe_sphere"
        ),
        mujoco.mj_name2id(
            many, mujoco.mjtObj.mjOBJ_GEOM, "probe_capsule"
        ),
    ]
    many_env = SimpleNamespace(model_cpu=many)
    worlds = 5
    many_xpos = torch.zeros((worlds, many.ngeom, 3), dtype=torch.float64)
    many_xmat = torch.eye(3, dtype=torch.float64).repeat(
        worlds, many.ngeom, 1, 1
    )
    for world in range(worlds):
        many_xpos[world, robot_ids[0]] = torch.tensor(
            [0.11 * world, -0.07, 0.03], dtype=torch.float64
        )
        many_xpos[world, robot_ids[1]] = torch.tensor(
            [-0.05, 0.04 * world, 0.15], dtype=torch.float64
        )
        for index, gid in enumerate(object_ids):
            many_xpos[world, gid] = torch.tensor(
                [
                    0.035 * (index % 7),
                    0.04 * ((index // 7) % 5),
                    0.01 * world,
                ],
                dtype=torch.float64,
            )
            if (index + world) % 3 == 0:
                many_xmat[world, gid] = torch.tensor(
                    [
                        [0.0, -1.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=torch.float64,
                )
    scalar = torch.stack(
        [
            _geom_box_sdf_min(
                config,
                many_env,
                robot_ids,
                gid,
                geom_xpos=many_xpos,
                geom_xmat=many_xmat,
                half_ext=torch.tensor(
                    many.geom_size[gid], dtype=torch.float64
                ),
            )
            for gid in object_ids
        ],
        dim=1,
    ).min(dim=1).values
    chunked = _geom_box_union_sdf_min(
        config,
        many_env,
        robot_ids,
        object_ids,
        geom_xpos=many_xpos,
        geom_xmat=many_xmat,
    )
    torch.testing.assert_close(chunked, scalar, atol=1e-10, rtol=0.0)

    non_box = mujoco.MjModel.from_xml_string(
        """
        <mujoco><worldbody>
          <body name="object">
            <geom name="object_collision" type="box" size="1 1 1"/>
            <geom name="object_collision_bad" type="sphere" size="0.1"/>
          </body>
        </worldbody></mujoco>
        """
    )
    try:
        _resolve_object_collision_geom_ids(non_box, "union")
    except ValueError as exc:
        assert "requires box geoms" in str(exc)
    else:
        raise AssertionError("union resolver must reject non-box object geoms")

    print("E175 multi-geom SDF tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
