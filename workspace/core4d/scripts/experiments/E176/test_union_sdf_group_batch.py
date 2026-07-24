#!/usr/bin/env python3
"""Numerical regression for E176 batched robot-geom union-SDF groups."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import torch

from spider.config import Config
import spider.simulators.mjwp as mjwp


MODEL_XML = """
<mujoco>
  <asset>
    <mesh name="tetra"
          vertex="0 0 0  0.12 0 0  0 0.10 0  0 0 0.08"
          face="0 1 2  0 1 3  0 2 3  1 2 3"/>
  </asset>
  <worldbody>
    <body name="robot">
      <geom name="probe_sphere" type="sphere" size="0.10"/>
      <geom name="probe_capsule" type="capsule" size="0.05 0.10"/>
      <geom name="probe_mesh" type="mesh" mesh="tetra"/>
    </body>
    <body name="object">
      <geom name="object_collision" type="box" size="0.5 0.4 0.3"/>
      <geom name="object_collision_side" type="box" size="0.2 0.3 0.2"/>
    </body>
  </worldbody>
</mujoco>
"""


def geom_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def main() -> int:
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    sphere = geom_id(model, "probe_sphere")
    capsule = geom_id(model, "probe_capsule")
    mesh = geom_id(model, "probe_mesh")
    object_ids = [
        geom_id(model, "object_collision"),
        geom_id(model, "object_collision_side"),
    ]
    config = Config()
    config.device = "cpu"
    env = SimpleNamespace(model_cpu=model)

    worlds = 5
    xpos = torch.zeros((worlds, model.ngeom, 3), dtype=torch.float64)
    xmat = torch.eye(3, dtype=torch.float64).repeat(
        worlds, model.ngeom, 1, 1
    )
    xpos[:, sphere, 0] = torch.tensor(
        [0.0, 0.35, 0.8, 1.2, -0.7], dtype=torch.float64
    )
    xpos[:, capsule, 1] = torch.tensor(
        [0.0, 0.5, 0.9, -0.6, 1.1], dtype=torch.float64
    )
    xpos[:, mesh, 2] = torch.tensor(
        [0.0, 0.4, 0.75, 1.0, -0.5], dtype=torch.float64
    )
    xpos[:, object_ids[1], :] = torch.tensor(
        [0.65, -0.15, 0.25], dtype=torch.float64
    )
    theta = torch.tensor(0.37, dtype=torch.float64)
    c, s = torch.cos(theta), torch.sin(theta)
    xmat[:, object_ids[1]] = torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )

    groups = [
        [sphere],
        [capsule],
        [mesh],
        [sphere, capsule],
        [capsule, mesh],
        [sphere, capsule, mesh],
        [sphere, capsule],
    ]
    expected = {
        tuple(group): mjwp._geom_box_union_sdf_min(
            config,
            env,
            group,
            object_ids,
            geom_xpos=xpos,
            geom_xmat=xmat,
        )
        for group in groups
    }

    with patch.object(
        mjwp,
        "_geom_box_union_sdf_min",
        wraps=mjwp._geom_box_union_sdf_min,
    ) as legacy:
        batched = mjwp._batched_geom_box_union_sdf_cache(
            config,
            env,
            groups,
            object_ids,
            geom_xpos=xpos,
            geom_xmat=xmat,
        )
        assert legacy.call_count == 0

    assert list(batched) == list(expected)
    for key, reference in expected.items():
        torch.testing.assert_close(
            batched[key],
            reference,
            atol=0.0,
            rtol=0.0,
        )

    moved = xpos.clone()
    moved[:, sphere, 0] += 0.25
    next_tick = mjwp._batched_geom_box_union_sdf_cache(
        config,
        env,
        groups,
        object_ids,
        geom_xpos=moved,
        geom_xmat=xmat,
    )
    assert not torch.equal(
        batched[(sphere,)],
        next_tick[(sphere,)],
    )
    print("E176 union-SDF group batching tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
