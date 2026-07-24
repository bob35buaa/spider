#!/usr/bin/env python3
"""Regression test for E176 tick-local exact union-SDF memoization."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import torch

from spider.config import Config
import spider.simulators.mjwp as mjwp


MODEL_XML = """
<mujoco>
  <worldbody>
    <body name="robot">
      <geom name="probe_a" type="sphere" size="0.10"/>
      <geom name="probe_b" type="capsule" size="0.05 0.10"/>
    </body>
    <body name="object">
      <geom name="object_collision" type="box" size="0.5 0.4 0.3"/>
      <geom name="object_collision_side" type="box" size="0.2 0.3 0.2"/>
    </body>
  </worldbody>
</mujoco>
"""


def main() -> int:
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    probe_a = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "probe_a"
    )
    probe_b = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "probe_b"
    )
    object_ids = [
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision"
        ),
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision_side"
        ),
    ]
    config = Config()
    config.device = "cpu"
    env = SimpleNamespace(model_cpu=model)
    xpos = torch.zeros((3, model.ngeom, 3), dtype=torch.float64)
    xmat = torch.eye(3, dtype=torch.float64).repeat(
        3, model.ngeom, 1, 1
    )
    xpos[:, probe_a, 0] = torch.tensor([0.0, 0.8, 1.4])
    xpos[:, probe_b, 1] = torch.tensor([0.0, 0.7, 1.2])
    xpos[:, object_ids[1], 0] = 0.65

    tick_cache: dict[tuple[int, ...], torch.Tensor] = {}
    original = mjwp._geom_box_union_sdf_min
    with patch.object(
        mjwp,
        "_geom_box_union_sdf_min",
        wraps=original,
    ) as wrapped:
        first = mjwp._cached_geom_box_union_sdf_min(
            tick_cache,
            config,
            env,
            [probe_a],
            object_ids,
            geom_xpos=xpos,
            geom_xmat=xmat,
        )
        repeated = mjwp._cached_geom_box_union_sdf_min(
            tick_cache,
            config,
            env,
            [probe_a],
            object_ids,
            geom_xpos=xpos,
            geom_xmat=xmat,
        )
        other_set = mjwp._cached_geom_box_union_sdf_min(
            tick_cache,
            config,
            env,
            [probe_b],
            object_ids,
            geom_xpos=xpos,
            geom_xmat=xmat,
        )
        combined = mjwp._cached_geom_box_union_sdf_min(
            tick_cache,
            config,
            env,
            [probe_a, probe_b],
            object_ids,
            geom_xpos=xpos,
            geom_xmat=xmat,
        )
        # Exact tuple caching only reuses identical groups. Cross-group
        # composition is handled by the explicit E176 batch path.
        assert wrapped.call_count == 3
    assert first.data_ptr() == repeated.data_ptr()
    torch.testing.assert_close(first, repeated, atol=0.0, rtol=0.0)
    assert other_set.shape == first.shape
    torch.testing.assert_close(
        combined,
        torch.minimum(first, other_set),
        atol=0.0,
        rtol=0.0,
    )

    # A new tick must use a fresh cache and observe the new geom pose.
    moved = xpos.clone()
    moved[:, probe_a, 0] += 0.4
    next_tick: dict[tuple[int, ...], torch.Tensor] = {}
    refreshed = mjwp._cached_geom_box_union_sdf_min(
        next_tick,
        config,
        env,
        [probe_a],
        object_ids,
        geom_xpos=moved,
        geom_xmat=xmat,
    )
    expected = original(
        config,
        env,
        [probe_a],
        object_ids,
        geom_xpos=moved,
        geom_xmat=xmat,
    )
    torch.testing.assert_close(refreshed, expected, atol=0.0, rtol=0.0)
    assert not torch.equal(first, refreshed)
    print("E176 union-SDF tick cache tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
