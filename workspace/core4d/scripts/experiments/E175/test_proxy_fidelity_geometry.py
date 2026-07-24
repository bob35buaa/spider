#!/usr/bin/env python3
"""CPU checks for E175 offline proxy-fidelity geometry helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

RUNNERS = Path(__file__).resolve().parents[2] / "eval" / "runners"
sys.path.insert(0, str(RUNNERS))

from eval_E175_proxy_fidelity import (  # noqa: E402
    nearest_object_surface,
    paired_object_geom_ids,
    ref_fk_contact_points,
    resize_time_nearest,
)


MODEL_XML = """
<mujoco>
  <default>
    <geom contype="0" conaffinity="0"/>
  </default>
  <worldbody>
    <body name="robot">
      <geom name="probe" type="sphere" size="0.1"/>
    </body>
    <body name="left_wrist_yaw_link">
      <geom type="sphere" size="0.01"/>
    </body>
    <body name="right_wrist_yaw_link" pos="1 0 0"
          quat="0.7071067811865476 0 0 0.7071067811865475">
      <geom type="sphere" size="0.01"/>
    </body>
    <body name="object">
      <geom name="object_collision" type="box" size="0.5 0.5 0.1"/>
      <geom name="object_collision_wall" type="box"
            pos="1 0 0" size="0.1 0.5 0.5"/>
    </body>
  </worldbody>
  <contact>
    <pair geom1="probe" geom2="object_collision"/>
  </contact>
</mujoco>
"""


def main() -> int:
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    primary = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision"
    )
    wall = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision_wall"
    )
    probe = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "probe")

    result = nearest_object_surface(
        model, data, np.asarray([0.95, 0.0, 0.45]), [primary, wall]
    )
    assert result.geom_id == wall
    assert result.geom_name == "object_collision_wall"
    assert abs(result.surface_distance_m - 0.05) < 1e-12
    assert result.signed_distance_m < 0.0

    assert paired_object_geom_ids(model, [probe], [primary, wall]) == [primary]
    contact = ref_fk_contact_points(
        model,
        np.zeros((1, model.nq), dtype=np.float64),
        ["left_wrist_yaw_link", "right_wrist_yaw_link"],
        np.asarray([0.05, 0.0, 0.0]),
        uses_eef_offset=True,
    )
    np.testing.assert_allclose(contact[0, 0], [0.05, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(contact[0, 1], [1.0, 0.05, 0.0], atol=1e-12)

    resized = resize_time_nearest(np.asarray([[0], [1], [2]]), 5)
    np.testing.assert_array_equal(resized[:, 0], [0, 0, 1, 2, 2])
    print("E175 proxy-fidelity geometry tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
