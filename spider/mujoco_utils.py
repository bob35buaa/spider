# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utils for mujoco.

Author: Chaoyi Pan
Date: 2025-11-01
"""

from contextlib import contextmanager

import mujoco
import mujoco.viewer


def get_viewer(show_viewer: bool, model: mujoco.MjModel, data: mujoco.MjData):
    if show_viewer:
        run_viewer = lambda: mujoco.viewer.launch_passive(model, data)
    else:
        cam = mujoco.MjvCamera()
        cam.type = 2
        cam.fixedcamid = 0

        @contextmanager
        def run_viewer():
            yield type(
                "DummyViewer",
                (),
                {"is_running": lambda: True, "sync": lambda: None, "cam": 0},
            )

    return run_viewer


# Holosoma G1 PD gains (from holosoma/config_values/robot.py:488-523).
# Keys are joint-name SUBSTRINGS — match against MuJoCo actuator name.
HOLOSOMA_G1_STIFFNESS = {
    "hip_yaw": 40.179238471,
    "hip_roll": 99.098427777,
    "hip_pitch": 40.179238471,
    "knee": 99.098427777,
    "ankle_pitch": 28.501246196,
    "ankle_roll": 28.501246196,
    "waist_yaw": 40.179238471,
    "waist_roll": 28.501246196,
    "waist_pitch": 28.501246196,
    "shoulder_pitch": 14.250623098,
    "shoulder_roll": 14.250623098,
    "shoulder_yaw": 14.250623098,
    "elbow": 14.250623098,
    "wrist_roll": 14.250623098,
    "wrist_pitch": 16.778327481,
    "wrist_yaw": 16.778327481,
}
HOLOSOMA_G1_DAMPING = {
    "hip_yaw": 2.557889765,
    "hip_roll": 6.308801854,
    "hip_pitch": 2.557889765,
    "knee": 6.308801854,
    "ankle_pitch": 1.814445687,
    "ankle_roll": 1.814445687,
    "waist_yaw": 2.557889765,
    "waist_roll": 1.814445687,
    "waist_pitch": 1.814445687,
    "shoulder_pitch": 0.907222843,
    "shoulder_roll": 0.907222843,
    "shoulder_yaw": 0.907222843,
    "elbow": 0.907222843,
    "wrist_roll": 0.907222843,
    "wrist_pitch": 1.068141502,
    "wrist_yaw": 1.068141502,
}


def _match_pd_key(actuator_name: str, gain_table: dict) -> float | None:
    """Find a substring match (longest first) in gain_table keys."""
    for key in sorted(gain_table.keys(), key=len, reverse=True):
        if key in actuator_name:
            return gain_table[key]
    return None


def apply_holosoma_g1_pd(model_cpu: mujoco.MjModel, verbose: bool = True) -> int:
    """Override actuator gainprm/biasprm with Holosoma G1 PD gains.

    MuJoCo position actuator parameters:
      gainprm[0] = kp (stiffness)
      biasprm[1] = -kp (so torque = kp*(qpos_target - qpos))
      biasprm[2] = -kd (damping)
    Object actuators (no name match) are left unchanged.

    Returns the number of robot actuators overridden.
    """
    n_overridden = 0
    for i in range(model_cpu.nu):
        name = model_cpu.actuator(i).name
        kp = _match_pd_key(name, HOLOSOMA_G1_STIFFNESS)
        kd = _match_pd_key(name, HOLOSOMA_G1_DAMPING)
        if kp is None or kd is None:
            if verbose:
                print(f"[holosoma_pd] skip actuator {i} '{name}' (no match)")
            continue
        # position actuator: kp via gainprm[0], -kp via biasprm[1]
        model_cpu.actuator_gainprm[i, 0] = kp
        model_cpu.actuator_biasprm[i, 1] = -kp
        model_cpu.actuator_biasprm[i, 2] = -kd
        n_overridden += 1
        if verbose:
            print(f"[holosoma_pd] {name}: kp={kp:.3f}, kd={kd:.3f}")
    return n_overridden
