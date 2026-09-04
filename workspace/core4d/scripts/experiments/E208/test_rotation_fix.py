#!/usr/bin/env python3
"""E208: verify the holosoma rotation-augmentation fix is semantically correct.

Must run under the hsretargeting conda python (that is the interpreter the
retargeter uses, and the one whose scipy version caused the break):

    /mnt/ali-sh-1/dataset/zeus/xiayb/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python \
        workspace/core4d/scripts/experiments/E208/test_rotation_fix.py

The fix changes `holosoma .../src/utils.py:346` from
    R.from_euler("z", rotation_list)        # (N,)  -> ValueError on scipy>=1.15
to
    R.from_euler("z", rotation_list[:, None])   # (N, 1)

The point of this test is NOT "does it stop raising" -- it is "does it apply the
per-frame yaw the surrounding code intends": full angle before the object starts
moving, exponential decay with rotation_tau=25 afterwards, positions untouched.
"""

from __future__ import annotations

import sys

import numpy as np

HOLOSOMA = "/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma/src/holosoma_retargeting/holosoma_retargeting"
sys.path.insert(0, HOLOSOMA)

from scipy.spatial.transform import Rotation as R  # noqa: E402
from src.utils import augment_object_poses  # noqa: E402

N, MOVING_IDX, ROTATION_TAU = 200, 40, 25
FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    if not ok:
        FAILURES.append(label)


def yaw_deg(poses: np.ndarray) -> np.ndarray:
    return np.degrees(R.from_quat(poses[:, :4], scalar_first=True).as_euler("ZYX")[:, 0])


def main() -> int:
    import scipy
    print(f"scipy {scipy.__version__}\n")

    # object_poses layout is [qw, qx, qy, qz, x, y, z] (see utils.py:332 vs :347)
    poses = np.zeros((N, 7))
    poses[:, 0] = 1.0
    poses[:, -3:] = np.linspace(0, 1, N)[:, None]
    root = np.array([-1.0, 0.0, 0.0])

    # 1) pure rotation: runs at all, and has the intended decay shape
    out = augment_object_poses(poses, MOVING_IDX, root,
                               local_translation=np.zeros(3), rotation_initial=np.pi / 4)
    y = yaw_deg(out)
    check("full angle before motion onset", abs(y[0] - 45.0) < 1e-6, f"yaw[0]={y[0]:.4f} deg (want 45)")
    check("angle still full at onset", abs(y[MOVING_IDX] - 45.0) < 1e-6,
          f"yaw[{MOVING_IDX}]={y[MOVING_IDX]:.4f} deg (want 45)")
    want = 45.0 * np.exp(-1.0)
    check("decays with rotation_tau=25", abs(y[MOVING_IDX + ROTATION_TAU] - want) < 1e-6,
          f"yaw[{MOVING_IDX + ROTATION_TAU}]={y[MOVING_IDX + ROTATION_TAU]:.4f} deg (want 45*e^-1={want:.4f})")
    check("anchored back near zero at the end", abs(y[-1]) < 0.5, f"yaw[-1]={y[-1]:.4f} deg (want ~0)")
    check("rotation leaves positions alone", np.allclose(out[:, -3:], poses[:, -3:]),
          f"max |dpos| = {np.abs(out[:, -3:] - poses[:, -3:]).max():.2e} m")
    check("quaternions stay unit-norm", np.allclose(np.linalg.norm(out[:, :4], axis=1), 1.0),
          f"max |‖q‖-1| = {np.abs(np.linalg.norm(out[:, :4], axis=1) - 1).max():.2e}")

    # 2) rotation_initial == 0 must be a strict no-op -- this is exactly why the
    #    three trans_* variants kept working while rot_* crashed
    same = augment_object_poses(poses, MOVING_IDX, root,
                                local_translation=np.zeros(3), rotation_initial=0)
    check("rotation_initial=0 is a no-op", np.allclose(same, poses),
          "trans_* variants never entered the broken branch")

    # 3) the shipped rot_0 config is +45 deg AND 0.2 m lateral, not a pure rotation
    combo = augment_object_poses(poses, MOVING_IDX, root,
                                 local_translation=np.array([0.0, 0.2, 0.0]),
                                 rotation_initial=np.pi / 4)
    dpos = float(np.linalg.norm(combo[0, -3:] - poses[0, -3:]))
    check("rot_0 carries its documented 0.2 m lateral offset too", abs(dpos - 0.2) < 1e-6,
          f"approach |dpos|={dpos:.4f} m, yaw[0]={yaw_deg(combo)[0]:.2f} deg "
          "(rot_* is rotation+translation by upstream design, not pure yaw)")

    # 4) negative angle (rot_1) is symmetric
    neg = augment_object_poses(poses, MOVING_IDX, root,
                               local_translation=np.zeros(3), rotation_initial=-np.pi / 4)
    check("rot_1 (-45 deg) is the mirror of rot_0", abs(yaw_deg(neg)[0] + 45.0) < 1e-6,
          f"yaw[0]={yaw_deg(neg)[0]:.4f} deg (want -45)")

    print(f"\n{6 + 3 - len(FAILURES)}/9 checks pass")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
