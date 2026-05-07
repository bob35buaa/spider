#!/usr/bin/env python3
"""Full anchor: remove both XY translation AND yaw rotation from CORE4D trajectories.

Extends anchor_pelvis.py by also removing the pelvis yaw rotation at each frame,
keeping the robot always facing its initial direction. Object position and orientation
are counter-rotated to maintain the relative geometry.

Usage:
    uv run workspace/core4d/scripts/convert/anchor_pelvis_full.py
"""

import os
import numpy as np
from scipy.spatial.transform import Rotation

PROCESSED = "/home/ubuntu/Workspace/spider/example_datasets/processed/core4d/unitree_g1/humanoid_object"

CASES = {
    "box025": "box025_person1",
    "bucket010": "bucket010_person1",
    "chair022": "chair022_person1",
    "desk005": "desk005_person2",
}


def quat_to_yaw(q: np.ndarray) -> float:
    """Extract yaw (rotation around z-axis) from wxyz quaternion."""
    # MuJoCo uses w,x,y,z ordering
    r = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # scipy uses x,y,z,w
    euler = r.as_euler("ZYX")  # intrinsic Z-Y-X = yaw, pitch, roll
    return euler[0]


def rotate_quat_by_yaw(q: np.ndarray, yaw: float) -> np.ndarray:
    """Apply a yaw rotation to a wxyz quaternion."""
    r_orig = Rotation.from_quat([q[1], q[2], q[3], q[0]])
    r_yaw = Rotation.from_euler("z", yaw)
    r_new = r_yaw * r_orig
    q_new = r_new.as_quat()  # x,y,z,w
    return np.array([q_new[3], q_new[0], q_new[1], q_new[2]])  # w,x,y,z


def rotate_2d(xy: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a 2D vector by angle (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * xy[0] - s * xy[1], s * xy[0] + c * xy[1]])


def full_anchor_trajectory(qpos: np.ndarray) -> np.ndarray:
    """Remove global XY translation AND yaw rotation from trajectory.

    qpos shape: (T, 43)
    - [0:3] = pelvis pos (x, y, z)
    - [3:7] = pelvis quat (w, x, y, z)
    - [36:39] = object pos (x, y, z)
    - [39:43] = object quat (w, x, y, z)
    """
    anchored = qpos.copy()
    T = qpos.shape[0]

    pelvis_xy_0 = qpos[0, :2].copy()
    yaw_0 = quat_to_yaw(qpos[0, 3:7])

    for t in range(T):
        # --- Step 1: Compute deltas ---
        delta_xy = qpos[t, :2] - pelvis_xy_0
        yaw_t = quat_to_yaw(qpos[t, 3:7])
        delta_yaw = yaw_t - yaw_0

        # --- Step 2: Remove pelvis XY translation ---
        anchored[t, 0] = pelvis_xy_0[0]
        anchored[t, 1] = pelvis_xy_0[1]
        # Keep pelvis z unchanged
        anchored[t, 2] = qpos[t, 2]

        # --- Step 3: Remove pelvis yaw rotation ---
        anchored[t, 3:7] = rotate_quat_by_yaw(qpos[t, 3:7], -delta_yaw)

        # --- Step 4: Transform object position ---
        # Object position relative to pelvis in world frame
        obj_rel_world = qpos[t, 36:38] - qpos[t, 0:2]
        # Rotate by -delta_yaw to stay in initial frame
        obj_rel_rotated = rotate_2d(obj_rel_world, -delta_yaw)
        # New object world position = anchored pelvis + rotated relative
        anchored[t, 36] = anchored[t, 0] + obj_rel_rotated[0]
        anchored[t, 37] = anchored[t, 1] + obj_rel_rotated[1]
        # Keep object z unchanged
        anchored[t, 38] = qpos[t, 38]

        # --- Step 5: Transform object orientation ---
        anchored[t, 39:43] = rotate_quat_by_yaw(qpos[t, 39:43], -delta_yaw)

    return anchored


def main():
    for case, task in CASES.items():
        src_path = f"{PROCESSED}/{task}/0/trajectory_kinematic.npz"
        dst_path = f"{PROCESSED}/{task}/0/trajectory_kinematic_fullanchor.npz"

        src = np.load(src_path)
        qpos = src["qpos"]
        T = qpos.shape[0]

        # Full anchor
        qpos_anchored = full_anchor_trajectory(qpos)

        # Recompute qvel (simplified finite difference)
        fps = 30
        qvel = np.zeros((T, 41))
        for t in range(1, T):
            qvel[t, :3] = (qpos_anchored[t, :3] - qpos_anchored[t-1, :3]) * fps
            qvel[t, 6:35] = (qpos_anchored[t, 7:36] - qpos_anchored[t-1, 7:36]) * fps
            qvel[t, 35:38] = (qpos_anchored[t, 36:39] - qpos_anchored[t-1, 36:39]) * fps

        np.savez(
            dst_path,
            qpos=qpos_anchored,
            qvel=qvel,
            ctrl=src["ctrl"],
            contact=src["contact"],
            contact_pos=src["contact_pos"],
        )

        # Stats
        yaw_orig = [quat_to_yaw(qpos[t, 3:7]) for t in range(T)]
        yaw_anch = [quat_to_yaw(qpos_anchored[t, 3:7]) for t in range(T)]
        yaw_range_orig = np.degrees(max(yaw_orig) - min(yaw_orig))
        yaw_range_anch = np.degrees(max(yaw_anch) - min(yaw_anch))

        xy_disp_orig = np.linalg.norm(qpos[-1, :2] - qpos[0, :2])
        xy_disp_anch = np.linalg.norm(qpos_anchored[-1, :2] - qpos_anchored[0, :2])

        print(f"{case}:")
        print(f"  XY disp: {xy_disp_orig:.3f}m → {xy_disp_anch:.4f}m")
        print(f"  Yaw range: {yaw_range_orig:.1f}° → {yaw_range_anch:.1f}°")
        print(f"  Obj start: ({qpos_anchored[0,36]:.3f}, {qpos_anchored[0,37]:.3f}, {qpos_anchored[0,38]:.3f})")
        print()


if __name__ == "__main__":
    main()
