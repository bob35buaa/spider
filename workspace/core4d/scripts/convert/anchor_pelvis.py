#!/usr/bin/env python3
"""Preprocess CORE4D trajectories: remove walking by anchoring pelvis xy.

Strategy: At each frame, subtract the pelvis xy displacement (relative to frame 0)
from the pelvis and object positions. This makes the trajectory "in-place" — the robot
performs the same upper-body/leg motions but without global translation.

The object position is also shifted by the same amount, preserving the relative
hand-object geometry.

Usage:
    uv run workspace/core4d/scripts/convert/anchor_pelvis.py
"""

import os
import numpy as np

PROCESSED = "/home/ubuntu/Workspace/spider/example_datasets/processed/core4d/unitree_g1/humanoid_object"

CASES = {
    "box025": "box025_person1",
    "bucket010": "bucket010_person1",
    "chair022": "chair022_person1",
    "desk005": "desk005_person2",
}


def anchor_trajectory(qpos: np.ndarray) -> np.ndarray:
    """Remove global xy translation from trajectory.

    qpos shape: (T, 43)
    - [0:3] = pelvis pos (x, y, z)
    - [36:39] = object pos (x, y, z)

    Strategy: subtract pelvis_xy[t] - pelvis_xy[0] from both pelvis and object at each frame.
    This keeps frame 0 unchanged, and subsequent frames have pelvis xy "anchored" near start.
    """
    anchored = qpos.copy()

    # Compute xy displacement relative to frame 0
    pelvis_xy_0 = qpos[0, :2].copy()

    for t in range(qpos.shape[0]):
        # Delta from start
        delta_xy = qpos[t, :2] - pelvis_xy_0

        # Subtract from pelvis
        anchored[t, 0] -= delta_xy[0]
        anchored[t, 1] -= delta_xy[1]

        # Subtract from object (keep relative geometry)
        anchored[t, 36] -= delta_xy[0]
        anchored[t, 37] -= delta_xy[1]

    return anchored


def main():
    for case, task in CASES.items():
        src_path = f"{PROCESSED}/{task}/0/trajectory_kinematic.npz"
        dst_path = f"{PROCESSED}/{task}/0/trajectory_kinematic_anchored.npz"

        src = np.load(src_path)
        qpos = src["qpos"]

        # Anchor
        qpos_anchored = anchor_trajectory(qpos)

        # Recompute qvel from anchored qpos
        fps = 30
        qvel = np.zeros((qpos.shape[0], 41))  # nv=41
        # Simple finite difference (same as core4d.py)
        for t in range(1, qpos.shape[0]):
            # For freejoint: use mj_differentiatePos ideally, but for xy removal, simple diff is fine
            qvel[t, :3] = (qpos_anchored[t, :3] - qpos_anchored[t-1, :3]) * fps
            qvel[t, 3:6] = 0  # angular vel (simplified)
            qvel[t, 6:35] = (qpos_anchored[t, 7:36] - qpos_anchored[t-1, 7:36]) * fps
            # Object vel
            qvel[t, 35:38] = (qpos_anchored[t, 36:39] - qpos_anchored[t-1, 36:39]) * fps
            qvel[t, 38:41] = 0  # object angular vel

        # Keep other fields unchanged
        ctrl = src["ctrl"]
        contact = src["contact"]
        contact_pos = src["contact_pos"]

        np.savez(
            dst_path,
            qpos=qpos_anchored,
            qvel=qvel,
            ctrl=ctrl,
            contact=contact,
            contact_pos=contact_pos,
        )

        # Stats
        orig_disp = np.linalg.norm(qpos[-1, :2] - qpos[0, :2])
        anch_disp = np.linalg.norm(qpos_anchored[-1, :2] - qpos_anchored[0, :2])
        print(f"{case}: orig pelvis disp={orig_disp:.3f}m → anchored={anch_disp:.4f}m")
        print(f"  obj start: ({qpos_anchored[0,36]:.3f}, {qpos_anchored[0,37]:.3f}, {qpos_anchored[0,38]:.3f})")
        print(f"  obj end:   ({qpos_anchored[-1,36]:.3f}, {qpos_anchored[-1,37]:.3f}, {qpos_anchored[-1,38]:.3f})")


if __name__ == "__main__":
    main()
