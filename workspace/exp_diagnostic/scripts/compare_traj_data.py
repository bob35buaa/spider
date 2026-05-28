"""Compare trajectory_kinematic.npz, scene XML, and contact masks between box021 (failing)
and box023/box025 (passing). Print structural summary so we can see what is qualitatively
different about box021.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider")
TASKS = ROOT / "example_datasets/processed/core4d/unitree_g1/humanoid_object"

CASES = {
    "box021_D003_18029_p2_e087m10": "d003_box021_20231018_029_p2_upperobj_e083_m10_e087",
    "box023_p2_GUARD_OK":           "box023_person2",
    "box025_p2_partial":            "box025_person2",
    "box023_p2_e083":               "box023_person2_upperobj_e083",
}

def load_npz(path: Path):
    if not path.exists():
        return None
    return dict(np.load(path, allow_pickle=True))

def summarize_traj(name, d):
    print(f"\n=== {name} ===")
    print(f"keys: {sorted(d.keys())}")
    for k in sorted(d.keys()):
        v = d[k]
        try:
            print(f"  {k}: shape={v.shape} dtype={v.dtype}")
        except Exception:
            print(f"  {k}: {type(v).__name__} {v}")
    qpos = d.get("qpos")
    obj_pose = d.get("object_pose")
    if qpos is not None:
        print(f"  qpos[0] head: {qpos[0, :7]}")
        print(f"  qpos pelvis z range: {qpos[:, 2].min():.3f} .. {qpos[:, 2].max():.3f}  (mean {qpos[:, 2].mean():.3f})")
        if qpos.shape[1] >= 30:
            # left wrist roughly at idx ~26-28 depending on robot (G1)
            pass
    if obj_pose is not None and obj_pose.ndim == 2 and obj_pose.shape[1] >= 7:
        print(f"  object_pose xyz frame0: {obj_pose[0, :3]}  framelast: {obj_pose[-1, :3]}")
        print(f"  object z range: {obj_pose[:, 2].min():.3f} .. {obj_pose[:, 2].max():.3f}  (mean {obj_pose[:, 2].mean():.3f})")
        print(f"  object dz frame_last-frame_0: {obj_pose[-1, 2] - obj_pose[0, 2]:+.3f}")
        # detect lift episode
        z = obj_pose[:, 2]
        z_init = z[:5].mean()
        z_above = z > (z_init + 0.05)
        print(f"  object z > z_init+5cm in {z_above.mean()*100:.1f}% of frames")
    # contact mask?
    for k in d.keys():
        if "contact" in k.lower() or "mask" in k.lower():
            v = d[k]
            try:
                if v.ndim >= 1:
                    print(f"  CONTACT-LIKE {k}: shape={v.shape} sum/mean={float(np.asarray(v).mean()):.3f}")
            except Exception:
                pass


def main():
    for label, task_dir in CASES.items():
        path = TASKS / task_dir / "0" / "trajectory_kinematic.npz"
        d = load_npz(path)
        if d is None:
            print(f"\n[MISSING] {label}: {path}")
            continue
        summarize_traj(label, d)


if __name__ == "__main__":
    main()
