#!/usr/bin/env python3
"""Verify face selection: recompute top face on (raw) contact_pos in object-local
frame using full 3D argmax vs xy-only argmax, for box021 cases vs box023.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import mujoco

REPO = Path("/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider")
BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"

def q_apply_inv(q, v):
    q = np.asarray(q, np.float64)
    q = q / max(np.linalg.norm(q), 1e-8)
    qi = q.copy(); qi[1:] *= -1.0
    qv = qi[1:]
    t = 2.0 * np.cross(qv, v)
    return v + qi[0] * t + np.cross(qv, t)

def face_xy(p, half):
    rel = np.abs(p[:2]) / np.clip(half[:2], 1e-8, None)
    a = int(np.argmax(rel))
    s = '+' if p[a] >= 0 else '-'
    return f"{s}{'xy'[a]}"

def face_xyz(p, half):
    rel = np.abs(p) / np.clip(half, 1e-8, None)
    a = int(np.argmax(rel))
    s = '+' if p[a] >= 0 else '-'
    return f"{s}{'xyz'[a]}"

def analyze(task: str) -> None:
    tdir = BASE / task
    if not (tdir / "scene.xml").exists():
        print(f"SKIP {task} (no scene.xml)"); return
    if not (tdir / "0/trajectory_kinematic.npz").exists():
        print(f"SKIP {task} (no traj)"); return
    model = mujoco.MjModel.from_xml_path(str(tdir / "scene.xml"))
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")
    if g < 0:
        print(f"SKIP {task} (no object_collision)"); return
    half = model.geom_size[g, :3].astype(np.float64)
    traj = np.load(tdir / "0/trajectory_kinematic.npz")
    if "contact_pos" not in traj or "contact" not in traj:
        print(f"SKIP {task} (no contact)"); return
    qpos = traj["qpos"]
    qpos = qpos.reshape(-1, qpos.shape[-1])
    cpos = traj["contact_pos"].reshape(qpos.shape[0], -1, 3)
    cflag = traj["contact"].reshape(qpos.shape[0], -1)
    # last 7: free joint of object (pos + quat). Use last 7 of qpos.
    op = qpos[:, -7:-4]; oq = qpos[:, -4:]
    from collections import Counter
    counts_xy = [Counter(), Counter()]
    counts_xyz = [Counter(), Counter()]
    # Aggregate per "hand": left = even sites? We just collect over all contact sites per hand.
    # The full sensor layout: 10 sites bimanual. We split into left/right halves.
    nC = cpos.shape[1]
    half_n = nC // 2
    for t in range(qpos.shape[0]):
        for h in range(nC):
            if not bool(cflag[t, h]):
                continue
            wpos = cpos[t, h]
            if not np.all(np.isfinite(wpos)):
                continue
            loc = q_apply_inv(oq[t], wpos - op[t])
            hand = 0 if h < half_n else 1
            counts_xy[hand][face_xy(loc, half)] += 1
            counts_xyz[hand][face_xyz(loc, half)] += 1
    print(f"\n{task}  half=[{half[0]:.3f},{half[1]:.3f},{half[2]:.3f}]")
    for hi, name in enumerate(("L", "R")):
        cx = counts_xy[hi]
        c3 = counts_xyz[hi]
        n = sum(c3.values())
        if n == 0:
            print(f"  {name}: no contact"); continue
        top3 = sorted(c3.items(), key=lambda kv: -kv[1])[:3]
        topxy = sorted(cx.items(), key=lambda kv: -kv[1])[:3]
        print(f"  {name}  N={n:4d}  xy_top={topxy}  xyz_top={top3}  z_frac={(c3.get('+z',0)+c3.get('-z',0))/n:.2%}")


if __name__ == "__main__":
    cases = sys.argv[1:] or [
        "d003_box021_20231018_029_p2",
        "d003_box021_20231011_035_p2",
        "d003_box021_20231018_030_p1",
        "d003_box021_20231020_019_p2",
        "d003_box021_20231020_020_p2",
        "box023_person2",
        "box023_person1",
    ]
    for c in cases:
        analyze(c)
