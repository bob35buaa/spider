"""Compare original vs B-path-repaired trajectory_kinematic.npz under the
G1-Feasibility gate AND a world-up-face supplementary metric.

The gate's `top_face_frac` is hardcoded to local +z. For box021 (quat ~90° X)
the world-up face is local +y. We therefore report a supplementary
`world_up_face_frac` that, for each frame, checks if the chosen face's outward
normal aligns with world +z (cos angle >= 0.9). This gives a fair view of
"hand-on-top-of-box" geometry regardless of object quat.

We DO NOT modify the gate itself; we report both numbers.

Usage:
    python gate_compare_b_path.py <task_orig> <task_repaired>
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import json

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import numpy as np
import mujoco
import xml.etree.ElementTree as ET

REPO = Path("/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider")
sys.path.insert(0, str(REPO / "workspace/exp_diagnostic/scripts"))
from g1_feasibility_gate import evaluate, fmt_row, GATE  # noqa: E402

EEF_OFFSET = np.array([0.05, 0.0, 0.0])
TASKS_DIR = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"


def world_up_face_frac(task_dir: Path):
    p_scene = task_dir / "scene.xml"
    p_npz = task_dir / "0" / "trajectory_kinematic.npz"
    if not p_scene.exists() or not p_npz.exists():
        return None
    half = None
    for body in ET.parse(p_scene).getroot().iter("body"):
        if body.attrib.get("name") == "object":
            for g in body.iter("geom"):
                if g.attrib.get("type") == "box" and "collision" in g.attrib.get("name", ""):
                    half = np.array([float(x) for x in g.attrib["size"].split()])
                    break
    m = mujoco.MjModel.from_xml_path(str(p_scene))
    d = mujoco.MjData(m)
    bidL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    bidR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    qpos = np.load(p_npz)["qpos"]
    T = qpos.shape[0]
    cnt_L = 0; cnt_R = 0
    for t in range(T):
        d.qpos[:] = qpos[t]; mujoco.mj_forward(m, d)
        mat = d.xmat[obj_bid].reshape(3, 3)
        obj_p = d.xpos[obj_bid].copy()
        for hand, bid, cnt_name in [("L", bidL, "L"), ("R", bidR, "R")]:
            eef_w = d.xpos[bid] + d.xmat[bid].reshape(3,3) @ EEF_OFFSET
            loc = mat.T @ (eef_w - obj_p)
            norm = loc / half
            ax = int(np.argmax(np.abs(norm)))
            sgn = float(np.sign(norm[ax])) or 1.0
            # outward normal of chosen face, in world frame
            local_normal = np.zeros(3); local_normal[ax] = sgn
            world_normal = mat @ local_normal
            if world_normal[2] >= 0.9:  # face is "top" in world
                if hand == "L":
                    cnt_L += 1
                else:
                    cnt_R += 1
    return {"L_world_up_face_frac": cnt_L / T, "R_world_up_face_frac": cnt_R / T}


def main():
    tasks = sys.argv[1:]
    rows = []
    for t in tasks:
        td = TASKS_DIR / t
        r = evaluate(td)
        wuf = world_up_face_frac(td)
        if wuf is not None:
            r.update(wuf)
        rows.append(r)
    print(f"Gate: {GATE}\n")
    for r in rows:
        print(fmt_row(r))
        if "L_world_up_face_frac" in r:
            print(f"  +world_up_face_frac  L={r['L_world_up_face_frac']*100:5.1f}%  "
                  f"R={r['R_world_up_face_frac']*100:5.1f}%")
    out = REPO / "workspace/exp_diagnostic/findings/08_B_path_gate_compare.json"
    out.write_text(json.dumps({"rows": [{
        "task": r.get("task"),
        "T": r.get("T"),
        "L": r.get("L"),
        "R": r.get("R"),
        "pelvis_z_min": r.get("pelvis_z_min"),
        "gate_pass": r.get("gate_pass"),
        "gate_reject_reasons": r.get("gate_reject_reasons"),
        "L_world_up_face_frac": r.get("L_world_up_face_frac"),
        "R_world_up_face_frac": r.get("R_world_up_face_frac"),
    } for r in rows]}, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
