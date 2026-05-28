"""Render top + side views of box + wrist positions across frames as 2D scatter.
No 3D MuJoCo rendering — just matplotlib to keep it portable.

Output PNG per case at workspace/exp_diagnostic/findings/04_overlay_{case}.png.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

from pathlib import Path
import numpy as np
import mujoco
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path("/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider")
TASKS = ROOT / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
OUT = ROOT / "workspace/exp_diagnostic/findings"
OUT.mkdir(parents=True, exist_ok=True)

CASES = [
    ("box021_18029_p2_FAIL", "d003_box021_20231018_029_p2_upperobj_e083_m10_e087"),
    ("box021_11035_p2_FAIL", "d003_box021_20231011_035_p2_upperobj_e083"),
    ("box021_20019_p1_FAIL", "d003_box021_20231020_019_p1_upperobj_e083"),
    ("box023_p2_OK", "box023_person2"),
    ("box025_p2_partial", "box025_person2"),
]


def parse_object_collision(scene_xml):
    tree = ET.parse(scene_xml)
    root = tree.getroot()
    for body in root.iter("body"):
        if body.attrib.get("name") == "object":
            for g in body.iter("geom"):
                if g.attrib.get("type") == "box" and "collision" in g.attrib.get("name", "").lower():
                    return tuple(float(x) for x in g.attrib["size"].split())
    return None


def render(label, task_dir):
    p_scene = TASKS / task_dir / "scene.xml"
    p_npz = TASKS / task_dir / "0" / "trajectory_kinematic.npz"
    half = np.array(parse_object_collision(p_scene))
    d = dict(np.load(p_npz))
    qpos = d["qpos"]
    T = qpos.shape[0]
    contact_pos = d["contact_pos"]  # (T, 2, 3) raw mocap-derived

    model = mujoco.MjModel.from_xml_path(str(p_scene))
    data = mujoco.MjData(model)
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    bidL = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    bidR = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")

    # Per-frame: wrists in object local frame
    Lloc = np.zeros((T, 3))
    Rloc = np.zeros((T, 3))
    CPloc = np.zeros((T, 2, 3))
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        obj_pos = data.xpos[obj_bid].copy()
        obj_mat = data.xmat[obj_bid].reshape(3, 3)
        Lloc[t] = obj_mat.T @ (data.xpos[bidL].copy() - obj_pos)
        Rloc[t] = obj_mat.T @ (data.xpos[bidR].copy() - obj_pos)
        CPloc[t, 0] = obj_mat.T @ (contact_pos[t, 0] - obj_pos)
        CPloc[t, 1] = obj_mat.T @ (contact_pos[t, 1] - obj_pos)

    # 3-panel: XY top, XZ side, YZ side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = [("X (local)", "Y (local)", (0, 1)),
              ("X (local)", "Z (local)", (0, 2)),
              ("Y (local)", "Z (local)", (1, 2))]
    for ax, (xl, yl, (ix, iy)) in zip(axes, titles):
        # box rectangle
        ax.add_patch(Rectangle((-half[ix], -half[iy]), 2*half[ix], 2*half[iy],
                               fill=False, edgecolor="black", linewidth=1.5))
        # IK wrists in local frame
        ax.scatter(Lloc[:, ix], Lloc[:, iy], s=10, c="red", alpha=0.5, label="L wrist (IK)")
        ax.scatter(Rloc[:, ix], Rloc[:, iy], s=10, c="blue", alpha=0.5, label="R wrist (IK)")
        # raw mocap contact_pos
        ax.scatter(CPloc[:, 0, ix], CPloc[:, 0, iy], s=10, c="orange", marker="x", alpha=0.5, label="L contact_pos")
        ax.scatter(CPloc[:, 1, ix], CPloc[:, 1, iy], s=10, c="green", marker="x", alpha=0.5, label="R contact_pos")
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        lim = max(half) * 1.7
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{label}  (object local frame, half_ext={half.round(3).tolist()})", fontsize=12)
    plt.tight_layout()
    p_out = OUT / f"04_overlay_{label}.png"
    plt.savefig(p_out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p_out}")


for label, td in CASES:
    render(label, td)
