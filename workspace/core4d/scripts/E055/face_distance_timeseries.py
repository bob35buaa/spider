#!/usr/bin/env python3
"""Plot signed distance from L/R palm to each box face across all frames.

Face naming (plane it lies in + sign of perpendicular axis):
  +yz / -yz : faces in yz plane, normal = ±Ox
  +xz / -xz : faces in xz plane, normal = ±Oy   (+xz = the face whose outward normal is +Oy)
  +xy / -xy : faces in xy plane, normal = ±Oz   (+xy = TOP, -xy = BOTTOM)

Signed distance convention:
  > 0 = palm is OUTSIDE the box on that face's side
  < 0 = palm is INSIDE the box (or on the opposite side)
  = 0 = palm exactly on the face

The "operative" face for grasping is the one with the smallest |signed_distance|
AND whose sign is ≥ 0 (palm is touching from outside, not inside).

Output: workspace/core4d/results/E055/box023_person1/face_distance_timeseries.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[4]
SCENE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person1/scene.xml"
TRAJ = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person1/0/trajectory_kinematic.npz"
OUT = REPO / "workspace/core4d/results/E055/box023_person1/face_distance_timeseries.png"


def main() -> None:
    qpos = np.load(TRAJ)["qpos"]
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(m)
    L_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_palm")
    R_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
    obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    half = np.array([0.1786, 0.1830, 0.2060])

    T = qpos.shape[0]
    L_signed = np.zeros((T, 6))
    R_signed = np.zeros((T, 6))
    L_local_all = np.zeros((T, 3))
    R_local_all = np.zeros((T, 3))

    for t in range(T):
        data.qpos[:] = qpos[t]; mujoco.mj_forward(m, data)
        obj_pos = data.xpos[obj_bid].copy()
        obj_R = data.xmat[obj_bid].copy().reshape(3, 3)
        L_local = obj_R.T @ (data.site_xpos[L_sid] - obj_pos)
        R_local = obj_R.T @ (data.site_xpos[R_sid] - obj_pos)
        L_local_all[t] = L_local; R_local_all[t] = R_local
        # signed dist: + outside, - inside
        # +yz, -yz, +xz, -xz, +xy, -xy
        L_signed[t] = [
            L_local[0] - half[0], -L_local[0] - half[0],
            L_local[1] - half[1], -L_local[1] - half[1],
            L_local[2] - half[2], -L_local[2] - half[2],
        ]
        R_signed[t] = [
            R_local[0] - half[0], -R_local[0] - half[0],
            R_local[1] - half[1], -R_local[1] - half[1],
            R_local[2] - half[2], -R_local[2] - half[2],
        ]

    face_names = ["+yz", "-yz", "+xz", "-xz", "+xy", "-xy"]
    face_colors = ["#d62728", "#ff9896", "#2ca02c", "#98df8a", "#1f77b4", "#9ecae1"]

    fig, axes = plt.subplots(2, 1, figsize=(15, 11), sharex=True)

    intent = (21, 78)

    for ax, signed, label in [(axes[0], L_signed, "L"), (axes[1], R_signed, "R")]:
        for i, fname in enumerate(face_names):
            ax.plot(signed[:, i] * 100, label=fname, color=face_colors[i],
                    lw=2 if fname in ("+xz", "-yz", "-xy") else 1.2,
                    ls="-" if fname[0] == "+" else "--")
        ax.axhline(0, color="black", lw=0.5)
        ax.axvspan(intent[0], intent[1], color="orange", alpha=0.15, label="intent window")
        ax.set_ylabel(f"{label} palm signed dist (cm)\n+ = outside, - = inside")
        ax.set_title(f"{label} palm: signed distance to each box face vs frame  "
                     f"(box023, intent window {intent[0]}-{intent[1]})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, ncol=4)
        ax.set_ylim(-50, 30)

    axes[1].set_xlabel("frame")
    fig.tight_layout()
    plt.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"wrote {OUT}")

    # Print summary: for each frame in intent, which face has smallest |dist|?
    # and which face has smallest dist with sign ≥ 0 (true outside contact)?
    print()
    print("=== Inside intent window: per-frame closest face for L ===")
    print(f"{'frame':>5}  {'L_local (m)':>22}  {'closest |d|':>20}  {'closest outside (≥0)':>25}")
    for t in range(intent[0], intent[1] + 1, 6):
        loc = L_local_all[t]
        d = L_signed[t] * 100
        idx_abs = int(np.argmin(np.abs(d)))
        outside_mask = d >= 0
        if outside_mask.any():
            outside_idx = int(np.argmin(np.where(outside_mask, np.abs(d), 1e9)))
            outside_str = f"{face_names[outside_idx]}  ({d[outside_idx]:+.1f}cm)"
        else:
            outside_str = "(palm fully inside box)"
        print(f"{t:>5}  ({loc[0]:+.2f},{loc[1]:+.2f},{loc[2]:+.2f})  "
              f"{face_names[idx_abs]} ({d[idx_abs]:+.1f}cm)  {outside_str}")


if __name__ == "__main__":
    main()
