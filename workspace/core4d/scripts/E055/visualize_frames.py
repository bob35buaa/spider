#!/usr/bin/env python3
"""Visualize all coordinate frames at a single box023_person1 frame.

Renders 4 panels:
  - 3D scene with world / object / pelvis axes + labeled box faces + L/R palms
  - Top-down (XY world) view
  - Front (XZ world) view
  - Side (YZ world) view

Output: workspace/core4d/results/E055/box023_person1/frames_t50.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO = Path(__file__).resolve().parents[4]
SCENE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person1/scene.xml"
TRAJ = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person1/0/trajectory_kinematic.npz"
OUT_DIR = REPO / "workspace/core4d/results/E055/box023_person1"


def draw_box_faces(ax, center, R, half, alpha=0.15):
    """Draw a 3D box (oriented) with each face labeled in its local frame."""
    # 8 corners in local frame
    corners = np.array([
        [+1, +1, +1], [+1, +1, -1], [+1, -1, +1], [+1, -1, -1],
        [-1, +1, +1], [-1, +1, -1], [-1, -1, +1], [-1, -1, -1],
    ]) * half[None, :]
    world = corners @ R.T + center

    # 6 faces named by the PLANE they lie in + sign of perpendicular axis position.
    # E.g. "+yz" = face in yz-plane at +x side (normal points +x).
    face_defs = {
        "+yz": ([0, 1, 3, 2], "tab:red", "+yz (normal = +x)"),
        "-yz": ([4, 5, 7, 6], "tab:red", "-yz (normal = -x)"),
        "+xz": ([0, 1, 5, 4], "tab:green", "+xz (normal = +y)"),
        "-xz": ([2, 3, 7, 6], "tab:green", "-xz (normal = -y)"),
        "+xy": ([0, 2, 6, 4], "tab:blue", "+xy = TOP"),
        "-xy": ([1, 3, 7, 5], "tab:blue", "-xy = BOTTOM"),
    }
    for key, (idx, color, label) in face_defs.items():
        verts = [world[idx]]
        poly = Poly3DCollection(verts, alpha=alpha, facecolor=color, edgecolor="k", lw=0.5)
        ax.add_collection3d(poly)
        # face center label
        face_c = world[idx].mean(axis=0)
        ax.text(*face_c, key, color=color, fontsize=8, ha="center", weight="bold")


def draw_axes(ax, origin, R, length=0.3, label="", lw=2):
    """Draw an XYZ frame at origin with rotation R (cols are world-frame x,y,z axes)."""
    colors = ["red", "green", "blue"]
    names = ["x", "y", "z"]
    for i in range(3):
        axis = R[:, i] * length
        ax.quiver(*origin, *axis, color=colors[i], arrow_length_ratio=0.15, lw=lw)
        ax.text(*(origin + axis * 1.15), f"{label}{names[i]}", color=colors[i], fontsize=9)


def main(t: int = 50) -> None:
    qpos = np.load(TRAJ)["qpos"]
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(m)
    data.qpos[:] = qpos[t]
    mujoco.mj_forward(m, data)

    L_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_palm")
    R_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
    pelvis_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")

    pelvis_pos = data.xpos[pelvis_bid].copy()
    pelvis_R = data.xmat[pelvis_bid].copy().reshape(3, 3)
    obj_pos = data.xpos[obj_bid].copy()
    obj_R = data.xmat[obj_bid].copy().reshape(3, 3)
    L_world = data.site_xpos[L_sid].copy()
    R_world = data.site_xpos[R_sid].copy()

    box_half = np.array([0.1786, 0.1830, 0.2060])

    fig = plt.figure(figsize=(18, 13))

    # === Panel 1: 3D, tight zoom around the box ===
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.set_title(f"3D scene (frame t={t})\n"
                  f"obj_pos=({obj_pos[0]:.2f},{obj_pos[1]:.2f},{obj_pos[2]:.2f})  "
                  f"pelvis=({pelvis_pos[0]:.2f},{pelvis_pos[1]:.2f},{pelvis_pos[2]:.2f})")
    # Object frame + faces
    draw_axes(ax1, obj_pos, obj_R, length=0.25, label="O_", lw=3)
    draw_box_faces(ax1, obj_pos, obj_R, box_half, alpha=0.15)
    # Pelvis (small marker, only forward axis to avoid clutter)
    ax1.scatter(*pelvis_pos, c="black", s=120, marker="^", label="pelvis")
    pv = pelvis_R[:, 0] * 0.30
    ax1.quiver(*pelvis_pos, *pv, color="black", arrow_length_ratio=0.2, lw=2)
    ax1.text(*(pelvis_pos + pv * 1.15), "Px (robot forward)", color="black", fontsize=8)
    # Hands
    ax1.scatter(*L_world, c="cyan", s=150, marker="o", label="L_palm", edgecolor="black")
    ax1.scatter(*R_world, c="magenta", s=150, marker="o", label="R_palm", edgecolor="black")
    # robot→box arrow
    ax1.quiver(*pelvis_pos, *(obj_pos - pelvis_pos), color="orange",
               arrow_length_ratio=0.08, lw=2, alpha=0.7)
    # World axes inset (small, at corner)
    world_origin = np.array([obj_pos[0] - 0.5, obj_pos[1] - 0.5, 0.0])
    draw_axes(ax1, world_origin, np.eye(3), length=0.20, label="W_", lw=2)
    ax1.text(*world_origin, "WORLD", fontsize=9, weight="bold", color="black")
    ax1.set_xlabel("World X (m)"); ax1.set_ylabel("World Y (m)"); ax1.set_zlabel("World Z (m)")
    ax1.legend(loc="upper left", fontsize=8)
    # Tight bounds around objects of interest
    pts = np.vstack([obj_pos, pelvis_pos, L_world, R_world])
    pad = 0.4
    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
    ax1.set_xlim(xs.min() - pad, xs.max() + pad)
    ax1.set_ylim(ys.min() - pad, ys.max() + pad)
    ax1.set_zlim(0, max(zs.max(), 1.0) + pad)
    ax1.set_box_aspect([1, 1, 0.7])
    ax1.view_init(elev=20, azim=-60)

    # === Panel 2: TOP-DOWN (world XY plane) ===
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("TOP-DOWN (look from +Z)\n"
                  "rendered camera is at +Y direction (robot back faces camera)")
    # box footprint as 4 oriented edges
    bottom_corners_local = np.array([
        [+1, +1, -1], [+1, -1, -1], [-1, -1, -1], [-1, +1, -1]
    ]) * box_half[None, :]
    bcw = bottom_corners_local @ obj_R.T + obj_pos
    ax2.fill(bcw[:, 0], bcw[:, 1], alpha=0.25, color="orange",
             edgecolor="black", lw=2, label="box footprint")
    # Mark each EDGE of the footprint with the +/-x or +/-y face label of the BOX
    # corner indices: 0=(+x+y), 1=(+x-y), 2=(-x-y), 3=(-x+y)
    edge_labels = [
        ("+yz", "red", [0, 1]),    # face in yz plane, +x side (normal = +x)
        ("-yz", "red", [2, 3]),    # face in yz plane, -x side (normal = -x)
        ("+xz", "green", [0, 3]),  # face in xz plane, +y side (normal = +y)
        ("-xz", "green", [1, 2]),  # face in xz plane, -y side (normal = -y)
    ]
    for label, color, idx in edge_labels:
        mid = (bcw[idx[0]] + bcw[idx[1]]) / 2
        ax2.text(mid[0], mid[1], label, fontsize=11, ha="center", va="center",
                 color=color, weight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=1.5))
    # box object frame arrows (Ox, Oy) drawn from box center
    for ax_idx, (label, color) in enumerate([("Ox", "red"), ("Oy", "green")]):
        v = obj_R[:2, ax_idx] * 0.15
        ax2.annotate("", xy=(obj_pos[0] + v[0], obj_pos[1] + v[1]),
                     xytext=(obj_pos[0], obj_pos[1]),
                     arrowprops=dict(arrowstyle="->", color=color, lw=2))
    ax2.text(obj_pos[0], obj_pos[1] + 0.04, "obj origin", fontsize=8, color="orange",
             ha="center", weight="bold")
    # pelvis with forward arrow
    ax2.scatter(pelvis_pos[0], pelvis_pos[1], c="black", s=250, marker="^",
                label="pelvis (▲ = robot)", zorder=5)
    pv = pelvis_R[:2, 0] * 0.30
    ax2.annotate("", xy=(pelvis_pos[0] + pv[0], pelvis_pos[1] + pv[1]),
                 xytext=(pelvis_pos[0], pelvis_pos[1]),
                 arrowprops=dict(arrowstyle="->", color="black", lw=2.5))
    ax2.text(pelvis_pos[0] + pv[0] + 0.03, pelvis_pos[1] + pv[1],
             "Px (robot facing)", color="black", fontsize=9, weight="bold")
    # hands
    ax2.scatter(L_world[0], L_world[1], c="cyan", s=200, marker="o", edgecolor="black",
                label="L_palm", zorder=5)
    ax2.scatter(R_world[0], R_world[1], c="magenta", s=200, marker="o", edgecolor="black",
                label="R_palm", zorder=5)
    # robot→box
    ax2.annotate("", xy=(obj_pos[0], obj_pos[1]), xytext=(pelvis_pos[0], pelvis_pos[1]),
                 arrowprops=dict(arrowstyle="->", color="orange", lw=1.5, alpha=0.6))
    ax2.text((pelvis_pos[0] + obj_pos[0]) / 2 + 0.03,
             (pelvis_pos[1] + obj_pos[1]) / 2,
             "robot→box", color="orange", fontsize=9, alpha=0.8)
    # World axis inset
    ax2.annotate("", xy=(0.10, 0.10), xytext=(0.02, 0.10),
                 xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax2.annotate("", xy=(0.02, 0.18), xytext=(0.02, 0.10),
                 xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="->", color="green", lw=2))
    ax2.text(0.11, 0.10, "Wx", color="red", weight="bold",
             transform=ax2.transAxes, fontsize=10, va="center")
    ax2.text(0.02, 0.20, "Wy", color="green", weight="bold",
             transform=ax2.transAxes, fontsize=10, ha="center")
    ax2.set_xlabel("World X (m)"); ax2.set_ylabel("World Y (m)")
    ax2.set_aspect("equal")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    # Tight bounds: include both robot and box with padding
    pad = 0.25
    ax2.set_xlim(min(pelvis_pos[0], bcw[:, 0].min()) - pad, max(pelvis_pos[0], bcw[:, 0].max()) + pad)
    ax2.set_ylim(min(pelvis_pos[1], bcw[:, 1].min()) - pad, max(pelvis_pos[1], bcw[:, 1].max()) + pad)

    # === Panel 3: ref video frame for spatial anchor ===
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title(f"Rendered ref view (visualization_kinematic.mp4 at t≈{t/30:.2f}s)")
    ref_jpg = OUT_DIR / "ref_diagnosis" / "refonly_t1.65s.jpg"
    if ref_jpg.exists():
        ax3.imshow(plt.imread(ref_jpg))
        ax3.axis("off")
    else:
        ax3.text(0.5, 0.5, "ref_diagnosis/refonly_t1.65s.jpg not found",
                 ha="center", va="center", transform=ax3.transAxes)

    # === Panel 4: legend / data summary ===
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    L_local = obj_R.T @ (L_world - obj_pos)
    R_local = obj_R.T @ (R_world - obj_pos)
    # signed distance to each face (positive = outside that face)
    def face_distances(local):
        return {
            "+yz": local[0] - box_half[0],
            "-yz": -local[0] - box_half[0],
            "+xz": local[1] - box_half[1],
            "-xz": -local[1] - box_half[1],
            "+xy": local[2] - box_half[2],
            "-xy": -local[2] - box_half[2],
        }
    L_d = face_distances(L_local)
    R_d = face_distances(R_local)
    L_closest = min(L_d, key=lambda k: abs(L_d[k]))
    R_closest = min(R_d, key=lambda k: abs(R_d[k]))

    summary = f"""COORDINATE FRAMES at frame t={t}

WORLD frame (MuJoCo, right-handed):
  Wx red   horizontal,   Wy green forward,   Wz blue up

OBJECT (box023) frame:
  origin = ({obj_pos[0]:+.2f}, {obj_pos[1]:+.2f}, {obj_pos[2]:+.2f}) m
  Ox red, Oy green, Oz blue (rotated wrt world!)
  half-sizes = ({box_half[0]*100:.0f}, {box_half[1]*100:.0f}, {box_half[2]*100:.0f}) cm

FACE NAMING (face = the plane it lies in, with sign for which side):
  +yz / -yz : faces in yz-plane, normal = ±Ox
  +xz / -xz : faces in xz-plane, normal = ±Oy
  +xy / -xy : faces in xy-plane, normal = ±Oz   (-xy = BOTTOM, +xy = TOP)

PELVIS (G1) frame:
  origin = ({pelvis_pos[0]:+.2f}, {pelvis_pos[1]:+.2f}, {pelvis_pos[2]:+.2f}) m
  Px = robot forward (black arrow)

HANDS in WORLD:
  L_palm = ({L_world[0]:+.2f}, {L_world[1]:+.2f}, {L_world[2]:+.2f}) m  [cyan]
  R_palm = ({R_world[0]:+.2f}, {R_world[1]:+.2f}, {R_world[2]:+.2f}) m  [magenta]

HANDS in OBJECT frame (= rotate world hand into box's local axes):
  L_local = ({L_local[0]:+.2f}, {L_local[1]:+.2f}, {L_local[2]:+.2f}) m
    signed distance to each face (+ = outside, - = inside box):
      +yz: {L_d['+yz']*100:+5.1f}cm   -yz: {L_d['-yz']*100:+5.1f}cm
      +xz: {L_d['+xz']*100:+5.1f}cm   -xz: {L_d['-xz']*100:+5.1f}cm
      +xy: {L_d['+xy']*100:+5.1f}cm   -xy: {L_d['-xy']*100:+5.1f}cm
    → CLOSEST face = {L_closest} (|d|={abs(L_d[L_closest])*100:.1f}cm)

  R_local = ({R_local[0]:+.2f}, {R_local[1]:+.2f}, {R_local[2]:+.2f}) m
      +yz: {R_d['+yz']*100:+5.1f}cm   -yz: {R_d['-yz']*100:+5.1f}cm
      +xz: {R_d['+xz']*100:+5.1f}cm   -xz: {R_d['-xz']*100:+5.1f}cm
      +xy: {R_d['+xy']*100:+5.1f}cm   -xy: {R_d['-xy']*100:+5.1f}cm
    → CLOSEST face = {R_closest} (|d|={abs(R_d[R_closest])*100:.1f}cm)
"""
    ax4.text(0.0, 1.0, summary, family="monospace", fontsize=9,
             verticalalignment="top", transform=ax4.transAxes)

    fig.tight_layout()
    out = OUT_DIR / f"frames_t{t}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
