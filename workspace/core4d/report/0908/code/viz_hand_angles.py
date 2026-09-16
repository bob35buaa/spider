"""Sweep hand/object close-up camera angles so we can pick the clearest one.

Renders the right-hand/object contact close-up for every method at a set of
(azimuth-offset, elevation) camera angles, and tiles them into one contact sheet:
rows = angle, columns = methods. Uses the same alignment / meshes as the paper
figure. Pick an (az_offset, el) and pass it to viz_paper_figure.py via
--hand-az-offset / --hand-el.

Run:
    PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa .venv/bin/python \
        workspace/core4d/report/0908/code/viz_hand_angles.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse  # noqa: E402

import numpy as np  # noqa: E402

if not hasattr(np, "infty"):
    np.infty = np.inf

import mujoco  # noqa: E402
import pyrender  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_methods_compare import DEFAULT_SCENE, apply_T, facing_yaw, load_qpos, method_qpos_paths  # noqa: E402
from viz_paper_figure import LABEL, _render_meshes, build_alignment, find_font, project  # noqa: E402
from viz_mixed_robot_smplx import _geom_rgb, look_at, object_world_mesh  # noqa: E402

METHODS = ["gmr", "omniretarget", "sbto", "spider"]

# candidate camera angles: (azimuth offset from robot facing [deg], elevation [deg]).
# FRONT 3/4 corner views only (behind-views mirror left/right and are confusing).
# Moderate elevation reveals the box TOP face (contact height); azimuth reveals a
# SIDE face (lateral contact). The LEFT arm is hidden so only the right hand shows.
ANGLES = [
    (-30, 22), (-45, 22), (-60, 22), (-75, 22),
    (-30, 30), (-45, 30), (-60, 30), (-45, 15),
]

LEFT_ARM = ["left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
            "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link",
            "left_wrist_yaw_link"]


def robot_meshes_hide(model, data, hide_bids):
    """Robot visual mesh geoms in world frame, grouped by material colour, EXCLUDING
    the object and any body in hide_bids (used to drop the left arm)."""
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    groups: dict = {}
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        bid = int(model.geom_bodyid[gid])
        if bid == obj_body or bid in hide_bids:
            continue
        mid = model.geom_dataid[gid]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        v = model.mesh_vert[va:va + vn].reshape(-1, 3)
        f = model.mesh_face[fa:fa + fn].reshape(-1, 3)
        vw = v @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        rgb = _geom_rgb(model, gid)
        g = groups.setdefault(rgb, [[], [], 0])
        g[0].append(vw); g[1].append(f + g[2]); g[2] += vn
    return [(np.concatenate(v), np.concatenate(f), rgb) for rgb, (v, f, _) in groups.items()]


def main():
    ap = argparse.ArgumentParser(description="Hand close-up camera-angle sweep sheet.")
    ap.add_argument("--case", default="box021_20231011_037")
    ap.add_argument("--scene", default=str(DEFAULT_SCENE))
    ap.add_argument("--file-frame", type=int, default=12)
    ap.add_argument("--res", type=int, default=440, help="per-cell render resolution")
    ap.add_argument("--hand-radius", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    scene = args.scene
    qpos, ref, regs, (w0, w1) = build_alignment(scene, args.case)
    ref_i = w0 + args.file_frame
    facing = facing_yaw(ref)
    yfov = np.pi / 4.0
    _m0 = mujoco.MjModel.from_xml_path(scene)
    rh_bid = mujoco.mj_name2id(_m0, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    hide_bids = {mujoco.mj_name2id(_m0, mujoco.mjtObj.mjOBJ_BODY, n) for n in LEFT_ARM}
    renderer = pyrender.OffscreenRenderer(args.res, args.res)

    # per method: forward once, cache transformed meshes (LEFT arm hidden) + contact point
    cache = {}
    for m in METHODS:
        lag, T = regs[m]
        model = mujoco.MjModel.from_xml_path(scene)
        data = mujoco.MjData(model)
        data.qpos[:] = qpos[m][ref_i - lag]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        robot = [(apply_T(v, T), f, rgb) for v, f, rgb in robot_meshes_hide(model, data, hide_bids)]
        ov, of, _ = object_world_mesh(model, data)
        ov = apply_T(ov, T)
        rhand = apply_T(data.xpos[rh_bid][None], T)[0]
        floor_z = float(np.concatenate([g[0] for g in robot] + [ov], 0)[:, 2].min())
        nearest = ov[np.argmin(np.linalg.norm(ov - rhand, axis=1))]
        contact = 0.5 * (rhand + nearest)
        cache[m] = (robot, ov, of, floor_z, contact, rhand)

    dist = args.hand_radius / np.tan(yfov / 2.0)
    cells = {}
    for (azo, el) in ANGLES:
        haz, hel = np.radians(facing + azo), np.radians(el)
        hd = np.array([np.cos(hel) * np.cos(haz), np.cos(hel) * np.sin(haz), np.sin(hel)])
        for m in METHODS:
            robot, ov, of, floor_z, contact, rhand = cache[m]
            pose = look_at(contact + hd * dist, contact, [0, 0, 1])
            img = _render_meshes(robot, ov, of, floor_z, pose, yfov, renderer).convert("RGB")
            # mark the RIGHT hand (target of the close-up) with a red ring
            hx, hy = project(rhand, pose, yfov, args.res, args.res)
            dd = ImageDraw.Draw(img)
            rr = int(args.res * 0.045)
            dd.ellipse([hx - rr, hy - rr, hx + rr, hy + rr], outline=(230, 40, 40), width=5)
            cells[(azo, el, m)] = img
        print(f"[angle] az_offset={azo:+4d} el={el:2d} done")
    renderer.delete()

    # tile: rows=angle, cols=methods; left gutter for angle label, top row for method names
    R = args.res
    gutter, header = 190, 54
    font = find_font(26)
    hfont = find_font(30)
    sheet = Image.new("RGB", (gutter + 4 * R, header + len(ANGLES) * R), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for c, m in enumerate(METHODS):
        dr.text((gutter + c * R + 12, 14), LABEL[m], fill=(20, 20, 20), font=hfont)
    for r, (azo, el) in enumerate(ANGLES):
        y = header + r * R
        dr.text((10, y + R // 2 - 20), f"az{azo:+d}\nel{el}", fill=(20, 20, 20), font=font)
        for c, m in enumerate(METHODS):
            sheet.paste(cells[(azo, el, m)], (gutter + c * R, y))
    out_dir = Path(args.out) if args.out else \
        REPO / f"workspace/core4d/report/0908/paper_results/viz/{args.case}_p2/paper_figure"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "hand_angle_sweep.png"
    sheet.save(out)
    print(f"[ok] {out}  ({len(ANGLES)} angles x {len(METHODS)} methods)")


if __name__ == "__main__":
    main()
