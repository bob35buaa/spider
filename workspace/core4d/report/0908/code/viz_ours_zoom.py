"""Standalone high-res render of the "Ours" (SPIDER) hand/object contact close-up.

Reuses the exact alignment + dedicated close-up camera from viz_paper_figure so the
output matches the zoom circle in 3_annotated.png, but rendered at high resolution and
saved on its own (both a square and a circular-masked variant).

Run (headless):
    PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa .venv/bin/python \
        workspace/core4d/report/0908/code/viz_ours_zoom.py
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
import trimesh  # noqa: E402

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_style import BOX_COLOR, soft_phong_lights  # noqa: E402
from viz_methods_compare import DEFAULT_SCENE, apply_T, facing_yaw  # noqa: E402
from viz_paper_figure import (LEFT_ARM, build_alignment, circular_whole,  # noqa: E402
                              render_panel, robot_meshes_hide)
from viz_mixed_robot_smplx import look_at, object_world_mesh  # noqa: E402

ORANGE = (233, 126, 34)


def render_closeup_nofloor(scene, qpos_m, lag, T, ref_i, res, yfov, hand_cdir,
                           hand_radius, rh_bid, hide_bids, bg):
    """Render the right-hand/object close-up with NO floor.

    Same contact-centered camera as viz_paper_figure.render_panel's close-up, but the
    checker floor is dropped and the background is transparent (bg='transparent') or
    solid white (bg='white'). Returns a PIL RGBA image (side res)."""
    import pyrender  # noqa: E402
    from PIL import Image

    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_m[ref_i - lag]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    robot_noleft = [(apply_T(v, T), f, rgb) for v, f, rgb in robot_meshes_hide(model, data, hide_bids)]
    ov, of, _ = object_world_mesh(model, data)
    ov = apply_T(ov, T)
    rhand = apply_T(data.xpos[rh_bid][None], T)[0]

    allv = np.concatenate([g[0] for g in robot_noleft] + [ov], 0)
    floor_z = float(allv[:, 2].min())                       # lighting reference only
    nearest = ov[np.argmin(np.linalg.norm(ov - rhand, axis=1))]
    contact = 0.5 * (rhand + nearest)
    hdist = hand_radius / np.tan(yfov / 2.0)
    cam_pose = look_at(contact + hand_cdir * hdist, contact, [0, 0, 1])

    bg_color = [1.0, 1.0, 1.0, 1.0] if bg == "white" else [0.0, 0.0, 0.0, 0.0]
    mat = lambda c, r=0.55: pyrender.MetallicRoughnessMaterial(
        baseColorFactor=c, metallicFactor=0.0, roughnessFactor=r, doubleSided=True)
    scene_pr = pyrender.Scene(bg_color=bg_color, ambient_light=[0.4, 0.4, 0.42])
    for rv, rf, rgb in robot_noleft:
        scene_pr.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(rv, rf, process=False),
                                                material=mat(list(rgb) + [1.0]), smooth=False))
    scene_pr.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(ov, of, process=False),
                                            material=mat(BOX_COLOR), smooth=False))
    cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
    scene_pr.add(cam, pose=cam_pose)
    tgt = cam_pose[:3, 3] - cam_pose[:3, 2]
    for light, pose in soft_phong_lights(look_at, tgt, [0, 0, 1], floor_z, up="z"):
        scene_pr.add(light, pose=pose)
    renderer = pyrender.OffscreenRenderer(res, res)
    color, depth = renderer.render(scene_pr, flags=pyrender.constants.RenderFlags.SHADOWS_DIRECTIONAL)
    renderer.delete()
    rgb = color[:, :, :3]
    if bg == "white":
        return Image.fromarray(rgb.copy(), "RGB").convert("RGBA")
    # transparent: alpha=0 where nothing was hit (depth==0)
    alpha = (depth > 0).astype(np.uint8) * 255
    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba.copy(), "RGBA")


def main():
    ap = argparse.ArgumentParser(description="Standalone Ours hand close-up.")
    ap.add_argument("--case", default="box021_20231011_037")
    ap.add_argument("--scene", default=str(DEFAULT_SCENE))
    ap.add_argument("--file-frame", type=int, default=12)
    ap.add_argument("--res", type=int, default=1400, help="close-up render resolution")
    ap.add_argument("--hand-az-offset", type=float, default=-126.1)
    ap.add_argument("--hand-el", type=float, default=5.0)
    ap.add_argument("--hand-radius", type=float, default=0.25)
    ap.add_argument("--border-w", type=int, default=10)
    ap.add_argument("--bg", choices=["floor", "transparent", "white"], default="transparent",
                    help="'floor' keeps the checker floor+sky; 'transparent'/'white' drop "
                         "the floor and render on an empty background")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    import pyrender  # noqa: E402

    scene = args.scene
    qpos, ref, regs, (w0, w1) = build_alignment(scene, args.case)
    ref_i = w0 + args.file_frame
    assert w0 <= ref_i < w1, f"frame {args.file_frame} -> ref {ref_i} outside [{w0},{w1})"

    facing = facing_yaw(ref)
    az, el = np.radians(facing + 45.0), np.radians(12.0)
    cdir = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    haz, hel = np.radians(facing + args.hand_az_offset), np.radians(args.hand_el)
    hand_cdir = np.array([np.cos(hel) * np.cos(haz), np.cos(hel) * np.sin(haz), np.sin(hel)])

    _m = mujoco.MjModel.from_xml_path(scene)
    rh_bid = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    foot_gid = next(gid for gid in range(_m.ngeom)
                    if _m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH
                    and (mujoco.mj_id2name(_m, mujoco.mjtObj.mjOBJ_BODY, _m.geom_bodyid[gid]) or "")
                    == "left_ankle_roll_link")
    hide_bids = {mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_BODY, n) for n in LEFT_ARM}

    yfov = np.pi / 4.0
    lag, T = regs["spider"]
    if args.bg == "floor":
        renderer = pyrender.OffscreenRenderer(args.res, args.res)
        _, _, closeup = render_panel(scene, qpos["spider"], lag, T, ref_i, args.res, yfov, cdir,
                                     1.0, renderer, rh_bid, foot_gid, hand_cdir, args.hand_radius,
                                     hide_bids)
        renderer.delete()
        suffix = ""
    else:
        closeup = render_closeup_nofloor(scene, qpos["spider"], lag, T, ref_i, args.res, yfov,
                                         hand_cdir, args.hand_radius, rh_bid, hide_bids, args.bg)
        suffix = f"_{args.bg}"

    out_dir = Path(args.out_dir) if args.out_dir else \
        REPO / f"workspace/core4d/report/0908/paper_results/viz/{args.case}_p2/paper_figure"
    out_dir.mkdir(parents=True, exist_ok=True)

    sq = out_dir / f"ours_zoom_square{suffix}.png"
    closeup.save(sq)
    print(f"[ok] {sq}")

    circ = circular_whole(closeup, args.res, ORANGE, args.border_w)
    cp = out_dir / f"ours_zoom_circle{suffix}.png"
    circ.save(cp)                                  # RGBA, transparent outside the circle
    print(f"[ok] {cp}")


if __name__ == "__main__":
    main()
