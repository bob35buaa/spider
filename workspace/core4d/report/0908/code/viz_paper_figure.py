"""Assemble the 4-method comparison paper figure from a single aligned frame.

2x2 grid (matches the reference figure layout):
    top-left  GMR            top-right   OmniRetarget
    bottom-left DynaRetarget  bottom-right Ours (SPIDER)

Each panel is the object-aligned robot+object render (bg style) at one frame, with
a colored method label. A circular close-up inset zooms the G1 LEFT-hand / object
contact region, with an arrow pointing to it. Three variants are written:
    1_plain     : grid + labels only
    2_zoom      : grid + labels + circular close-up inset
    3_annotated : grid + labels + inset + text annotation per panel

Panels are re-rendered here (same alignment/camera as viz_methods_compare) so the
projected hand pixel that anchors the inset/arrow is exact.

Run (headless):
    PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa .venv/bin/python \
        workspace/core4d/report/0908/code/viz_paper_figure.py
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
import pyrender  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_style import BOX_COLOR, make_checker_floor, soft_phong_lights  # noqa: E402
from viz_methods_compare import (DEFAULT_SCENE, facing_yaw, load_qpos,  # noqa: E402
                                 method_qpos_paths, object_world_traj,
                                 register_object, yaw_trans_fit, apply_T)
from viz_mixed_robot_smplx import _geom_rgb, look_at, object_world_mesh, robot_world_meshes  # noqa: E402

# left-arm bodies hidden in the hand close-up so only the RIGHT hand shows
LEFT_ARM = ["left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
            "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link",
            "left_wrist_yaw_link"]


def robot_meshes_hide(model, data, hide_bids):
    """World-frame robot mesh geoms grouped by material colour, excluding the object
    and any body in hide_bids (drops the left arm for the hand close-up)."""
    obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    groups: dict = {}
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        bid = int(model.geom_bodyid[gid])
        if bid == obj or bid in hide_bids:
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

# --- figure layout / labels (method name text fixed per the user's spec) ---
GRID = [["gmr", "omniretarget"], ["sbto", "spider"]]      # rows: [TL,TR],[BL,BR]
LABEL = {"gmr": "GMR", "omniretarget": "OmniRetarget",
         "sbto": "DynaRetarget", "spider": "Ours"}
# pastel label backgrounds (Ours highlighted green), matching the reference look
LABEL_BG = {"gmr": (255, 235, 205), "omniretarget": (219, 226, 245),
            "sbto": (255, 250, 205), "spider": (205, 245, 215)}
# per-panel annotation lines for variant 3 (edit freely). each line: (text, ok).
# ok True -> green check, False -> red cross.
ANNOT = {
    "gmr":          [("Not in contact", False), ("Feet floating", False)],
    "omniretarget": [("Loose contact", False), ("Feet floating", False)],
    "sbto":         [("Penetration", False)],
    "spider":       [("Clean contact", True), ("No major artifacts", True)],
}
# methods that also get a foot/ground close-up inset (feet visibly float)
FOOT_ZOOM = {"gmr", "omniretarget"}

ORANGE = (233, 126, 34)
GREEN = (40, 170, 70)
RED = (210, 55, 45)


def find_font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def project(pt, cam_pose, yfov, W, H):
    """World point -> (px,py) pixel for a pyrender PerspectiveCamera (aspect 1)."""
    Rcw = cam_pose[:3, :3]
    eye = cam_pose[:3, 3]
    pc = Rcw.T @ (pt - eye)
    z = -pc[2]
    f = 1.0 / np.tan(yfov / 2.0)
    x = (pc[0] / z) * f
    y = (pc[1] / z) * f
    return (x + 1) / 2 * W, (1 - (y + 1) / 2) * H


def build_alignment(scene, case):
    """Recompute regs + common window + camera exactly as viz_methods_compare."""
    methods = ["spider", "omniretarget", "sbto", "gmr"]
    paths = method_qpos_paths(case)
    qpos = {m: load_qpos(paths[m]) for m in methods}
    ref = qpos["spider"]
    ref_obj = object_world_traj(scene, ref)
    ref_root0 = ref[0, 0:3]
    regs = {}
    for m in methods:
        q = qpos[m]
        if m == "spider":
            regs[m] = (0, np.eye(4))
            continue
        obj = object_world_traj(scene, q)
        native = (len(q) == len(ref)) and (np.linalg.norm(q[0, 0:3] - ref_root0) < 0.10)
        if native:
            n = min(len(obj), len(ref_obj))
            T, _, _ = yaw_trans_fit(obj[:n], ref_obj[:n])
            regs[m] = (0, T)
        else:
            lag, T, _, _, _ = register_object(obj, ref_obj)
            regs[m] = (lag, T)
    w0 = max(max(0, regs[m][0]) for m in methods)
    w1 = min(min(len(ref), len(qpos[m]) + regs[m][0]) for m in methods)
    return qpos, ref, regs, (w0, w1)


def foot_sole_world(model, data, foot_gid):
    """Lowest vertex (world) of the foot mesh geom -- the sole point over the floor."""
    did = model.geom_dataid[foot_gid]
    va, vn = model.mesh_vertadr[did], model.mesh_vertnum[did]
    v = model.mesh_vert[va:va + vn].reshape(-1, 3)
    vw = v @ data.geom_xmat[foot_gid].reshape(3, 3).T + data.geom_xpos[foot_gid]
    return vw[np.argmin(vw[:, 2])]


def _render_meshes(robot, ov, of, floor_z, cam_pose, yfov, renderer):
    """Build a pyrender scene from prepared world meshes and render one bg frame."""
    mat = lambda c, r=0.55: pyrender.MetallicRoughnessMaterial(
        baseColorFactor=c, metallicFactor=0.0, roughnessFactor=r, doubleSided=True)
    scene_pr = pyrender.Scene(bg_color=[0.90, 0.91, 0.93, 1.0], ambient_light=[0.4, 0.4, 0.42])
    scene_pr.add(pyrender.Mesh.from_trimesh(make_checker_floor(floor_z, extent=12.0, repeats=14, up="z")))
    for rv, rf, rgb in robot:
        scene_pr.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(rv, rf, process=False),
                                                material=mat(list(rgb) + [1.0]), smooth=False))
    scene_pr.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(ov, of, process=False),
                                            material=mat(BOX_COLOR), smooth=False))
    cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
    scene_pr.add(cam, pose=cam_pose)
    tgt = cam_pose[:3, 3] - cam_pose[:3, 2]                  # a point in front of the camera
    for light, pose in soft_phong_lights(look_at, tgt, [0, 0, 1], floor_z, up="z"):
        scene_pr.add(light, pose=pose)
    color, _ = renderer.render(scene_pr, flags=pyrender.constants.RenderFlags.SHADOWS_DIRECTIONAL)
    return Image.fromarray(color[:, :, :3].copy())


def render_panel(scene, qpos_m, lag, T, ref_i, res, yfov, cdir, dist, renderer, rh_bid, foot_gid,
                 hand_cdir, hand_radius, hide_bids):
    """Render main panel + a separate right-hand/object contact close-up (different angle).

    Returns (panel_img, {'hand':(px,py),'foot':(px,py)}, hand_closeup_img)."""
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_m[ref_i - lag]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    robot = [(apply_T(v, T), f, rgb) for v, f, rgb in robot_world_meshes(model, data)]
    robot_noleft = [(apply_T(v, T), f, rgb) for v, f, rgb in robot_meshes_hide(model, data, hide_bids)]
    ov, of, _ = object_world_mesh(model, data)
    ov = apply_T(ov, T)
    rhand = apply_T(data.xpos[rh_bid][None], T)[0]          # right hand (wrist)
    foot = apply_T(foot_sole_world(model, data, foot_gid)[None], T)[0]

    rmean = np.concatenate([g[0] for g in robot], 0).mean(0)
    target = (rmean + ov.mean(0)) / 2.0
    allv = np.concatenate([g[0] for g in robot] + [ov], 0)
    floor_z = float(allv[:, 2].min())

    cam_pose = look_at(target + cdir * dist, target, [0, 0, 1])
    panel = _render_meshes(robot, ov, of, floor_z, cam_pose, yfov, renderer)

    # dedicated hand/object contact close-up: tighter camera, chosen angle, LEFT arm
    # hidden (only the right hand), centered on the interface (midpoint of the right
    # hand and the nearest object vertex).
    nearest = ov[np.argmin(np.linalg.norm(ov - rhand, axis=1))]
    contact = 0.5 * (rhand + nearest)
    hdist = hand_radius / np.tan(yfov / 2.0)
    hpose = look_at(contact + hand_cdir * hdist, contact, [0, 0, 1])
    closeup = _render_meshes(robot_noleft, ov, of, floor_z, hpose, yfov, renderer)

    return (panel,
            {"hand": project(rhand, cam_pose, yfov, res, res),
             "foot": project(foot, cam_pose, yfov, res, res)},
            closeup)


def circular_inset(src_img, cx, cy, src_size, out_size, border_color, border_w):
    """Crop a square (side src_size) centered at (cx,cy) from src_img, circular-mask it."""
    W, H = src_img.size
    half = src_size // 2
    x0 = int(np.clip(cx - half, 0, W - src_size))
    y0 = int(np.clip(cy - half, 0, H - src_size))
    patch = src_img.crop((x0, y0, x0 + src_size, y0 + src_size)).resize((out_size, out_size))
    mask = Image.new("L", (out_size, out_size), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, out_size, out_size], fill=255)
    out = Image.new("RGBA", (out_size, out_size), (0, 0, 0, 0))
    out.paste(patch, (0, 0), mask)
    ImageDraw.Draw(out).ellipse([border_w // 2, border_w // 2,
                                 out_size - border_w // 2, out_size - border_w // 2],
                                outline=border_color, width=border_w)
    return out


def circular_whole(src_img, out_size, border_color, border_w):
    """Circular-mask an already-framed square image (e.g. a dedicated close-up render)."""
    patch = src_img.resize((out_size, out_size))
    mask = Image.new("L", (out_size, out_size), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, out_size, out_size], fill=255)
    out = Image.new("RGBA", (out_size, out_size), (0, 0, 0, 0))
    out.paste(patch, (0, 0), mask)
    ImageDraw.Draw(out).ellipse([border_w // 2, border_w // 2,
                                 out_size - border_w // 2, out_size - border_w // 2],
                                outline=border_color, width=border_w)
    return out


def arrow(draw, p0, p1, color, width, head=22):
    draw.line([p0, p1], fill=color, width=width)
    d = np.array(p1, float) - np.array(p0, float)
    n = np.linalg.norm(d)
    if n < 1e-6:
        return
    d /= n
    perp = np.array([-d[1], d[0]])
    tip = np.array(p1, float)
    a = tip - d * head + perp * head * 0.6
    b = tip - d * head - perp * head * 0.6
    draw.polygon([tuple(tip), tuple(a), tuple(b)], fill=color)


def main():
    ap = argparse.ArgumentParser(description="Assemble the 4-method comparison paper figure.")
    ap.add_argument("--case", default="box021_20231011_037")
    ap.add_argument("--scene", default=str(DEFAULT_SCENE))
    ap.add_argument("--file-frame", type=int, default=12,
                    help="saved-frame index (frame_XXXX.png) to base the figure on")
    ap.add_argument("--res", type=int, default=1100, help="per-panel render resolution")
    ap.add_argument("--out", default=None)
    # crop (remove empty sky/floor); fractions of the rendered panel
    ap.add_argument("--crop", default="0.06,0.11,0.94,0.88",
                    help="left,top,right,bottom fractions to keep")
    ap.add_argument("--zoom-src", type=int, default=300, help="foot close-up source square px (full-res)")
    ap.add_argument("--zoom-out", type=int, default=300, help="close-up inset diameter px")
    # dedicated hand/object close-up camera (rendered from a different angle so the
    # contact reads clearly, unlike a crop of the 3/4 panel view)
    ap.add_argument("--hand-az-offset", type=float, default=-126.1,
                    help="hand close-up azimuth = robot facing yaw + this offset (deg)")
    ap.add_argument("--hand-el", type=float, default=5.0, help="hand close-up elevation deg")
    ap.add_argument("--hand-radius", type=float, default=0.25,
                    help="world half-size framed by the hand close-up (m); larger = more "
                         "box context so the hand-object contact relationship is visible")
    args = ap.parse_args()

    scene = args.scene
    qpos, ref, regs, (w0, w1) = build_alignment(scene, args.case)
    ref_i = w0 + args.file_frame
    assert w0 <= ref_i < w1, f"frame {args.file_frame} -> ref {ref_i} outside [{w0},{w1})"

    facing = facing_yaw(ref)
    az, el = np.radians(facing + 45.0), np.radians(12.0)
    cdir = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    # hand close-up camera direction (different angle to reveal the contact)
    haz, hel = np.radians(facing + args.hand_az_offset), np.radians(args.hand_el)
    hand_cdir = np.array([np.cos(hel) * np.cos(haz), np.cos(hel) * np.sin(haz), np.sin(hel)])

    # right hand (wrist) body + left foot mesh geom for the close-ups
    _m = mujoco.MjModel.from_xml_path(scene)
    rh_bid = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    foot_gid = next(gid for gid in range(_m.ngeom)
                    if _m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH
                    and (mujoco.mj_id2name(_m, mujoco.mjtObj.mjOBJ_BODY, _m.geom_bodyid[gid]) or "")
                    == "left_ankle_roll_link")
    hide_bids = {mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_BODY, n) for n in LEFT_ARM}
    ref_frames = list(range(w0, w1))
    sample = ref_frames[:: max(1, len(ref_frames) // 12)] or ref_frames
    renderer = pyrender.OffscreenRenderer(args.res, args.res)
    yfov = np.pi / 4.0
    radius = 0.0
    for m in ["spider", "omniretarget", "sbto", "gmr"]:
        lag, T = regs[m]
        model = mujoco.MjModel.from_xml_path(scene)
        data = mujoco.MjData(model)
        for ri in sample:
            data.qpos[:] = qpos[m][ri - lag]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            robot = [(apply_T(v, T), f, rgb) for v, f, rgb in robot_world_meshes(model, data)]
            ov, _, _ = object_world_mesh(model, data)
            ov = apply_T(ov, T)
            rmean = np.concatenate([g[0] for g in robot], 0).mean(0)
            tgt = (rmean + ov.mean(0)) / 2.0
            allv = np.concatenate([g[0] for g in robot] + [ov], 0)
            radius = max(radius, float(np.linalg.norm(allv - tgt, axis=1).max()))
    dist = radius / np.tan(yfov / 2.0) * 1.10
    print(f"[fig] ref_i={ref_i} facing={facing:.1f} az={np.degrees(az):.1f} dist={dist:.2f}")

    # render every panel; keep full-res image + projected right-hand / foot pixels
    panels = {}
    for m in ["gmr", "omniretarget", "sbto", "spider"]:
        lag, T = regs[m]
        img, kp, closeup = render_panel(scene, qpos[m], lag, T, ref_i, args.res, yfov, cdir, dist,
                                        renderer, rh_bid, foot_gid, hand_cdir, args.hand_radius,
                                        hide_bids)
        panels[m] = (img, kp, closeup)
        print(f"[panel] {m:12s} hand_px=({kp['hand'][0]:.0f},{kp['hand'][1]:.0f}) "
              f"foot_px=({kp['foot'][0]:.0f},{kp['foot'][1]:.0f})")
    renderer.delete()

    cl, ct, cr, cb = (float(x) for x in args.crop.split(","))
    W = args.res
    box = (int(cl * W), int(ct * W), int(cr * W), int(cb * W))
    cw, ch = box[2] - box[0], box[3] - box[1]
    label_font = find_font(max(22, cw // 16))
    annot_font = find_font(max(18, cw // 22))

    out_dir = Path(args.out) if args.out else \
        REPO / f"workspace/core4d/report/0908/paper_results/viz/{args.case}_p2/paper_figure"
    out_dir.mkdir(parents=True, exist_ok=True)

    def place_inset(base, im, dr, target_px, corner_xy):
        """Composite a circular close-up at corner_xy with an arrow to target_px."""
        px, py = target_px
        inset = circular_inset(base, px, py, args.zoom_src, args.zoom_out, ORANGE, 8)
        ix, iy = corner_xy
        tx, ty = px - box[0], py - box[1]                 # target in cropped coords
        cx, cy = ix + args.zoom_out / 2, iy + args.zoom_out / 2
        d = np.array([tx - cx, ty - cy], float)
        d /= max(np.linalg.norm(d), 1e-6)
        start = (cx + d[0] * args.zoom_out / 2, cy + d[1] * args.zoom_out / 2)
        arrow(dr, start, (tx, ty), ORANGE, 7)
        im.alpha_composite(inset, (int(ix), int(iy)))

    def label_panel(base, m, with_zoom, with_annot):
        im = base.crop(box).convert("RGBA")
        dr = ImageDraw.Draw(im)
        # close-up insets (right side). Hand/object contact is a DEDICATED render from
        # a different angle (a panel crop hides the contact); foot/ground is a panel crop.
        if with_zoom:
            margin = int(cw * 0.02)
            hx0 = cw - args.zoom_out - margin
            # hand close-up (whole dedicated render), with arrow to the hand in the panel
            hand_inset = circular_whole(panels[m][2], args.zoom_out, ORANGE, 8)
            hcx, hcy = hx0 + args.zoom_out / 2, margin + args.zoom_out / 2
            tx, ty = panels[m][1]["hand"][0] - box[0], panels[m][1]["hand"][1] - box[1]
            dd = np.array([tx - hcx, ty - hcy], float)
            dd /= max(np.linalg.norm(dd), 1e-6)
            arrow(dr, (hcx + dd[0] * args.zoom_out / 2, hcy + dd[1] * args.zoom_out / 2),
                  (tx, ty), ORANGE, 7)
            im.alpha_composite(hand_inset, (int(hx0), int(margin)))
            if m in FOOT_ZOOM:
                place_inset(base, im, dr, panels[m][1]["foot"],
                            (hx0, margin + args.zoom_out + int(ch * 0.02)))
        # annotation lines (top-left)
        if with_annot:
            ax, ay = int(cw * 0.03), int(ch * 0.03)
            lh = int(getattr(annot_font, "size", 20) * 1.28)
            for i, (atxt, ok) in enumerate(ANNOT[m]):
                mark = "✓ " if ok else "✗ "
                col = GREEN if ok else RED
                mb = dr.textbbox((0, 0), mark, font=annot_font)
                y = ay + i * lh
                dr.text((ax, y), mark, fill=col, font=annot_font)
                dr.text((ax + (mb[2] - mb[0]), y), atxt, fill=(20, 20, 20), font=annot_font)
        # method label flush to the bottom-left corner (drawn last, on top)
        txt = LABEL[m]
        tb = dr.textbbox((0, 0), txt, font=label_font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = int(th * 0.45)
        boxw, boxh = tw + 2 * pad, th + 2 * pad
        dr.rectangle([0, ch - boxh, boxw, ch], fill=LABEL_BG[m])
        dr.text((pad, ch - boxh + pad - tb[1]), txt, fill=(20, 20, 20), font=label_font)
        return im

    def assemble(with_zoom, with_annot, name):
        gap = max(6, cw // 90)
        grid = Image.new("RGBA", (2 * cw + gap, 2 * ch + gap), (255, 255, 255, 255))
        for r, rowm in enumerate(GRID):
            for c, m in enumerate(rowm):
                panel = label_panel(panels[m][0], m, with_zoom, with_annot)
                grid.alpha_composite(panel, (c * (cw + gap), r * (ch + gap)))
        out = out_dir / f"{name}.png"
        grid.convert("RGB").save(out)
        print(f"[ok] {out}")

    assemble(False, False, "1_plain")
    assemble(True, False, "2_zoom")
    assemble(True, True, "3_annotated")


if __name__ == "__main__":
    main()
