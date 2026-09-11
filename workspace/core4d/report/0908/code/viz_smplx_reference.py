"""SMPLX reference render for CORE4D box021_20231011_037 (both persons + object).

Renders the ground-truth human motion: person1 + person2 SMPLX meshes and the
posed object mesh, single fixed 3/4 view, pure white background, per-frame PNGs
+ mp4. Companion to viz_dual_robot_clean.py (the retargeted-robot version).

Vertices are pre-baked in the CORE4D pose npz (no smplx forward needed); only the
SMPLX topology (faces) is loaded, from the GMR-hosted SMPLX_NEUTRAL.npz. World is
Y-up; the pair carries the box along +Z.

Any CORE4D sequence + data paths are configurable via CLI (see --help). Run
headless with osmesa:
    PYOPENGL_PLATFORM=osmesa .venv/bin/python \
        workspace/core4d/report/0908/code/viz_smplx_reference.py \
        --case box021_20231011_037 --data-root <CORE4D_Real> --smplx-model <NPZ>
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import shutil

import sys

import numpy as np

if not hasattr(np, "infty"):        # pyrender uses np.infty, removed in NumPy 2.0
    np.infty = np.inf

import trimesh
import pyrender
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_style import (BOX_COLOR, add_common_args, default_out,  # noqa: E402
                          make_checker_floor, object_mesh_path, parse_case,
                          seq_dir, soft_phong_lights)
from smplx_min import SMPLXModel  # noqa: E402

_SMPLX_KEYS = ["global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "transl"]

# CORE4D-Instructions official human palette; object recoloured to slate blue.
P1_COLOR = [135 / 255, 235 / 255, 156 / 255, 1.0]   # green  (person1)
P2_COLOR = [117 / 255, 157 / 255, 231 / 255, 1.0]   # blue   (person2)
OBJ_COLOR = BOX_COLOR                                # slate blue (object)


def default_shape_verts(model: SMPLXModel, person: dict) -> np.ndarray:
    """(T,V,3) vertices re-posed at the neutral shape from a CORE4D pose dict."""
    n = len(person["global_orient"])
    return np.stack([model.forward({k: person[k][t] for k in _SMPLX_KEYS})
                     for t in range(n)])


def parse_args():
    p = argparse.ArgumentParser(description="SMPLX reference render (both persons + object).")
    add_common_args(p)
    p.add_argument("--primary", default="p2", help="output-dir person tag (default: p2)")
    p.add_argument("--window", default=None, help="frame window 'lo,hi' (default: full clip)")
    p.add_argument("--baked-shape", action="store_true",
                   help="use the subjects' baked betas instead of the neutral betas=0 shape")
    p.add_argument("--object-mesh", default=None, help="override object .obj path")
    return p.parse_args()


def look_at(eye, target, up):
    """4x4 camera-to-world pose (OpenGL convention: camera looks down -Z)."""
    eye, target, up = map(lambda a: np.asarray(a, float), (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4)
    M[:3, 0] = s
    M[:3, 1] = u
    M[:3, 2] = -f
    M[:3, 3] = eye
    return M


def main():
    args = parse_args()
    obj, date, seq, category = parse_case(args.case)
    SEQ = seq_dir(args.data_root, date, seq)
    OBJ_MESH = Path(args.object_mesh) if args.object_mesh else \
        object_mesh_path(args.data_root, obj, category)
    SMPLX_NEUTRAL = Path(args.smplx_model)
    OUT = Path(args.out) if args.out else default_out(args.case, args.primary)
    STAGE = Path(f"/tmp/smplx_stage_{args.case}")
    RES, FPS, PREVIEW_N = args.res, args.fps, args.preview
    DEFAULT_SHAPE = not args.baked_shape
    WINDOW = tuple(int(x) for x in args.window.split(",")) if args.window else None

    STAGE.mkdir(parents=True, exist_ok=True)
    d1 = np.load(SEQ / "person1_poses.npz", allow_pickle=True)["arr_0"].item()
    d2 = np.load(SEQ / "person2_poses.npz", allow_pickle=True)["arr_0"].item()
    if DEFAULT_SHAPE:
        model = SMPLXModel(str(SMPLX_NEUTRAL))
        p1, p2 = default_shape_verts(model, d1), default_shape_verts(model, d2)
        print("[shape] re-posed both persons at neutral betas=0")
    else:
        p1, p2 = d1["vertices"], d2["vertices"]
    obj_poses = np.load(SEQ / "smooth_objposes.npy")
    faces = np.load(SMPLX_NEUTRAL, allow_pickle=True)["f"].astype(np.int64)
    obj_mesh = trimesh.load(OBJ_MESH, process=False)
    obj_v0, obj_f = np.asarray(obj_mesh.vertices), np.asarray(obj_mesh.faces)

    T = min(len(p1), len(p2), len(obj_poses))
    lo, hi = (0, T) if WINDOW is None else (WINDOW[0], min(WINDOW[1], T))

    obj_c0 = obj_v0.mean(0)

    def obj_verts(t):
        return obj_v0 @ obj_poses[t][:3, :3].T + obj_poses[t][:3, 3]

    # Fixed vertical center: lock the camera target's Y (up) to a constant so the
    # floor line does NOT bob as the box is lifted (following object Y made the pair
    # look like they levitate mid-sequence). Follow only horizontally (X, Z).
    target_y = float(np.median([(p1[t].mean(0)[1] + p2[t].mean(0)[1]) / 2 for t in range(lo, hi)]))

    def frame_centroid(t):
        oc = obj_c0 @ obj_poses[t][:3, :3].T + obj_poses[t][:3, 3]
        c = (p1[t].mean(0) + p2[t].mean(0) + oc) / 3.0
        c[1] = target_y
        return c

    # Follow camera: frontal 3/4 view (holosoma visualize_retarget_smpl style),
    # per-frame lookat on the horizontally-following centroid; distance fits the
    # widest single-frame spread over the window.
    yfov = np.pi / 3.0                            # 60 deg, like the reference
    direction = np.array([1.0, 0.3, -1.0])        # front-right, slightly elevated
    direction /= np.linalg.norm(direction)
    sub = list(range(lo, hi, max(1, (hi - lo) // 16)))
    radius = max(
        np.linalg.norm(np.ptp(np.concatenate([p1[t], p2[t], obj_verts(t)], 0), axis=0)) / 2.0
        for t in sub)
    dist = radius / np.tan(yfov / 2.0) * 1.25

    floor_y = float(min(min(p1[t][:, 1].min(), p2[t][:, 1].min()) for t in sub))
    floor_trimesh = make_checker_floor(floor_y, extent=10.0, repeats=12, up="y")

    frame_ts = list(range(lo, hi))
    if PREVIEW_N > 0:
        idx = np.linspace(0, len(frame_ts) - 1, PREVIEW_N).round().astype(int)
        frame_ts = [frame_ts[i] for i in idx]
        print(f"[preview] rendering {PREVIEW_N} sampled frames")

    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
    renderer = pyrender.OffscreenRenderer(RES, RES)
    # CARI4D-style soft Phong: low metallic, mid-high roughness, smooth normals.
    mat = lambda c, r=0.55: pyrender.MetallicRoughnessMaterial(
        baseColorFactor=c, metallicFactor=0.0, roughnessFactor=r, doubleSided=True)
    shadow_flag = pyrender.constants.RenderFlags.SHADOWS_DIRECTIONAL

    def render_version(mode: str):
        assert mode in ("bg", "white")
        with_bg = mode == "bg"
        stage_dir = STAGE / f"smplx_frames_{mode}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        bg_col = [0.90, 0.91, 0.93, 1.0] if with_bg else [1.0, 1.0, 1.0, 1.0]
        frames = []
        for t in frame_ts:
            scene = pyrender.Scene(bg_color=bg_col, ambient_light=[0.4, 0.4, 0.42])
            if with_bg:
                scene.add(pyrender.Mesh.from_trimesh(floor_trimesh))
            scene.add(pyrender.Mesh.from_trimesh(
                trimesh.Trimesh(p1[t], faces, process=False), material=mat(P1_COLOR), smooth=True))
            scene.add(pyrender.Mesh.from_trimesh(
                trimesh.Trimesh(p2[t], faces, process=False), material=mat(P2_COLOR), smooth=True))
            scene.add(pyrender.Mesh.from_trimesh(
                trimesh.Trimesh(obj_verts(t), obj_f, process=False), material=mat(OBJ_COLOR), smooth=True))
            target = frame_centroid(t)
            scene.add(camera, pose=look_at(target + direction * dist, target, [0, 1, 0]))
            for light, pose in soft_phong_lights(look_at, target, [0, 1, 0], floor_y, up="y"):
                scene.add(light, pose=pose)
            color, _ = renderer.render(scene, flags=shadow_flag if with_bg else 0)
            frames.append(color[:, :, :3].copy())
            imageio.imwrite(stage_dir / f"frame_{frame_ts.index(t):04d}.png", color[:, :, :3])
        return frames, stage_dir

    for mode in ("bg", "white"):
        frames, stage_dir = render_version(mode)
        out_frame_dir = OUT / f"smplx_frames_{mode}"
        out_frame_dir.mkdir(parents=True, exist_ok=True)
        for png in sorted(stage_dir.glob("frame_*.png")):
            shutil.copy2(png, out_frame_dir / png.name)
        if PREVIEW_N == 0:
            mp4_stage = STAGE / f"smplx_reference_{mode}.mp4"
            imageio.mimsave(mp4_stage, frames, fps=FPS, quality=9)
            shutil.copy2(mp4_stage, OUT / f"smplx_reference_{mode}.mp4")
            print(f"[ok] {mode}: [{lo},{hi}) {len(frames)} frames -> {out_frame_dir}  "
                  f"video -> {OUT / f'smplx_reference_{mode}.mp4'}")
        else:
            print(f"[ok] {mode}: {len(frames)} preview frames -> {out_frame_dir}")
    renderer.delete()


if __name__ == "__main__":
    main()
