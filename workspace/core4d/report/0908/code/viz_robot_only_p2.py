"""Single-robot (p2 only) clean render: the SPIDER-CEM G1 + its manipulated object.

Companion to viz_mixed_robot_smplx.py / viz_smplx_reference.py: renders ONLY person2's
retargeted G1 robot + the shared object (no human, no second robot), white background,
per-frame PNGs. Same pyrender style (silver G1 links, slate-blue box, soft Phong) and
the SAME camera as the mixed render -- it reads the mixed render's camera_align.json
(cem_az_deg / cem_el_deg) so p2 faces the camera identically across the three figures.

Run (headless):
    PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa .venv/bin/python \
        workspace/core4d/report/0908/code/viz_robot_only_p2.py --case <case> \
        --exp E198 --cem-dir <dir> --scene-prefix dcv3_omnirt_v1_ref_fk --out <OUT>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse
import json

import numpy as np

if not hasattr(np, "infty"):        # pyrender uses np.infty, removed in NumPy 2.0
    np.infty = np.inf

import mujoco
import trimesh
import pyrender
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_style import (BOX_COLOR, add_common_args, default_out,  # noqa: E402
                          make_checker_floor, soft_phong_lights)
from viz_mixed_robot_smplx import (look_at, mean_facing_yaw,  # noqa: E402
                                   object_world_mesh, robot_world_meshes)

REPO = Path(__file__).resolve().parents[5]
DEFAULT_PROCESSED_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"


def parse_args():
    p = argparse.ArgumentParser(description="Single p2 robot + object render (white bg).")
    add_common_args(p)      # --case --data-root --smplx-model --out --res --fps --preview
    p.add_argument("--primary", default="p2", help="person whose robot is rendered (default: p2)")
    p.add_argument("--exp", default="E170", help="retarget experiment id (default: E170)")
    p.add_argument("--cem-dir", default=None, help="CEM results dir (default derived from --exp)")
    p.add_argument("--cem-suffix", default="PRG", help="CEM npz name suffix (default: PRG)")
    p.add_argument("--processed-root", default=str(DEFAULT_PROCESSED_ROOT))
    p.add_argument("--scene-prefix", default="dcv3_omnirt_v1_ref_fk")
    p.add_argument("--window", default=None, help="cem frame window 'lo,hi' (default: full)")
    p.add_argument("--cam-az-offset", type=float, default=45.0,
                   help="fallback azimuth = robot facing yaw + offset (deg) when no align json")
    p.add_argument("--cam-el", type=float, default=12.0)
    return p.parse_args()


def main():
    args = parse_args()
    CEM_DIR = Path(args.cem_dir) if args.cem_dir else \
        REPO / f"workspace/core4d/results/{args.exp}/s6_downstream/cem/full"
    scene = (Path(args.processed_root) / f"{args.scene_prefix}_{args.case}_{args.primary}"
             / f"scene_act_{args.exp}_lowerbody_physics.xml")
    OUT = Path(args.out) if args.out else default_out(args.case, args.primary)
    OUT.mkdir(parents=True, exist_ok=True)
    RES, PREVIEW_N = args.res, args.preview

    q = np.load(CEM_DIR / f"{args.exp}_{args.case}_{args.primary}_{args.cem_suffix}.npz")["qpos"]
    q = q[:, 0, :] if q.ndim == 3 else q
    lo, hi = (0, len(q)) if args.window is None else tuple(int(x) for x in args.window.split(","))

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)

    # Camera: reuse the mixed render's exact viewpoint (cem az/el) if its align json is
    # present in OUT; else derive azimuth from this robot's own facing yaw.
    align_path = OUT / "camera_align.json"
    if align_path.exists():
        j = json.loads(align_path.read_text())
        az, el = np.radians(j["cem_az_deg"]), np.radians(j["cem_el_deg"])
        print(f"[cam] aligned to mixed render via {align_path}: az={j['cem_az_deg']:.1f} el={j['cem_el_deg']:.1f}")
    else:
        az = np.radians(mean_facing_yaw(q) + args.cam_az_offset)
        el = np.radians(args.cam_el)
        print(f"[cam] no align json; az={np.degrees(az):.1f} el={np.degrees(el):.1f}")
    cdir = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    yfov = np.pi / 4.0

    def frame_meshes(f):
        data.qpos[:] = q[f]; data.qvel[:] = 0.0; mujoco.mj_forward(model, data)
        return robot_world_meshes(model, data), object_world_mesh(model, data)

    sample = list(range(lo, hi, max(1, (hi - lo) // 16)))
    radius, floor_z = 0.0, np.inf
    for f in sample:
        robot, (ov, _, _) = frame_meshes(f)
        allv = np.concatenate([g[0] for g in robot] + [ov], 0)
        tgt = (np.concatenate([g[0] for g in robot], 0).mean(0) + ov.mean(0)) / 2.0
        radius = max(radius, float(np.linalg.norm(allv - tgt, axis=1).max()))
        floor_z = min(floor_z, float(allv[:, 2].min()))
    dist = radius / np.tan(yfov / 2.0) * 1.15
    floor_z = float(floor_z)

    frame_fs = list(range(lo, hi))
    if PREVIEW_N > 0:
        idx = np.linspace(0, len(frame_fs) - 1, PREVIEW_N).round().astype(int)
        frame_fs = [frame_fs[i] for i in idx]

    renderer = pyrender.OffscreenRenderer(RES, RES)
    mat = lambda c, r=0.55: pyrender.MetallicRoughnessMaterial(
        baseColorFactor=c, metallicFactor=0.0, roughnessFactor=r, doubleSided=True)

    frame_dir = OUT / "robot_frames_white"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frame_fs):
        robot, (ov, of, _orgb) = frame_meshes(f)
        target = (np.concatenate([g[0] for g in robot], 0).mean(0) + ov.mean(0)) / 2.0
        scene_pr = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.4, 0.4, 0.42])
        for rv, rf, rgb in robot:
            scene_pr.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(rv, rf, process=False),
                                                    material=mat(list(rgb) + [1.0]), smooth=False))
        scene_pr.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(ov, of, process=False),
                                                material=mat(BOX_COLOR), smooth=False))
        scene_pr.add(pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0),
                     pose=look_at(target + cdir * dist, target, [0, 0, 1]))
        for light, pose in soft_phong_lights(look_at, target, [0, 0, 1], floor_z, up="z"):
            scene_pr.add(light, pose=pose)
        color, depth = renderer.render(scene_pr, flags=0)
        rgb = color[:, :, :3].copy()
        rgb[depth == 0] = 255
        imageio.imwrite(frame_dir / f"frame_{i:04d}.png", rgb)
    renderer.delete()
    print(f"[ok] white: cem {args.primary}[{lo},{hi}) {len(frame_fs)} frames -> {frame_dir}")


if __name__ == "__main__":
    main()
