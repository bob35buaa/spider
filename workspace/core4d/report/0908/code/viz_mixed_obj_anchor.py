"""Mixed render (object-anchored partner): primary robot + partner SMPLX human + box,
for CORE4D sequences where ONLY the primary person was retargeted (no partner robot).

Unlike viz_mixed_robot_smplx.py (which anchors the partner human to the PARTNER's
retargeted robot pelvis), this variant anchors the partner human to the SHARED OBJECT:
each frame the human is mapped from mocap into the primary's CEM frame through the box's
own 6-DOF pose (cem_object_pose @ mocap_object_pose^-1). The human therefore keeps its
exact mocap pose relative to the box -- hands stay on the box -- without needing any
partner-side retargeting. Same pyrender style / camera as viz_mixed_robot_smplx.py.

Run:
    PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa .venv/bin/python \
        workspace/core4d/report/0908/code/viz_mixed_obj_anchor.py --case <case> \
        --primary p1 --partner p2 --exp E198 --cem-dir <dir> --cem-suffix G1A2 \
        --scene-prefix dcv3_omnirt_v2_ref_fk --primary-trim <m> --human-scale 0.8 --out <OUT>
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

if not hasattr(np, "infty"):
    np.infty = np.inf

import trimesh
import pyrender
import mujoco
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_style import (BOX_COLOR, add_common_args, default_out,  # noqa: E402
                          make_checker_floor, parse_case, seq_dir, soft_phong_lights)
from smplx_min import SMPLXModel  # noqa: E402
from viz_mixed_robot_smplx import (look_at, mean_facing_yaw, object_world_mesh,  # noqa: E402
                                   robot_world_meshes, P1_COLOR, _SMPLX_KEYS)

REPO = Path(__file__).resolve().parents[5]
DEFAULT_PROCESSED_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"


def parse_args():
    p = argparse.ArgumentParser(description="Object-anchored mixed render (primary robot + partner human + box).")
    add_common_args(p)
    p.add_argument("--primary", default="p1", help="person whose robot is rendered (default: p1)")
    p.add_argument("--partner", default="p2", help="SMPLX human person, object-anchored (default: p2)")
    p.add_argument("--exp", default="E198")
    p.add_argument("--cem-dir", required=True)
    p.add_argument("--cem-suffix", default="G1A2")
    p.add_argument("--processed-root", default=str(DEFAULT_PROCESSED_ROOT))
    p.add_argument("--scene-prefix", default="dcv3_omnirt_v2_ref_fk")
    p.add_argument("--primary-trim", type=int, required=True,
                   help="primary cem[t] <-> mocap[primary_trim + t]")
    p.add_argument("--window", default=None, help="primary cem frame window 'lo,hi' (default: full)")
    p.add_argument("--human-scale", type=float, default=0.8,
                   help="uniform scale of the partner mesh about its grounded feet (default: 0.8)")
    p.add_argument("--baked-shape", action="store_true")
    p.add_argument("--cam-az-offset", type=float, default=45.0)
    p.add_argument("--cam-el", type=float, default=12.0)
    return p.parse_args()


def object_body_pose(model, data):
    """(R 3x3, t 3) world pose of the object body (convention-independent, from mj)."""
    ob = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    return data.xmat[ob].reshape(3, 3).copy(), data.xpos[ob].copy()


def main():
    args = parse_args()
    obj, date, seq, _ = parse_case(args.case)
    SEQ = seq_dir(args.data_root, date, seq)
    OUT = Path(args.out) if args.out else default_out(args.case, args.primary)
    OUT.mkdir(parents=True, exist_ok=True)
    RES, PREVIEW_N = args.res, args.preview
    CEM = Path(args.cem_dir)
    scene = (Path(args.processed_root) / f"{args.scene_prefix}_{args.case}_{args.primary}"
             / f"scene_act_{args.exp}_lowerbody_physics.xml")

    # primary robot + object
    qp = np.load(CEM / f"{args.exp}_{args.case}_{args.primary}_{args.cem_suffix}.npz")["qpos"]
    qp = qp[:, 0, :] if qp.ndim == 3 else qp
    model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model)

    # partner human (neutral re-pose by default) + shared object mocap trajectory
    partner_file = SEQ / f"person{args.partner[1:]}_poses.npz"
    pose = np.load(partner_file, allow_pickle=True)["arr_0"].item()
    smplx_faces = np.load(args.smplx_model, allow_pickle=True)["f"].astype(np.int64)
    if args.baked_shape:
        ph_verts = pose["vertices"]
    else:
        sm = SMPLXModel(str(args.smplx_model))
        n = len(pose["global_orient"])
        ph_verts = np.stack([sm.forward({k: pose[k][t] for k in _SMPLX_KEYS}) for t in range(n)])
        print("[shape] re-posed partner at neutral betas=0")
    obj_moc = np.load(SEQ / "smooth_objposes.npy")       # (T,4,4) y-up mocap object pose

    lo, hi = (0, len(qp)) if args.window is None else tuple(int(x) for x in args.window.split(","))

    def frame_data(f):
        """primary robot meshes, object mesh, and the object-anchored partner verts."""
        data.qpos[:] = qp[f]; data.qvel[:] = 0.0; mujoco.mj_forward(model, data)
        robot = robot_world_meshes(model, data)
        ov, of, orgb = object_world_mesh(model, data)
        R_cem, t_cem = object_body_pose(model, data)
        m = args.primary_trim + f                         # mocap frame
        R_moc, t_moc = obj_moc[m][:3, :3], obj_moc[m][:3, 3]
        # map partner verts: cem_obj_pose @ mocap_obj_pose^-1 @ v
        v = ((ph_verts[m] - t_moc) @ R_moc) @ R_cem.T + t_cem
        return robot, (ov, of, orgb), v

    # camera (same convention as viz_mixed_robot_smplx): primary faces the camera
    yfov = np.pi / 4.0
    facing = mean_facing_yaw(qp)
    az, el = np.radians(facing + args.cam_az_offset), np.radians(args.cam_el)
    cdir = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    print(f"[cam] primary facing yaw={facing:.1f}deg -> az={np.degrees(az):.1f}deg el={args.cam_el:.1f}deg")

    # Emit camera_align.json so the companion smplx / robot_only renders reproduce this
    # viewpoint. The cem->mocap direction mapping goes through the shared object frame at
    # a mid-window frame: d_mocap = R_moc @ R_cem^T @ d_cem.
    f_mid = (lo + hi) // 2
    data.qpos[:] = qp[f_mid]; data.qvel[:] = 0.0; mujoco.mj_forward(model, data)
    R_cem_mid, _ = object_body_pose(model, data)
    R_moc_mid = obj_moc[args.primary_trim + f_mid][:3, :3]
    cdir_mocap = (R_moc_mid @ R_cem_mid.T @ cdir).tolist()
    (OUT / "camera_align.json").write_text(json.dumps({
        "cam_dir_mocap": cdir_mocap,
        "yfov_deg": float(np.degrees(yfov)),
        "dist_mult": 1.15,
        "source": "viz_mixed_obj_anchor.py",
        "cem_az_deg": float(np.degrees(az)),
        "cem_el_deg": float(args.cam_el),
    }, indent=2))

    sample = list(range(lo, hi, max(1, (hi - lo) // 16)))
    floor_z = np.inf
    for f in sample:
        robot, (ov, _, _), _ = frame_data(f)
        allv = np.concatenate([g[0] for g in robot] + [ov], 0)
        floor_z = min(floor_z, float(allv[:, 2].min()))
    floor_z = float(floor_z)

    def anchored_human(f, scale):
        _, _, v = frame_data(f)
        v = v.copy()
        v[:, 2] += floor_z - v[:, 2].min()                # drop feet to the physical floor
        if scale != 1.0:
            foot = np.array([v[:, 0].mean(), v[:, 1].mean(), floor_z])
            v = foot + (v - foot) * scale
        return v

    radius = 0.0
    for f in sample:
        robot, (ov, _, _), _ = frame_data(f)
        rall = np.concatenate([g[0] for g in robot] + [ov, anchored_human(f, args.human_scale)], 0)
        radius = max(radius, np.linalg.norm(np.ptp(rall, axis=0)) / 2.0)
    dist = radius / np.tan(yfov / 2.0) * 1.15
    floor_trimesh = make_checker_floor(floor_z, extent=10.0, repeats=12, up="z")  # noqa: F841 (white-only)

    frame_fs = list(range(lo, hi))
    if PREVIEW_N > 0:
        idx = np.linspace(0, len(frame_fs) - 1, PREVIEW_N).round().astype(int)
        frame_fs = [frame_fs[i] for i in idx]

    renderer = pyrender.OffscreenRenderer(RES, RES)
    mat = lambda c, r=0.55: pyrender.MetallicRoughnessMaterial(
        baseColorFactor=c, metallicFactor=0.0, roughnessFactor=r, doubleSided=True)
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)

    frame_dir = OUT / "mixed_frames_white"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frame_fs):
        robot, (ov, of, _orgb), _ = frame_data(f)
        hv = anchored_human(f, args.human_scale)
        rmean = np.concatenate([g[0] for g in robot], 0).mean(0)
        target = (rmean + hv.mean(0) + ov.mean(0)) / 3.0
        scene_pr = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.4, 0.4, 0.42])
        for rv, rf, rgb in robot:
            scene_pr.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(rv, rf, process=False),
                                                    material=mat(list(rgb) + [1.0]), smooth=False))
        scene_pr.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(hv, smplx_faces, process=False),
                                                material=mat(P1_COLOR), smooth=True))
        scene_pr.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(ov, of, process=False),
                                                material=mat(BOX_COLOR), smooth=False))
        scene_pr.add(camera, pose=look_at(target + cdir * dist, target, [0, 0, 1]))
        for light, pose_l in soft_phong_lights(look_at, target, [0, 0, 1], floor_z, up="z"):
            scene_pr.add(light, pose=pose_l)
        color, depth = renderer.render(scene_pr, flags=0)
        rgb = color[:, :, :3].copy(); rgb[depth == 0] = 255
        imageio.imwrite(frame_dir / f"frame_{i:04d}.png", rgb)
    renderer.delete()
    print(f"[ok] white: primary {args.primary}[{lo},{hi}) {len(frame_fs)} frames -> {frame_dir}")


if __name__ == "__main__":
    main()
