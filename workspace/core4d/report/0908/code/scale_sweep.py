"""One-frame human-scale sweep for the mixed render, so a scale can be chosen before
committing to the full clip. Reproduces viz_mixed_robot_smplx.main()'s setup (neutral
re-pose anchor, mocap->cem rotation, floor grounding, camera) EXACTLY, computes the
camera once (fixed across scales for clean comparison), then renders a single frame F
at each --scales value to  <out>/scale_sweep/frame<F>_s<scale>.png .

Usage:
    PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa .venv/bin/python \
        workspace/core4d/report/0908/code/scale_sweep.py \
        --case box001_20231023_108 --exp E198 --cem-dir <dir> \
        --scene-prefix dcv3_omnirt_v1_ref_fk --p1-trim-start 19 --p1-to-p2 -12 \
        --frame 50 --scales 1.0,0.9,0.85,0.8,0.75,0.7 --out <OUT>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse

import numpy as np

if not hasattr(np, "infty"):
    np.infty = np.inf

import trimesh
import pyrender
import mujoco
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_style import BOX_COLOR, make_checker_floor, seq_dir, soft_phong_lights  # noqa: E402
from smplx_min import SMPLXModel  # noqa: E402
import viz_mixed_robot_smplx as M  # noqa: E402

REPO = Path(__file__).resolve().parents[5]
DEFAULT_PROCESSED_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
DEFAULT_DATA_ROOT = ("/mnt/ali-sh-1/usr/xiayibo/xyb_data_tidal_alsh/other-datasets/"
                     "mocap_data/CORE4D/CORE4D_Real")
DEFAULT_SMPLX = ("/mnt/ali-sh-1/usr/xiayibo/xyb_data_tidal_alsh/other-datasets/"
                 "mocap_data/human_model_files/smplx/SMPLX_NEUTRAL.npz")


def parse_args():
    p = argparse.ArgumentParser(description="One-frame human-scale sweep for mixed render.")
    p.add_argument("--case", required=True)
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--smplx-model", default=DEFAULT_SMPLX)
    p.add_argument("--primary", default="p2")
    p.add_argument("--partner", default="p1")
    p.add_argument("--exp", default="E198")
    p.add_argument("--cem-dir", required=True)
    p.add_argument("--cem-suffix", default="G1A2")
    p.add_argument("--processed-root", default=str(DEFAULT_PROCESSED_ROOT))
    p.add_argument("--scene-prefix", default="dcv3_omnirt_v1_ref_fk")
    p.add_argument("--p1-trim-start", type=int, default=19)
    p.add_argument("--p1-to-p2", type=int, default=-12)
    p.add_argument("--cam-az-offset", type=float, default=45.0)
    p.add_argument("--cam-el", type=float, default=12.0)
    p.add_argument("--frame", type=int, default=50, help="primary(p2) cem frame to render")
    p.add_argument("--scales", default="1.0,0.9,0.85,0.8,0.75,0.7")
    p.add_argument("--res", type=int, default=1080)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    M.P1_TRIM_START, M.P1_TO_P2 = args.p1_trim_start, args.p1_to_p2
    OUT = Path(args.out)
    sweep_dir = OUT / "scale_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    SEQ = seq_dir(args.data_root, *args.case.split("_")[1:3])
    partner_file = SEQ / f"person{args.partner[1:]}_poses.npz"

    pose1 = np.load(partner_file, allow_pickle=True)["arr_0"].item()
    p1_joints = pose1["joints"]
    smplx_faces = np.load(args.smplx_model, allow_pickle=True)["f"].astype(np.int64)
    # neutral re-pose (matches delivered default shape)
    model = SMPLXModel(str(args.smplx_model))
    n = len(pose1["global_orient"])
    verts_pel = [model.forward({k: pose1[k][t] for k in M._SMPLX_KEYS}, return_joints=True)
                 for t in range(n)]
    p1_verts = np.stack([vp[0] for vp in verts_pel])
    p1_pel_moc = np.stack([vp[1][0] for vp in verts_pel])
    print("[shape] re-posed partner at neutral betas=0")

    cem = Path(args.cem_dir)
    q2 = np.load(cem / f"{args.exp}_{args.case}_{args.primary}_{args.cem_suffix}.npz")["qpos"][:, 0, :]
    q1 = np.load(cem / f"{args.exp}_{args.case}_{args.partner}_{args.cem_suffix}.npz")["qpos"][:, 0, :]
    sp = Path(args.processed_root)
    scene_primary = sp / f"{args.scene_prefix}_{args.case}_{args.primary}" / f"scene_act_{args.exp}_lowerbody_physics.xml"
    scene_partner = sp / f"{args.scene_prefix}_{args.case}_{args.partner}" / f"scene_act_{args.exp}_lowerbody_physics.xml"
    m2 = mujoco.MjModel.from_xml_path(str(scene_primary)); d2 = mujoco.MjData(m2)
    m1 = mujoco.MjModel.from_xml_path(str(scene_partner)); d1 = mujoco.MjData(m1)
    pel1 = mujoco.mj_name2id(m1, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    p1_pel_cem = np.empty((len(q1), 3))
    for i, q in enumerate(q1):
        d1.qpos[:] = q; d1.qvel[:] = 0.0; mujoco.mj_forward(m1, d1); p1_pel_cem[i] = d1.xpos[pel1]
    Rinv, yaw = M.mocap_to_cem_rot(q1, p1_pel_cem, p1_joints)

    # floor_z from robot+object over a clip sample (identical to main())
    lo, hi = 0, len(q2)
    sample = list(range(lo, hi, max(1, (hi - lo) // 16)))
    floor_z = np.inf
    for f in sample:
        d2.qpos[:] = q2[f]; d2.qvel[:] = 0.0; mujoco.mj_forward(m2, d2)
        allv = np.concatenate([g[0] for g in M.robot_world_meshes(m2, d2)]
                              + [M.object_world_mesh(m2, d2)[0]], 0)
        floor_z = min(floor_z, allv[:, 2].min())
    floor_z = float(floor_z)

    def human_cem(f, scale):
        t1 = f - args.p1_to_p2
        r = args.p1_trim_start + t1
        pel_cem = p1_pel_cem[t1]; pel_moc = p1_pel_moc[r]
        v = (Rinv @ (p1_verts[r] - pel_moc).T).T + pel_cem
        v[:, 2] += floor_z - v[:, 2].min()
        if scale != 1.0:
            anchor = np.array([pel_cem[0], pel_cem[1], floor_z])
            v = anchor + (v - anchor) * scale
        return v

    # camera FIXED across scales (computed at scale 1.0) so only the human size changes
    yfov = np.pi / 4.0
    facing = M.mean_facing_yaw(q2)
    az, el = np.radians(facing + args.cam_az_offset), np.radians(args.cam_el)
    cdir = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    radius = 0.0
    for f in sample:
        d2.qpos[:] = q2[f]; d2.qvel[:] = 0.0; mujoco.mj_forward(m2, d2)
        robot = M.robot_world_meshes(m2, d2); ov = M.object_world_mesh(m2, d2)[0]
        rall = np.concatenate([g[0] for g in robot] + [ov, human_cem(f, 1.0)], 0)
        radius = max(radius, np.linalg.norm(np.ptp(rall, axis=0)) / 2.0)
    dist = radius / np.tan(yfov / 2.0) * 1.15

    renderer = pyrender.OffscreenRenderer(args.res, args.res)
    mat = lambda c, r=0.55: pyrender.MetallicRoughnessMaterial(
        baseColorFactor=c, metallicFactor=0.0, roughnessFactor=r, doubleSided=True)
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)

    F = args.frame
    d2.qpos[:] = q2[F]; d2.qvel[:] = 0.0; mujoco.mj_forward(m2, d2)
    robot = M.robot_world_meshes(m2, d2)
    ov, of, _ = M.object_world_mesh(m2, d2)
    rmean = np.concatenate([g[0] for g in robot], 0).mean(0)
    for s in [float(x) for x in args.scales.split(",")]:
        hv = human_cem(F, s)
        target = (rmean + hv.mean(0) + ov.mean(0)) / 3.0
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.4, 0.4, 0.42])
        for rv, rf, rgb in robot:
            scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(rv, rf, process=False),
                                                 material=mat(list(rgb) + [1.0]), smooth=False))
        scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(hv, smplx_faces, process=False),
                                             material=mat(M.P1_COLOR), smooth=True))
        scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(ov, of, process=False),
                                             material=mat(BOX_COLOR), smooth=False))
        scene.add(camera, pose=M.look_at(target + cdir * dist, target, [0, 0, 1]))
        for light, pose in soft_phong_lights(M.look_at, target, [0, 0, 1], floor_z, up="z"):
            scene.add(light, pose=pose)
        color, depth = renderer.render(scene, flags=0)
        rgb = color[:, :, :3].copy(); rgb[depth == 0] = 255
        outp = sweep_dir / f"frame{F:04d}_s{s:.2f}.png"
        imageio.imwrite(outp, rgb)
        print(f"[ok] scale={s:.2f} -> {outp}")
    renderer.delete()


if __name__ == "__main__":
    main()
