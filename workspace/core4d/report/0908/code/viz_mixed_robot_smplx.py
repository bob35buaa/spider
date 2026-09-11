"""Mixed render: p2 as retargeted G1 robot + p1 as SMPLX human + shared object.

Rendered in the CEM (robot scene_act) frame -- the frame where p2's robot and the
object live natively and where p1's robot is already well co-registered with p2
(the E170 paired re-anchor, ~2.6 deg). We keep p2's robot + object as-is and swap
p1's robot for p1's ground-truth SMPLX mesh, brought into the CEM frame by
anchoring it to p1's own robot pelvis each frame (position) + a global mocap->CEM
rotation. All bodies are real size (G1 ~1.3 m, human ~1.7 m, real box).

Single 3/4 follow view (Z-up), pure white background, per-frame PNGs + mp4.

Run:
    PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa .venv/bin/python \
        workspace/core4d/report/0908/code/viz_mixed_robot_smplx.py
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse
import sys

import numpy as np

if not hasattr(np, "infty"):        # pyrender uses np.infty, removed in NumPy 2.0
    np.infty = np.inf

import trimesh
import pyrender
import mujoco
import imageio.v2 as imageio
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_style import (BOX_COLOR, add_common_args, default_out,  # noqa: E402
                          make_checker_floor, parse_case, seq_dir, soft_phong_lights)
from smplx_min import SMPLXModel  # noqa: E402

REPO = Path(__file__).resolve().parents[5]
DEFAULT_PROCESSED_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
_SMPLX_KEYS = ["global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "transl"]

# Reassigned from CLI in main(); module-level mocap_to_cem_rot / nested human_cem read these.
P1_TRIM_START = 109     # p1 cem[t] <-> mocap[P1_TRIM_START + t]
P1_TO_P2 = 15           # p1 cem[t] <-> p2 cem[P1_TO_P2 + t]  (paired re-anchor offset)

P1_COLOR = [0.30, 0.55, 0.85, 1.0]      # blue (human = p1)
OBJ_COLOR = BOX_COLOR                   # slate blue (object)
R_ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)


def parse_args():
    p = argparse.ArgumentParser(
        description="Mixed render: p2 robot + p1 SMPLX human + shared object.")
    add_common_args(p)      # --case --data-root --smplx-model --out --res --fps --preview
    p.add_argument("--primary", default="p2", help="robot person driving the object (default: p2)")
    p.add_argument("--partner", default="p1", help="SMPLX human person (default: p1)")
    p.add_argument("--exp", default="E170", help="retarget experiment id (default: E170)")
    p.add_argument("--cem-dir", default=None, help="CEM results dir (default derived from --exp)")
    p.add_argument("--processed-root", default=str(DEFAULT_PROCESSED_ROOT),
                   help="processed humanoid_object root holding the scene dirs")
    p.add_argument("--scene-prefix", default="dcv3_omnirt_v1_ref_fk",
                   help="scene dir name prefix before <case>_<person>")
    p.add_argument("--window", default=None, help="cem frame window 'lo,hi' (default: auto)")
    p.add_argument("--baked-shape", action="store_true",
                   help="use partner's baked betas instead of the neutral betas=0 shape")
    p.add_argument("--p1-trim-start", type=int, default=109,
                   help="partner mocap<->cem trim offset (paired-export specific; default: 109)")
    p.add_argument("--p1-to-p2", type=int, default=15,
                   help="partner cem <-> primary cem offset (paired-export specific; default: 15)")
    return p.parse_args()


def look_at(eye, target, up):
    eye, target, up = map(lambda a: np.asarray(a, float), (eye, target, up))
    f = target - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = s, u, -f, eye
    return M


def mocap_to_cem_rot(p1_q, p1_pelvis_cem, p1_joints):
    """Global rotation mapping mocap directions -> CEM directions, from person1's
    pelvis path (walked path is non-degenerate in yaw). Returns R_inv (mocap->cem)."""
    pc = p1_pelvis_cem                                   # (N,3) cem
    pm = p1_joints[P1_TRIM_START:P1_TRIM_START + len(pc), 0, :]   # (N,3) mocap
    A2 = (R_ZUP_TO_YUP @ pc.T).T[:, [0, 2]]              # up-corrected cem, horiz (x,z)
    B2 = pm[:, [0, 2]]
    Ac, Bc = A2 - A2.mean(0), B2 - B2.mean(0)
    U, _, Vt = np.linalg.svd(Bc.T @ Ac / len(A2))
    d = np.sign(np.linalg.det(U @ Vt))
    R2 = U @ np.diag([1, d]) @ Vt
    th = np.arctan2(R2[1, 0], R2[0, 0])
    cy, sy = np.cos(th), np.sin(th)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    R = Ry @ R_ZUP_TO_YUP                                # cem -> mocap
    return R.T, np.degrees(th)                           # mocap -> cem


SILVER = (0.7, 0.7, 0.7)


def _geom_rgb(model, gid):
    """The rendered base color of a geom: its material rgba if any, else geom_rgba.
    Bright-green fingertip/contact marker geoms (geom_rgba ~ (0,1,0), no material)
    are remapped to silver so the robot hands look like normal links."""
    mid = int(model.geom_matid[gid])
    rgba = model.mat_rgba[mid] if mid >= 0 else model.geom_rgba[gid]
    rgb = tuple(np.round(rgba[:3], 3))
    if mid < 0 and rgb[1] > 0.8 and rgb[0] < 0.3 and rgb[2] < 0.3:
        return SILVER
    return rgb


def robot_world_meshes(model, data):
    """Robot visual mesh geoms in world (cem) frame, grouped by material colour so
    the G1 keeps its real silver-links / black-joints look (matches robot_dual);
    excludes the object. Returns list of (verts, faces, rgb)."""
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    groups: dict = {}
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if int(model.geom_bodyid[gid]) == obj_body:
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


def object_world_mesh(model, data):
    """(verts, faces, rgb) of the object visual mesh geom in world (cem) frame."""
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) == obj_body and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[gid]
            va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
            v = model.mesh_vert[va:va + vn].reshape(-1, 3)
            f = model.mesh_face[fa:fa + fn].reshape(-1, 3)
            vw = v @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
            return vw, f, _geom_rgb(model, gid)
    raise RuntimeError("object visual mesh not found")


def main():
    global P1_TRIM_START, P1_TO_P2
    args = parse_args()
    obj, date, seq, category = parse_case(args.case)
    SEQ = seq_dir(args.data_root, date, seq)
    SMPLX_NEUTRAL = Path(args.smplx_model)
    CEM_DIR = Path(args.cem_dir) if args.cem_dir else \
        REPO / f"workspace/core4d/results/{args.exp}/s6_downstream/cem/full"
    scene_primary = (Path(args.processed_root) / f"{args.scene_prefix}_{args.case}_{args.primary}"
                     / f"scene_act_{args.exp}_lowerbody_physics.xml")
    scene_partner = (Path(args.processed_root) / f"{args.scene_prefix}_{args.case}_{args.partner}"
                     / f"scene_act_{args.exp}_lowerbody_physics.xml")
    OUT = Path(args.out) if args.out else default_out(args.case, args.primary)
    RES, FPS, PREVIEW_N = args.res, args.fps, args.preview
    DEFAULT_SHAPE = not args.baked_shape
    WINDOW = tuple(int(x) for x in args.window.split(",")) if args.window else None
    P1_TRIM_START, P1_TO_P2 = args.p1_trim_start, args.p1_to_p2
    partner_file = SEQ / f"person{args.partner[1:]}_poses.npz"

    OUT.mkdir(parents=True, exist_ok=True)
    pose1 = np.load(partner_file, allow_pickle=True)["arr_0"].item()
    p1_verts, p1_joints = pose1["vertices"], pose1["joints"]
    smplx_faces = np.load(SMPLX_NEUTRAL, allow_pickle=True)["f"].astype(np.int64)
    # Neutral-shape re-pose: swap baked verts for betas=0, and anchor by the
    # matching default-shape pelvis (joint 0). p1_joints stays as-is for the
    # betas-independent mocap->cem yaw fit.
    p1_pel_moc = p1_joints[:, 0]
    if DEFAULT_SHAPE:
        model = SMPLXModel(str(SMPLX_NEUTRAL))
        n = len(pose1["global_orient"])
        verts_pel = [model.forward({k: pose1[k][t] for k in _SMPLX_KEYS}, return_joints=True)
                     for t in range(n)]
        p1_verts = np.stack([vp[0] for vp in verts_pel])
        p1_pel_moc = np.stack([vp[1][0] for vp in verts_pel])
        print("[shape] re-posed partner at neutral betas=0")

    q2 = np.load(CEM_DIR / f"{args.exp}_{args.case}_{args.primary}_PRG.npz")["qpos"][:, 0, :]
    q1 = np.load(CEM_DIR / f"{args.exp}_{args.case}_{args.partner}_PRG.npz")["qpos"][:, 0, :]

    m2 = mujoco.MjModel.from_xml_path(str(scene_primary)); d2 = mujoco.MjData(m2)
    m1 = mujoco.MjModel.from_xml_path(str(scene_partner)); d1 = mujoco.MjData(m1)
    pel1 = mujoco.mj_name2id(m1, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    pel2 = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    # person1 robot pelvis path (cem) for the mocap->cem rotation
    p1_pel_cem = np.empty((len(q1), 3))
    for i, q in enumerate(q1):
        d1.qpos[:] = q; d1.qvel[:] = 0.0; mujoco.mj_forward(m1, d1); p1_pel_cem[i] = d1.xpos[pel1]
    Rinv, yaw = mocap_to_cem_rot(q1, p1_pel_cem, p1_joints)
    print(f"mocap->cem rotation yaw={yaw:.1f}deg")

    def human_cem(f):
        """person1 SMPLX vertices mapped into the CEM frame at p2 cem index f."""
        t1 = f - P1_TO_P2                     # p1 cem index
        r = P1_TRIM_START + t1                # mocap index
        pel_cem = p1_pel_cem[t1]
        pel_moc = p1_pel_moc[r]
        return (Rinv @ (p1_verts[r] - pel_moc).T).T + pel_cem, r

    lo = P1_TO_P2
    hi = min(len(q2), P1_TO_P2 + (len(q1) - 0))          # p1 cem valid: t1 in [0,len(q1))
    hi = min(hi, len(q2))
    if WINDOW is not None:
        lo, hi = WINDOW

    # camera: cem Z-up, 3/4 elevated view (azimuth 135, elevation +12), follow centroid
    yfov = np.pi / 4.0
    az, el = np.radians(135.0), np.radians(12.0)
    cdir = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
    renderer = pyrender.OffscreenRenderer(RES, RES)
    # CARI4D-style soft Phong (low metallic, mid-high roughness).
    mat = lambda c, r=0.55: pyrender.MetallicRoughnessMaterial(
        baseColorFactor=c, metallicFactor=0.0, roughnessFactor=r, doubleSided=True)
    shadow_flag = pyrender.constants.RenderFlags.SHADOWS_DIRECTIONAL

    def frame_meshes(f):
        d2.qpos[:] = q2[f]; d2.qvel[:] = 0.0; mujoco.mj_forward(m2, d2)
        robot = robot_world_meshes(m2, d2)                 # list of (v,f,rgb) by material
        ov, of, orgb = object_world_mesh(m2, d2)
        hv, _ = human_cem(f)
        return robot, (ov, of, orgb), hv

    sample = list(range(lo, hi, max(1, (hi - lo) // 16)))
    radius = 0.0
    floor_z = np.inf
    for f in sample:
        robot, (ov, _, _), hv = frame_meshes(f)
        rall = np.concatenate([g[0] for g in robot] + [ov, hv], 0)
        radius = max(radius, np.linalg.norm(np.ptp(rall, axis=0)) / 2.0)
        floor_z = min(floor_z, rall[:, 2].min())
    dist = radius / np.tan(yfov / 2.0) * 1.15
    floor_trimesh = make_checker_floor(float(floor_z), extent=10.0, repeats=12, up="z")

    frame_fs = list(range(lo, hi))
    if PREVIEW_N > 0:
        idx = np.linspace(0, len(frame_fs) - 1, PREVIEW_N).round().astype(int)
        frame_fs = [frame_fs[i] for i in idx]
        print(f"[preview] rendering {PREVIEW_N} sampled frames")

    def render_version(mode: str):
        assert mode in ("bg", "white")
        with_bg = mode == "bg"
        frame_dir = OUT / f"mixed_frames_{mode}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        bg_col = [0.90, 0.91, 0.93, 1.0] if with_bg else [1.0, 1.0, 1.0, 1.0]
        frames = []
        for i, f in enumerate(frame_fs):
            robot, (ov, of, _orgb), hv = frame_meshes(f)
            rmean = np.concatenate([g[0] for g in robot], 0).mean(0)
            target = (rmean + hv.mean(0) + ov.mean(0)) / 3.0
            scene = pyrender.Scene(bg_color=bg_col, ambient_light=[0.4, 0.4, 0.42])
            if with_bg:
                scene.add(pyrender.Mesh.from_trimesh(floor_trimesh))
            for rv, rf, rgb in robot:                      # G1 silver links + black joints
                scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(rv, rf, process=False),
                                                     material=mat(list(rgb) + [1.0]), smooth=False))
            scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(hv, smplx_faces, process=False),
                                                 material=mat(P1_COLOR), smooth=True))
            scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(ov, of, process=False),
                                                 material=mat(BOX_COLOR), smooth=False))
            scene.add(camera, pose=look_at(target + cdir * dist, target, [0, 0, 1]))
            for light, pose in soft_phong_lights(look_at, target, [0, 0, 1], float(floor_z), up="z"):
                scene.add(light, pose=pose)
            color, depth = renderer.render(scene, flags=shadow_flag if with_bg else 0)
            rgb = color[:, :, :3].copy()
            if not with_bg:
                rgb[depth == 0] = 255             # pure white background
            frames.append(rgb)
            imageio.imwrite(frame_dir / f"frame_{i:04d}.png", rgb)
        return frames, frame_dir

    for mode in ("bg", "white"):
        frames, frame_dir = render_version(mode)
        if PREVIEW_N == 0:
            mp4 = OUT / f"mixed_robot_smplx_{mode}.mp4"
            imageio.mimsave(mp4, frames, fps=FPS, quality=9)
            print(f"[ok] {mode}: cem p2[{lo},{hi}) {len(frames)} frames -> {frame_dir}  video -> {mp4}")
        else:
            print(f"[ok] {mode}: {len(frames)} preview frames -> {frame_dir}")
    renderer.delete()


if __name__ == "__main__":
    main()
