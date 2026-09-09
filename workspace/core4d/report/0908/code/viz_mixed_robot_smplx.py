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

import numpy as np
import trimesh
import pyrender
import mujoco
import imageio.v2 as imageio
from scipy.spatial.transform import Rotation as Rot

RAW = Path("/mnt/ali-sh-1/usr/xiayibo/xyb_data_tidal_alsh/other-datasets/mocap_data/CORE4D/CORE4D_Real")
SEQ = RAW / "human_object_motions" / "20231011" / "037"
SMPLX_NEUTRAL = Path("/mnt/ali-sh-1/usr/xiayibo/xyb_data_tidal_alsh/other-datasets/"
                     "mocap_data/human_model_files/smplx/SMPLX_NEUTRAL.npz")

REPO = Path(__file__).resolve().parents[5]
CEM_DIR = REPO / "workspace/core4d/results/E170/s6_downstream/cem/full"
HUM = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
SCENE_P2 = HUM / "dcv3_omnirt_v1_ref_fk_box021_20231011_037_p2" / "scene_act_E170_lowerbody_physics.xml"
SCENE_P1 = HUM / "dcv3_omnirt_v1_ref_fk_box021_20231011_037_p1" / "scene_act_E170_lowerbody_physics.xml"

OUT = (Path(__file__).resolve().parents[1] / "paper_results" / "viz"
       / "box021_20231011_037_p2")
FRAME_DIR = OUT / "mixed_frames"

RES = 1080
FPS = 20
WINDOW = None

# p1 cem[t] <-> mocap[109+t]; p1 cem[t] <-> p2 cem[15+t]  (E170 paired trims / re-anchor)
P1_TRIM_START = 109
P1_TO_P2 = 15

ROBOT_COLOR = [0.82, 0.84, 0.88, 1.0]   # light gray (robot = p2)
P1_COLOR = [0.30, 0.55, 0.85, 1.0]      # blue (human = p1)
OBJ_COLOR = [0.55, 0.60, 0.68, 1.0]     # slate
R_ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)


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
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    pose1 = np.load(SEQ / "person1_poses.npz", allow_pickle=True)["arr_0"].item()
    p1_verts, p1_joints = pose1["vertices"], pose1["joints"]
    smplx_faces = np.load(SMPLX_NEUTRAL, allow_pickle=True)["f"].astype(np.int64)

    q2 = np.load(CEM_DIR / "E170_box021_20231011_037_p2_PRG.npz")["qpos"][:, 0, :]
    q1 = np.load(CEM_DIR / "E170_box021_20231011_037_p1_PRG.npz")["qpos"][:, 0, :]

    m2 = mujoco.MjModel.from_xml_path(str(SCENE_P2)); d2 = mujoco.MjData(m2)
    m1 = mujoco.MjModel.from_xml_path(str(SCENE_P1)); d1 = mujoco.MjData(m1)
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
        pel_moc = p1_joints[r, 0]
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
    mat = lambda c: pyrender.MetallicRoughnessMaterial(
        baseColorFactor=c, metallicFactor=0.0, roughnessFactor=0.6)
    lights = [(np.array([0.6, 0.9, 1.0]), 5.5),
              (np.array([-0.9, -0.2, 0.3]), 1.0),
              (np.array([0.1, -1.2, 0.5]), 1.3)]

    def frame_meshes(f):
        d2.qpos[:] = q2[f]; d2.qvel[:] = 0.0; mujoco.mj_forward(m2, d2)
        robot = robot_world_meshes(m2, d2)                 # list of (v,f,rgb) by material
        ov, of, orgb = object_world_mesh(m2, d2)
        hv, _ = human_cem(f)
        return robot, (ov, of, orgb), hv

    sample = list(range(lo, hi, max(1, (hi - lo) // 16)))
    radius = 0.0
    for f in sample:
        robot, (ov, _, _), hv = frame_meshes(f)
        rall = np.concatenate([g[0] for g in robot] + [ov, hv], 0)
        radius = max(radius, np.linalg.norm(np.ptp(rall, axis=0)) / 2.0)
    dist = radius / np.tan(yfov / 2.0) * 1.15

    frames = []
    for f in range(lo, hi):
        robot, (ov, of, orgb), hv = frame_meshes(f)
        rmean = np.concatenate([g[0] for g in robot], 0).mean(0)
        target = (rmean + hv.mean(0) + ov.mean(0)) / 3.0
        scene = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.15, 0.15, 0.15])
        for rv, rf, rgb in robot:                          # G1 silver links + black joints
            scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(rv, rf, process=False),
                                                 material=mat(list(rgb) + [1.0]), smooth=False))
        scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(hv, smplx_faces, process=False),
                                             material=mat(P1_COLOR), smooth=True))
        scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(ov, of, process=False),
                                             material=mat(list(orgb) + [1.0]), smooth=False))
        cam_pose = look_at(target + cdir * dist, target, [0, 0, 1])
        scene.add(camera, pose=cam_pose)
        for dvec, inten in lights:
            scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=inten),
                      pose=look_at(target + dvec, target, [0, 0, 1]))
        color, depth = renderer.render(scene)
        rgb = color[:, :, :3].copy()
        rgb[depth == 0] = 255                 # pure white background (no-geometry pixels)
        frames.append(rgb)
        imageio.imwrite(FRAME_DIR / f"frame_{f - lo:04d}.png", rgb)
    renderer.delete()

    mp4 = OUT / "mixed_robot_smplx.mp4"
    imageio.mimsave(mp4, frames, fps=FPS, quality=9)
    print(f"[ok] cem frames p2[{lo},{hi})  {len(frames)} frames -> {FRAME_DIR}")
    print(f"[ok] video -> {mp4}")


if __name__ == "__main__":
    main()
