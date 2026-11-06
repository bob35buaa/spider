"""Render specific UPPER-BODY vs OBJECT penetration frames as still images.

Given a scene_act XML + qpos npz + frame index, render the G1 robot + its object
in the paper pyrender style, from a 3/4 view framed on the object. Two passes per
frame so the penetration is unambiguous:
  - "solid"  : opaque object (the clip is visible as the link surface disappearing
               into the box face)
  - "xray"   : semi-transparent object + the penetrating link(s) painted red (the
               embedded forearm/elbow shows through the box)

Run (headless):
    PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa .venv/bin/python \
        workspace/core4d/report/0908/code/render_upperbody_pen.py \
        --scene <scene.xml> --qpos <qpos.npz> --frames 72 --kind spider \
        --highlight left_elbow_link --out <dir> --tag box023_040_p2_spider
"""
from __future__ import annotations

import os
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse
import sys
from pathlib import Path

import numpy as np
if not hasattr(np, "infty"):
    np.infty = np.inf

import mujoco
import trimesh
import pyrender
import imageio.v2 as imageio
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_style import BOX_COLOR, make_checker_floor, soft_phong_lights  # noqa: E402
from viz_mixed_robot_smplx import look_at  # noqa: E402

SILVER = (0.7, 0.7, 0.7)
HILITE = [0.88, 0.12, 0.12, 1.0]       # red for the penetrating link


def _geom_rgb(model, gid):
    mid = int(model.geom_matid[gid])
    rgba = model.mat_rgba[mid] if mid >= 0 else model.geom_rgba[gid]
    rgb = tuple(np.round(rgba[:3], 3))
    if mid < 0 and rgb[1] > 0.8 and rgb[0] < 0.3 and rgb[2] < 0.3:
        return SILVER
    return rgb


def robot_geoms(model, data, highlight):
    """Per-geom robot visual meshes in world frame, tagged highlight if the geom's
    body name contains any highlight token."""
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    out = []
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if int(model.geom_bodyid[gid]) == obj_body:
            continue
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                  int(model.geom_bodyid[gid])) or ""
        mid = model.geom_dataid[gid]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        v = model.mesh_vert[va:va + vn].reshape(-1, 3)
        f = model.mesh_face[fa:fa + fn].reshape(-1, 3)
        vw = v @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        hit = any(tok in bname for tok in highlight)
        out.append((vw, f, _geom_rgb(model, gid), hit))
    return out


def object_mesh(model, data):
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) == obj_body and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[gid]
            va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
            v = model.mesh_vert[va:va + vn].reshape(-1, 3)
            f = model.mesh_face[fa:fa + fn].reshape(-1, 3)
            vw = v @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
            return vw, f
    raise RuntimeError("no object mesh")


def facing_yaw(qpos_row):
    quat = qpos_row[3:7][[1, 2, 3, 0]]
    fwd = Rot.from_quat(quat).apply([1.0, 0.0, 0.0])
    return float(np.degrees(np.arctan2(fwd[1], fwd[0])))


def load_qpos(path):
    q = np.load(path)["qpos"]
    return q[:, 0, :] if q.ndim == 3 else q


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--qpos", required=True)
    p.add_argument("--frames", required=True, help="comma-separated frame indices")
    p.add_argument("--highlight", default="elbow,wrist",
                   help="comma body-name tokens to paint red + aim the camera at")
    p.add_argument("--tag", required=True, help="filename prefix")
    p.add_argument("--out", required=True)
    p.add_argument("--res", type=int, default=1200)
    p.add_argument("--az", type=float, default=None)
    p.add_argument("--az-offset", type=float, default=35.0)
    p.add_argument("--el", type=float, default=14.0)
    p.add_argument("--zoom", type=float, default=1.0,
                   help="<1 zooms IN on the highlighted link (tighter crop)")
    return p.parse_args()


def main():
    a = parse_args()
    highlight = [t for t in a.highlight.split(",") if t]
    frames = [int(x) for x in a.frames.split(",")]
    qpos = load_qpos(a.qpos)
    model = mujoco.MjModel.from_xml_path(a.scene)
    data = mujoco.MjData(model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    yfov = np.pi / 4.0
    renderer = pyrender.OffscreenRenderer(a.res, a.res)
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
    shadow = pyrender.constants.RenderFlags.SHADOWS_DIRECTIONAL

    def mat(c, rough=0.55, alpha=None):
        c = list(c[:3]) + [alpha if alpha is not None else (c[3] if len(c) > 3 else 1.0)]
        return pyrender.MetallicRoughnessMaterial(
            baseColorFactor=c, metallicFactor=0.0, roughnessFactor=rough,
            alphaMode="BLEND" if (c[3] < 1.0) else "OPAQUE", doubleSided=True)

    for fr in frames:
        data.qpos[:] = qpos[fr]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        robot = robot_geoms(model, data, highlight)
        ov, of = object_mesh(model, data)

        # camera target: the highlighted link centroid (fall back to object center)
        hv = [g[0] for g in robot if g[3]]
        link_c = np.concatenate(hv, 0).mean(0) if hv else ov.mean(0)
        target = (link_c + ov.mean(0)) / 2.0
        allv = np.concatenate([g[0] for g in robot] + [ov], 0)
        floor_z = float(allv[:, 2].min())
        radius = float(np.linalg.norm(
            np.concatenate(hv + [ov], 0) - target, axis=1).max()) if hv else \
            float(np.linalg.norm(allv - target, axis=1).max())
        radius *= 1.6 * a.zoom
        dist = radius / np.tan(yfov / 2.0) * 1.1

        az = np.radians(a.az if a.az is not None else facing_yaw(qpos[fr]) + a.az_offset)
        el = np.radians(a.el)
        cdir = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        floor_trimesh = make_checker_floor(floor_z, extent=12.0, repeats=14, up="z")

        for kind in ("solid", "xray"):
            bg = [0.90, 0.91, 0.93, 1.0]
            scene_pr = pyrender.Scene(bg_color=bg, ambient_light=[0.4, 0.4, 0.42])
            scene_pr.add(pyrender.Mesh.from_trimesh(floor_trimesh))
            for vw, f, rgb, hit in robot:
                col = HILITE if (hit and kind == "xray") else (list(rgb) + [1.0])
                scene_pr.add(pyrender.Mesh.from_trimesh(
                    trimesh.Trimesh(vw, f, process=False),
                    material=mat(col), smooth=False))
            obj_alpha = 0.4 if kind == "xray" else 1.0
            scene_pr.add(pyrender.Mesh.from_trimesh(
                trimesh.Trimesh(ov, of, process=False),
                material=mat(BOX_COLOR, alpha=obj_alpha), smooth=False))
            scene_pr.add(camera, pose=look_at(target + cdir * dist, target, [0, 0, 1]))
            for light, pose in soft_phong_lights(look_at, target, [0, 0, 1], floor_z, up="z"):
                scene_pr.add(light, pose=pose)
            color, _ = renderer.render(scene_pr, flags=shadow)
            fn = out / f"{a.tag}_f{fr:03d}_{kind}.png"
            imageio.imwrite(fn, color[:, :, :3])
            print(f"[ok] {fn}")
    renderer.delete()


if __name__ == "__main__":
    main()
