"""SMPLX reference render for CORE4D box021_20231011_037 (both persons + object).

Renders the ground-truth human motion: person1 + person2 SMPLX meshes and the
posed object mesh, single fixed 3/4 view, pure white background, per-frame PNGs
+ mp4. Companion to viz_dual_robot_clean.py (the retargeted-robot version).

Vertices are pre-baked in the CORE4D pose npz (no smplx forward needed); only the
SMPLX topology (faces) is loaded, from the GMR-hosted SMPLX_NEUTRAL.npz. World is
Y-up; the pair carries the box along +Z.

Run in an isolated env (keeps the shared venv clean):
    PYOPENGL_PLATFORM=osmesa uv run --no-sync \
        --with pyrender --with trimesh --with imageio --with numpy --with pyopengl \
        python workspace/core4d/report/0908/code/viz_smplx_reference.py
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import shutil

import numpy as np
import trimesh
import pyrender
import imageio.v2 as imageio

RAW = Path("/mnt/ali-sh-1/usr/xiayibo/xyb_data_tidal_alsh/other-datasets/mocap_data/CORE4D/CORE4D_Real")
SEQ = RAW / "human_object_motions" / "20231011" / "037"
OBJ_MESH = RAW / "object_models" / "box" / "box021_m.obj"
SMPLX_NEUTRAL = Path("/mnt/ali-sh-1/usr/xiayibo/xyb_data_tidal_alsh/other-datasets/"
                     "mocap_data/human_model_files/smplx/SMPLX_NEUTRAL.npz")

OUT = (Path(__file__).resolve().parents[1] / "paper_results" / "viz"
       / "box021_20231011_037_p2")
FRAME_DIR = OUT / "smplx_frames"
# Render to local disk first (the networked workspace stalls on per-frame writes),
# then bulk-copy to FRAME_DIR / OUT at the end.
STAGE = Path("/tmp/smplx_stage_box021")
FRAME_STAGE = STAGE / "smplx_frames"

RES = 1080
FPS = 20
# Robot render covers raw frames [109, 217) (p1 trim window). Keep None for the
# full sequence; set to (109, 217) to line up 1:1 with the robot clip.
WINDOW = None

# CORE4D-Instructions official palette (dataset_utils/pyt3d_wrapper.py)
P1_COLOR = [135 / 255, 235 / 255, 156 / 255, 1.0]   # green  (person1)
P2_COLOR = [117 / 255, 157 / 255, 231 / 255, 1.0]   # blue   (person2)
OBJ_COLOR = [224 / 255, 193 / 255, 240 / 255, 1.0]  # purple (object)


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
    FRAME_STAGE.mkdir(parents=True, exist_ok=True)
    p1 = np.load(SEQ / "person1_poses.npz", allow_pickle=True)["arr_0"].item()["vertices"]
    p2 = np.load(SEQ / "person2_poses.npz", allow_pickle=True)["arr_0"].item()["vertices"]
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

    # Ground plane at the feet level (a large matte-gray quad) so the pair stands
    # on a floor instead of floating in a white void (matches the reference).
    floor_y = float(min(min(p1[t][:, 1].min(), p2[t][:, 1].min()) for t in sub))
    G = 50.0
    floor_v = np.array([[-G, floor_y, -G], [G, floor_y, -G], [G, floor_y, G], [-G, floor_y, G]])
    floor_f = np.array([[0, 2, 1], [0, 3, 2]])          # +Y up-facing normal

    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
    renderer = pyrender.OffscreenRenderer(RES, RES)
    mat = lambda c, r=0.6: pyrender.MetallicRoughnessMaterial(
        baseColorFactor=c, metallicFactor=0.0, roughnessFactor=r, doubleSided=True)
    floor_mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.72, 0.73, 0.76, 1.0], metallicFactor=0.0, roughnessFactor=1.0,
        doubleSided=True)

    frames = []
    for t in range(lo, hi):
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.35, 0.35, 0.35])
        scene.add(pyrender.Mesh.from_trimesh(
            trimesh.Trimesh(floor_v, floor_f, process=False), material=floor_mat))
        scene.add(pyrender.Mesh.from_trimesh(
            trimesh.Trimesh(p1[t], faces, process=False), material=mat(P1_COLOR), smooth=True))
        scene.add(pyrender.Mesh.from_trimesh(
            trimesh.Trimesh(p2[t], faces, process=False), material=mat(P2_COLOR), smooth=True))
        scene.add(pyrender.Mesh.from_trimesh(
            trimesh.Trimesh(obj_verts(t), obj_f, process=False), material=mat(OBJ_COLOR), smooth=True))
        target = frame_centroid(t)
        cam_pose = look_at(target + direction * dist, target, [0, 1, 0])
        scene.add(camera, pose=cam_pose)
        # CORE4D/holosoma lighting: strong point light above + directional fill.
        pl_pose = np.eye(4); pl_pose[:3, 3] = [target[0], floor_y + 4.0, target[2]]
        scene.add(pyrender.PointLight(color=[1, 1, 1], intensity=30.0), pose=pl_pose)
        scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.5),
                  pose=look_at(target + np.array([0.6, 1.0, 0.5]), target, [0, 1, 0]))
        color, _ = renderer.render(scene)        # Scene bg_color is white
        frames.append(color[:, :, :3].copy())
        imageio.imwrite(FRAME_STAGE / f"frame_{t - lo:04d}.png", color[:, :, :3])
    renderer.delete()

    mp4_stage = STAGE / "smplx_reference.mp4"
    imageio.mimsave(mp4_stage, frames, fps=FPS, quality=9)

    # Bulk-copy staged outputs to the (slow) network workspace.
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    for png in sorted(FRAME_STAGE.glob("frame_*.png")):
        shutil.copy2(png, FRAME_DIR / png.name)
    shutil.copy2(mp4_stage, OUT / "smplx_reference.mp4")
    print(f"[ok] window [{lo},{hi})  {len(frames)} frames -> {FRAME_DIR}")
    print(f"[ok] video -> {OUT / 'smplx_reference.mp4'}")


if __name__ == "__main__":
    main()
