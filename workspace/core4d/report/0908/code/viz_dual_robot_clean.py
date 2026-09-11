"""Clean two-person robot render for a CORE4D case (SPIDER-CEM, both persons).

Renders BOTH persons' retargeted G1 robots + the shared object in one MuJoCo
scene, single 3/4 view, pure white background, per-frame PNGs + mp4. No contact
overlay, no A/B comparison (SPIDER only) -- the "clean" deliverable.

The two persons are optimized independently (different trim windows, per-person
object trajectory). We re-anchor person1 into person2's frame:
  1. Recover the temporal offset between the two object trajectories.
  2. Fit ONE constant rigid transform T mapping person1's object onto person2's.
  3. Apply T to person1's floating base over the overlap window; drive person2
     and the shared object directly from person2's result.

Run (headless):
    MUJOCO_GL=osmesa .venv/bin/python \
        workspace/core4d/report/0908/code/viz_dual_robot_clean.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/convert"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_dual_data import generate_scene_dual_robot  # noqa: E402
from render_style import add_common_args, beautify_robot_scene_xml, default_out  # noqa: E402

DEFAULT_PROCESSED_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"

# Reassigned from CLI in main(); load_cem / render() read these module globals.
CASE = "box021_20231011_037"
PRIMARY = "p2"          # rendered as robot1 (drives the shared object)
PARTNER = "p1"          # re-anchored into primary's frame as robot2 (r2_)
EXP = "E170"
CEM_DIR = REPO / "workspace/core4d/results/E170/s6_downstream/cem/full"
H, W = 1080, 1080
FPS = 20
# 3/4 view; azimuth/elevation reused from the contact-overlay body cam.
CAM = {"azimuth": 135.0, "elevation": -14.0, "distance": 2.7, "lookat_z": 0.8}
PREVIEW_N = 0


def parse_args():
    p = argparse.ArgumentParser(description="Clean two-person robot render (both persons + object).")
    add_common_args(p)      # --case --data-root --smplx-model --out --res --fps --preview
    p.add_argument("--primary", default="p2", help="person driving the shared object (default: p2)")
    p.add_argument("--partner", default="p1", help="person re-anchored as robot2 (default: p1)")
    p.add_argument("--exp", default="E170", help="retarget experiment id (default: E170)")
    p.add_argument("--cem-dir", default=None,
                   help="CEM results dir (default: <repo>/workspace/core4d/results/<exp>/s6_downstream/cem/full)")
    p.add_argument("--processed-root", default=str(DEFAULT_PROCESSED_ROOT),
                   help="processed humanoid_object root holding the scene dirs")
    p.add_argument("--scene-prefix", default="dcv3_omnirt_v1_ref_fk",
                   help="scene dir name prefix before <case>_<person>")
    return p.parse_args()


def load_cem(person: str) -> np.ndarray:
    """(T, 42) qpos of the selected CEM rollout: base7 + joints29 + obj(trans3+eulZYX3)."""
    q = np.load(CEM_DIR / f"{EXP}_{CASE}_{person}_PRG.npz")["qpos"]
    return q[:, 0, :] if q.ndim == 3 else q


def obj_transforms(qpos: np.ndarray) -> np.ndarray:
    """(T,4,4) world transform of the object from qpos[36:39] trans + [39:42] euler ZYX."""
    T = np.tile(np.eye(4), (len(qpos), 1, 1))
    T[:, :3, 3] = qpos[:, 36:39]
    T[:, :3, :3] = Rot.from_euler("ZYX", qpos[:, 39:42]).as_matrix()
    return T


def align_partner(part: np.ndarray, prim: np.ndarray):
    """Recover (offset, T, window) mapping the partner (shorter) onto the primary.

    partner frame t corresponds to primary frame offset+t; T is the single rigid
    transform (partner-object -> primary-object) averaged over the overlap."""
    Tp, Tq = obj_transforms(part), obj_transforms(prim)
    n = len(Tp)
    best = None
    for o in range(0, len(Tq) - n + 1):
        Ti = Tq[o:o + n] @ np.linalg.inv(Tp)          # partner-obj -> primary-obj, per frame
        rots = Rot.from_matrix(Ti[:, :3, :3])
        ang = (rots * rots.mean().inv()).magnitude()   # rotational spread (rad)
        tv = Ti[:, :3, 3].std(axis=0).sum()
        score = np.rad2deg(ang.mean()) + tv * 57.3
        if best is None or score < best[0]:
            best = (score, o, Ti, np.rad2deg(ang.mean()), np.rad2deg(ang.std()), tv)
    _, offset, Ti, ang_mean, ang_std, tv = best
    Tmean = np.eye(4)
    Tmean[:3, :3] = Rot.from_matrix(Ti[:, :3, :3]).mean().as_matrix()
    Tmean[:3, 3] = Ti[:, :3, 3].mean(axis=0)
    return offset, Tmean, n, ang_mean, ang_std, tv


def apply_T_to_base(base7: np.ndarray, T: np.ndarray) -> np.ndarray:
    """base7 = [x,y,z, qw,qx,qy,qz]; return T @ base as the same 7-vector."""
    M = np.eye(4)
    M[:3, :3] = Rot.from_quat(base7[[4, 5, 6, 3]]).as_matrix()   # wxyz -> xyzw
    M[:3, 3] = base7[:3]
    M = T @ M
    out = np.empty(7)
    out[:3] = M[:3, 3]
    q = Rot.from_matrix(M[:3, :3]).as_quat()                     # xyzw
    out[3:] = q[[3, 0, 1, 2]]                                    # -> wxyz
    return out


def prep_scene(model) -> int:
    """Hide sites and object collision proxies; keep the (now-tiled) floor. Returns
    the floor geom id so the white-bg pass can hide it per-render."""
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    floor_gid = -1
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name == "floor":
            floor_gid = gid
        if int(model.geom_bodyid[gid]) == obj_body:
            model.geom_rgba[gid, 3] = 0.0 if "collision" in name else 1.0
    for sid in range(model.nsite):
        model.site_rgba[sid, 3] = 0.0
    return floor_gid


def build_dual_qpos(prim: np.ndarray, part: np.ndarray, offset: int, n: int, T: np.ndarray):
    """Assemble (n,78) dual qpos over the overlap window.

    R1[0:36]=primary, R2[36:72]=re-anchored partner, OBJ[72:78]=primary object."""
    dual = np.zeros((n, 78))
    for t in range(n):
        p = prim[offset + t]
        q = part[t]
        dual[t, 0:7] = p[0:7]                      # R1 base (primary)
        dual[t, 7:36] = p[7:36]                    # R1 joints
        dual[t, 36:43] = apply_T_to_base(q[0:7], T)  # R2 base (re-anchored partner)
        dual[t, 43:72] = q[7:36]                   # R2 joints
        dual[t, 72:78] = p[36:42]                  # shared object (primary)
    return dual


def render(dual_qpos: np.ndarray, model, floor_gid: int, mode: str) -> list[np.ndarray]:
    """Render the dual-robot clip.

    mode="bg":    tiled floor + gradient sky + soft shadow + reflection.
    mode="white": floor hidden, no shadow/reflection, background painted white.
    """
    assert mode in ("bg", "white")
    with_bg = mode == "bg"
    if floor_gid >= 0:
        model.geom_rgba[floor_gid, 3] = 1.0 if with_bg else 0.0

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation, cam.distance = CAM["azimuth"], CAM["elevation"], CAM["distance"]
    r1_pel = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    r2_pel = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r2_pelvis")
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")

    frames = []
    for q in dual_qpos:
        data.qpos[:] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        # frame both robots + object
        pts = np.stack([data.xpos[r1_pel], data.xpos[r2_pel], data.xpos[obj_body]])
        cam.lookat = np.array([pts[:, 0].mean(), pts[:, 1].mean(), CAM["lookat_z"]])
        renderer.update_scene(data, camera=cam)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1 if with_bg else 0
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1 if with_bg else 0
        rgb = renderer.render().copy()
        if not with_bg:
            # pure white background via segmentation mask
            renderer.enable_segmentation_rendering()
            seg = renderer.render()[:, :, 0]
            renderer.disable_segmentation_rendering()
            rgb[seg < 0] = 255
        frames.append(rgb)
    renderer.close()
    return frames


def main():
    global CASE, PRIMARY, PARTNER, EXP, CEM_DIR, H, W, FPS, PREVIEW_N
    args = parse_args()
    CASE, PRIMARY, PARTNER, EXP = args.case, args.primary, args.partner, args.exp
    CEM_DIR = Path(args.cem_dir) if args.cem_dir else \
        REPO / f"workspace/core4d/results/{EXP}/s6_downstream/cem/full"
    H = W = args.res
    FPS, PREVIEW_N = args.fps, args.preview
    scene_dir = Path(args.processed_root) / f"{args.scene_prefix}_{CASE}_{PRIMARY}"
    src_scene = scene_dir / f"scene_act_{EXP}_lowerbody_physics.xml"
    dual_scene = scene_dir / "scene_dual_render.xml"          # asset paths relative to scene_dir
    dual_scene_styled = scene_dir / "scene_dual_render_styled.xml"
    OUT = Path(args.out) if args.out else default_out(CASE, PRIMARY)
    OUT.mkdir(parents=True, exist_ok=True)

    prim = load_cem(PRIMARY)
    part = load_cem(PARTNER)
    offset, T, n, ang_mean, ang_std, tv = align_partner(part, prim)

    # p1 (partner) trim window is known from the paired export: source frames 109..217.
    print(f"=== {CASE}: re-anchor {PARTNER} -> {PRIMARY} ===")
    print(f"  primary({PRIMARY}) frames={len(prim)}  partner({PARTNER}) frames={len(part)}")
    print(f"  temporal offset: {PARTNER}[t] <-> {PRIMARY}[{offset}+t]   overlap n={n}")
    print(f"  partner window in primary index: [{offset}, {offset + n})")
    print(f"  rigid-fit residual: rot mean={ang_mean:.2f} deg std={ang_std:.2f} deg  "
          f"trans std={tv:.4f} m")
    print(f"  T trans={T[:3, 3]}  rot(ZYX deg)={Rot.from_matrix(T[:3, :3]).as_euler('ZYX', degrees=True)}")

    generate_scene_dual_robot(str(src_scene), str(dual_scene))
    beautify_robot_scene_xml(str(dual_scene), str(dual_scene_styled))
    model = mujoco.MjModel.from_xml_path(str(dual_scene_styled))
    floor_gid = prep_scene(model)

    dual_qpos = build_dual_qpos(prim, part, offset, n, T)
    if PREVIEW_N > 0:
        idx = np.linspace(0, len(dual_qpos) - 1, PREVIEW_N).round().astype(int)
        dual_qpos = dual_qpos[idx]
        print(f"[preview] rendering {PREVIEW_N} sampled frames")

    for mode in ("bg", "white"):
        frame_dir = OUT / f"robot_frames_{mode}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frames = render(dual_qpos, model, floor_gid, mode)
        for i, f in enumerate(frames):
            imageio.imwrite(frame_dir / f"frame_{i:04d}.png", f)
        if PREVIEW_N == 0:
            mp4 = OUT / f"robot_dual_{mode}.mp4"
            imageio.mimsave(mp4, frames, fps=FPS, quality=9)
            print(f"[ok] {mode}: {len(frames)} frames -> {frame_dir}  video -> {mp4}")
        else:
            print(f"[ok] {mode}: {len(frames)} preview frames -> {frame_dir}")


if __name__ == "__main__":
    main()
