"""Four-method p2/G1 comparison render (SPIDER vs OmniRetarget vs SBTO vs GMR).

Renders ONE person (p2), ONLY the G1 robot + its manipulated object, for each of
four retargeting methods, in the same pyrender style as the mixed deliverable --
so the four clips read as one comparison figure.

All four methods store the G1 as MuJoCo qpos (7 base + 29 G1 joints + 6 object
[trans3 + euler ZYX]) and are rendered on the SAME scene model
(scene_act_E170_lowerbody_physics.xml: G1 29dof + box021), so robot mesh /
materials / object mesh are identical across methods.

Alignment (OBJECT-align; reference = SPIDER):
  Each method is registered so its OBJECT world trajectory coincides with SPIDER's
  (planar yaw + 3D translation + integer frame lag), then that single world rigid
  transform is applied to the method's robot AND object meshes. This fixes the box
  in one world location across all four clips; each robot falls where its method
  places it relative to the shared box. SPIDER/OmniRetarget/SBTO share the pipeline
  frame (identity); GMR (object-unaware, robot ~1.4 m from the box) gets a real
  transform and will visibly sit offset from the box -- the honest depiction of its
  near-zero object contact.

Temporal alignment: frame index (native for the same-trim methods; GMR mapped by
its recovered lag), rendered over the common window -> equal-length clips.

Run (headless):
    PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa .venv/bin/python \
        workspace/core4d/report/0908/code/viz_methods_compare.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse  # noqa: E402

import numpy as np  # noqa: E402

if not hasattr(np, "infty"):        # pyrender uses np.infty, removed in NumPy 2.0
    np.infty = np.inf

import mujoco  # noqa: E402
import trimesh  # noqa: E402
import pyrender  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_style import BOX_COLOR, make_checker_floor, soft_phong_lights  # noqa: E402
from viz_mixed_robot_smplx import look_at, object_world_mesh, robot_world_meshes  # noqa: E402

DEFAULT_CASE = "box021_20231011_037"
PERSON = "p2"
SBTO_ROOT = Path("/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/sbto")
GMR_ROOT = Path("/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/GMR")

DEFAULT_SCENE = (REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
                 / f"dcv3_omnirt_v1_ref_fk_{DEFAULT_CASE}_{PERSON}"
                 / "scene_act_E170_lowerbody_physics.xml")

METHODS = ["spider", "omniretarget", "sbto", "gmr"]     # SPIDER first (reference)


def method_qpos_paths(case: str) -> dict[str, Path]:
    return {
        "spider": REPO / f"workspace/core4d/results/E170/s6_downstream/cem/full/E170_{case}_{PERSON}_PRG.npz",
        "omniretarget": (REPO / "workspace/core4d/report/0908/paper_results/omni_scene_act_qpos"
                         / f"{case}_{PERSON}_omnirt_scene_act_qpos.npz"),
        "sbto": SBTO_ROOT / f"paper_results/_sbto_scene_qpos/{case}_{PERSON}_omnirt_scene_act_qpos.npz",
        "gmr": GMR_ROOT / f"out/core4d_g1/paper_metrics/qpos/{case}_{PERSON}.qpos.npz",
    }


def load_qpos(path: Path) -> np.ndarray:
    q = np.load(path)["qpos"]
    q = q[:, 0, :] if q.ndim == 3 else q
    assert q.shape[1] == 42, f"expected 42-col qpos, got {q.shape} in {path}"
    return q


def yaw_trans_fit(src: np.ndarray, dst: np.ndarray):
    """Planar yaw (about Z) + HORIZONTAL translation mapping src->dst (both Nx3).

    The vertical (z) translation is forced to 0 so each method keeps its native
    floor-relative height -- object-align must not lift/sink a method, or it would
    fake the foot-ground clearance (floating / ground penetration) in the render.
    Returns (T 4x4, yaw_deg, residual_m)."""
    s2 = src[:, :2] - src[:, :2].mean(0)
    d2 = dst[:, :2] - dst[:, :2].mean(0)
    U, _, Vt = np.linalg.svd(s2.T @ d2)
    sgn = np.sign(np.linalg.det(Vt.T @ U.T))
    R2 = Vt.T @ np.diag([1, sgn]) @ U.T
    th = np.arctan2(R2[1, 0], R2[0, 0])
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    t = dst.mean(0) - R @ src.mean(0)
    t[2] = 0.0                                   # no vertical shift (keep native z)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    resid = float(np.linalg.norm((src @ R.T + t) - dst, axis=1).mean())
    return T, np.degrees(th), resid


def object_world_traj(scene: str, qpos: np.ndarray) -> np.ndarray:
    """(T,3) world position of the object body over the clip via mj_forward."""
    m = mujoco.MjModel.from_xml_path(scene)
    d = mujoco.MjData(m)
    obj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    out = np.empty((len(qpos), 3))
    for t, q in enumerate(qpos):
        d.qpos[:] = q
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        out[t] = d.xpos[obj]
    return out


def register_object(method_obj: np.ndarray, ref_obj: np.ndarray, maxlag: int = 15):
    """Object-align: yaw+trans+lag mapping method object-world -> ref object-world.
    Returns (lag, T, yaw, resid, n)."""
    best = None
    need = max(30, len(method_obj) // 2)
    for lag in range(-maxlag, maxlag + 1):
        pairs = [(t, t + lag) for t in range(len(method_obj)) if 0 <= t + lag < len(ref_obj)]
        if len(pairs) < need:
            continue
        src = method_obj[[t for t, _ in pairs]]
        dst = ref_obj[[r for _, r in pairs]]
        T, yaw, resid = yaw_trans_fit(src, dst)
        if best is None or resid < best[3]:
            best = (lag, T, yaw, resid, len(pairs))
    return best


def facing_yaw(qpos: np.ndarray) -> float:
    quats = qpos[:, 3:7][:, [1, 2, 3, 0]]
    fwd = Rot.from_quat(quats).apply(np.tile([1.0, 0.0, 0.0], (len(qpos), 1)))
    return float(np.degrees(np.arctan2(fwd[:, 1].mean(), fwd[:, 0].mean())))


def apply_T(verts: np.ndarray, T: np.ndarray) -> np.ndarray:
    return verts @ T[:3, :3].T + T[:3, 3]


def parse_args():
    p = argparse.ArgumentParser(description="Four-method p2/G1 comparison render (object-aligned).")
    p.add_argument("--case", default=DEFAULT_CASE)
    p.add_argument("--scene", default=str(DEFAULT_SCENE))
    p.add_argument("--out", default=None)
    p.add_argument("--res", type=int, default=1080)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--preview", type=int, default=0)
    p.add_argument("--az", type=float, default=None,
                   help="camera azimuth deg (default: reference facing yaw + --az-offset)")
    p.add_argument("--az-offset", type=float, default=45.0)
    p.add_argument("--el", type=float, default=12.0, help="camera elevation deg (default: 12)")
    return p.parse_args()


def main():
    args = parse_args()
    scene = args.scene
    paths = method_qpos_paths(args.case)
    qpos = {m: load_qpos(paths[m]) for m in METHODS}
    ref = qpos["spider"]
    ref_obj = object_world_traj(scene, ref)

    # Alignment. TEMPORAL lag: same-trim methods (same length + start pose as the
    # reference) are frame-index aligned -> lag 0 (their object-motion phase differs
    # only via physics vs kinematic, which must NOT shift them in time); GMR (own
    # frame/trim) gets a lag from object correlation. SPATIAL: an object-align yaw+
    # translation at that lag places every method's box on the reference box.
    ref_root0 = ref[0, 0:3]
    regs = {}
    for m in METHODS:
        q = qpos[m]
        if m == "spider":
            regs[m] = (0, np.eye(4))
            print(f"[align] {m:12s} len={len(q):3d} REFERENCE")
            continue
        obj = object_world_traj(scene, q)
        native = (len(q) == len(ref)) and (np.linalg.norm(q[0, 0:3] - ref_root0) < 0.10)
        if native:
            lag = 0
            pairs = [(t, t) for t in range(min(len(obj), len(ref_obj)))]
            T, yaw, resid = yaw_trans_fit(obj[[t for t, _ in pairs]], ref_obj[[r for _, r in pairs]])
            n = len(pairs)
            tag = "NATIVE-time"
        else:
            lag, T, yaw, resid, n = register_object(obj, ref_obj)
            tag = "REGISTERED"
        regs[m] = (lag, T)
        print(f"[align] {m:12s} len={len(q):3d} {tag} lag={lag:+d} yaw={yaw:+.1f}deg "
              f"obj_resid={resid:.3f}m overlap={n}")

    # common ref-index window (ref index i maps to method frame i-lag)
    w0 = max(max(0, regs[m][0]) for m in METHODS)
    w1 = min(min(len(ref), len(qpos[m]) + regs[m][0]) for m in METHODS)
    assert w1 > w0, f"empty common window [{w0},{w1})"
    print(f"[window] common ref-index window [{w0},{w1}) -> {w1 - w0} frames per method")

    facing = facing_yaw(ref)
    az = np.radians(args.az if args.az is not None else facing + args.az_offset)
    el = np.radians(args.el)
    cdir = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    print(f"[cam] reference facing yaw={facing:.1f}deg -> az={np.degrees(az):.1f}deg el={args.el:.1f}deg")

    out_dir = Path(args.out) if args.out else \
        REPO / f"workspace/core4d/report/0908/paper_results/viz/{args.case}_{PERSON}/methods_compare"

    ref_frames = list(range(w0, w1))
    if args.preview > 0:
        idx = np.linspace(0, len(ref_frames) - 1, args.preview).round().astype(int)
        ref_frames = [ref_frames[i] for i in idx]

    yfov = np.pi / 4.0
    renderer = pyrender.OffscreenRenderer(args.res, args.res)
    mat = lambda c, r=0.55: pyrender.MetallicRoughnessMaterial(
        baseColorFactor=c, metallicFactor=0.0, roughnessFactor=r, doubleSided=True)
    shadow_flag = pyrender.constants.RenderFlags.SHADOWS_DIRECTIONAL
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)

    def meshes_fn(m):
        lag, T = regs[m]
        model = mujoco.MjModel.from_xml_path(scene)
        data = mujoco.MjData(model)

        def frame_meshes(ref_i):
            data.qpos[:] = qpos[m][ref_i - lag]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            robot = [(apply_T(v, T), f, rgb) for v, f, rgb in robot_world_meshes(model, data)]
            ov, of, orgb = object_world_mesh(model, data)
            return robot, (apply_T(ov, T), of, orgb)
        return frame_meshes

    # Common camera DISTANCE + floor across methods (uniform scale), but a per-frame
    # target that follows each method's own robot+object midpoint so every clip stays
    # well framed (GMR's robot sits ~2 m off the box; a single static shot covering
    # both would shrink everyone). The object-aligned box anchors the left of frame.
    sample = ref_frames[:: max(1, len(ref_frames) // 12)] or ref_frames
    radius, floor_z = 0.0, np.inf
    for m in METHODS:
        fm = meshes_fn(m)
        for ri in sample:
            robot, (ov, _, _) = fm(ri)
            rmean = np.concatenate([g[0] for g in robot], 0).mean(0)
            tgt = (rmean + ov.mean(0)) / 2.0
            allv = np.concatenate([g[0] for g in robot] + [ov], 0)
            radius = max(radius, float(np.linalg.norm(allv - tgt, axis=1).max()))
            floor_z = min(floor_z, float(allv[:, 2].min()))
    dist = radius / np.tan(yfov / 2.0) * 1.10
    floor_trimesh = make_checker_floor(floor_z, extent=12.0, repeats=14, up="z")
    print(f"[cam] common dist={dist:.2f} radius={radius:.2f} floor_z={floor_z:.2f}")

    for m in METHODS:
        frame_meshes = meshes_fn(m)
        for mode in ("bg", "white"):
            with_bg = mode == "bg"
            bg_col = [0.90, 0.91, 0.93, 1.0] if with_bg else [1.0, 1.0, 1.0, 1.0]
            frame_dir = out_dir / m / f"frames_{mode}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            frames = []
            for k, ri in enumerate(ref_frames):
                robot, (ov, of, _orgb) = frame_meshes(ri)
                rmean = np.concatenate([g[0] for g in robot], 0).mean(0)
                target = (rmean + ov.mean(0)) / 2.0
                scene_pr = pyrender.Scene(bg_color=bg_col, ambient_light=[0.4, 0.4, 0.42])
                if with_bg:
                    scene_pr.add(pyrender.Mesh.from_trimesh(floor_trimesh))
                for rv, rf, rgb in robot:
                    scene_pr.add(pyrender.Mesh.from_trimesh(
                        trimesh.Trimesh(rv, rf, process=False),
                        material=mat(list(rgb) + [1.0]), smooth=False))
                scene_pr.add(pyrender.Mesh.from_trimesh(
                    trimesh.Trimesh(ov, of, process=False), material=mat(BOX_COLOR), smooth=False))
                scene_pr.add(camera, pose=look_at(target + cdir * dist, target, [0, 0, 1]))
                for light, pose in soft_phong_lights(look_at, target, [0, 0, 1], floor_z, up="z"):
                    scene_pr.add(light, pose=pose)
                color, depth = renderer.render(scene_pr, flags=shadow_flag if with_bg else 0)
                rgb = color[:, :, :3].copy()
                if not with_bg:
                    rgb[depth == 0] = 255
                frames.append(rgb)
                imageio.imwrite(frame_dir / f"frame_{k:04d}.png", rgb)
            if args.preview == 0:
                mp4 = out_dir / m / f"{m}_{mode}.mp4"
                imageio.mimsave(mp4, frames, fps=args.fps, quality=9)
                print(f"[ok] {m} {mode}: {len(frames)} frames -> {frame_dir}  video -> {mp4}")
            else:
                print(f"[ok] {m} {mode}: {len(frames)} preview frames -> {frame_dir}")
    renderer.delete()


if __name__ == "__main__":
    main()
