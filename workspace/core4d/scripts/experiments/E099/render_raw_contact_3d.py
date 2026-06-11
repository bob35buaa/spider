"""E099 raw_contact 3D 可视化生成器。

每 case 输出：
- {case}_4view.png: 4 视角静态 PNG (isometric / top / side / front)
- {case}_turntable.mp4: 36 帧 turntable 视频 ~3 秒

叠加内容（obj local frame）：
- box collision wireframe
- raw 10 指尖散点（按帧颜色渐变 + L/R 区分形状）
- palm site 位置（trajectory_kinematic.npz contact_pos in obj local）作对比
- per-frame fingertip vote 多数面高亮（绿）
- 每个面的 face label 文字
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent / "E098"))

import face_utils as fu  # noqa: E402
from case_to_raw import parse_case  # noqa: E402
from fingertip_face_vote import (  # noqa: E402
    LEFT_FINGERTIP_INDICES,
    RIGHT_FINGERTIP_INDICES,
    world_to_local_points,
    yup_to_zup_points,
    yup_to_zup_T,
)


HAND_COLORS = {"L": "tab:blue", "R": "tab:red"}
VOTE_HIGHLIGHT = "tab:green"


def _draw_box(ax, half: np.ndarray, lw: float = 1.0, color: str = "black") -> None:
    hx, hy, hz = half
    corners = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        ax.plot(
            [corners[a,0], corners[b,0]],
            [corners[a,1], corners[b,1]],
            [corners[a,2], corners[b,2]],
            color=color, lw=lw, alpha=0.5,
        )


def _face_polygon(face: str, half: np.ndarray) -> np.ndarray:
    axis = "xyz".index(face[1])
    sign = 1.0 if face[0] == "+" else -1.0
    coord = sign * half[axis]
    others = [i for i in (0, 1, 2) if i != axis]
    polys = []
    for s1, s2 in [(-1, -1), (-1, 1), (1, 1), (1, -1)]:
        pt = [0.0, 0.0, 0.0]
        pt[axis] = coord
        pt[others[0]] = s1 * half[others[0]]
        pt[others[1]] = s2 * half[others[1]]
        polys.append(pt)
    return np.array(polys)


def _draw_face(ax, face: str, half: np.ndarray, color: str, alpha: float = 0.25) -> None:
    poly = _face_polygon(face, half)
    coll = Poly3DCollection([poly], alpha=alpha, edgecolor=color, linewidth=2.0)
    coll.set_facecolor(color)
    ax.add_collection3d(coll)


def _draw_face_labels(ax, half: np.ndarray) -> None:
    """六面中心写 face label."""
    for axis, axis_name in enumerate("xyz"):
        for sign, sign_name in [(1.0, "+"), (-1.0, "-")]:
            pos = np.zeros(3)
            pos[axis] = sign * half[axis] * 1.15
            ax.text(pos[0], pos[1], pos[2], f"{sign_name}{axis_name}",
                    fontsize=7, color="gray", ha="center", va="center")


def _draw_scene(ax, case_data: dict, view_elev: float, view_azim: float, title: str) -> None:
    half = np.array(case_data["half"])
    ax.view_init(elev=view_elev, azim=view_azim)
    _draw_box(ax, half)
    _draw_face_labels(ax, half)

    for hand_name in ["L", "R"]:
        info = case_data[hand_name]
        pts_all = info["points_local"]  # list of (5, 3) or empty
        if not pts_all:
            continue
        arr = np.concatenate(pts_all, axis=0)  # (n_total_tips, 3)
        ax.scatter(
            arr[:, 0], arr[:, 1], arr[:, 2],
            c=HAND_COLORS[hand_name], s=8, alpha=0.5,
            label=f"{hand_name} fingertips ({arr.shape[0]})",
        )
        if info["palm_local"] is not None and len(info["palm_local"]) > 0:
            palm = np.array(info["palm_local"])
            ax.scatter(palm[:, 0], palm[:, 1], palm[:, 2],
                       c=HAND_COLORS[hand_name], marker="x", s=20, alpha=0.6,
                       label=f"{hand_name} palm")
        if info["vote_face"]:
            _draw_face(ax, info["vote_face"], half, VOTE_HIGHLIGHT, alpha=0.18)

    lim = 1.4 * max(half)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    summary = []
    for hand in ["L", "R"]:
        info = case_data[hand]
        if info["vote_face"]:
            summary.append(
                f"{hand}: vote={info['vote_face']} ({info['vote_frac']*100:.0f}%, n={info['n_contact']})"
            )
        else:
            summary.append(f"{hand}: no contact")
    ax.text2D(0.02, 0.98, "\n".join(summary), transform=ax.transAxes,
              va="top", ha="left", fontsize=9, family="monospace",
              bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    ax.set_title(title, fontsize=10)


def render_static(case_data: dict, out_path: Path) -> None:
    case = case_data["case"]
    fig = plt.figure(figsize=(14, 12))
    views = [(25, 45, "isometric"), (90, -90, "top (xy)"),
             (0, -90, "side (xz)"), (0, 0, "front (yz)")]
    for i, (el, az, name) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        _draw_scene(ax, case_data, el, az, f"{case} | {name}")
    handles = [
        plt.Line2D([], [], color=HAND_COLORS["L"], marker="o", linestyle="", label="L fingertips"),
        plt.Line2D([], [], color=HAND_COLORS["R"], marker="o", linestyle="", label="R fingertips"),
        plt.Line2D([], [], color="black", marker="x", linestyle="", label="palm (FK)"),
        plt.Line2D([], [], color=VOTE_HIGHLIGHT, marker="s", linestyle="", markersize=10,
                   label="vote face"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def render_turntable(case_data: dict, out_path: Path, n_frames: int = 36, fps: int = 12) -> None:
    case = case_data["case"]
    tmp = out_path.parent / f"_frames_{case}"
    tmp.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n_frames):
        az = -90 + i * (360.0 / n_frames)
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        _draw_scene(ax, case_data, view_elev=20.0, view_azim=az, title=f"{case} | az={az:.0f}°")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fp = tmp / f"f{i:03d}.png"
        fig.savefig(fp, dpi=90)
        plt.close(fig)
        paths.append(fp)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=7) as w:
        for fp in paths:
            w.append_data(imageio.imread(fp))
    for fp in paths:
        fp.unlink()
    tmp.rmdir()


def collect_case_data(case: str, scene_xml: Path, contact_thresh: float = 0.02) -> dict | None:
    """读 raw mocap + obj poses + processed traj，组装可视化所需数据."""
    spec = parse_case(case)
    if spec.note.startswith("SKIP"):
        return None
    if not spec.raw_ok():
        return None

    raw = np.load(spec.person_npz, allow_pickle=True)["arr_0"].item()
    joints_all = raw["joints"]  # (T, 127, 3) Y-up
    T_raw = joints_all.shape[0]
    L_tips_zup = yup_to_zup_points(joints_all[:, LEFT_FINGERTIP_INDICES, :])
    R_tips_zup = yup_to_zup_points(joints_all[:, RIGHT_FINGERTIP_INDICES, :])
    T_obj_yup = np.load(spec.obj_poses_npy)
    T_obj_zup = yup_to_zup_T(T_obj_yup)
    T = min(T_raw, T_obj_zup.shape[0])
    L_tips_zup = L_tips_zup[:T]; R_tips_zup = R_tips_zup[:T]; T_obj_zup = T_obj_zup[:T]
    L_local = world_to_local_points(L_tips_zup, T_obj_zup)  # (T,5,3)
    R_local = world_to_local_points(R_tips_zup, T_obj_zup)

    # 拿 box half 从 scene.xml
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")
    half = model.geom_size[gid, :3].astype(np.float64)

    # palm site：从 trajectory_kinematic.npz 读 contact_pos（world），转 obj local
    case_dir = scene_xml.parent
    traj_path = case_dir / "0" / "trajectory_kinematic.npz"
    palm_L_local = []
    palm_R_local = []
    if traj_path.is_file():
        traj = np.load(traj_path, allow_pickle=True)
        # contact_pos 已是 world; obj 在 traj 中位置：qpos[obj_qadr+0:3]; quat: qpos[obj_qadr+3:7]
        # 这里用 spider 仓库的 obj pose（与 raw mocap 不同时基，但同一物体类）
        obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
        obj_jnt = model.body_jntadr[obj_body]
        obj_qadr = int(model.jnt_qposadr[obj_jnt])
        qpos = traj["qpos"]
        contact_pos = traj.get("contact_pos", None)
        contact = traj.get("contact", None)
        if contact_pos is not None:
            T_traj = qpos.shape[0]
            data_mj = mujoco.MjData(model)
            for t in range(T_traj):
                data_mj.qpos[:] = qpos[t]
                mujoco.mj_forward(model, data_mj)
                obj_pos = data_mj.qpos[obj_qadr:obj_qadr+3].astype(np.float64)
                obj_quat = data_mj.qpos[obj_qadr+3:obj_qadr+7].astype(np.float64)
                for hand, store in [(0, palm_L_local), (1, palm_R_local)]:
                    if contact is not None and not bool(contact[t, hand]):
                        continue
                    w = np.asarray(contact_pos[t, hand], dtype=np.float64)
                    if not np.all(np.isfinite(w)):
                        continue
                    # local = R^T (w - t)
                    out = np.zeros(3)
                    qinv = obj_quat.copy(); qinv[1:] *= -1
                    mujoco.mju_rotVecQuat(out, w - obj_pos, qinv)
                    store.append(out)

    # face vote (close to surface only)
    def per_hand(local_pts: np.ndarray) -> dict:
        kept_per_frame: list[np.ndarray] = []  # only contact frames keep all 5 tips
        labels = []
        for t in range(T):
            tip_labels = []
            for k in range(5):
                p = local_pts[t, k]
                if not np.all(np.isfinite(p)):
                    continue
                tip_labels.append(fu.face_label(p, half))
            if not tip_labels:
                continue
            most = Counter(tip_labels).most_common(1)[0][0]
            axis = "xyz".index(most[1])
            sign = 1.0 if most[0] == "+" else -1.0
            face_coord = sign * half[axis]
            in_contact = False
            for k in range(5):
                p = local_pts[t, k]
                if not np.all(np.isfinite(p)):
                    continue
                if fu.face_label(p, half) != most:
                    continue
                d = abs(p[axis] - face_coord)
                others = [i for i in (0, 1, 2) if i != axis]
                in_plane = (
                    abs(p[others[0]]) <= half[others[0]] + 0.05
                    and abs(p[others[1]]) <= half[others[1]] + 0.05
                )
                if in_plane and d <= contact_thresh:
                    in_contact = True
                    break
            if in_contact:
                kept_per_frame.append(local_pts[t])  # (5,3)
                labels.append(most)
        if not labels:
            return {"points_local": [], "vote_face": "", "vote_frac": 0.0,
                    "n_contact": 0, "palm_local": []}
        c = Counter(labels)
        top, n = c.most_common(1)[0]
        return {
            "points_local": kept_per_frame,
            "vote_face": top,
            "vote_frac": n / len(labels),
            "n_contact": len(labels),
            "palm_local": [],  # 填充见下
        }

    L = per_hand(L_local)
    R = per_hand(R_local)
    L["palm_local"] = palm_L_local
    R["palm_local"] = palm_R_local
    return {
        "case": case,
        "half": [float(x) for x in half],
        "obj_name": spec.obj_name,
        "L": L,
        "R": R,
    }


def read_manifest(path: Path) -> list[str]:
    cases = []
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            cases.append(row["case_name"])
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("workspace/core4d/scripts/E098/historical_case_manifest.tsv"),
    )
    ap.add_argument(
        "--scene-base",
        type=Path,
        default=Path("example_datasets/processed/core4d/unitree_g1/humanoid_object"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("workspace/core4d/results/E099/visuals/raw_contact_3d"),
    )
    ap.add_argument("--no-mp4", action="store_true")
    ap.add_argument("--only-png", action="store_true", help="alias of --no-mp4")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    skip_mp4 = args.no_mp4 or args.only_png

    cases = read_manifest(args.manifest)
    for case in cases:
        scene = args.scene_base / case / "scene.xml"
        if not scene.is_file():
            print(f"[SKIP] {case} (no scene.xml)")
            continue
        print(f"[VIZ ] {case}", flush=True)
        try:
            data = collect_case_data(case, scene)
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            continue
        if data is None:
            print(f"  [SKIP] missing raw or geometry mismatch")
            continue
        png = args.out / f"{case}_4view.png"
        render_static(data, png)
        print(f"  -> {png}")
        if not skip_mp4:
            mp4 = args.out / f"{case}_turntable.mp4"
            render_turntable(data, mp4)
            print(f"  -> {mp4}")


if __name__ == "__main__":
    main()
