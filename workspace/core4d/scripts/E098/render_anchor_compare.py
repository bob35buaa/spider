"""3D anchor 对比可视化：每 case 出一段 turntable mp4 + 1 张静态 4 视角 PNG。

每只手用不同颜色，box collision wireframe + 旧选中面 (xy-only) + 新选中面 (3D) 高亮。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


HAND_COLORS = {"L": "tab:blue", "R": "tab:red"}
OLD_HIGHLIGHT_COLOR = "tab:orange"  # 旧面 outline
NEW_HIGHLIGHT_COLOR = "tab:green"  # 新面 outline


def _draw_box(ax, half: np.ndarray, lw: float = 1.0, color: str = "black") -> None:
    hx, hy, hz = half
    corners = np.array(
        [
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
        ]
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for a, b in edges:
        ax.plot(
            [corners[a, 0], corners[b, 0]],
            [corners[a, 1], corners[b, 1]],
            [corners[a, 2], corners[b, 2]],
            color=color, lw=lw, alpha=0.5,
        )


def _face_polygon(face: str, half: np.ndarray) -> np.ndarray:
    axis = "xyz".index(face[1])
    sign = 1.0 if face[0] == "+" else -1.0
    coord = sign * half[axis]
    others = [i for i in (0, 1, 2) if i != axis]
    polys: list[list[float]] = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            pt = [0.0, 0.0, 0.0]
            pt[axis] = coord
            pt[others[0]] = s1 * half[others[0]]
            pt[others[1]] = s2 * half[others[1]]
            polys.append(pt)
    # 调整顺序为环
    polys = [polys[0], polys[1], polys[3], polys[2]]
    return np.array(polys)


def _draw_face(ax, face: str, half: np.ndarray, color: str, alpha: float = 0.25) -> None:
    poly = _face_polygon(face, half)
    coll = Poly3DCollection([poly], alpha=alpha, edgecolor=color, linewidth=2.0)
    coll.set_facecolor(color)
    ax.add_collection3d(coll)


def _draw_scene(ax, case_data: dict, view_elev: float, view_azim: float, title: str) -> None:
    half = np.array(case_data["half"])
    ax.view_init(elev=view_elev, azim=view_azim)
    _draw_box(ax, half)

    # 高亮 old vs new 主面
    for hand_idx, hand_name in enumerate(["L", "R"]):
        if hand_name not in case_data:
            continue
        info = case_data[hand_name]
        pts = np.array(info["points_local"])
        if len(pts) == 0:
            continue
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=HAND_COLORS[hand_name], s=12, alpha=0.6,
            label=f"{hand_name} contact ({len(pts)})",
        )
        # 旧面：用 outline 单层；新面：fill
        if info["old_top_face"]:
            _draw_face(ax, info["old_top_face"], half, OLD_HIGHLIGHT_COLOR, alpha=0.05)
        if info["new_top_face"]:
            _draw_face(ax, info["new_top_face"], half, NEW_HIGHLIGHT_COLOR, alpha=0.20)

    # 限定视野 ±1.2*half
    lim = 1.4 * max(half)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("x_local")
    ax.set_ylabel("y_local")
    ax.set_zlabel("z_local")
    # 写出 old vs new 主面摘要
    summary_lines = []
    for hand in ["L", "R"]:
        if hand in case_data:
            info = case_data[hand]
            tag = "DIFFER" if info["differs"] else "same"
            summary_lines.append(
                f"{hand}: old={info['old_top_face']}({info['old_top_count']}) "
                f"new={info['new_top_face']}({info['new_top_count']}) {tag}"
            )
    ax.text2D(0.02, 0.98, "\n".join(summary_lines), transform=ax.transAxes,
              va="top", ha="left", fontsize=9, family="monospace",
              bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    ax.set_title(title, fontsize=10)


def render_static(case_data: dict, out_path: Path) -> None:
    """4 视角 PNG."""
    case = case_data["case"]
    fig = plt.figure(figsize=(14, 12))
    views = [
        (25, 45, "isometric"),
        (90, -90, "top (xy)"),
        (0, -90, "side (xz)"),
        (0, 0, "front (yz)"),
    ]
    for i, (el, az, name) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        _draw_scene(ax, case_data, el, az, f"{case} | {name}")
    # 全局 legend
    handles = [
        plt.Line2D([], [], color=HAND_COLORS["L"], marker="o", linestyle="", label="L contact"),
        plt.Line2D([], [], color=HAND_COLORS["R"], marker="o", linestyle="", label="R contact"),
        plt.Line2D([], [], color=OLD_HIGHLIGHT_COLOR, marker="s", linestyle="",
                   label="OLD face (xy-only B1)", markersize=10),
        plt.Line2D([], [], color=NEW_HIGHLIGHT_COLOR, marker="s", linestyle="",
                   label="NEW face (3D B1-fixed)", markersize=10),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def render_turntable(case_data: dict, out_path: Path, n_frames: int = 36, fps: int = 12) -> None:
    """方位角 0..360° 转一圈出 mp4."""
    case = case_data["case"]
    tmp_dir = out_path.parent / f"_frames_{case}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n_frames):
        az = -90 + i * (360.0 / n_frames)
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        _draw_scene(ax, case_data, view_elev=20.0, view_azim=az, title=f"{case} | az={az:.0f}°")
        # legend per frame
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fp = tmp_dir / f"f{i:03d}.png"
        fig.savefig(fp, dpi=90)
        plt.close(fig)
        paths.append(fp)
    # 拼 mp4
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=7) as writer:
        for fp in paths:
            writer.append_data(imageio.imread(fp))
    # 清理临时帧
    for fp in paths:
        fp.unlink()
    tmp_dir.rmdir()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--refit-dir",
        type=Path,
        default=Path("workspace/core4d/results/E098/anchor_refit"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("workspace/core4d/results/E098/visuals/anchor_compare"),
    )
    ap.add_argument("--no-mp4", action="store_true", help="skip turntable mp4")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    json_files = sorted(args.refit_dir.glob("*.json"))
    for jf in json_files:
        case_data = json.loads(jf.read_text())
        if case_data.get("status") != "ok":
            print(f"[SKIP] {jf.name}: {case_data.get('status')}")
            continue
        case = case_data["case"]
        png_path = args.out / f"{case}_4view.png"
        print(f"[PNG ] {png_path}")
        render_static(case_data, png_path)
        if not args.no_mp4:
            mp4_path = args.out / f"{case}_turntable.mp4"
            print(f"[MP4 ] {mp4_path}")
            render_turntable(case_data, mp4_path)


if __name__ == "__main__":
    main()
