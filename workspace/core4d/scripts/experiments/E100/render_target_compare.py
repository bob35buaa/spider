"""E100 render_target_compare.py：Tier 1 case 出 raw vs new target 对比 PNG。

每 case 4-panel 静态图：左上=isometric，右上=top，左下=side，右下=front。
叠加：
- box wireframe + face labels
- raw palm_local (黑 ×)
- palm-based target (橙圆)
- fingertip-based new target (绿方)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent / "E098"))
sys.path.insert(0, str(THIS_DIR.parent / "E099"))


def _draw_box(ax, half, color="black"):
    hx, hy, hz = half
    corners = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        ax.plot([corners[a,0], corners[b,0]],
                [corners[a,1], corners[b,1]],
                [corners[a,2], corners[b,2]], color=color, lw=1.0, alpha=0.5)


def _draw_face(ax, face, half, color, alpha):
    axis = "xyz".index(face[1])
    sign = 1.0 if face[0] == "+" else -1.0
    coord = sign * half[axis]
    others = [i for i in (0,1,2) if i != axis]
    polys = []
    for s1, s2 in [(-1,-1),(-1,1),(1,1),(1,-1)]:
        pt = [0.0, 0.0, 0.0]
        pt[axis] = coord
        pt[others[0]] = s1 * half[others[0]]
        pt[others[1]] = s2 * half[others[1]]
        polys.append(pt)
    coll = Poly3DCollection([np.array(polys)], alpha=alpha, edgecolor=color, linewidth=2)
    coll.set_facecolor(color)
    ax.add_collection3d(coll)


def _scene(ax, half, palm_L, palm_R, t_palm_L, t_palm_R, t_finger_L, t_finger_R,
           vote_L, vote_R, palm_face_L, palm_face_R, elev, azim, title):
    ax.view_init(elev=elev, azim=azim)
    _draw_box(ax, half)
    if vote_L:
        _draw_face(ax, vote_L, half, "tab:green", alpha=0.10)
    if vote_R and vote_R != vote_L:
        _draw_face(ax, vote_R, half, "tab:green", alpha=0.10)
    if palm_face_L and palm_face_L != vote_L:
        _draw_face(ax, palm_face_L, half, "tab:orange", alpha=0.08)
    if palm_face_R and palm_face_R != vote_R and palm_face_R != palm_face_L:
        _draw_face(ax, palm_face_R, half, "tab:orange", alpha=0.08)

    if palm_L is not None:
        ax.scatter(palm_L[:,0], palm_L[:,1], palm_L[:,2], c="black", marker="x", s=8, alpha=0.4)
    if palm_R is not None:
        ax.scatter(palm_R[:,0], palm_R[:,1], palm_R[:,2], c="dimgray", marker="x", s=8, alpha=0.4)
    if t_palm_L is not None:
        ax.scatter(t_palm_L[:,0], t_palm_L[:,1], t_palm_L[:,2], c="tab:orange", marker="o", s=10, alpha=0.5, label="L palm-target")
    if t_palm_R is not None:
        ax.scatter(t_palm_R[:,0], t_palm_R[:,1], t_palm_R[:,2], c="tab:red", marker="o", s=10, alpha=0.5, label="R palm-target")
    if t_finger_L is not None:
        ax.scatter(t_finger_L[:,0], t_finger_L[:,1], t_finger_L[:,2], c="tab:green", marker="s", s=14, alpha=0.7, label="L finger-target")
    if t_finger_R is not None:
        ax.scatter(t_finger_R[:,0], t_finger_R[:,1], t_finger_R[:,2], c="tab:blue", marker="s", s=14, alpha=0.7, label="R finger-target")

    lim = 1.4 * max(half)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title, fontsize=10)


def render_case(case: str, target_dir: Path, out_path: Path) -> None:
    npz_path = target_dir / case / "spider_contact_target_object_local.npz"
    summ_path = target_dir / case / "summary.json"
    import json
    d = np.load(npz_path)
    target_finger = d["spider_contact_target_object_local"]  # (T,2,3)
    palm_local = d["palm_local_record"]
    active = d["active"]
    summary = json.loads(summ_path.read_text())
    half = np.array(summary["half"])
    vote_L = summary["vote_L"]
    vote_R = summary["vote_R"]
    palm_face_L = summary["palm_vote_L"]
    palm_face_R = summary["palm_vote_R"]

    # Compute palm-based target by re-projecting (replicate logic)
    from build_fingertip_aware_target import project_to_face  # noqa: E402
    T = palm_local.shape[0]
    target_palm = np.zeros_like(palm_local)
    for t in range(T):
        target_palm[t, 0] = project_to_face(palm_local[t, 0], palm_face_L, half)
        target_palm[t, 1] = project_to_face(palm_local[t, 1], palm_face_R, half)

    palm_L = palm_local[active[:, 0], 0] if active[:, 0].any() else None
    palm_R = palm_local[active[:, 1], 1] if active[:, 1].any() else None
    t_palm_L = target_palm[active[:, 0], 0] if active[:, 0].any() else None
    t_palm_R = target_palm[active[:, 1], 1] if active[:, 1].any() else None
    t_finger_L = target_finger[active[:, 0], 0] if active[:, 0].any() else None
    t_finger_R = target_finger[active[:, 1], 1] if active[:, 1].any() else None

    fig = plt.figure(figsize=(14, 12))
    views = [(25, 45, "isometric"), (90, -90, "top (xy)"),
             (0, -90, "side (xz)"), (0, 0, "front (yz)")]
    for i, (el, az, name) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        title = (f"{case} | {name}\n"
                 f"L: palm={palm_face_L} → finger={vote_L}  R: palm={palm_face_R} → finger={vote_R}")
        _scene(ax, half, palm_L, palm_R, t_palm_L, t_palm_R, t_finger_L, t_finger_R,
               vote_L, vote_R, palm_face_L, palm_face_R, el, az, title)
    handles = [
        plt.Line2D([], [], color="black", marker="x", linestyle="", label="palm (FK)"),
        plt.Line2D([], [], color="tab:orange", marker="o", linestyle="", label="palm-based target"),
        plt.Line2D([], [], color="tab:green", marker="s", linestyle="", label="fingertip-based target (new)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--target-dir",
        type=Path,
        default=Path("workspace/core4d/results/E100/fingertip_targets"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("workspace/core4d/results/E100/visuals/target_compare"),
    )
    ap.add_argument(
        "--tier1",
        nargs="+",
        default=[
            "d003_box021_20231018_030_p1",
            "d003_box021_20231020_020_p2",
            "d003_box021_20231018_028_p2",
            "box021_person1",
            "box023_person1",
            "box023_person2",
            "e091_box004_20231003_2_083_p2",
            "e091_box004_20231003_2_083_p1",
            "e091_box004_20231003_2_082_p1",
        ],
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for case in args.tier1:
        npz = args.target_dir / case / "spider_contact_target_object_local.npz"
        if not npz.is_file():
            print(f"[SKIP] {case} (no target NPZ)")
            continue
        out = args.out_dir / f"{case}_target_compare.png"
        print(f"[PNG ] {out}")
        render_case(case, args.target_dir, out)


if __name__ == "__main__":
    main()
