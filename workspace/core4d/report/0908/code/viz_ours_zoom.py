"""Standalone high-res render of the "Ours" (SPIDER) hand/object contact close-up.

Reuses the exact alignment + dedicated close-up camera from viz_paper_figure so the
output matches the zoom circle in 3_annotated.png, but rendered at high resolution and
saved on its own (both a square and a circular-masked variant).

Run (headless):
    PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa .venv/bin/python \
        workspace/core4d/report/0908/code/viz_ours_zoom.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse  # noqa: E402

import numpy as np  # noqa: E402

if not hasattr(np, "infty"):
    np.infty = np.inf

import mujoco  # noqa: E402

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_methods_compare import DEFAULT_SCENE, facing_yaw  # noqa: E402
from viz_paper_figure import (LEFT_ARM, build_alignment, circular_whole,  # noqa: E402
                              render_panel)

ORANGE = (233, 126, 34)


def main():
    ap = argparse.ArgumentParser(description="Standalone Ours hand close-up.")
    ap.add_argument("--case", default="box021_20231011_037")
    ap.add_argument("--scene", default=str(DEFAULT_SCENE))
    ap.add_argument("--file-frame", type=int, default=12)
    ap.add_argument("--res", type=int, default=1400, help="close-up render resolution")
    ap.add_argument("--hand-az-offset", type=float, default=-126.1)
    ap.add_argument("--hand-el", type=float, default=5.0)
    ap.add_argument("--hand-radius", type=float, default=0.25)
    ap.add_argument("--border-w", type=int, default=10)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    import pyrender  # noqa: E402

    scene = args.scene
    qpos, ref, regs, (w0, w1) = build_alignment(scene, args.case)
    ref_i = w0 + args.file_frame
    assert w0 <= ref_i < w1, f"frame {args.file_frame} -> ref {ref_i} outside [{w0},{w1})"

    facing = facing_yaw(ref)
    az, el = np.radians(facing + 45.0), np.radians(12.0)
    cdir = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    haz, hel = np.radians(facing + args.hand_az_offset), np.radians(args.hand_el)
    hand_cdir = np.array([np.cos(hel) * np.cos(haz), np.cos(hel) * np.sin(haz), np.sin(hel)])

    _m = mujoco.MjModel.from_xml_path(scene)
    rh_bid = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    foot_gid = next(gid for gid in range(_m.ngeom)
                    if _m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH
                    and (mujoco.mj_id2name(_m, mujoco.mjtObj.mjOBJ_BODY, _m.geom_bodyid[gid]) or "")
                    == "left_ankle_roll_link")
    hide_bids = {mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_BODY, n) for n in LEFT_ARM}

    yfov = np.pi / 4.0
    lag, T = regs["spider"]
    renderer = pyrender.OffscreenRenderer(args.res, args.res)
    _, _, closeup = render_panel(scene, qpos["spider"], lag, T, ref_i, args.res, yfov, cdir,
                                 1.0, renderer, rh_bid, foot_gid, hand_cdir, args.hand_radius,
                                 hide_bids)
    renderer.delete()

    out_dir = Path(args.out_dir) if args.out_dir else \
        REPO / f"workspace/core4d/report/0908/paper_results/viz/{args.case}_p2/paper_figure"
    out_dir.mkdir(parents=True, exist_ok=True)

    sq = out_dir / "ours_zoom_square.png"
    closeup.save(sq)
    print(f"[ok] {sq}")

    circ = circular_whole(closeup, args.res, ORANGE, args.border_w)
    cp = out_dir / "ours_zoom_circle.png"
    circ.save(cp)                                  # RGBA, transparent outside the circle
    print(f"[ok] {cp}")


if __name__ == "__main__":
    main()
