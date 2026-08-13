#!/usr/bin/env python3
"""Offline 2x2-factorial MP4 render for E198 (osmesa headless).

For each case, tiles the four arms A0 / G1 / A2 / G1+A2 in a 2x2 grid and writes
E198_{case_id}_4cell.mp4. Reads rollout qpos + scene from e198_arm_cache.tsv.
Run headless: MUJOCO_GL=osmesa .venv/bin/python render_g1a2.py [--cases ...] [--all]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2]))  # eval.core
import e198_common as C  # noqa: E402
from eval.core.core_metrics import npz_qpos  # noqa: E402

CACHE = C.RESULTS_E198 / "s6_downstream/eval/full_factorial/e198_arm_cache.tsv"
OUTDIR = C.RESULTS_E198 / "s6_downstream/render/full_factorial"
ARMS = ("A0", "G1", "A2", "G1A2")
ARM_LABEL = {"A0": "A0 / PRG", "G1": "G1 (gravcomp)", "A2": "A2 (hand-gate)", "G1A2": "G1+A2"}
W, H, FPS, MAX_FRAMES = 480, 360, 20, 160


def render_arm(qpos_path: Path, scene_xml: Path, n_target: int) -> list[np.ndarray]:
    qpos, _ = npz_qpos(qpos_path)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    n = len(qpos)
    idxs = np.linspace(0, n - 1, min(n, n_target)).astype(int)
    frames = []
    for i in idxs:
        data.qpos[:] = qpos[i][: model.nq]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = data.xpos[obj] if obj >= 0 else data.xpos[0]
        cam.distance, cam.azimuth, cam.elevation = 3.2, 130.0, -18.0
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())
    renderer.close()
    return frames


def label(img: np.ndarray, text: str, ok: str) -> np.ndarray:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width - 1, 16], fill=(20, 20, 20))
    d.text((4, 3), f"{text}  {ok}", fill=(255, 255, 255))
    return np.asarray(im)


def tile(frames_by_arm: dict[str, list[np.ndarray]], passes: dict[str, str]) -> list[np.ndarray]:
    n = min(len(frames_by_arm[a]) for a in ARMS)
    out = []
    for i in range(n):
        cells = [label(frames_by_arm[a][i], ARM_LABEL[a], passes.get(a, "")) for a in ARMS]
        top = np.concatenate([cells[0], cells[1]], axis=1)
        bot = np.concatenate([cells[2], cells[3]], axis=1)
        out.append(np.concatenate([top, bot], axis=0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="", help="comma-separated case_ids; default = box024 (P0)")
    ap.add_argument("--all", action="store_true", help="render all 59 cases")
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    args = ap.parse_args()

    rows = C.read_tsv(CACHE)
    by_case: dict[str, dict[str, dict[str, str]]] = {}
    for r in rows:
        by_case.setdefault(r["case_id"], {})[r["arm"]] = r
    all_cases = [c for c, arms in by_case.items() if set(arms) >= set(ARMS)]

    if args.all:
        cases = sorted(all_cases)
    elif args.cases:
        cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        cases = sorted(c for c in all_cases if c.startswith("box024"))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    done = 0
    for cid in cases:
        arms = by_case.get(cid, {})
        if set(arms) < set(ARMS):
            print(f"[skip] {cid}: missing arms {set(ARMS) - set(arms)}", file=sys.stderr)
            continue
        frames = {}
        passes = {}
        for a in ARMS:
            r = arms[a]
            frames[a] = render_arm(C.repo_path(r["outdir_npz"]), C.repo_path(r["scene_xml"]), args.max_frames)
            g = r.get("numeric_release_pass_12gate", "")
            passes[a] = "[12/12]" if str(g).lower() in ("true", "1") else "[fail]"
        grid = tile(frames, passes)
        out = OUTDIR / f"E198_{cid}_4cell.mp4"
        imageio.mimwrite(out, grid, fps=FPS, codec="libx264", quality=7)
        done += 1
        print(f"[{done}/{len(cases)}] wrote {out.name} ({len(grid)} frames)")
    print(f"[render-complete] {done} MP4 -> {C.rel(OUTDIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
