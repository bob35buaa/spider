#!/usr/bin/env python3
"""Side-by-side MP4 render: E198 G1+A2 physics rollout vs original OmniRetarget ref.

For each case, tiles [G1+A2 physics | OmniRetarget reference] left-to-right and writes
E198_{case_id}_G1A2_vs_omni.mp4. Both cells come from the SAME G1A2 rollout npz
(qpos[:,0,:] = physics, qpos[:,1,:] = omni reference), replayed in the identical
gravcomp scene / camera. Reads rollout qpos + scene from e198_arm_cache.tsv.

Run headless: MUJOCO_GL=osmesa .venv/bin/python render_g1a2_vs_omni.py [--cases ...]
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
OUTDIR = C.RESULTS_E198 / "s6_downstream/render/G1A2_vs_omni"
DEFAULT_CASES = [
    "box024_20231011_026_p1", "box024_20231011_026_p2",
    "box024_20231011_027_p1", "box024_20231011_027_p2",
    "box024_20231011_028_p2",
]
W, H, FPS, MAX_FRAMES = 480, 360, 20, 160


def render_cell(qpos: np.ndarray, model: mujoco.MjModel, n_target: int) -> list[np.ndarray]:
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


def label(img: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width - 1, 16], fill=(20, 20, 20))
    d.text((4, 3), text, fill=(255, 255, 255))
    return np.asarray(im)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="", help="comma-separated case_ids; default = the 5 requested")
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    args = ap.parse_args()

    rows = C.read_tsv(CACHE)
    g1a2 = {r["case_id"]: r for r in rows if r["arm"] == "G1A2"}
    cases = [c.strip() for c in args.cases.split(",") if c.strip()] or DEFAULT_CASES

    OUTDIR.mkdir(parents=True, exist_ok=True)
    done = 0
    for cid in cases:
        r = g1a2.get(cid)
        if r is None:
            print(f"[skip] {cid}: no G1A2 row in cache", file=sys.stderr)
            continue
        phys, ref = npz_qpos(C.repo_path(r["outdir_npz"]))
        if ref is None:
            print(f"[skip] {cid}: rollout npz has no reference channel", file=sys.stderr)
            continue
        model = mujoco.MjModel.from_xml_path(str(C.repo_path(r["scene_xml"])))
        g = str(r.get("numeric_release_pass_12gate", "")).lower()
        gate = "[12/12]" if g in ("true", "1") else "[fail]"
        left = render_cell(phys, model, args.max_frames)
        right = render_cell(ref, model, args.max_frames)
        n = min(len(left), len(right))
        grid = [
            np.concatenate(
                [label(left[i], f"G1+A2 (physics)  {gate}"), label(right[i], "OmniRetarget (ref)")],
                axis=1,
            )
            for i in range(n)
        ]
        out = OUTDIR / f"E198_{cid}_G1A2_vs_omni.mp4"
        imageio.mimwrite(out, grid, fps=FPS, codec="libx264", quality=7)
        done += 1
        print(f"[{done}/{len(cases)}] wrote {out.name} ({len(grid)} frames)")
    print(f"[render-complete] {done} MP4 -> {C.rel(OUTDIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
