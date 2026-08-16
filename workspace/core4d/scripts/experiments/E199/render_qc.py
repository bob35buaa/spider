#!/usr/bin/env python3
"""E199 visual QC: render each completed CEM rollout into its scene_act and emit
an mp4 + keyframe strip. Used to eyeball penetration/floating/jitter and to
compare augmented variants against the same-case orig.

Renders the sim rollout (outdir_npz qpos, main tick) in the E199 rubber_hull+PRG
scene. Offscreen EGL. Selects variants from the priority manifest.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e199_common as C  # noqa: E402

W, H = 640, 480


def rollout_qpos(npz_path: Path, nq: int) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as data:
        q = np.asarray(data["qpos"], dtype=np.float64)
    if q.ndim == 3:      # (T, intra_tick, nq) -> main tick
        q = q[:, 0, :]
    return q[:, :nq]


def render_row(row: dict, out_dir: Path, n_key: int = 4) -> dict:
    scene = C.repo_path(row["scene_act"])
    npz = C.repo_path(row["outdir_npz"])
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    q = rollout_qpos(npz, model.nq)
    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    # frame the object
    oid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    data.qpos[:] = q[0]; data.qvel[:] = 0.0; mujoco.mj_forward(model, data)
    cam.lookat[:] = data.xpos[oid] if oid >= 0 else data.subtree_com[0]
    cam.distance = 3.2; cam.azimuth = 135.0; cam.elevation = -18.0

    tag = f"{row['object_key']}_{row['aug_variant']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / f"{tag}.mp4"
    frames = []
    for i in range(len(q)):
        data.qpos[:] = q[i]; data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())
    imageio.mimwrite(mp4, frames, fps=20, codec="libx264", quality=7)
    # keyframe strip (evenly spaced)
    idxs = np.linspace(0, len(q) - 1, n_key).astype(int)
    strip = np.concatenate([frames[i] for i in idxs], axis=1)
    strip_path = out_dir / f"{tag}_keyframes.jpg"
    imageio.imwrite(strip_path, strip)
    renderer.close()
    return {"tag": tag, "frames": len(q), "mp4": C.rel(mp4), "keyframes": C.rel(strip_path)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default="box024,box021,box004,bucket003",
                    help="comma list of object_keys to render (orig + its aug variants)")
    ap.add_argument("--out", type=Path, default=C.RESULTS / "s6_downstream/render/qc")
    ap.add_argument("--n-keyframes", type=int, default=5)
    ap.add_argument("--manifest", type=Path, default=C.FULL_MANIFEST,
                    help="manifest to render from (default pilot; use fullscale manifest for plan229)")
    ap.add_argument("--max-per-object", type=int, default=0,
                    help="cap rendered rows per object (0 = no cap); samples distinct cases first")
    args = ap.parse_args()

    wanted = {o.strip() for o in args.objects.split(",") if o.strip()}
    rows = [r for r in C.read_tsv(args.manifest)
            if r["object_key"] in wanted and C.repo_path(r["outdir_npz"]).is_file()]
    rows.sort(key=lambda r: (r["object_key"], r.get("case_id", ""), r["aug_variant"]))
    if args.max_per_object > 0:
        capped: list[dict] = []
        per_obj: dict[str, int] = {}
        for r in rows:
            obj = r["object_key"]
            if per_obj.get(obj, 0) >= args.max_per_object:
                continue
            per_obj[obj] = per_obj.get(obj, 0) + 1
            capped.append(r)
        rows = capped
    done = []
    for r in rows:
        try:
            info = render_row(r, args.out, args.n_keyframes)
            done.append(info)
            print(f"[render] {info['tag']} frames={info['frames']} -> {info['mp4']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {r['object_key']} {r['aug_variant']}: {type(exc).__name__}: {exc}", file=sys.stderr)
    C.write_json(args.out / "render_index.json", done)
    print(f"[done] rendered {len(done)} rollouts -> {C.rel(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
