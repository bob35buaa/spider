#!/usr/bin/env python3
"""E201 · render frames (+ full mp4) for VLM pre-screen (L2/L3-review rollouts).

For each rollout in the review queue it renders the full sim rollout into its
scene_act (same single camera as E199 render_qc), writes:
  * video.mp4        — the full-length rollout (native fps) for human eyeballing
  * f000.jpg ...     — frames sampled at FPS (default 5), capped at MAX_FRAMES
                       (uniform downsample if more), in temporal order, for the VLM
  * frames.json      — sampled indices + timestamps + mp4 path
under  vlm_review/frames/{exp}/{case_id}#{variant}/ .

Parallelism is by SHARDS, not a process pool: MuJoCo's EGL context does not
survive a multiprocessing fork-pool (workers segfault on the 2nd task and the
parent hangs). Instead launch N independent processes, each rendering its own
`--shard I --num-shards N` slice serially (serial renderer create+close per
rollout is stable — same as E199 render_qc). The wrapper fans these out.

Usage (single shard / serial):
    .venv/bin/python .../render_frames_for_vlm.py --exp E199 [--fps 5] [--max-frames 32]
Usage (one of N shards):
    .venv/bin/python .../render_frames_for_vlm.py --exp E199 --shard 0 --num-shards 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E199"))
import e199_common as C  # noqa: E402

VLM_DIR = REPO / "workspace/core4d/results/E201/vlm_review"
W, H = 960, 720   # per-view; two views hconcat -> 1920x720 (API resizes to --image-size)
JPEG_QUALITY = 95  # NB: imageio/pillow JPEG quality is 0-100 (was mistakenly 8 -> block noise)

# Two complementary camera angles (azimuth 90° apart) so the VLM can judge
# hand-object contact depth (single view was systematically misreading grasp).
# Each is (azimuth_deg, elevation_deg); shared lookat=object, distance below.
VIEWS = [(135.0, -18.0), (225.0, -18.0)]
CAM_DISTANCE = 3.2


def read_tsv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def rollout_qpos(npz_path: Path, nq: int) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as data:
        q = np.asarray(data["qpos"], dtype=np.float64)
    if q.ndim == 3:  # (T, intra_tick, nq) -> main tick
        q = q[:, 0, :]
    return q[:, :nq]


def sample_indices(n_total: int, duration_s: float, fps: float, max_frames: int) -> list[int]:
    """FPS sampling over the rollout, then cap at max_frames (uniform)."""
    if n_total <= 1:
        return [0] if n_total == 1 else []
    native_fps = (n_total - 1) / duration_s if duration_s and duration_s > 0 else 30.0
    stride = max(1, round(native_fps / fps))
    idxs = list(range(0, n_total, stride))
    if idxs[-1] != n_total - 1:
        idxs.append(n_total - 1)
    if len(idxs) > max_frames:
        idxs = sorted(set(int(round(x)) for x in np.linspace(0, n_total - 1, max_frames)))
    return idxs


def render_one(row: dict, out_dir: Path, fps: float, max_frames: int) -> dict:
    scene = C.repo_path(row["scene_act"])
    npz = C.repo_path(row["qpos_path"])
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    q = rollout_qpos(npz, model.nq)
    n_total = len(q)
    dur = float(row["duration_s"]) if row.get("duration_s") else 0.0
    native_fps = (n_total - 1) / dur if dur > 0 else 30.0
    idxs = sample_indices(n_total, dur, fps, max_frames)
    idx_set = set(idxs)

    renderer = mujoco.Renderer(model, height=H, width=W)
    oid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    data.qpos[:] = q[0]; data.qvel[:] = 0.0; mujoco.mj_forward(model, data)
    lookat = (data.xpos[oid] if oid >= 0 else data.subtree_com[0]).copy()
    cams = []
    for azimuth, elevation in VIEWS:
        c = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, c)
        c.lookat[:] = lookat
        c.distance = CAM_DISTANCE; c.azimuth = azimuth; c.elevation = elevation
        cams.append(c)

    out_dir.mkdir(parents=True, exist_ok=True)
    all_frames = []       # full rollout -> mp4 (views side-by-side)
    frame_paths = []      # sampled -> jpg (for VLM), each = views side-by-side
    timestamps = []
    for i in range(n_total):
        data.qpos[:] = q[i]; data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        views = []
        for c in cams:
            renderer.update_scene(data, camera=c)
            # disable shadows: the default light's shadow map aliases into floor
            # stripes that read as noise to the VLM.
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            views.append(renderer.render())
        img = np.concatenate(views, axis=1) if len(views) > 1 else views[0]
        all_frames.append(img)
        if i in idx_set:
            j = len(frame_paths)
            fp = out_dir / f"f{j:03d}.jpg"
            imageio.imwrite(fp, img, quality=JPEG_QUALITY)
            frame_paths.append(str(fp))
            timestamps.append(round(i / native_fps, 3) if native_fps else i)
    renderer.close()

    mp4 = out_dir / "video.mp4"
    imageio.mimwrite(mp4, all_frames, fps=max(1, round(native_fps)), codec="libx264", quality=7)

    meta = {
        "n_frames": len(frame_paths), "n_total": n_total, "native_fps": round(native_fps, 2),
        "sample_fps": fps, "frame_indices": idxs, "timestamps_s": timestamps,
        "frame_paths": frame_paths, "video": C.rel(mp4),
    }
    (out_dir / "frames.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def _task(row: dict, frames_root: Path, fps: float, max_frames: int) -> tuple[str, dict]:
    key = f"{row['case_id']}#{row['aug_variant']}"
    out_dir = frames_root / key
    try:
        if not row.get("qpos_path") or not row.get("scene_act"):
            raise FileNotFoundError("missing qpos_path/scene_act in queue row")
        meta = render_one(row, out_dir, fps, max_frames)
        return key, {"dir": C.rel(out_dir), "n_frames": meta["n_frames"],
                     "video": meta["video"], "layer": row["layer"], "object_key": row["object_key"]}
    except Exception as exc:  # noqa: BLE001
        return key, {"error": f"{type(exc).__name__}: {exc}", "layer": row.get("layer", "")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E199")
    ap.add_argument("--queue", type=Path, default=None)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--max-frames", type=int, default=32)
    ap.add_argument("--shard", type=int, default=0, help="this shard index [0, num_shards)")
    ap.add_argument("--num-shards", type=int, default=1, help="total shards (independent processes)")
    ap.add_argument("--out-root", type=Path, default=None)
    args = ap.parse_args()

    queue = args.queue or (VLM_DIR / f"{args.exp}_review_queue.tsv")
    rows = read_tsv(queue)
    frames_root = args.out_root or (VLM_DIR / "frames" / args.exp)
    frames_root.mkdir(parents=True, exist_ok=True)

    # deterministic shard slice (round-robin keeps object mix even across shards)
    shard_rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]

    index = {}
    ok = 0
    for done, row in enumerate(shard_rows, start=1):
        key, info = _task(row, frames_root=frames_root, fps=args.fps, max_frames=args.max_frames)
        index[key] = info
        if "error" in info:
            print(f"[error] {key}: {info['error']}", file=sys.stderr)
        else:
            ok += 1
        if done % 5 == 0 or done == len(shard_rows):
            print(f"[render s{args.shard}] {done}/{len(shard_rows)} (ok={ok})", flush=True)

    # per-shard index; wrapper merges shard indices after all finish
    idx_path = frames_root / (f"_index.shard{args.shard}.json" if args.num_shards > 1 else "_index.json")
    idx_path.write_text(json.dumps(index, ensure_ascii=False, indent=2))
    print(f"[done s{args.shard}] rendered {ok}/{len(shard_rows)} -> {C.rel(idx_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
