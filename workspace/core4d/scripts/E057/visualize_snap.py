#!/usr/bin/env python3
"""E057: Render side-by-side ref vs snap MuJoCo videos for bucket005_s2_person1.

Top row = ref qpos. Bottom row = snapped qpos. Two cameras per row.

Output:
  workspace/core4d/results/E057/bucket005_s2_person1/snap_visualization.mp4
"""

from __future__ import annotations

from pathlib import Path

import imageio
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[4]
CASE = "bucket005_s2_person1"
SCENE = REPO / f"example_datasets/processed/core4d/unitree_g1/humanoid_object/{CASE}/scene.xml"
RESULTS = REPO / f"workspace/core4d/results/E057/{CASE}"

W, H = 720, 480
FPS = 30


def render_pair(model, data, qpos_seq, label):
    model.vis.global_.offwidth = W
    model.vis.global_.offheight = H
    renderer = mujoco.Renderer(model, height=H, width=W)
    cam_names = ["front", "side"]
    available = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                 for i in range(model.ncam)]
    cams = [c if c in available else None for c in cam_names]
    frames = []
    for t, q in enumerate(qpos_seq):
        data.qpos[:] = q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        row = []
        for cam in cams:
            if cam is None:
                renderer.update_scene(data)
            else:
                renderer.update_scene(data, camera=cam)
            row.append(renderer.render())
        frames.append(np.concatenate(row, axis=1))
    return frames


def main() -> None:
    npz = np.load(RESULTS / "warmstart_qpos.npz")
    qpos_ref = npz["qpos_ref"]
    qpos_snap = npz["qpos_snap"]
    intent = npz["intent_window"]
    print(f"[viz] T={qpos_ref.shape[0]}, intent={tuple(intent.tolist())}")

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION

    print("[viz] rendering ref...")
    ref_frames = render_pair(model, data, qpos_ref, "ref")
    print("[viz] rendering snap...")
    snap_frames = render_pair(model, data, qpos_snap, "snap")

    combined = []
    for t in range(len(ref_frames)):
        in_intent = intent[0] <= t <= intent[1]
        bar = np.zeros((10, ref_frames[t].shape[1], 3), dtype=np.uint8)
        bar[:] = (0, 200, 0) if in_intent else (60, 60, 60)
        top = np.concatenate([bar, ref_frames[t]], axis=0)
        bot = snap_frames[t]
        combined.append(np.concatenate([top, bot], axis=0))

    out = RESULTS / "snap_visualization.mp4"
    imageio.mimsave(out, combined, fps=FPS, codec="libx264", quality=8)
    print(f"[viz] wrote {out}")


if __name__ == "__main__":
    main()
