#!/usr/bin/env python3
"""Render keyframe comparison for E018 results vs E017-d and reference."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import interp1d


def render_keyframes(
    npz_path: str,
    xml_path: str,
    output_prefix: str,
    mpc_indices: tuple[int, ...] = (0, 3, 5, 8, 10),
    width: int = 720,
    height: int = 480,
) -> None:
    data = np.load(npz_path)
    qpos = data["qpos"]  # (n_mpc, ctrl_steps, nq)
    n_mpc, ctrl_steps, nq = qpos.shape

    model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    for mpc_idx in mpc_indices:
        if mpc_idx >= n_mpc:
            continue
        # take the first frame of this MPC tick
        mj_data.qpos[:] = qpos[mpc_idx, 0]
        mj_data.qvel[:] = 0
        mujoco.mj_forward(model, mj_data)
        renderer.update_scene(mj_data, camera=-1)
        img = renderer.render()
        out_path = f"{output_prefix}_mpc{mpc_idx}.png"
        import imageio
        imageio.imwrite(out_path, img)
        print(f"  Wrote: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    render_keyframes(args.npz, args.xml, args.prefix)


if __name__ == "__main__":
    main()
