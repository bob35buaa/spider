#!/usr/bin/env python3
"""Extract Holosoma converter input from a Spider MJWarp trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Spider trajectory_mjwp.npz")
    parser.add_argument("--output", required=True, type=Path, help="Output npz with qpos + fps")
    parser.add_argument(
        "--env-index",
        default=0,
        type=int,
        help="Env index for --layout legacy-env with (T, N, 43) qpos",
    )
    parser.add_argument(
        "--layout",
        choices=("flatten", "legacy-env"),
        default="flatten",
        help="Interpret 3D qpos as (control_ticks, ctrl_steps, 43) and flatten, or legacy (T, env, 43)",
    )
    parser.add_argument("--fps", default=60.0, type=float, help="Spider input frame rate")
    return parser.parse_args()


def extract_qpos(qpos: np.ndarray, env_index: int, layout: str) -> np.ndarray:
    if qpos.ndim == 3:
        if qpos.shape[2] != 43:
            raise ValueError(f"expected qpos shape (..., 43), got {qpos.shape}")
        if layout == "flatten":
            # MJWP saves each control tick as (ctrl_steps, qpos_dim); keep all substeps.
            qpos = qpos.reshape(-1, qpos.shape[2])
        else:
            if env_index < 0 or env_index >= qpos.shape[1]:
                raise ValueError(f"env_index={env_index} out of range for qpos shape {qpos.shape}")
            qpos = qpos[:, env_index, :]
    elif qpos.ndim != 2:
        raise ValueError(f"expected qpos shape (T, 43) or (T, N, 43), got {qpos.shape}")

    if qpos.shape[1] != 43:
        raise ValueError(f"expected flattened qpos width 43, got {qpos.shape[1]} from shape {qpos.shape}")
    if qpos.shape[0] < 2:
        raise ValueError(f"need at least 2 frames, got {qpos.shape[0]}")
    return np.asarray(qpos, dtype=np.float32)


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        if "qpos" not in data.files:
            raise KeyError(f"{args.input} does not contain qpos")
        raw_qpos_shape = list(data["qpos"].shape)
        qpos = extract_qpos(data["qpos"], args.env_index, args.layout)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, qpos=qpos, fps=np.asarray(args.fps, dtype=np.float32))

    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "raw_qpos_shape": raw_qpos_shape,
                "qpos_shape": list(qpos.shape),
                "fps": args.fps,
                "env_index": args.env_index,
                "layout": args.layout,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
