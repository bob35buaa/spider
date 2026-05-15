#!/usr/bin/env python3
"""E077-only fixed trim: box023 person2 retargeted[42:178].

Do not use this as the general CORE4D preprocessing trim step. The general
source of SPIDER CORE4D references is Holosoma's pipeline, especially
workspace/pipeline/trim_no_contact.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[4]
DEFAULT_IN = (
    REPO
    / "workspace/core4d/results/E077/holosoma_box023_person2/retargeted/"
    / "20231008-045-person2-Box023_with_obj_original.npz"
)
DEFAULT_OUT = (
    REPO
    / "workspace/core4d/results/E077/holosoma_box023_person2/trimmed/"
    / "20231008-045-person2-Box023_with_obj_original.npz"
)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--trim-start", type=int, default=42)
    parser.add_argument("--trim-frames", type=int, default=136)
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    start = args.trim_start
    end = start + args.trim_frames
    if data["qpos"].shape[0] < end:
        raise ValueError(f"Need at least {end} frames, got {data['qpos'].shape[0]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for key in data.files:
        value = data[key]
        if hasattr(value, "shape") and value.ndim > 0 and value.shape[0] == data["qpos"].shape[0]:
            payload[key] = value[start:end]
        else:
            payload[key] = value
    np.savez(args.output, **payload)
    print(f"Wrote {args.output}")
    print(f"slice: [{start}:{end}]")
    print(f"qpos: {data['qpos'].shape} -> {payload['qpos'].shape}")
    if "human_joints" in payload:
        print(f"human_joints: {payload['human_joints'].shape}")
    if "fps" in payload:
        print(f"fps: {int(payload['fps'])}")


if __name__ == "__main__":
    main()
