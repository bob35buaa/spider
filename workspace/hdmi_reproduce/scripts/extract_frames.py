#!/usr/bin/env python3
"""Extract frames from HDMI visualization video at specified timestamps.

Usage:
    python workspace/hdmi_reproduce/scripts/extract_frames.py \
        workspace/hdmi_reproduce/results/R011/visualization_hdmi.mp4 \
        --times 2 3 4

    # Custom output dir:
    python workspace/hdmi_reproduce/scripts/extract_frames.py \
        workspace/hdmi_reproduce/results/R011/visualization_hdmi.mp4 \
        --times 1 2 3 4 5 --outdir /tmp/frames
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import numpy as np


def extract_frames(
    video_path: str,
    times: list[float],
    outdir: str | None = None,
) -> list[str]:
    """Extract frames at specified times (seconds) from video."""
    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    fps = meta["fps"]
    n_frames = reader.count_frames()
    duration = n_frames / fps

    print(f"Video: {video_path}")
    print(f"  FPS: {fps}, frames: {n_frames}, duration: {duration:.1f}s")

    if outdir is None:
        outdir = str(Path(video_path).parent)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    for t in times:
        frame_idx = int(t * fps)
        if frame_idx >= n_frames:
            print(f"  t={t}s: frame {frame_idx} out of range (max {n_frames-1}), using last")
            frame_idx = n_frames - 1

        frame = reader.get_data(frame_idx)
        fname = f"frame_t{t:.1f}s.png"
        path = str(Path(outdir) / fname)
        imageio.imwrite(path, frame)
        saved.append(path)
        print(f"  t={t:.1f}s -> frame {frame_idx} -> {path}")

    reader.close()

    # Also create a montage if multiple frames
    if len(saved) > 1:
        frames = [imageio.imread(p) for p in saved]
        montage = np.concatenate(frames, axis=1)
        montage_path = str(Path(outdir) / "montage.png")
        imageio.imwrite(montage_path, montage)
        print(f"  Montage ({len(frames)} frames) -> {montage_path}")

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract video frames at timestamps")
    parser.add_argument("video", help="Path to visualization_hdmi.mp4")
    parser.add_argument(
        "--times", "-t", nargs="+", type=float, default=[2.0, 3.0, 4.0],
        help="Timestamps in seconds (default: 2 3 4)",
    )
    parser.add_argument("--outdir", "-o", help="Output directory (default: same as video)")
    args = parser.parse_args()

    extract_frames(args.video, args.times, args.outdir)


if __name__ == "__main__":
    main()
