#!/usr/bin/env python3
"""Infer the Holosoma trim window by matching trimmed qpos to untrimmed qpos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def find_matches(untrimmed: np.ndarray, trimmed: np.ndarray) -> list[int]:
    if untrimmed.ndim != 2 or trimmed.ndim != 2:
        raise ValueError("Expected qpos arrays with shape (T, D)")
    if untrimmed.shape[1] != trimmed.shape[1]:
        raise ValueError(
            f"qpos DOF mismatch: untrimmed={untrimmed.shape}, trimmed={trimmed.shape}"
        )
    if trimmed.shape[0] > untrimmed.shape[0]:
        raise ValueError(
            f"trimmed sequence is longer than untrimmed: {trimmed.shape[0]} > {untrimmed.shape[0]}"
        )

    matches: list[int] = []
    limit = untrimmed.shape[0] - trimmed.shape[0] + 1
    for start in range(limit):
        if np.allclose(untrimmed[start : start + trimmed.shape[0]], trimmed):
            matches.append(start)
    return matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--untrimmed", type=Path, required=True)
    parser.add_argument("--trimmed", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--expect-start", type=int)
    parser.add_argument("--expect-frames", type=int)
    parser.add_argument("--print-tsv", action="store_true")
    args = parser.parse_args()

    untrimmed_npz = np.load(args.untrimmed, allow_pickle=True)
    trimmed_npz = np.load(args.trimmed, allow_pickle=True)
    untrimmed_qpos = untrimmed_npz["qpos"]
    trimmed_qpos = trimmed_npz["qpos"]

    matches = find_matches(untrimmed_qpos, trimmed_qpos)
    if not matches:
        raise RuntimeError(
            f"Could not match trimmed qpos {args.trimmed} inside untrimmed {args.untrimmed}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous trim window for {args.trimmed}: possible starts={matches}"
        )

    trim_start = matches[0]
    trim_frames = int(trimmed_qpos.shape[0])
    trim_end = trim_start + trim_frames

    if args.expect_start is not None and trim_start != args.expect_start:
        raise RuntimeError(
            f"Unexpected trim_start: inferred={trim_start}, expected={args.expect_start}"
        )
    if args.expect_frames is not None and trim_frames != args.expect_frames:
        raise RuntimeError(
            f"Unexpected trim_frames: inferred={trim_frames}, expected={args.expect_frames}"
        )

    info = {
        "untrimmed": str(args.untrimmed),
        "trimmed": str(args.trimmed),
        "untrimmed_frames": int(untrimmed_qpos.shape[0]),
        "trimmed_frames": trim_frames,
        "trim_start": trim_start,
        "trim_end": trim_end,
        "trim_frames": trim_frames,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(info, f, indent=2)

    if args.print_tsv:
        print(f"{trim_start}\t{trim_frames}")
    else:
        print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
