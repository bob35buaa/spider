#!/usr/bin/env python3
"""Write an explicit CORE4D trim-window manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--trim-start", type=int, required=True)
    parser.add_argument("--trim-frames", type=int, required=True)
    parser.add_argument("--source", default="case-file")
    args = parser.parse_args()

    if args.trim_start < 0:
        raise ValueError(f"trim-start must be non-negative, got {args.trim_start}")
    if args.trim_frames <= 0:
        raise ValueError(f"trim-frames must be positive, got {args.trim_frames}")

    info = {
        "source": args.source,
        "trim_start": args.trim_start,
        "trim_end": args.trim_start + args.trim_frames,
        "trim_frames": args.trim_frames,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(info, f, indent=2)

    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
