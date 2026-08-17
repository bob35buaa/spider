#!/usr/bin/env python3
"""E201 · build the VLM pre-screen request JSONL for call_api_imitate_redaccel.

One line per rollout. Each request_info:
  images   : the rollout's frame paths (temporal order, from frames.json)
  messages : [user(prompt + one "[帧k] <image>" line per frame, in order),
              assistant("")]  -- the trailing assistant is the GT placeholder the
              API strips; messages must be even. <image> tokens are consumed in
              order against images.
  metadata : provenance (exp/case_id/variant/layer/... ) echoed to the output.

Usage:
    .venv/bin/python workspace/core4d/scripts/experiments/E201/build_vlm_requests.py --exp E199
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
SCRIPT_DIR = Path(__file__).resolve().parent
VLM_DIR = REPO / "workspace/core4d/results/E201/vlm_review"
PROMPT_FILE = SCRIPT_DIR / "vlm_review_prompt.txt"


def read_tsv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E199")
    ap.add_argument("--queue", type=Path, default=None)
    ap.add_argument("--frames-root", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    prompt = PROMPT_FILE.read_text(encoding="utf-8").strip()
    queue = args.queue or (VLM_DIR / f"{args.exp}_review_queue.tsv")
    frames_root = args.frames_root or (VLM_DIR / "frames" / args.exp)
    out = args.out or (VLM_DIR / "requests" / f"{args.exp}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = read_tsv(queue)
    n_written = n_skip = 0
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            key = f"{row['case_id']}#{row['aug_variant']}"
            fmeta = frames_root / key / "frames.json"
            if not fmeta.is_file():
                n_skip += 1
                print(f"[skip] no frames for {key}", file=sys.stderr)
                continue
            meta = json.loads(fmeta.read_text())
            frame_paths = meta["frame_paths"]
            if not frame_paths:
                n_skip += 1
                continue
            # prompt + one labelled <image> line per frame (temporal order)
            frame_block = "\n".join(f"[帧{j}] <image>" for j in range(len(frame_paths)))
            content = (
                f"{prompt}\n\n以下是按时间先后顺序的 {len(frame_paths)} 帧："
                f"\n{frame_block}"
            )
            request = {
                "images": frame_paths,
                "messages": [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": ""},  # GT placeholder (stripped)
                ],
                "metadata": {
                    "exp": row["exp"], "object_key": row["object_key"],
                    "case_id": row["case_id"], "aug_variant": row["aug_variant"],
                    "group": row["group"], "layer": row["layer"],
                    "family_flag": row["family_flag"],
                    "wide_failed": row["wide_failed"], "narrow_failed": row["narrow_failed"],
                    "n_frames": len(frame_paths),
                },
            }
            fh.write(json.dumps(request, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[done] wrote {out} ({n_written} requests; skipped {n_skip} without frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
