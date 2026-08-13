#!/usr/bin/env python3
"""Extract grasp/lift/carry/place 4-phase contact sheets from E198 4-cell MP4s.

Each 4-cell MP4 already tiles A0/G1/A2/G1+A2 (2x2). This stacks four phase frames
(≈8/35/62/90%) vertically with a phase label, producing one contact sheet PNG per
case for visual review, and a visual_review.tsv index.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import e198_common as C  # noqa: E402

RENDER = C.RESULTS_E198 / "s6_downstream/render/full_factorial"
SHEETS = RENDER / "keyframes"
PHASES = [("grasp", 0.08), ("lift", 0.35), ("carry", 0.62), ("place", 0.90)]


def contact_sheet(mp4: Path, scale: float = 0.5) -> np.ndarray:
    reader = imageio.get_reader(mp4)
    n = reader.count_frames()
    rows = []
    for name, frac in PHASES:
        fr = reader.get_data(min(n - 1, int(frac * n)))
        im = Image.fromarray(fr)
        im = im.resize((int(im.width * scale), int(im.height * scale)))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 70, 16], fill=(10, 60, 10))
        d.text((4, 3), name, fill=(255, 255, 255))
        rows.append(np.asarray(im))
    reader.close()
    return np.concatenate(rows, axis=0)


def main() -> int:
    SHEETS.mkdir(parents=True, exist_ok=True)
    mp4s = sorted(RENDER.glob("E198_*_4cell.mp4"))
    index = []
    for mp4 in mp4s:
        case_id = mp4.stem.replace("E198_", "").replace("_4cell", "")
        out = SHEETS / f"{case_id}_4phase.png"
        imageio.imwrite(out, contact_sheet(mp4))
        index.append({"case_id": case_id, "object_key": case_id.split("_")[0],
                      "sheet": C.rel(out), "mp4": C.rel(mp4), "observation": ""})
        print(f"[sheet] {case_id}")
    C.write_tsv(RENDER / "visual_review.tsv", index)
    print(f"[done] {len(index)} contact sheets -> {C.rel(SHEETS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
