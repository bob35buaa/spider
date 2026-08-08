#!/usr/bin/env python3
"""E194: build 4-cell (A0/G1/G2/G3) keyframe montages for the key cases.

For each key case, extracts 4 timepoints (grasp / lift-peak / carry / place) from
each arm's replay MP4 and tiles them into a labelled grid (rows = arms, cols =
phases). This is the plan's mandatory visual comparison — is the box back at the
reference height (C1), does the far end still droop (C3), is the hand holding vs
merely resting (C4).

A0 MP4s come from the landed E172/E173 renders; G1/G2/G3 from render_E194_arms.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e194_common as C  # noqa: E402

# case_id -> A0 baseline MP4 (already rendered by E172/E173)
A0_MP4 = {
    "box024_20231011_026_p1": "workspace/core4d/results/E173/s6_downstream/render/full/E173_box024_20231011_026_p1_PRG_full.mp4",
    "box004_20231003_2_082_p1": "workspace/core4d/results/E172/s6_downstream/render/full/E172_box004_20231003_2_082_p1_PRG_full.mp4",
}
KEY_CASES = ["box024_20231011_026_p1", "box004_20231003_2_082_p1"]
PHASES = [("grasp", 0.15), ("lift", 0.40), ("carry", 0.65), ("place", 0.90)]
E194_RENDER = "workspace/core4d/results/E194/s6_downstream/render/full"


def duration_s(mp4: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(mp4)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip() or 0.0)


def extract(mp4: Path, t: float, dst: Path) -> bool:
    try:
        subprocess.run(["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(mp4),
                        "-frames:v", "1", str(dst)],
                       capture_output=True, check=True)
        return dst.is_file()
    except subprocess.CalledProcessError:
        return False


def arm_mp4(case_id: str, arm: str) -> Path | None:
    if arm == "A0":
        p = C.repo_path(A0_MP4[case_id]) if case_id in A0_MP4 else None
    else:
        p = C.repo_path(f"{E194_RENDER}/E194_{case_id}_{arm}_full.mp4")
    return p if p and p.is_file() else None


def build_montage(case_id: str, out_png: Path) -> bool:
    from PIL import Image, ImageDraw  # local import; pillow ships with imageio pipeline

    arms = ["A0", "G1", "G2", "G3"]
    with tempfile.TemporaryDirectory() as td:
        tiles: dict[tuple[str, str], Path] = {}
        cell_w = cell_h = None
        for arm in arms:
            mp4 = arm_mp4(case_id, arm)
            if mp4 is None:
                continue
            dur = duration_s(mp4)
            for phase, frac in PHASES:
                dst = Path(td) / f"{arm}_{phase}.png"
                if extract(mp4, max(0.0, dur * frac), dst):
                    tiles[(arm, phase)] = dst
                    if cell_w is None:
                        im = Image.open(dst)
                        cell_w, cell_h = im.size
        if cell_w is None:
            print(f"[montage] {case_id}: no frames extracted (no MP4s yet)", file=sys.stderr)
            return False

        pad, label = 6, 26
        cols, rows = len(PHASES), len(arms)
        W = label + cols * (cell_w + pad) + pad
        H = label + rows * (cell_h + pad) + pad
        canvas = Image.new("RGB", (W, H), (18, 18, 22))
        draw = ImageDraw.Draw(canvas)
        for ci, (phase, _) in enumerate(PHASES):
            draw.text((label + ci * (cell_w + pad) + cell_w // 2 - 15, 6), phase, fill=(230, 230, 230))
        for ri, arm in enumerate(arms):
            y = label + ri * (cell_h + pad) + cell_h // 2
            draw.text((6, y), arm, fill=(120, 220, 160) if arm != "A0" else (220, 180, 120))
            for ci, (phase, _) in enumerate(PHASES):
                tile = tiles.get((arm, phase))
                x = label + ci * (cell_w + pad) + pad
                yy = label + ri * (cell_h + pad) + pad
                if tile is not None:
                    canvas.paste(Image.open(tile).resize((cell_w, cell_h)), (x, yy))
        out_png.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_png)
        print(f"[montage] {case_id} -> {C.rel(out_png)} ({len(tiles)}/{rows*cols} cells)")
        return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", default=KEY_CASES)
    parser.add_argument("--out-dir", default=f"{E194_RENDER}/keyframes")
    args = parser.parse_args()
    ok = 0
    for case_id in args.cases:
        out = C.repo_path(args.out_dir) / f"{case_id}_A0_G1_G2_G3_4cell.png"
        if build_montage(case_id, out):
            ok += 1
    print(f"[keyframes] built {ok}/{len(args.cases)} montages")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
