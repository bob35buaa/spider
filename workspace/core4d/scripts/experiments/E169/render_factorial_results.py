#!/usr/bin/env python3
"""Render E169 full trajectories locally and optionally build 8-cell montages."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "E168"))
from render_a100_cem_videos import render_row, repo_path  # noqa: E402


REPO = Path(__file__).resolve().parents[5]
DEFAULT_MANIFEST = REPO / "workspace/core4d/results/E169/manifests/factorial_analysis_manifest.tsv"
CELL_ORDER = ("B0", "P", "R", "G", "PR", "PG", "RG", "PRG")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def montage(case_id: str, rows: list[dict[str, str]], output: Path, overwrite: bool) -> bool:
    by_cell = {row["cell_id"]: repo_path(row["video"]) for row in rows}
    missing = [cell for cell in CELL_ORDER if cell not in by_cell or not by_cell[cell].is_file()]
    if missing:
        print(f"[montage-not-ready] {case_id}: missing={','.join(missing)}")
        return False
    if output.is_file() and not overwrite:
        print(f"[montage-existing] {case_id} -> {output}")
        return True
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y", "-nostdin"]
    for cell in CELL_ORDER:
        command += ["-i", str(by_cell[cell])]
    filters = []
    for index, cell in enumerate(CELL_ORDER):
        filters.append(
            f"[{index}:v]scale=640:360:force_original_aspect_ratio=decrease,"
            f"pad=640:360:(ow-iw)/2:(oh-ih)/2:black,setsar=1,"
            f"drawtext=text='{cell}':x=16:y=14:fontsize=28:fontcolor=white:"
            f"box=1:boxcolor=black@0.65[v{index}]"
        )
    layout = "|".join(
        ("0_0", "w0_0", "w0+w1_0", "w0+w1+w2_0", "0_h0", "w0_h0", "w0+w1_h0", "w0+w1+w2_h0")
    )
    filters.append("".join(f"[v{i}]" for i in range(8)) + f"xstack=inputs=8:layout={layout}:fill=black[out]")
    command += [
        "-filter_complex", ";".join(filters), "-map", "[out]", "-an",
        "-c:v", "libx264", "-crf", "22", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-shortest", str(output),
    ]
    subprocess.run(command, check=True)
    print(f"[montage-ok] {case_id} -> {output}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("available", "full"), default="available")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cases", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--montage", action="store_true")
    args = parser.parse_args()
    rows = read_rows(repo_path(args.manifest))
    selectors = set(args.cases)
    selected = [
        row for row in rows
        if row["cell_id"] != "B0"
        and (not selectors or row["case_id"] in selectors or row["variant"] in selectors)
    ]
    if args.limit:
        selected = selected[: args.limit]
    counts: Counter[str] = Counter()
    for row in selected:
        needed = [repo_path(row[key]) for key in ("outdir_npz", "config_act", "scene_act", "trajectory")]
        if not all(path.is_file() for path in needed):
            counts["not_ready"] += 1
            print(f"[not-ready] {row['variant']}")
            continue
        output = repo_path(row["video"])
        if output.is_file() and not args.overwrite:
            counts["existing"] += 1
            continue
        if args.dry_run:
            counts["ready"] += 1
            print(f"[ready] {row['variant']} -> {output}")
            continue
        try:
            frames, fps = render_row(row, out_path=output, max_frames=args.max_frames)
            counts["rendered"] += 1
            print(f"[render-ok] {row['variant']}: frames={frames} fps={fps}")
        except Exception as exc:
            counts["failed"] += 1
            print(f"[render-failed] {row['variant']}: {exc}", file=sys.stderr)
    if args.montage and not args.dry_run:
        for case_id in sorted({row["case_id"] for row in rows}):
            case_rows = [row for row in rows if row["case_id"] == case_id]
            output = REPO / f"workspace/core4d/results/E169/render/full/montage/{case_id}_8cell.mp4"
            try:
                counts["montage_ok" if montage(case_id, case_rows, output, args.overwrite) else "montage_not_ready"] += 1
            except Exception as exc:
                counts["montage_failed"] += 1
                print(f"[montage-failed] {case_id}: {exc}", file=sys.stderr)
    print("summary " + " ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    if counts["failed"] or counts["montage_failed"]:
        return 1
    if args.stage == "full" and counts["not_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
