#!/usr/bin/env python3
"""Render E170 new PRG results and build 28 E168-vs-PRG paired videos."""

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
DEFAULT_MANIFEST = REPO / "workspace/core4d/results/E170/s6_downstream/manifests/analysis_manifest.tsv"
PAIR_ROOT = REPO / "workspace/core4d/results/E170/s6_downstream/render/full/paired"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def paired_video(row: dict[str, str], *, overwrite: bool) -> str:
    baseline, prg = repo_path(row["e168_baseline_video"]), repo_path(row["video"])
    if not baseline.is_file() or not prg.is_file():
        return "not_ready"
    output = PAIR_ROOT / f"{row['case_id']}_E168_vs_E170_PRG.mp4"
    if output.is_file() and not overwrite:
        return "existing"
    output.parent.mkdir(parents=True, exist_ok=True)
    filters = (
        "[0:v]scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2:black,setsar=1,"
        "drawtext=text='E168 B0':x=18:y=16:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.65[v0];"
        "[1:v]scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2:black,setsar=1,"
        "drawtext=text='E170 PRG':x=18:y=16:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.65[v1];"
        "[v0][v1]hstack=inputs=2[out]"
    )
    subprocess.run(["ffmpeg", "-y", "-nostdin", "-i", str(baseline), "-i", str(prg), "-filter_complex", filters, "-map", "[out]", "-an", "-c:v", "libx264", "-crf", "22", "-preset", "medium", "-pix_fmt", "yuv420p", "-shortest", str(output)], check=True)
    return "rendered"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stage", choices=("available", "full"), default="available")
    parser.add_argument("--cases", nargs="*", default=[])
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selectors = set(args.cases)
    rows = [row for row in read_rows(repo_path(args.manifest)) if not selectors or row["case_id"] in selectors]
    counts: Counter[str] = Counter()
    for row in rows:
        if row["execution_source"] == "E170":
            required = [repo_path(row[key]) for key in ("outdir_npz", "config_act", "scene_act", "trajectory")]
            if not all(path.is_file() for path in required):
                counts["new_not_ready"] += 1
                continue
            output = repo_path(row["video"])
            if output.is_file() and not args.overwrite:
                counts["new_existing"] += 1
            elif args.dry_run:
                counts["new_ready"] += 1
            else:
                try:
                    render_row(row, out_path=output, max_frames=args.max_frames)
                    counts["new_rendered"] += 1
                except Exception as exc:
                    counts["new_failed"] += 1
                    print(f"[render-failed] {row['case_id']}: {exc}", file=sys.stderr)
        else:
            counts["reuse_reference_ready" if repo_path(row["video"]).is_file() else "reuse_reference_missing"] += 1
        if args.dry_run:
            if repo_path(row["e168_baseline_video"]).is_file() and repo_path(row["video"]).is_file():
                counts["paired_ready"] += 1
            else:
                counts["paired_not_ready"] += 1
        else:
            try:
                counts[f"paired_{paired_video(row, overwrite=args.overwrite)}"] += 1
            except Exception as exc:
                counts["paired_failed"] += 1
                print(f"[paired-failed] {row['case_id']}: {exc}", file=sys.stderr)
    print("summary " + " ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    if counts["new_failed"] or counts["paired_failed"] or counts["reuse_reference_missing"]:
        return 1
    if args.stage == "full" and (counts["new_not_ready"] or counts["paired_not_ready"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
