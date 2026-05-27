#!/usr/bin/env python3
"""Index E028b online rollout videos and create contact sheets."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
RESULTS = REPO / "workspace/core4d_collab_retarget/results/E028b_anchor_refit"
MANIFEST = RESULTS / "manifest.tsv"
COMPARISON = RESULTS / "comparison.csv"
ONLINE_DIR = RESULTS / "online_video"


def read_manifest() -> dict[str, dict[str, str]]:
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return {row["variant"]: row for row in csv.DictReader(f, delimiter="\t")}


def read_comparison() -> dict[str, dict[str, str]]:
    if not COMPARISON.is_file():
        return {}
    with COMPARISON.open("r", encoding="utf-8") as f:
        return {row["variant"]: row for row in csv.DictReader(f)}


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=REPO, text=True).strip()


def make_sheet(video: Path, sheet: Path, *, force: bool) -> None:
    if sheet.is_file() and sheet.stat().st_size > 0 and not force:
        return
    sheet.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            "fps=1,scale=320:-1,tile=5x2",
            "-frames:v",
            "1",
            str(sheet),
        ],
        cwd=REPO,
        check=True,
    )


def probe(video: Path) -> str:
    try:
        return run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,nb_frames",
                "-of",
                "csv=p=0",
                str(video),
            ]
        )
    except Exception as exc:
        return f"probe_failed:{exc}"


def fmt(row: dict[str, str], key: str, digits: int = 3) -> str:
    try:
        return f"{float(row[key]):.{digits}f}"
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variants", nargs="*")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = read_manifest()
    comparison = read_comparison()
    selected = args.variants or list(manifest.keys())
    lines = [
        "# E028b Online Video Evaluation",
        "",
        "Videos are direct `run_mjwp.py` online rollout outputs.",
        "",
        "| Variant | Face | Anchor Review | Ratio vs E028 | Video | Sheet | ffprobe | Epos m | Erot deg | Contact 5cm % | Deep pen % | Leg % | Fall | Diagnosis |",
        "|---------|------|---------------|---------------|-------|-------|---------|--------|----------|--------------|------------|-------|------|-----------|",
    ]
    for variant in selected:
        if variant not in manifest:
            raise ValueError(f"Unknown E028b variant: {variant}")
        video = ONLINE_DIR / f"{variant}.mp4"
        sheet = ONLINE_DIR / f"{variant}_sheet.jpg"
        if not video.is_file():
            lines.append(
                f"| `{variant}` | {manifest[variant]['anchor_face']} | {manifest[variant]['anchor_face_review']} | "
                f"{manifest[variant].get('anchor_distance_ratio_vs_e028', '')} | missing | missing |  |  |  |  |  |  |  | missing_video |"
            )
            continue
        make_sheet(video, sheet, force=args.force)
        row = comparison.get(variant, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{variant}`",
                    manifest[variant]["anchor_face"],
                    manifest[variant]["anchor_face_review"],
                    manifest[variant].get("anchor_distance_ratio_vs_e028", ""),
                    f"`{video.relative_to(REPO)}`",
                    f"`{sheet.relative_to(REPO)}`",
                    probe(video),
                    fmt(row, "paper_object_Epos_case_m"),
                    fmt(row, "paper_object_Erot_case_deg", 1),
                    fmt(row, "paper_omniretarget_contact_preservation_5cm_pct", 1),
                    fmt(row, "paper_omniretarget_robot_object_deep_penetration_duration_pct", 1),
                    fmt(row, "case_window_sim_leg_box_interference_frames_pct", 1),
                    row.get("E028b_robot_fall_detected", ""),
                    row.get("E028b_diagnostic_class", ""),
                ]
            )
            + " |"
        )
    ONLINE_DIR.mkdir(parents=True, exist_ok=True)
    out = ONLINE_DIR / "online_video_eval.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
