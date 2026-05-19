#!/usr/bin/env python3
"""Render E016 offline comparison videos and contact sheets.

This is a small wrapper around workspace/hdmi_reproduce/scripts/render_trajectory_video.py
so E016 paths are resolved from the manifest instead of hand-written shell.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
RESULTS = REPO / "workspace/core4d_collab_retarget/results/E016"
MANIFEST = RESULTS / "manifest.tsv"
COMPARISON = RESULTS / "comparison.csv"
VIS_DIR = RESULTS / "visual"
BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
RENDER_SCRIPT = REPO / "workspace/hdmi_reproduce/scripts/render_trajectory_video.py"


def read_manifest() -> dict[str, dict[str, str]]:
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return {row["variant"]: row for row in csv.DictReader(f, delimiter="\t")}


def read_comparison() -> dict[str, dict[str, str]]:
    if not COMPARISON.is_file():
        return {}
    with COMPARISON.open("r", encoding="utf-8", newline="") as f:
        return {row["variant"]: row for row in csv.DictReader(f)}


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=REPO, env=env, check=True)


def render_variant(
    variant: str,
    row: dict[str, str],
    *,
    force: bool,
    width: int,
    height: int,
    fps: int,
) -> tuple[Path, Path]:
    task = row["derived_task"]
    scene_name = row["scene_name"]
    scene = BASE / task / f"{scene_name}.xml"
    kin = BASE / task / "0/trajectory_kinematic.npz"
    phys = RESULTS / f"{variant}.npz"
    out = VIS_DIR / f"{variant}_comparison.mp4"
    sheet_dir = VIS_DIR / f"{variant}_frames"
    sheet = sheet_dir / "sheet.jpg"

    for path in (scene, kin, phys):
        if not path.is_file():
            raise FileNotFoundError(path)

    VIS_DIR.mkdir(parents=True, exist_ok=True)
    sheet_dir.mkdir(parents=True, exist_ok=True)
    if force or not out.is_file() or out.stat().st_size == 0:
        env = dict(**__import__("os").environ, MUJOCO_GL="egl")
        run(
            [
                sys.executable,
                str(RENDER_SCRIPT),
                "--scene",
                str(scene),
                "--kin",
                str(kin),
                "--phys",
                str(phys),
                "--output",
                str(out),
                "--width",
                str(width),
                "--height",
                str(height),
                "--fps",
                str(fps),
            ],
            env=env,
        )
    else:
        print(f"skip existing {out}")

    run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(out),
            "-vf",
            "fps=1,scale=320:-1,tile=4x2",
            "-frames:v",
            "1",
            str(sheet),
        ]
    )
    return out, sheet


def fmt_pct(row: dict[str, str], key: str) -> str:
    try:
        return f"{float(row[key]):.1f}"
    except Exception:
        return ""


def fmt_num(row: dict[str, str], key: str, digits: int = 3) -> str:
    try:
        return f"{float(row[key]):.{digits}f}"
    except Exception:
        return ""


def write_visual_index(
    rendered: list[tuple[str, Path, Path]],
    comparison: dict[str, dict[str, str]],
) -> None:
    lines = [
        "# E016 Visual Evaluation",
        "",
        "Offline side-by-side videos compare kinematic/reference qpos (left) with MJWarp output (right).",
        "",
        "| Variant | Video | Sheet | Epos m | Erot deg | contact 5cm % | deep pen % | leg % | diagnosis |",
        "|---------|-------|-------|--------|----------|--------------|------------|-------|-----------|",
    ]
    for variant, video, sheet in rendered:
        row = comparison.get(variant, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{variant}`",
                    f"`{video.relative_to(REPO)}`",
                    f"`{sheet.relative_to(REPO)}`",
                    fmt_num(row, "paper_object_Epos_case_m"),
                    fmt_num(row, "paper_object_Erot_case_deg", 1),
                    fmt_pct(row, "paper_omniretarget_contact_preservation_5cm_pct"),
                    fmt_pct(
                        row,
                        "paper_omniretarget_robot_object_deep_penetration_duration_pct",
                    ),
                    fmt_pct(row, "case_window_sim_leg_box_interference_frames_pct"),
                    row.get("E016_diagnostic_class", ""),
                ]
            )
            + " |"
        )
    (VIS_DIR / "visual_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variants", nargs="*", help="Variant names. Default: all.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    manifest = read_manifest()
    selected = args.variants or list(manifest.keys())
    comparison = read_comparison()
    rendered: list[tuple[str, Path, Path]] = []
    for variant in selected:
        if variant not in manifest:
            raise ValueError(f"Unknown E016 variant: {variant}")
        video, sheet = render_variant(
            variant,
            manifest[variant],
            force=args.force,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
        rendered.append((variant, video, sheet))
    write_visual_index(rendered, comparison)
    print(f"Wrote {VIS_DIR / 'visual_eval.md'}")


if __name__ == "__main__":
    main()
