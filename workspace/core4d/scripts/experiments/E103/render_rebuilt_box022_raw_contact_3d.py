#!/usr/bin/env python3
"""Render E103 Box022 raw fingertip preflight as 3D object-local evidence."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "E099"))

import case_to_raw  # noqa: E402
from render_raw_contact_3d import collect_case_data, render_static, render_turntable  # noqa: E402


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--core4d-root",
        type=Path,
        default=Path("/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real/human_object_motions"),
    )
    parser.add_argument("--no-mp4", action="store_true")
    args = parser.parse_args()

    case_to_raw.CORE4D_ROOT = args.core4d_root
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in read_tsv(args.preflight):
        case = row["target_task"]
        scene = Path(row["source_scene_xml"])
        if not scene.is_file():
            print(f"[SKIP] {case}: missing source_scene {scene}")
            continue
        data = collect_case_data(case, scene)
        if data is None:
            print(f"[SKIP] {case}: missing raw or skipped case")
            continue
        png = args.out_dir / f"{case}_raw_contact_3d_4view.png"
        render_static(data, png)
        mp4 = ""
        if not args.no_mp4:
            mp4_path = args.out_dir / f"{case}_raw_contact_3d_turntable.mp4"
            render_turntable(data, mp4_path)
            mp4 = str(mp4_path)
        rows.append(
            {
                "target_task": case,
                "decision": row.get("decision", ""),
                "reason": row.get("reason", ""),
                "source_scene_xml": str(scene),
                "L_contact": row.get("L_contact", ""),
                "R_contact": row.get("R_contact", ""),
                "png": str(png),
                "mp4": mp4,
            }
        )
        print(f"[VIZ] {case}: {png} {mp4}")

    lines = [
        "# E103 Rebuilt Box022 Raw Contact 3D Review",
        "",
        "Object-local 3D evidence uses the rebuilt clean `box022_person*` source scene collision half-extents.",
        "The expected result for current selected rows is no close fingertip contact, matching `rebuilt_box022_preflight.tsv`.",
        "",
        "| target_task | decision | L/R contact | source_scene | png | video |",
        "|---|---|---:|---|---|---|",
    ]
    for row in rows:
        video = f"[mp4]({Path(row['mp4']).name})" if row["mp4"] else ""
        lines.append(
            f"| `{row['target_task']}` | `{row['decision']}` | {row['L_contact']}/{row['R_contact']} | "
            f"`{Path(row['source_scene_xml']).parent.name}` | [png]({Path(row['png']).name}) | {video} |"
        )
    (args.out_dir / "REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"review -> {args.out_dir / 'REVIEW.md'}")


if __name__ == "__main__":
    main()
