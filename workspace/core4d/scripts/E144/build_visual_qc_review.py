#!/usr/bin/env python3
"""Build E144 visual-QC review TSV from locally inspected render packages."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_RUN_ROOT = Path("workspace/core4d/results/E144/E144_full_nonbox_raw_contact")
REVIEWER = "codex_local_visual_review_20260605"
REVIEW_SOURCE = "E144_visual_qc_render_sheet_local_review"

FIELDS = [
    "case_id",
    "retarget_variant_id",
    "target_variant_id",
    "visual_qc_status",
    "reviewer",
    "review_source",
    "review_notes",
    "video_path",
    "sheet_path",
]


APPROVED_NOTES = {
    "bucket003_20231018_001_p2": "sheet reviewed: stable bucket, no gross scene offset, robot remains upright",
    "bucket003_20231018_002_p2": "sheet reviewed: stable bucket, no gross scene offset, robot remains upright",
    "bucket003_20231020_064_p2": "sheet reviewed: bucket motion/tilt is coherent, no flyaway or collapse",
    "bucket003_20231020_065_p2": "sheet reviewed: bucket motion/tilt is coherent, no flyaway or collapse",
    "bucket004_20231002_021_p1": "sheet reviewed: lift/carry sequence stable, no object flyaway or robot collapse",
    "bucket004_20231002_021_p2": "sheet reviewed: carry/turn sequence stable, no gross lower-body or object failure",
    "bucket004_20231002_022_p1": "sheet reviewed: lift/place sequence stable, no object flyaway or robot collapse",
    "bucket004_20231002_022_p2": "sheet reviewed: carry sequence stable, no object flyaway or robot collapse",
    "bucket004_20231003_1_012_p1": "sheet reviewed: approach/lift/place sequence stable, no gross scene failure",
    "bucket004_20231003_1_012_p2": "sheet reviewed: approach/lift/place sequence stable, no gross scene failure",
    "bucket009_20231002_056_p2": "sheet reviewed: cylindrical bucket trajectory stable, no flyaway or robot collapse",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_rows(render_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    render_by_case = {row.get("case_id", ""): row for row in render_rows}
    missing = sorted(set(APPROVED_NOTES) - set(render_by_case))
    if missing:
        raise FileNotFoundError(f"missing render rows for approved cases: {missing}")
    for case_id, notes in sorted(APPROVED_NOTES.items()):
        render = render_by_case[case_id]
        if render.get("render_status") != "pass":
            raise ValueError(f"cannot approve visual QC without render pass: {case_id}")
        sheet = render.get("sheet_path", "")
        video = render.get("video_path", "")
        if not sheet or not Path(sheet).is_file():
            raise FileNotFoundError(f"missing sheet for {case_id}: {sheet}")
        if not video or not Path(video).is_file():
            raise FileNotFoundError(f"missing video for {case_id}: {video}")
        rows.append(
            {
                "case_id": case_id,
                "retarget_variant_id": render.get("retarget_variant_id", "omnirt_v1"),
                "target_variant_id": render.get("target_variant_id", "ref_fk"),
                "visual_qc_status": "pass",
                "reviewer": REVIEWER,
                "review_source": REVIEW_SOURCE,
                "review_notes": notes,
                "video_path": video,
                "sheet_path": sheet,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render-manifest-tsv",
        type=Path,
        default=DEFAULT_RUN_ROOT
        / "s4_gate_visual_qc/omnirt_v1/ref_fk/visual_qc_render/visual_qc_render_manifest.tsv",
    )
    parser.add_argument(
        "--out-tsv",
        type=Path,
        default=DEFAULT_RUN_ROOT / "s4_gate_visual_qc/omnirt_v1/ref_fk/visual_qc/visual_qc_review.tsv",
    )
    args = parser.parse_args()

    rows = build_rows(read_tsv(args.render_manifest_tsv))
    write_tsv(args.out_tsv, rows, FIELDS)
    summary = {
        "review_tsv": str(args.out_tsv),
        "rows": len(rows),
        "visual_qc_counts": dict(Counter(row["visual_qc_status"] for row in rows)),
        "reviewer": REVIEWER,
        "review_source": REVIEW_SOURCE,
    }
    print(summary)


if __name__ == "__main__":
    main()
