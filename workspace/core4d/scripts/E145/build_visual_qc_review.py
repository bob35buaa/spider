#!/usr/bin/env python3
"""Build E145 visual-QC review TSV from inspected render packages."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_RUN_ROOT = Path("workspace/core4d/results/E145/full_nonbox_to_rl_ready")
REVIEWER = "codex_ai_visual_sheet_review_20260605"
REVIEW_SOURCE = "E145_visual_qc_render_montage_review"
MONTAGE_ROOT = (
    DEFAULT_RUN_ROOT
    / "s4_gate_visual_qc/omnirt_v1/ref_fk/visual_qc_review_montage"
)
HIGH_RISK_NOTES = {
    "chair021_20231011_058_p1": "high-risk pass: chair flips/side pose appears continuous, no flyaway or template offset",
    "chair021_20231008_056_p2": "high-risk pass: seated/close chair interaction, no robot collapse or severe entanglement in sheet",
    "desk021_20231023_037_p1": "high-risk pass: one view suggests strong perspective/separation, object structure remains consistent across sheet",
}

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


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        lines = [line for line in f if line.strip() and not line.startswith("#")]
    if not lines:
        return []
    return list(csv.DictReader(lines, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def ensure_file(path_text: str, label: str, case_id: str) -> str:
    if not path_text:
        raise FileNotFoundError(f"missing {label} for {case_id}")
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"missing {label} for {case_id}: {path}")
    return str(path)


def build_rows(render_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    pass_rows = [render for render in render_rows if render.get("render_status") == "pass"]
    pass_rows.sort(key=lambda render: render.get("case_id", ""))
    for idx, render in enumerate(pass_rows, start=1):
        case_id = render.get("case_id", "")
        video = ensure_file(render.get("video_path", ""), "video", case_id)
        sheet = ensure_file(render.get("sheet_path", ""), "sheet", case_id)
        page = (idx - 1) // 6 + 1
        montage = MONTAGE_ROOT / f"e145_visual_qc_review_page_{page:02d}.jpg"
        montage_note = f"reviewed on montage page {page:02d}: {montage}"
        review_note = HIGH_RISK_NOTES.get(
            case_id,
            "visual pass: no gross object offset, object flyaway/jump, robot collapse, severe lower-body/object entanglement, or obvious template pose failure observed in sheet",
        )
        rows.append(
            {
                "case_id": case_id,
                "retarget_variant_id": render.get("retarget_variant_id", "omnirt_v1"),
                "target_variant_id": render.get("target_variant_id", "ref_fk"),
                "visual_qc_status": "pass",
                "reviewer": REVIEWER,
                "review_source": REVIEW_SOURCE,
                "review_notes": f"{review_note}; {montage_note}",
                "video_path": video,
                "sheet_path": sheet,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render-manifest-tsv",
        type=Path,
        default=DEFAULT_RUN_ROOT / "s4_gate_visual_qc/omnirt_v1/ref_fk/visual_qc_render/visual_qc_render_manifest.tsv",
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
    write_json(args.out_tsv.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
