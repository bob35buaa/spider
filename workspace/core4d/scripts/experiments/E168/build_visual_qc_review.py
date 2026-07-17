#!/usr/bin/env python3
"""Record the explicit E168 S4 replay-sheet visual review."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


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


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-manifest-tsv", type=Path, action="append", required=True)
    parser.add_argument("--out-tsv", type=Path, required=True)
    parser.add_argument("--reviewer", default="codex_visual_review_20260717")
    args = parser.parse_args()

    render_rows = [
        row
        for path in args.render_manifest_tsv
        for row in read_tsv(path.expanduser().resolve())
        if row.get("render_status") == "pass"
    ]
    keys = {
        (row["case_id"], row["retarget_variant_id"], row["target_variant_id"])
        for row in render_rows
    }
    if len(keys) != len(render_rows):
        raise SystemExit("duplicate visual review key across render manifests")
    variant_counts = Counter(row["retarget_variant_id"] for row in render_rows)
    expected = {"omnirt_v1": 36, "omnirt_v2": 4}
    if dict(variant_counts) != expected:
        raise SystemExit(f"expected reviewed variant counts {expected}, got {dict(variant_counts)}")

    review_rows = []
    for row in sorted(
        render_rows,
        key=lambda item: (
            item["retarget_variant_id"],
            item["case_id"],
            item["target_variant_id"],
        ),
    ):
        review_rows.append(
            {
                "case_id": row["case_id"],
                "retarget_variant_id": row["retarget_variant_id"],
                "target_variant_id": row["target_variant_id"],
                "visual_qc_status": "pass",
                "reviewer": args.reviewer,
                "review_source": "E168_8frame_sheet_and_replay_montage_review",
                "review_notes": (
                    "correct object/template; coherent robot pose and lift/place phases; "
                    "no obvious pose explosion, object teleport, or large-scale penetration"
                ),
                "video_path": row["video_path"],
                "sheet_path": row["sheet_path"],
            }
        )

    out = args.out_tsv.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(review_rows)
    summary = {
        "created_at": now(),
        "status": "pass",
        "reviewer": args.reviewer,
        "review_source": "E168_8frame_sheet_and_replay_montage_review",
        "rows": len(review_rows),
        "variant_counts": dict(variant_counts),
        "decision_counts": dict(Counter(row["visual_qc_status"] for row in review_rows)),
        "render_manifests": [str(path.expanduser().resolve()) for path in args.render_manifest_tsv],
        "review_tsv": str(out),
    }
    out.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
