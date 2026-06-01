#!/usr/bin/env python3
"""Create or apply S4 visual-QC decisions for target-gate rows."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

SCRIPT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "data_construction_v3")
for _path in (SCRIPT_ROOT / "lib", SCRIPT_ROOT / "state", SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from typing import Any

from common import SCHEMA_VERSION, json_dumps, read_tsv, timestamp, write_json, write_tsv


FIELDS = [
    "stage",
    "case_id",
    "retarget_variant_id",
    "target_variant_id",
    "target_task",
    "object_key",
    "object_name",
    "date",
    "seq",
    "person",
    "person_idx",
    "target_gate_status",
    "visual_qc_status",
    "reviewer",
    "review_source",
    "review_notes",
    "target_scene",
    "scene_act",
    "trajectory",
    "video_path",
    "sheet_path",
    "failure_mode",
    "schema_version",
    "updated_at",
]


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row.get("case_id", ""),
        row.get("retarget_variant_id", "shared") or "shared",
        row.get("target_variant_id", "ref_fk") or "ref_fk",
    )


def read_review_index(path: Path | None) -> dict[tuple[str, str, str], dict[str, str]]:
    if not path or not path.is_file():
        return {}
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in read_tsv(path):
        out[key(row)] = row
    return out


def normalize_status(value: str) -> str:
    text = value.strip().lower()
    if text in {"pass", "review", "reject", "not_run"}:
        return text
    if text in {"ok", "approved", "accept"}:
        return "pass"
    if text in {"fail", "failed", "bad"}:
        return "reject"
    if text in {"pending", "todo", ""}:
        return "review"
    raise ValueError(f"unsupported visual_qc_status: {value}")


def build_rows(
    gate_rows: list[dict[str, str]],
    review_index: dict[tuple[str, str, str], dict[str, str]],
    default_pass_status: str,
    review_source: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in gate_rows:
        gate_status = row.get("target_gate_status", "not_run") or "not_run"
        review = review_index.get(key(row), {})
        if review:
            visual_status = normalize_status(review.get("visual_qc_status", review.get("status", "")))
            source = review.get("review_source", review_source)
            notes = review.get("review_notes", review.get("notes", ""))
            reviewer = review.get("reviewer", "")
        elif gate_status == "pass":
            visual_status = normalize_status(default_pass_status)
            source = review_source
            notes = "machine gate pass; visual QC pending" if visual_status == "review" else ""
            reviewer = ""
        else:
            visual_status = "not_run"
            source = review_source
            notes = "target gate is not pass"
            reviewer = ""
        rows.append(
            {
                "stage": "S4_visual_qc",
                "case_id": row.get("case_id", ""),
                "retarget_variant_id": row.get("retarget_variant_id", ""),
                "target_variant_id": row.get("target_variant_id", "ref_fk"),
                "target_task": row.get("target_task", ""),
                "object_key": row.get("object_key", ""),
                "object_name": row.get("object_name", ""),
                "date": row.get("date", ""),
                "seq": row.get("seq", ""),
                "person": row.get("person", ""),
                "person_idx": row.get("person_idx", ""),
                "target_gate_status": gate_status,
                "visual_qc_status": visual_status,
                "reviewer": reviewer,
                "review_source": source,
                "review_notes": notes,
                "target_scene": row.get("target_scene", ""),
                "scene_act": row.get("scene_act", ""),
                "trajectory": row.get("trajectory", ""),
                "video_path": review.get("video_path", ""),
                "sheet_path": review.get("sheet_path", ""),
                "failure_mode": row.get("failure_mode", ""),
                "schema_version": SCHEMA_VERSION,
                "updated_at": timestamp(),
            }
        )
    rows.sort(key=lambda item: (item["visual_qc_status"], item["case_id"], item["retarget_variant_id"], item["target_variant_id"]))
    return rows


def markdown_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# S4 visual QC summary",
        "",
        f"- rows: `{summary['rows']}`",
        "",
        "## visual QC counts",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for key_, count in summary["visual_qc_counts"].items():
        lines.append(f"| `{key_}` | {count} |")
    lines.extend(["", "## review/reject rows", "", "| case | variant | target | status | notes |", "|---|---|---|---|---|"])
    for row in rows:
        if row["visual_qc_status"] != "pass":
            lines.append(
                f"| `{row['case_id']}` | `{row['retarget_variant_id']}` | `{row['target_variant_id']}` | `{row['visual_qc_status']}` | `{row['review_notes']}` |"
            )
    lines.extend(
        [
            "",
            "说明：visual QC 是 release checklist，不替代机器 target gate；机器 gate 与 visual QC 状态在 registry 中独立记录。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-gate-manifest-tsv", type=Path, required=True)
    parser.add_argument("--review-tsv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--default-pass-status", choices=("review", "pass", "not_run"), default="review")
    parser.add_argument("--review-source", default="visual_qc_manifest")
    args = parser.parse_args()

    gate_rows = read_tsv(args.target_gate_manifest_tsv)
    review_index = read_review_index(args.review_tsv)
    rows = build_rows(gate_rows, review_index, args.default_pass_status, args.review_source)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(out_dir / "visual_qc_manifest.tsv", rows, FIELDS)
    write_json(out_dir / "visual_qc_manifest.json", rows)
    summary = {
        "stage": "S4_visual_qc",
        "created_at": timestamp(),
        "schema_version": SCHEMA_VERSION,
        "rows": len(rows),
        "target_gate_counts": dict(Counter(row["target_gate_status"] for row in rows)),
        "visual_qc_counts": dict(Counter(row["visual_qc_status"] for row in rows)),
        "review_rows": len(review_index),
        "out_dir": str(out_dir),
    }
    write_json(out_dir / "visual_qc_summary.json", summary)
    (out_dir / "visual_qc_summary.md").write_text(markdown_summary(summary, rows), encoding="utf-8")
    print(json_dumps(summary))


if __name__ == "__main__":
    main()
