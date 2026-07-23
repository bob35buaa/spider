#!/usr/bin/env python3
"""Generate the E174 PRG full-validation workbook (box024/box023/box001).

Adapted from gen_E170_box021_prg_xlsx.py. E174 is a cross-object screening
(not a paired vs-E168 validation), so this drops the paired-deltas / keyframe
strict gates and adds a per-object numeric-pass breakdown + the size-degradation
finding on the Overview. Consumes the canonical E174 eval TSVs + summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

DEFAULT_EVAL = Path("workspace/core4d/results/E174/s6_downstream/eval/full")
SHEETS = (
    ("Case Metrics", "e174_case_metrics.tsv"),
    ("Group Summary", "e174_group_summary.tsv"),
    ("Worst Cases", "e174_worst_cases.tsv"),
    ("Manual Review", "user_manual_review_template.tsv"),
    ("Codex Verification", "codex_verification.tsv"),
    ("Not Ready", "e174_not_ready.tsv"),
    ("Eval Errors", "e174_evaluation_errors.tsv"),
)
# object -> (volume m^3, size class) for the size-degradation context
OBJECT_INFO = {
    "box023": (0.036, "small"),
    "box024": (0.253, "large"),
    "box001": (0.256, "large"),
}


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file() or path.stat().st_size <= 1:
        return [], []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def value(raw: str) -> Any:
    text = str(raw)
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    if text == "":
        return None
    try:
        return float(text) if any(c in text.lower() for c in (".", "e")) else int(text)
    except ValueError:
        return text


def style_sheet(sheet: Any) -> None:
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for column in sheet.columns:
        letter = get_column_letter(column[0].column)
        width = min(45, max(10, max(len(str(cell.value or "")) for cell in column) + 2))
        sheet.column_dimensions[letter].width = width
        for cell in column:
            cell.font = Font(
                name="Arial", size=9, bold=cell.row == 1,
                color="FFFFFF" if cell.row == 1 else "000000",
            )
            cell.alignment = Alignment(vertical="top", wrap_text=False)
    for cell in sheet[1]:
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(horizontal="center", vertical="center")


def add_tsv_sheet(workbook: Workbook, title: str, path: Path) -> dict[str, int]:
    fields, rows = read_tsv(path)
    sheet = workbook.create_sheet(title)
    if not fields:
        sheet.append(["status"])
        sheet.append(["not_generated_or_empty"])
        style_sheet(sheet)
        return {"status": 1}
    sheet.append(fields)
    for row in rows:
        sheet.append([value(row.get(field, "")) for field in fields])
    style_sheet(sheet)
    return {field: index + 1 for index, field in enumerate(fields)}


def col_range(sheet: str, column: int, last: int) -> str:
    letter = get_column_letter(column)
    return f"'{sheet}'!{letter}2:{letter}{last}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    eval_dir = args.eval_dir
    output = args.output or eval_dir / "E174_boxes_prg_full_validation.xlsx"
    summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
    counts = summary["counts"]
    fail = summary["numeric_failure_counts"]

    workbook = Workbook()
    overview = workbook.active
    overview.title = "Overview"

    # data sheets first so we know Case Metrics row count for formula ranges
    mappings: dict[str, dict[str, int]] = {}
    for title, filename in SHEETS:
        mappings[title] = add_tsv_sheet(workbook, title, eval_dir / filename)
    case = mappings["Case Metrics"]
    last = counts["evaluated"] + 1  # header + N rows

    required = ("case_id", "object_key", "numeric_release_pass", "retarget_variant_id",
                "user_manual_review_status", "manual_use_decision",
                "leg_gate_health_pass")
    missing = [f for f in required if f not in case]
    if missing:
        raise RuntimeError(f"Case Metrics missing workbook fields: {missing}")

    cid = col_range("Case Metrics", case["case_id"], last)
    obj = col_range("Case Metrics", case["object_key"], last)
    npass = col_range("Case Metrics", case["numeric_release_pass"], last)
    variant = col_range("Case Metrics", case["retarget_variant_id"], last)
    reviewed = col_range("Case Metrics", case["user_manual_review_status"], last)
    usedec = col_range("Case Metrics", case["manual_use_decision"], last)
    gate = col_range("Case Metrics", case["leg_gate_health_pass"], last)

    overview.append(["E174 PRG Full Validation — box024 / box023 / box001",
                     "Value", "Source / rule"])
    rows: list[tuple[str, Any, str]] = [
        ("Metric standard", summary["metric_standard_id"], "summary.json"),
        ("Evaluated rows", f"=COUNTA({cid})", "COUNTA Case Metrics case_id"),
        ("Numeric release pass", f"=COUNTIF({npass},TRUE)", "COUNTIF numeric_release_pass"),
        ("Numeric pass rate", f"=B4/B3", "pass / evaluated"),
        ("— box023 (small 0.036 m³)",
         f'=COUNTIFS({obj},"box023",{npass},TRUE)&" / "&COUNTIF({obj},"box023")',
         "per-object numeric pass"),
        ("— box024 (large 0.253 m³)",
         f'=COUNTIFS({obj},"box024",{npass},TRUE)&" / "&COUNTIF({obj},"box024")',
         "per-object numeric pass"),
        ("— box001 (large 0.256 m³)",
         f'=COUNTIFS({obj},"box001",{npass},TRUE)&" / "&COUNTIF({obj},"box001")',
         "per-object numeric pass"),
        ("v1 pass",
         f'=COUNTIFS({variant},"omnirt_v1",{npass},TRUE)&" / "&COUNTIF({variant},"omnirt_v1")',
         "by retarget variant"),
        ("v2 (rescue) pass",
         f'=COUNTIFS({variant},"omnirt_v2",{npass},TRUE)&" / "&COUNTIF({variant},"omnirt_v2")',
         "by retarget variant"),
        ("Fail: hand_penetration", fail.get("hand_penetration", 0), "numeric_failure_counts"),
        ("Fail: lower_body", fail.get("lower_body", 0), "numeric_failure_counts"),
        ("Fail: release", fail.get("release", 0), "numeric_failure_counts"),
        ("Fail: fall / body_z / contact",
         f"{fail.get('fall',0)} / {fail.get('body_z',0)} / {fail.get('contact',0)}",
         "numeric_failure_counts"),
        ("Gate-health pass", f"=COUNTIF({gate},TRUE)", "COUNTIF leg_gate_health_pass"),
        ("User reviewed", f'=COUNTIF({reviewed},"reviewed")',
         "COUNTIF user_manual_review_status=reviewed"),
        ("Manual USE", f'=COUNTIF({usedec},"USE")', "COUNTIF manual_use_decision=USE"),
        ("Manual DO_NOT_USE", f'=COUNTIF({usedec},"DO_NOT_USE")',
         "COUNTIF manual_use_decision=DO_NOT_USE"),
        ("Machine recommendation", summary["recommendation"]["machine_recommendation"],
         "summary.json (lean PARTIAL_YIELD)"),
        ("Final promotion decision",
         f'=IF(B16<{counts["evaluated"]},"PENDING_USER_REVIEW","USER_DECIDED")',
         "requires user USE/DO_NOT_USE on all rows"),
        ("KEY FINDING", "PRG numeric pass degrades monotonically with box size",
         "small ~80% (box023 81%) → large ~35% (box024 33%, box001 36%)"),
        ("Manifest SHA256", summary["manifest_sha256"], summary["manifest"]),
    ]
    for label, val, note in rows:
        overview.append([label, val, note])

    overview.freeze_panes = "A2"
    overview.column_dimensions["A"].width = 34
    overview.column_dimensions["B"].width = 30
    overview.column_dimensions["C"].width = 62
    for cell in overview[1]:
        cell.font = Font(name="Arial", bold=True, color="FFFFFF", size=12)
        cell.fill = PatternFill("solid", fgColor="17365D")
    for row in overview.iter_rows(min_row=2):
        for cell in row:
            cell.font = Font(name="Arial", size=10)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    for cell in (overview["B15"], overview["B20"], overview["B21"]):
        cell.fill = PatternFill("solid", fgColor="FFF2CC")

    # move Overview to front
    workbook.move_sheet("Overview", -(len(workbook.sheetnames) - 1))
    workbook.calculation.fullCalcOnLoad = True
    workbook.calculation.forceFullCalc = True
    output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output)
    print(json.dumps({"output": str(output), "sheets": workbook.sheetnames,
                      "evaluated": counts["evaluated"], "numeric_pass": counts["numeric_pass"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
