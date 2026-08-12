#!/usr/bin/env python3
"""Rebuild the E194 72-case workbook with E196 corrected G1 rows overlaid.

The E194 noPRG/PRG authorities and 72-case universe remain frozen. Only the
29 affected G1 rows are replaced by the E196 fail-closed corrected reruns.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
REPORT_DIR = ROOT / "workspace/core4d/scripts/eval/reports"
EVAL = ROOT / "workspace/core4d/results/E194/s6_downstream/eval/full_g1_expansion"
E196_EVAL = ROOT / "workspace/core4d/results/E196/s6_downstream/eval/full_reference_fix"
sys.path.insert(0, str(REPORT_DIR))
sys.path.insert(0, str(ROOT / "workspace/core4d/scripts/eval/runners"))
sys.path.insert(0, str(ROOT / "workspace/core4d/scripts/experiments/E194"))

import build_E194_three_arm_workbook as book  # noqa: E402
import eval_E194_three_arm_comparison as scorer  # noqa: E402
import e194_g1_expansion_common as C  # noqa: E402


def main() -> int:
    original = C.read_tsv(EVAL / "e194_three_arm_case_metrics.tsv")
    corrected = [r for r in C.read_tsv(E196_EVAL / "e196_reference_fix_case_metrics.tsv")
                 if r.get("arm") == "G1_corrected"]
    if len(original) != 216 or len(corrected) != 29:
        raise SystemExit(f"unexpected inputs: E194={len(original)}, corrected={len(corrected)}")
    corrected_by_case = {r["case_id"]: r for r in corrected}
    if len(corrected_by_case) != 29:
        raise SystemExit("duplicate corrected G1 case IDs")
    rows = []
    replaced = 0
    for row in original:
        if row.get("arm") == "G1" and row["case_id"] in corrected_by_case:
            replacement = dict(corrected_by_case[row["case_id"]])
            replacement["arm"] = "G1"
            replacement["source_exp"] = "E196"
            replacement["execution_source"] = "E196"
            replacement["reused_full"] = "false"
            rows.append(replacement)
            replaced += 1
        else:
            rows.append(dict(row))
    if replaced != 29 or len(rows) != 216:
        raise SystemExit(f"overlay cardinality failed: replaced={replaced}, rows={len(rows)}")

    arms = {arm: {r["case_id"]: r for r in rows if r.get("arm") == arm}
            for arm in ("noPRG", "PRG", "G1")}
    if any(len(group) != 72 for group in arms.values()):
        raise SystemExit({arm: len(group) for arm, group in arms.items()})
    paired = scorer.paired_rows(arms)
    by_object, migrations = scorer.summarize(paired)
    gate_detail, gate_overall = scorer.gate_tables(arms)
    if (len(paired), len(migrations), len(gate_detail), len(gate_overall)) != (144, 1728, 48, 4):
        raise SystemExit("derived comparison cardinality failed")

    authority_path = book.E173_BOX001_USE_AUTHORITY
    review_path = book.E194_G1_REVIEW
    authority_rows = C.read_tsv(authority_path)
    review_rows = C.read_tsv(review_path)
    box001_use_rows = {r["case_id"]: r for r in authority_rows}
    g1_reviews = {r["case_id"]: r for r in review_rows}
    authority_sha = hashlib.sha256(authority_path.read_bytes()).hexdigest()
    review_sha = hashlib.sha256(review_path.read_bytes()).hexdigest()
    payload = {"status": "pass", "metric_standard_id": "core4d-e154-physics-contact-v1",
               "arm_rows": {k: len(v) for k, v in arms.items()}, "paired_rows": len(paired),
               "gate_migration_rows": len(migrations), "corrected_g1_overlay_rows": replaced,
               "source": "E194 three-arm authorities with E196 corrected G1 overlay"}

    wb = book.Workbook()
    wb.calculation.fullCalcOnLoad = True
    wb.calculation.forceFullCalc = True
    wb.calculation.calcMode = "auto"
    book.add_readme(wb, payload, authority_sha, review_sha, corrected_overlay=True, corrected_count=replaced)
    book.add_arm_cases(wb, rows)
    book.add_paired(wb, paired, box001_use_rows, authority_sha, g1_reviews, review_sha)
    book.add_box001_human_review(wb, paired, box001_use_rows, g1_reviews, review_sha, corrected_overlay=True)
    book.add_prg_g1_failure_modes(wb, paired)
    book.add_by_object(wb)
    book.add_gate_summary(wb, migrations)
    book.add_12gate_comparison(wb)
    book.add_12gate_overall(wb)
    book.add_visual_review(wb)
    for ws in wb.worksheets:
        ws.sheet_view.showGridLines = False
        for row in ws.iter_rows():
            for cell in row:
                if cell.value is not None and cell.font.name != "Arial":
                    cell.font = book.Font(name="Arial", size=cell.font.sz or 10,
                                          bold=cell.font.bold, italic=cell.font.italic,
                                          color=cell.font.color)
    output = EVAL / "E194_noPRG_PRG_G1_comparison.xlsx"
    wb.save(output)
    # Keep a lightweight overlay manifest beside the workbook for auditability.
    manifest = EVAL / "e194_corrected_g1_overlay_manifest.json"
    manifest.write_text(json.dumps({**payload, "corrected_cases": sorted(corrected_by_case),
                                    "corrected_source": str(E196_EVAL / "e196_reference_fix_case_metrics.tsv")},
                                   indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
