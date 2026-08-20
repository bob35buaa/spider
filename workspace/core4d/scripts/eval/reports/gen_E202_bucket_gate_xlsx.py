#!/usr/bin/env python3
"""E202 bucket augmentation gate report -> xlsx.

Self-contained: consumes the E202 eval artifacts (e202_aug_case_metrics.tsv,
e202_aug_orig_deltas.tsv, e202_aug_eval_summary.json) + the E178 canonical orig
baseline (e178_case_metrics.tsv). Emits:
  - summary: overall + per-object aug vs E178 orig (mean/std/worst) + gate rates
             + feasibility distribution (C5/C6)
  - detail:  per aug rollout, 12-gate values + pass flags + delta vs same-case orig

Usage: .venv/bin/python workspace/core4d/scripts/eval/reports/gen_E202_bucket_gate_xlsx.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

REPO = Path(__file__).resolve().parents[5]
EVAL_DIR = REPO / "workspace/core4d/results/E202/s6_downstream/eval/full_augmentation"
CASE_METRICS = EVAL_DIR / "e202_aug_case_metrics.tsv"
DELTAS = EVAL_DIR / "e202_aug_orig_deltas.tsv"
SUMMARY = EVAL_DIR / "e202_aug_eval_summary.json"
E178_CASE_METRICS = REPO / "workspace/core4d/results/E178/s6_downstream/eval/full/e178_case_metrics.tsv"
OUT = EVAL_DIR / "E202_bucket_gate_report.xlsx"

KEY_METRICS = [
    ("track_obj_pos_err_cm_mean", "obj_pos_cm"),
    ("track_obj_ori_err_deg_mean", "obj_ori_deg"),
    ("track_root_pos_err_cm_mean", "root_pos_cm"),
    ("track_eef_pos_err_cm_mean", "eef_pos_cm"),
    ("hand_object_physics_contact_in_mask_frac", "contact"),
    ("hand_object_physics_penetration_3mm_frame_frac", "hand_pen"),
    ("leg_penetration_frac", "leg_pen"),
    ("fall_flag", "fall"),
]
GATES = ["fall", "object_pos", "object_ori", "contact", "hand_penetration", "lower_body"]

HDR = Font(bold=True, color="FFFFFF")
HDR_FILL = PatternFill("solid", fgColor="374151")
GOOD = PatternFill("solid", fgColor="C6EFCE")
BAD = PatternFill("solid", fgColor="FFC7CE")
SUB = PatternFill("solid", fgColor="E5E7EB")


def finite(v: Any) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else math.nan
    except (TypeError, ValueError):
        return math.nan


def read_tsv(p: Path) -> list[dict[str, str]]:
    with p.open(encoding="utf-8", newline="") as s:
        return list(csv.DictReader(s, delimiter="\t"))


def hcell(ws, r, c, v):
    cell = ws.cell(r, c, v)
    cell.font = HDR
    cell.fill = HDR_FILL
    cell.alignment = Alignment(horizontal="center")


def num(ws, r, c, v, fill=None):
    x = finite(v)
    cell = ws.cell(r, c, round(x, 3) if math.isfinite(x) else "")
    if fill:
        cell.fill = fill
    return cell


def write_summary(ws, aug: list[dict], orig: list[dict], summ: dict) -> None:
    d = summ["distribution"]
    ws.cell(1, 1, "E202 bucket 平移增强 — aug vs E178 orig（同 core_metrics）").font = Font(bold=True, size=13)
    ws.cell(2, 1, summ.get("confound_note", "")).font = Font(italic=True, size=9)
    r = 4
    hcell(ws, r, 1, "指标"); hcell(ws, r, 2, "orig mean"); hcell(ws, r, 3, "aug mean")
    hcell(ws, r, 4, "aug std"); hcell(ws, r, 5, "aug worst"); hcell(ws, r, 6, "Δ%")
    r += 1
    for mk, label in KEY_METRICS:
        o = d.get(f"orig_{mk}_mean"); a = d.get(f"aug_{mk}_mean")
        ws.cell(r, 1, label)
        num(ws, r, 2, o); num(ws, r, 3, a)
        num(ws, r, 4, d.get(f"aug_{mk}_std")); num(ws, r, 5, d.get(f"aug_{mk}_worst"))
        if o and finite(o) and abs(finite(o)) > 1e-9 and a is not None:
            pct = (finite(a) - finite(o)) / finite(o) * 100
            fill = BAD if (label == "obj_pos_cm" and pct > 25) else None
            num(ws, r, 6, pct, fill)
        r += 1
    ws.cell(r, 1, "12-gate 通过率"); num(ws, r, 2, d.get("orig_gate_pass_frac")); num(ws, r, 3, d.get("aug_gate_pass_frac"))
    r += 2

    # per-object
    hcell(ws, r, 1, "物体"); hcell(ws, r, 2, "n_aug"); hcell(ws, r, 3, "gate通过")
    hcell(ws, r, 4, "obj_pos aug"); hcell(ws, r, 5, "contact aug"); hcell(ws, r, 6, "fall worst"); hcell(ws, r, 7, "leg_pen worst")
    r += 1
    for obj, sub in d.get("by_object", {}).items():
        ws.cell(r, 1, obj).fill = SUB
        num(ws, r, 2, sub.get("n_aug"))
        gp = sub.get("aug_gate_pass_frac")
        num(ws, r, 3, gp, GOOD if finite(gp) >= 0.5 else BAD)
        num(ws, r, 4, sub.get("aug_track_obj_pos_err_cm_mean_mean"))
        num(ws, r, 5, sub.get("aug_hand_object_physics_contact_in_mask_frac_mean"))
        num(ws, r, 6, sub.get("aug_fall_flag_worst"), BAD if finite(sub.get("aug_fall_flag_worst")) >= 1 else None)
        num(ws, r, 7, sub.get("aug_leg_penetration_frac_worst"))
        r += 1
    r += 1

    # feasibility C6
    hcell(ws, r, 1, "可行性 C6"); hcell(ws, r, 2, "cases"); hcell(ws, r, 3, "trans0"); hcell(ws, r, 4, "trans1"); hcell(ws, r, 5, "trans2"); hcell(ws, r, 6, "built/possible")
    r += 1
    for obj, fv in summ.get("feasibility", {}).items():
        cases = list(fv.values())[0]["cases"] if fv else 0
        built = sum(v["built"] for v in fv.values())
        ws.cell(r, 1, obj).fill = SUB
        num(ws, r, 2, cases)
        for i, k in enumerate(("trans0", "trans1", "trans2")):
            num(ws, r, 3 + i, fv.get(k, {}).get("built"))
        ws.cell(r, 6, f"{built}/{cases * 3}")
        r += 1
    for c in range(1, 8):
        ws.column_dimensions[get_column_letter(c)].width = 16


def write_detail(ws, aug: list[dict], deltas: list[dict]) -> None:
    dmap = {(x["case_id"], x["aug_variant"]): x for x in deltas}
    cols = ["object_key", "case_id", "aug_variant"] + [lbl for _, lbl in KEY_METRICS] + \
           [f"{g}_gate" for g in GATES] + ["all_gates", "failure_modes", "Δobj_pos%"]
    for c, h in enumerate(cols, 1):
        hcell(ws, 1, c, h)
    r = 2
    for row in sorted(aug, key=lambda x: (x["object_key"], x["case_id"], x["aug_variant"])):
        ws.cell(r, 1, row["object_key"]); ws.cell(r, 2, row["case_id"]); ws.cell(r, 3, row["aug_variant"])
        c = 4
        for mk, _ in KEY_METRICS:
            num(ws, r, c, row.get(mk)); c += 1
        for g in GATES:
            v = str(row.get(f"{g}_gate_pass", "")).lower() in ("true", "1")
            ws.cell(r, c, "✓" if v else "✗").fill = GOOD if v else BAD
            c += 1
        allp = str(row.get("all_gates_pass", "")).lower() in ("true", "1")
        ws.cell(r, c, "PASS" if allp else "FAIL").fill = GOOD if allp else BAD; c += 1
        ws.cell(r, c, row.get("numeric_failure_modes", "")); c += 1
        dd = dmap.get((row["case_id"], row["aug_variant"]))
        if dd:
            num(ws, r, c, dd.get("pct_track_obj_pos_err_cm_mean"))
        r += 1
    ws.freeze_panes = "A2"
    for c in range(1, len(cols) + 1):
        ws.column_dimensions[get_column_letter(c)].width = 11 if c > 3 else 26


def main() -> int:
    aug = [r for r in read_tsv(CASE_METRICS) if r.get("aug_variant") != "orig"]
    deltas = read_tsv(DELTAS)
    summ = json.loads(SUMMARY.read_text())
    orig = read_tsv(E178_CASE_METRICS) if E178_CASE_METRICS.is_file() else []

    wb = Workbook()
    write_summary(wb.active, aug, orig, summ)
    wb.active.title = "summary"
    write_detail(wb.create_sheet("detail"), aug, deltas)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"[done] {len(aug)} aug rows -> {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
