#!/usr/bin/env python3
"""E216 chair/desk workbook: 0.1m-vs-0.2m comparison + E201 14-gate funnel.

chair/desk (Phase 2) scored aug-only (no orig rows), so unlike the box/bucket
workbook there is no orig-vs-aug sheet.  Instead:
  * Compare_0p1m_0p2m — per-object + pooled: 6-gate pass 0.1m vs 0.2m (E213 same
    selected arm), falls, metric deltas; plus the absolute 0.1m pass column.
  * Funnel           — the 59 trans rollouts re-scored through the E201 14-gate
    wide/narrow funnel (the downstream pre-filter caliber), layer distribution
    per object.
  * Detail           — per (case, variant): 4 hard + 10 banded gates + layer + g.

Usage:
    .venv/bin/python .../E216/gen_E216cd_trans_funnel_xlsx.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E201"))
import funnel_config as FC  # noqa: E402

EVAL_DIR = REPO / "workspace/core4d/results/E216/chair_desk/s6_downstream/eval/aug"
ROLLOUT_TSV = EVAL_DIR / "e216cd_trans_rollout.tsv"
COMPARE_JSON = EVAL_DIR / "e216cd_0p1m_vs_0p2m_summary.json"
DEFAULT_OUT = EVAL_DIR / "E216cd_trans_0p1m_vs_0p2m.xlsx"

HARD, BANDED = FC.HARD_GATES, FC.BANDED_GATES
GREEN = PatternFill("solid", fgColor="C6EFCE")
RED = PatternFill("solid", fgColor="FFC7CE")
NAVY = PatternFill("solid", fgColor="1F3864")
HEAD = Font(bold=True, color="FFFFFF")
LAYER_FILL = {"L1_reject": PatternFill("solid", fgColor="C00000"),
              "L2_review": PatternFill("solid", fgColor="FFD966"),
              "L3_review": PatternFill("solid", fgColor="F4B183")}


def finite(v: Any) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else math.nan
    except (TypeError, ValueError):
        return math.nan


def read_tsv(p: Path) -> list[dict[str, str]]:
    with p.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def classify(row: dict[str, Any]) -> dict[str, str]:
    h_ok, h_f = FC.hard_gate_result(row)
    n_ok, n_f = FC.banded_gate_result(row, "narrow")
    w_ok, w_f = FC.banded_gate_result(row, "wide")
    layer = "L1_reject" if (not h_ok or not w_ok) else ("L2_review" if not n_ok else "L3_review")
    return {"hard_failed": ",".join(h_f), "narrow_failed": ",".join(n_f),
            "wide_failed": ",".join(w_f), "layer": layer}


def _c(ws, r, c, v, fill=None, font=None):
    cell = ws.cell(row=r, column=c, value=v)
    cell.alignment = Alignment(horizontal="center", vertical="center")
    if fill:
        cell.fill = fill
    if font:
        cell.font = font
    return cell


def write_compare(ws, cmp: dict[str, Any]) -> None:
    ws.append(["E216 chair/desk 核心对照：0.1m vs 0.2m（E213 同选定臂、同评估器、6-gate）"])
    ws.append([cmp.get("note", "")])
    ws.append([f"n_paired={cmp.get('n_paired','')}  n_scored_0.1m={cmp.get('n_scored_0p1m','')}  failed={cmp.get('n_failed','')}"])
    ws.append([])
    heads = ["object", "n", "0.1m pass", "0.2m pass", "Δpass(pp)", "0.1m abs pass",
             "fall 0.1", "fall 0.2", "Δobj_pos(cm)", "Δobj_ori", "Δcontact", "Δleg_pen"]
    hr = ws.max_row + 1
    for c, h in enumerate(heads, 1):
        _c(ws, hr, c, h, NAVY, HEAD)
    comp = cmp.get("compare_by_object", {})
    absd = cmp.get("absolute_0p1m_by_object", {})
    order = [k for k in comp if k != "POOLED"] + (["POOLED"] if "POOLED" in comp else [])
    for obj in order:
        c = comp[obj]
        a = absd.get(obj, {})
        dpp = c.get("pass_rate_delta", 0) * 100
        r = ws.max_row + 1
        _c(ws, r, 1, obj, None, Font(bold=(obj == "POOLED")))
        _c(ws, r, 2, c.get("n", ""))
        _c(ws, r, 3, round(c.get("pass_rate_0p1m", 0), 3))
        _c(ws, r, 4, round(c.get("pass_rate_0p2m", 0), 3))
        _c(ws, r, 5, round(dpp, 1), GREEN if dpp >= -1 else RED)
        _c(ws, r, 6, round(a.get("pass_rate", 0), 3) if a else "")
        _c(ws, r, 7, c.get("falls_0p1m", ""))
        _c(ws, r, 8, c.get("falls_0p2m", ""))
        _c(ws, r, 9, round(c.get("delta_track_obj_pos_err_cm_mean_mean", 0), 3))
        _c(ws, r, 10, round(c.get("delta_track_obj_ori_err_deg_mean_mean", 0), 3))
        _c(ws, r, 11, round(c.get("delta_hand_object_physics_contact_in_mask_frac_mean", 0), 3))
        _c(ws, r, 12, round(c.get("delta_leg_penetration_frac_mean", 0), 3))
    ws.append([])
    ws.append(["结论：chair/desk 的 0.1m 显著劣于 0.2m（pooled −25.4pp，desk007 −66.7pp），与 box/bucket 相反。"])
    ws.append(["机制：失败全在 hand/leg 穿透；0.1m 身-家具净空更小→腿脚穿入桌子（osmesa 视觉证实）。0 fall。"])
    for i, w in enumerate([12, 5, 10, 10, 11, 12, 8, 8, 13, 11, 11, 10], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A6"


def write_funnel(ws, rows: list[dict[str, Any]]) -> None:
    ws.append(["E216 chair/desk · E201 14-gate 宽/窄漏斗（下游预筛同口径，aug-only 无 orig）"])
    ws.append([])
    objects = sorted({r["object_key"] for r in rows})
    heads = ["object", "n", "L1_reject", "L2_review", "L3(accept)", "L3 accept率",
             "WIDE pass", "NARROW pass"]
    hr = ws.max_row + 1
    for c, h in enumerate(heads, 1):
        _c(ws, hr, c, h, NAVY, HEAD)
    for obj in objects + ["POOLED"]:
        rs = rows if obj == "POOLED" else [r for r in rows if r["object_key"] == obj]
        n = len(rs)
        l1 = sum(1 for r in rs if r["layer"] == "L1_reject")
        l2 = sum(1 for r in rs if r["layer"] == "L2_review")
        l3 = sum(1 for r in rs if r["layer"] == "L3_review")
        wp = sum(1 for r in rs if not (r.get("hard_failed") or r.get("wide_failed")))
        npass = sum(1 for r in rs if not (r.get("hard_failed") or r.get("narrow_failed")))
        r = ws.max_row + 1
        _c(ws, r, 1, obj, None, Font(bold=(obj == "POOLED")))
        _c(ws, r, 2, n)
        _c(ws, r, 3, l1, LAYER_FILL["L1_reject"] if l1 else None)
        _c(ws, r, 4, l2, LAYER_FILL["L2_review"] if l2 else None)
        _c(ws, r, 5, l3, LAYER_FILL["L3_review"] if l3 else None)
        _c(ws, r, 6, round(l3 / n, 3) if n else "")
        _c(ws, r, 7, f"{wp}/{n}")
        _c(ws, r, 8, f"{npass}/{n}")
    for i, w in enumerate([12, 5, 11, 11, 11, 11, 10, 11], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A3"


def write_detail(ws, rows: list[dict[str, Any]]) -> None:
    header = ["object_key", "case_id", "aug_variant", "gravcomp", "layer"]
    for name, *_ in HARD:
        header += [name, f"{name}✓"]
    for name, _f, _o, nr, wd in BANDED:
        header += [name, f"{name} N({nr:g})", f"{name} W({wd:g})"]
    for c, h in enumerate(header, 1):
        cell = _c(ws, 1, c, h, NAVY, HEAD)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.freeze_panes = "F2"
    for ri, row in enumerate(rows, 2):
        _c(ws, ri, 1, row.get("object_key", ""))
        _c(ws, ri, 2, row.get("case_id", ""))
        _c(ws, ri, 3, row.get("aug_variant", ""))
        _c(ws, ri, 4, row.get("gravcomp", ""))
        _c(ws, ri, 5, row.get("layer", ""), LAYER_FILL.get(row.get("layer", "")), HEAD)
        c = 6
        hf = set((row.get("hard_failed") or "").split(","))
        for name, field, *_ in HARD:
            _c(ws, ri, c, str(row.get(field, "")) if field == "fall_flag" else round(finite(row.get(field)), 4) if math.isfinite(finite(row.get(field))) else "")
            c += 1
            ok = name not in hf
            _c(ws, ri, c, "PASS" if ok else "FAIL", GREEN if ok else RED)
            c += 1
        wf = set((row.get("wide_failed") or "").split(","))
        nf = set((row.get("narrow_failed") or "").split(","))
        for name, field, *_ in BANDED:
            v = finite(row.get(field))
            _c(ws, ri, c, round(v, 4) if math.isfinite(v) else "")
            c += 1
            _c(ws, ri, c, "PASS" if name not in nf else "FAIL", GREEN if name not in nf else RED)
            c += 1
            _c(ws, ri, c, "PASS" if name not in wf else "FAIL", GREEN if name not in wf else RED)
            c += 1
    widths = [10, 34, 11, 9, 11] + [12, 6] * len(HARD) + [12, 9, 9] * len(BANDED)
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def main() -> int:
    FC.assert_monotonic()
    if not ROLLOUT_TSV.is_file():
        raise SystemExit(f"missing {ROLLOUT_TSV}")
    rows = read_tsv(ROLLOUT_TSV)
    for r in rows:
        r.update(classify(r))
    rows.sort(key=lambda r: (r["object_key"], r["case_id"], r["aug_variant"]))
    cmp = json.loads(COMPARE_JSON.read_text(encoding="utf-8")) if COMPARE_JSON.is_file() else {}

    wb = Workbook()
    write_compare(wb.active, cmp)
    wb.active.title = "Compare_0p1m_0p2m"
    write_funnel(wb.create_sheet("Funnel"), rows)
    write_detail(wb.create_sheet("Detail"), rows)
    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(DEFAULT_OUT)
    print(json.dumps({"status": "PASS", "output": str(DEFAULT_OUT),
                      "sheets": wb.sheetnames, "rows": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
