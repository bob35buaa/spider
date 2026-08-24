#!/usr/bin/env python3
"""E204/E205 three-arm 14-gate comparison workbook (xlsx).

Reads the combined three_arm_rollout.tsv (from eval_E204E205_arm_ablation.py) and
builds an xlsx comparing noPRG (E204) / PRG (E178) / G1A2 (E205) on the E201
14-gate funnel over the 27 E178 bucket cases. Sheets:

  1. README        - arms, 14-gate thresholds, 口径 notes
  2. 14-Gate Summary - per-arm hard/wide/narrow pass + per-gate narrow pass counts
  3. Per-Object    - bucket003/004/007 x arm narrow-pass rate + key metric means
  4. Per-Case      - 27 cases x 3 arms, 14 metric values, narrow pass colored, layer
  5. Dist & Paired - per-arm mean/std/worst + paired deltas vs PRG (win/loss)

Usage:
    .venv/bin/python workspace/core4d/scripts/eval/reports/gen_E204E205_three_arm_workbook.py
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E201"))
import funnel_config as FC  # noqa: E402

EVAL_DIR = REPO / "workspace/core4d/results/E204/s6_downstream/eval/three_arm"
ROLLOUT_TSV = EVAL_DIR / "three_arm_rollout.tsv"
OUT_XLSX = EVAL_DIR / "E204E205_three_arm_14gate.xlsx"

ARMS = ["noPRG", "PRG", "G1A2"]  # display order (E204 / E178 / E205)
ARM_DESC = {
    "noPRG": "E204 · E167A, leg-PRG off, gravcomp0, no A2",
    "PRG": "E178 · leg-PRG on (baseline)",
    "G1A2": "E205 · PRG + G1 gravcomp + A2 hand-gate",
}
NAVY, BLUE, GREEN, RED, GRAY, WHITE = "1F3864", "2E5496", "C6EFCE", "FFC7CE", "D9D9D9", "FFFFFF"
# key metrics for distribution/paired sheets: (field, direction lower_is_better)
KEY = [
    ("track_obj_pos_err_cm_mean", True), ("track_obj_ori_err_deg_mean", True),
    ("track_root_pos_err_cm_mean", True), ("track_eef_pos_err_cm_mean", True),
    ("hand_object_physics_contact_in_mask_frac", False),
    ("hand_object_physics_penetration_3mm_frame_frac", True),
    ("leg_penetration_frac", True), ("body_z_err_p95_m", True),
]
# 14 gates in funnel order for per-gate columns
GATE_ORDER = [("fall", "fall_flag"), ("body_z", "body_z_err_p95_m"),
              ("ankle_jerk", "ankle_jerk_p95"), ("obj_speed", "obj_speed_max")] + \
    [(g[0], g[1]) for g in FC.BANDED_GATES]
NARROW_THR = {g[1]: (g[2], g[3]) for g in FC.BANDED_GATES}
HARD = {h[1]: h for h in FC.HARD_GATES}


def finite(v: Any) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else math.nan
    except (TypeError, ValueError):
        return math.nan


def gate_pass_narrow(gfield: str, value: Any) -> bool:
    if gfield == "fall_flag":
        return str(value).strip().lower() not in ("true", "1")
    if gfield in HARD:
        _n, _f, op, thr = HARD[gfield]
        return FC.passes(op, finite(value), thr)
    op, thr = NARROW_THR[gfield][0], NARROW_THR[gfield][1]
    return FC.passes(op, finite(value), thr)


def read_rows() -> list[dict[str, str]]:
    with ROLLOUT_TSV.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _title(ws, text: str, subtitle: str, ncol: int) -> None:
    ws["A1"] = text
    ws["A1"].font = Font(size=14, bold=True, color=WHITE)
    ws["A1"].fill = PatternFill("solid", fgColor=NAVY)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncol)
    ws["A2"] = subtitle
    ws["A2"].font = Font(size=9, italic=True, color="666666")
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=ncol)


def _hdr(ws, row: int, ncol: int) -> None:
    for cell in ws[row][:ncol]:
        cell.font = Font(bold=True, color=WHITE)
        cell.fill = PatternFill("solid", fgColor=BLUE)
        cell.alignment = Alignment(horizontal="center", wrap_text=True)


def sheet_readme(wb: Workbook) -> None:
    ws = wb.active
    ws.title = "README"
    _title(ws, "E204/E205 三 arm 14-gate 对比 (27 bucket case)",
           "复用 E178 的 27 case (bucket003=9/004=4/007=14), 同 contactAlignedTop 五段物体代理 + omnirt_v1 轨迹 + CEM1024x32seed0; 仅改 reward arm。E178 PRG 读 canonical tsv, E204/E205 用 public core_metrics 现评; 同一 14-gate 尺子。", 6)
    r = 4
    for arm in ARMS:
        ws.cell(r, 1, arm).font = Font(bold=True)
        ws.cell(r, 2, ARM_DESC[arm])
        r += 1
    r += 1
    ws.cell(r, 1, "14 gates = 4 hard + 10 banded(wide/narrow)").font = Font(bold=True)
    r += 1
    for name, field, op, thr in FC.HARD_GATES:
        ws.cell(r, 1, f"HARD {name}"); ws.cell(r, 2, f"{field} {op} {thr}"); r += 1
    for g in FC.BANDED_GATES:
        ws.cell(r, 1, f"BAND {g[0]}"); ws.cell(r, 2, f"{g[1]} {g[2]} narrow={g[3]} wide={g[4]}"); r += 1
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 70


def sheet_summary(wb: Workbook, rows: list[dict]) -> None:
    ws = wb.create_sheet("14-Gate Summary")
    gate_names = [g[0] for g in GATE_ORDER]
    cols = ["arm", "n", "hard", "wide_all", "narrow_all", "narrow_all_%", "L3_auto"] + gate_names
    _title(ws, "14-Gate 通过汇总 (per arm; 计数=通过该门的 case 数, narrow 口径)",
           "narrow_all = 27 case 中 4 硬门 + 10 窄带门全过的数量; 每列 = 单门 narrow 通过计数 (越高越好)", len(cols))
    ws.append([""] * len(cols))
    ws.append(cols)
    _hdr(ws, ws.max_row, len(cols))
    for arm in ARMS:
        ar = [r for r in rows if r["arm"] == arm]
        n = len(ar)
        hard = sum(r["hard_pass"] == "True" for r in ar)
        wide = sum(r["wide_pass"] == "True" for r in ar)
        narrow = sum(r["narrow_pass"] == "True" for r in ar)
        l3 = sum(r["layer"] == "L3_auto" for r in ar)
        line = [arm, n, hard, wide, narrow, f"{(narrow/n if n else 0):.0%}", l3]
        for _gn, gf in GATE_ORDER:
            line.append(sum(gate_pass_narrow(gf, r[gf]) for r in ar))
        ws.append(line)
    # highlight best per gate col (max)
    first_data = ws.max_row - len(ARMS) + 1
    ws.column_dimensions["A"].width = 8
    for i in range(len(cols)):
        ws.column_dimensions[chr(66 + i) if i else "A"].width = 10


def sheet_per_object(wb: Workbook, rows: list[dict]) -> None:
    ws = wb.create_sheet("Per-Object")
    metrics = [k for k, _ in KEY]
    cols = ["object", "arm", "n", "narrow_all", "narrow_%"] + [m.replace("track_", "").replace("_err", "").replace("hand_object_physics_", "") for m in metrics]
    _title(ws, "逐物体 × arm (bucket003/004/007)", "narrow_all=全门窄通过数; 指标为 case 均值 (obj/root/eef cm 或 deg, contact/pen/leg frac, body_z m)", len(cols))
    ws.append([""] * len(cols)); ws.append(cols); _hdr(ws, ws.max_row, len(cols))
    for obj in ["bucket003", "bucket004", "bucket007"]:
        for arm in ARMS:
            ar = [r for r in rows if r["arm"] == arm and r["object_key"] == obj]
            n = len(ar)
            if not n:
                continue
            narrow = sum(r["narrow_pass"] == "True" for r in ar)
            line = [obj, arm, n, narrow, f"{narrow/n:.0%}"]
            for m in metrics:
                vals = [finite(r[m]) for r in ar if math.isfinite(finite(r[m]))]
                line.append(round(statistics.mean(vals), 3) if vals else "")
            ws.append(line)
    ws.column_dimensions["A"].width = 12
    ws.column_dimensions["B"].width = 8


def sheet_per_case(wb: Workbook, rows: list[dict]) -> None:
    ws = wb.create_sheet("Per-Case")
    gate_fields = [gf for _gn, gf in GATE_ORDER]
    cols = ["case_id", "arm", "layer"] + [gn for gn, _gf in GATE_ORDER]
    _title(ws, "逐 case × arm 14-gate 值 (绿=窄门过, 红=不过)",
           "每 case 3 行 (noPRG/PRG/G1A2); 单元格值=该门指标, 底色=是否窄门通过", len(cols))
    ws.append([""] * len(cols)); ws.append(cols); _hdr(ws, ws.max_row, len(cols))
    by_case: dict[str, list[dict]] = {}
    for r in rows:
        by_case.setdefault(r["case_id"], []).append(r)
    for cid in sorted(by_case):
        for arm in ARMS:
            ar = [r for r in by_case[cid] if r["arm"] == arm]
            if not ar:
                continue
            r = ar[0]
            ws.append([cid, arm, r["layer"]] + [r[gf] for gf in gate_fields])
            ridx = ws.max_row
            for j, gf in enumerate(gate_fields):
                cell = ws.cell(ridx, 4 + j)
                ok = gate_pass_narrow(gf, r[gf])
                cell.fill = PatternFill("solid", fgColor=GREEN if ok else RED)
    ws.column_dimensions["A"].width = 30
    ws.freeze_panes = "D4"


def sheet_dist_paired(wb: Workbook, rows: list[dict]) -> None:
    ws = wb.create_sheet("Dist & Paired")
    _title(ws, "指标分布 (mean/std/worst) + 与 PRG 的 paired 对比",
           "worst = 最劣值 (lower_is_better 取 max, contact 取 min); paired: 每 case (arm - PRG) 均值 + win/loss (arm 更优的 case 数)", 8)
    by = {arm: {r["case_id"]: r for r in rows if r["arm"] == arm} for arm in ARMS}
    row = 4
    for field, lower_better in KEY:
        ws.cell(row, 1, field).font = Font(bold=True, color=WHITE)
        ws.cell(row, 1).fill = PatternFill("solid", fgColor=NAVY)
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
        row += 1
        head = ["arm", "mean", "std", "worst", "vs PRG Δmean", "arm-better cases", "worse cases", "ties"]
        for j, h in enumerate(head):
            c = ws.cell(row, 1 + j, h); c.font = Font(bold=True, color=WHITE); c.fill = PatternFill("solid", fgColor=BLUE)
        row += 1
        for arm in ARMS:
            vals = [finite(r[field]) for r in by[arm].values() if math.isfinite(finite(r[field]))]
            mean = statistics.mean(vals) if vals else math.nan
            std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            worst = (max(vals) if lower_better else min(vals)) if vals else math.nan
            dmean = win = lose = tie = ""
            if arm != "PRG":
                deltas, w, l, t = [], 0, 0, 0
                for cid, r in by[arm].items():
                    pr = by["PRG"].get(cid)
                    if not pr:
                        continue
                    a, p = finite(r[field]), finite(pr[field])
                    if not (math.isfinite(a) and math.isfinite(p)):
                        continue
                    d = a - p
                    deltas.append(d)
                    better = (d < 0) if lower_better else (d > 0)
                    worse = (d > 0) if lower_better else (d < 0)
                    if abs(d) < 1e-9:
                        t += 1
                    elif better:
                        w += 1
                    else:
                        l += 1
                dmean = round(statistics.mean(deltas), 4) if deltas else ""
                win, lose, tie = w, l, t
            ws.append([]) if False else None
            for j, v in enumerate([arm, round(mean, 4) if math.isfinite(mean) else "",
                                   round(std, 4), round(worst, 4) if math.isfinite(worst) else "",
                                   dmean, win, lose, tie]):
                ws.cell(row, 1 + j, v)
            row += 1
        row += 1
    ws.column_dimensions["A"].width = 42
    for c in "BCDEFGH":
        ws.column_dimensions[c].width = 15


def main() -> int:
    if not ROLLOUT_TSV.is_file():
        print(f"missing {ROLLOUT_TSV}; run eval_E204E205_arm_ablation.py first", file=sys.stderr)
        return 2
    rows = read_rows()
    wb = Workbook()
    sheet_readme(wb)
    sheet_summary(wb, rows)
    sheet_per_object(wb, rows)
    sheet_per_case(wb, rows)
    sheet_dist_paired(wb, rows)
    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT_XLSX)
    print(f"[done] wrote {OUT_XLSX.relative_to(REPO)} ({len(rows)} rows, {len(rows)//3} cases x 3 arms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
