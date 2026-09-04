#!/usr/bin/env python3
"""E206 P9 report: the two-arm workbook, formatted for reading rather than parsing.

Sheets:
  说明             what each sheet is + the C5a / C5b verdicts up front
  funnel_summary   per-arm L1/L2/L3 and 12-gate counts (COUNTIFS over `rollout`)
  per_gate         per gate: mean/median/std/worst per arm + paired delta
  per_object       per object x arm, with n and whether n permits a recommendation
  paired_delta     per case: PRG - noPRG on the decision-relevant metrics
  vs_E174_desk007  the C5a like-for-like table
  rollout          the full 130-row scored data (the source every formula points at)

Styling follows the established convention in this directory
(build_E194_three_arm_workbook.py): NAVY title bar, BLUE header, Excel Table with
autofilter + frozen panes, Arial throughout, and `add_delta_quality_formatting`'s
zero-centred green/red scale for signed deltas.

Reporting choices that are deliberate, not cosmetic:
  * `worst` sits next to every mean -- rule 5 forbids the mean alone.
  * root_pos also carries a median: two L1 fall cases (root error ~15 m) dominate
    its mean. Hiding that behind a mean is metric deception; dropping them is
    selective reporting. Both are shown, and the outlier rows are flagged red.
  * per_object rows with n < 3 are greyed and marked -- they must not be read as
    per-object recommendations.

Usage:
    .venv/bin/python .../gen_E206_two_arm_workbook.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E206"))
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E201"))

import e206_common as C  # noqa: E402
import funnel_config as FC  # noqa: E402

EVAL = C.S6_DIR / "eval/two_arm"
OUT = EVAL / "e206_two_arm.xlsx"

NAVY = "17365D"
BLUE = "4472C4"
LIGHT_BLUE = "D9EAF7"
LIGHT_GREEN = "E2F0D9"
LIGHT_RED = "FCE4D6"
LIGHT_AMBER = "FFF2CC"
LIGHT_GRAY = "E7E6E6"
WHITE = "FFFFFF"
GREEN_TXT = "1E7145"
RED_TXT = "9C0006"
THIN = Side(style="thin", color="B7B7B7")
ARIAL = "Arial"

# metric -> True when a LOWER value is better
LOWER_BETTER = {g[1]: (g[2] == "<=") for g in FC.BANDED_GATES}
LOWER_BETTER["body_z_err_p95_m"] = True

DECISION_FIELDS = [
    "leg_penetration_frac",
    "hand_object_physics_penetration_3mm_frame_frac",
    "hand_object_physics_contact_in_mask_frac",
    "hand_object_release_false_contact_3mm_frac",
    "body_z_err_p95_m",
    "track_obj_pos_err_cm_mean",
    "track_root_pos_err_cm_mean",
]


def f(v: Any) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else math.nan
    except (TypeError, ValueError):
        return math.nan


def truthy(v: Any) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes"}


def title(ws, text: str, subtitle: str, columns: int) -> None:
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=columns)
    ws["A1"] = text
    ws["A1"].font = Font(name=ARIAL, size=16, bold=True, color=WHITE)
    ws["A1"].fill = PatternFill("solid", fgColor=NAVY)
    ws["A1"].alignment = Alignment(vertical="center")
    ws.row_dimensions[1].height = 28
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=columns)
    ws["A2"] = subtitle
    ws["A2"].font = Font(name=ARIAL, size=9, italic=True, color="666666")
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[2].height = 34


def header(ws, row: int, columns: int) -> None:
    for cell in ws[row][:columns]:
        cell.font = Font(name=ARIAL, bold=True, color=WHITE, size=9)
        cell.fill = PatternFill("solid", fgColor=BLUE)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(bottom=THIN)
    ws.row_dimensions[row].height = 40


def finish_table(ws, start_row: int, end_row: int, end_col: int, name: str) -> None:
    ref = f"A{start_row}:{get_column_letter(end_col)}{end_row}"
    tbl = Table(displayName=name, ref=ref)
    tbl.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True,
                                        showFirstColumn=False, showLastColumn=False,
                                        showColumnStripes=False)
    ws.add_table(tbl)
    ws.auto_filter.ref = ref
    ws.freeze_panes = f"D{start_row + 1}"


def set_widths(ws, widths: dict[int, float], default: float = 12) -> None:
    for i in range(1, ws.max_column + 1):
        ws.column_dimensions[get_column_letter(i)].width = widths.get(i, default)


def delta_scale(ws, rng: str, lower_is_better: bool) -> None:
    """Zero-centred gradient: green = the arm improved, red = it regressed."""
    start, end = (("63BE7B", "F8696B") if lower_is_better else ("F8696B", "63BE7B"))
    ws.conditional_formatting.add(rng, ColorScaleRule(
        start_type="min", start_color=start,
        mid_type="num", mid_value=0, mid_color="FFEB84",
        end_type="max", end_color=end))


def body_font(ws, first_row: int, last_row: int, bold_first_col: bool = True) -> None:
    for r in range(first_row, last_row + 1):
        for c in range(1, ws.max_column + 1):
            cell = ws.cell(r, c)
            cell.font = Font(name=ARIAL, size=9, bold=bold_first_col and c == 1)
            cell.alignment = Alignment(vertical="center",
                                       horizontal="left" if c <= 3 else "right")


def main() -> int:
    rows = C.read_tsv(EVAL / "e206_two_arm_rollout.tsv")
    by = {(r["arm"], r["case_id"]): r for r in rows}
    cases = sorted({r["case_id"] for r in rows})
    paired = [c for c in cases if all((a, c) in by for a in C.ARMS)]
    c5a = json.loads((EVAL / "c5a_vs_e174_desk007.json").read_text(encoding="utf-8")) \
        if (EVAL / "c5a_vs_e174_desk007.json").is_file() else {}

    wb = Workbook()
    wb.remove(wb.active)

    # ------------------------------------------------------------ rollout ---
    cols = (["object_key", "case_id", "arm", "layer", "gate12_pass",
             "physics6_pass", "tracking6_pass", "narrow_pass"]
            + [g[1] for g in FC.BANDED_GATES]
            + ["body_z_err_p95_m", "fall_flag", "narrow_failed", "wide_failed"])
    ws = wb.create_sheet("rollout")
    title(ws, "E206 · 130 条 rollout 逐行结果",
          "两 arm 均由新鲜 rollout 经同一代码路径打分。P/F = 该门通过与否；"
          "layer 为 E201 14 门漏斗判定；本页是其余各页公式的数据源。", len(cols))
    ws.append([])
    ws.append(cols)
    header(ws, 4, len(cols))
    first = 5
    for r in sorted(rows, key=lambda x: (x["object_key"], x["case_id"], x["arm"])):
        ws.append([r.get(c, "") if c not in ("gate12_pass", "physics6_pass",
                                             "tracking6_pass", "narrow_pass")
                   else truthy(r.get(c)) for c in cols])
    last = ws.max_row
    for r in range(first, last + 1):
        for c in range(1, len(cols) + 1):
            cell = ws.cell(r, c)
            name = cols[c - 1]
            cell.font = Font(name=ARIAL, size=9)
            if name in ("gate12_pass", "physics6_pass", "tracking6_pass", "narrow_pass"):
                ok = cell.value is True
                cell.value = "P" if ok else "F"
                cell.fill = PatternFill("solid", fgColor=LIGHT_GREEN if ok else LIGHT_RED)
                cell.font = Font(name=ARIAL, size=9, bold=True,
                                 color=GREEN_TXT if ok else RED_TXT)
                cell.alignment = Alignment(horizontal="center")
            elif name == "layer":
                v = str(cell.value)
                fill = {"L3_auto": LIGHT_GREEN, "L2_review": LIGHT_AMBER,
                        "L1_reject": LIGHT_RED}.get(v, WHITE)
                cell.fill = PatternFill("solid", fgColor=fill)
                cell.alignment = Alignment(horizontal="center")
            elif name in LOWER_BETTER:
                cell.value = f(cell.value) if math.isfinite(f(cell.value)) else None
                cell.number_format = "0.000"
            elif name == "fall_flag":
                if truthy(cell.value):
                    cell.fill = PatternFill("solid", fgColor=LIGHT_RED)
                    cell.font = Font(name=ARIAL, size=9, bold=True, color=RED_TXT)
    finish_table(ws, 4, last, len(cols), "rollout_tbl")
    set_widths(ws, {1: 11, 2: 30, 3: 8, 4: 11, len(cols) - 1: 26, len(cols): 26}, 11)
    R = "rollout"          # sheet name used by every COUNTIFS below
    col_of = {n: get_column_letter(i + 1) for i, n in enumerate(cols)}
    rng = lambda n: f"{R}!${col_of[n]}${first}:${col_of[n]}${last}"  # noqa: E731

    # ---------------------------------------------------- funnel_summary ---
    ws = wb.create_sheet("funnel_summary", 0)
    scols = ["arm", "n", "L3_auto", "L2_review", "L1_reject", "narrow_pass",
             "gate12_pass", "physics6_pass", "tracking6_pass"]
    title(ws, "E206 · 双 arm 漏斗汇总",
          "计数为对 rollout 页的 COUNTIFS 公式，改动源数据会自动重算。"
          "L3=自动接受 / L2=待人工 / L1=拒绝。", len(scols))
    ws.append([])
    ws.append(scols)
    header(ws, 4, len(scols))
    for i, arm in enumerate(C.ARMS):
        r = 5 + i
        a = f'"{arm}"'
        ws.cell(r, 1, arm)
        ws.cell(r, 2, f'=COUNTIF({rng("arm")},{a})')
        ws.cell(r, 3, f'=COUNTIFS({rng("arm")},{a},{rng("layer")},"L3_auto")')
        ws.cell(r, 4, f'=COUNTIFS({rng("arm")},{a},{rng("layer")},"L2_review")')
        ws.cell(r, 5, f'=COUNTIFS({rng("arm")},{a},{rng("layer")},"L1_reject")')
        for j, field in enumerate(["narrow_pass", "gate12_pass", "physics6_pass",
                                   "tracking6_pass"]):
            ws.cell(r, 6 + j, f'=COUNTIFS({rng("arm")},{a},{rng(field)},"P")')
    ws.cell(7, 1, "Δ (PRG − noPRG)")
    for c in range(2, len(scols) + 1):
        L = get_column_letter(c)
        ws.cell(7, c, f"={L}6-{L}5")
    body_font(ws, 5, 7)
    for c in range(1, len(scols) + 1):
        ws.cell(7, c).fill = PatternFill("solid", fgColor=LIGHT_BLUE)
        ws.cell(7, c).font = Font(name=ARIAL, size=9, bold=True)
    delta_scale(ws, f"C7:I7", lower_is_better=False)
    ws.cell(7, 5).value = "=E6-E5"          # L1: fewer is better
    delta_scale(ws, "E7:E7", lower_is_better=True)

    # verdict banner
    legpen = {arm: 100.0 * sum(1 for c in paired
                               if f(by[(arm, c)]["leg_penetration_frac"]) <= 0.20)
              / len(paired) for arm in C.ARMS}
    dpp = legpen["prg"] - legpen["noprg"]
    l3d = (sum(1 for c in paired if by[("prg", c)]["layer"] == "L3_auto")
           - sum(1 for c in paired if by[("noprg", c)]["layer"] == "L3_auto"))
    verdict = ("PRG 胜（C5b 达标）" if dpp >= 10 and l3d >= 0
               else "noPRG 胜" if dpp <= -10 else "无分离")
    lines = [
        ("C5b · arm 判据", f"leg_pen narrow {legpen['noprg']:.1f}% → {legpen['prg']:.1f}% "
                          f"({dpp:+.1f} pp，门 ≥ +10)；L3 Δ {l3d:+d} （门 ≥ 0）  ⇒  {verdict}",
         LIGHT_GREEN if "胜" in verdict and dpp > 0 else LIGHT_AMBER),
        ("C5b · 代价", "hand_pen narrow −9.2 pp、root_pos −4.6 pp、obj_pos −4.6 pp —— "
                      "PRG 把穿透从腿部分转移到手，不是免费改善", LIGHT_RED),
    ]
    if c5a:
        lines.insert(0, (
            "C5a · vs E174 desk007",
            f"contact_in_mask {c5a['e174_prg_baseline']['mean']} → {c5a['e206_prg']['mean']}"
            f"（{c5a['mean_improvement']:+}，门 +{c5a['bar_absolute_improvement']}）"
            f"  ⇒  {c5a['verdict'].upper()}；E174 有 5/9 例接触恰为 0.0000（F7 指纹）",
            LIGHT_GREEN if c5a["verdict"] == "pass" else LIGHT_RED))
    row = 10
    for label, text, fill in lines:
        ws.cell(row, 1, label).font = Font(name=ARIAL, size=10, bold=True)
        ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=len(scols))
        cell = ws.cell(row, 2, text)
        cell.font = Font(name=ARIAL, size=9)
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        for c in range(1, len(scols) + 1):
            ws.cell(row, c).fill = PatternFill("solid", fgColor=fill)
            ws.cell(row, c).border = Border(bottom=THIN)
        ws.row_dimensions[row].height = 30
        row += 1
    set_widths(ws, {1: 22, 2: 16}, 13)

    # ------------------------------------------------------------ per_gate ---
    ws = wb.create_sheet("per_gate")
    gcols = ["gate", "field", "方向", "narrow", "wide"]
    for arm in C.ARMS:
        gcols += [f"{arm}_mean", f"{arm}_median", f"{arm}_std", f"{arm}_worst",
                  f"{arm}_narrow%"]
    gcols += ["Δ narrow pp", "Δ mean", "Δ median"]
    title(ws, "E206 · 逐门统计（两 arm）",
          "每门同时给 mean / median / std / worst —— 只报均值会被离群主导（见 root_pos）。"
          "「Δ narrow pp」与「Δ mean」按门方向着色：绿=PRG 更好，红=更差。", len(gcols))
    ws.append([])
    ws.append(gcols)
    header(ws, 4, len(gcols))
    gate_defs = list(FC.BANDED_GATES) + [("body_z", "body_z_err_p95_m", "<=", 0.20, 0.20)]
    for name, field, op, narrow, wide in gate_defs:
        lower = op == "<="
        rec: list[Any] = [name, field, "越小越好" if lower else "越大越好", narrow, wide]
        stat: dict[str, dict[str, float]] = {}
        for arm in C.ARMS:
            v = [f(by[(arm, c)][field]) for c in paired]
            v = [x for x in v if math.isfinite(x)]
            s = {"mean": statistics.mean(v), "median": statistics.median(v),
                 "std": statistics.pstdev(v) if len(v) > 1 else 0.0,
                 "worst": max(v) if lower else min(v)}
            stat[arm] = s
            npass = 100.0 * sum(1 for c in paired
                                if FC.passes(op, f(by[(arm, c)][field]), narrow)) / len(paired)
            rec += [round(s["mean"], 4), round(s["median"], 4), round(s["std"], 4),
                    round(s["worst"], 4), round(npass, 1)]
        n_pct, p_pct = rec[9], rec[14]
        rec += [round(p_pct - n_pct, 1),
                round(stat["prg"]["mean"] - stat["noprg"]["mean"], 4),
                round(stat["prg"]["median"] - stat["noprg"]["median"], 4)]
        ws.append(rec)
    last = ws.max_row
    body_font(ws, 5, last)
    for r in range(5, last + 1):
        for c in (6, 7, 8, 9, 11, 12, 13, 14, 17, 18):
            ws.cell(r, c).number_format = "0.000"
        for c in (10, 15, 16):
            ws.cell(r, c).number_format = "0.0"
    # direction-aware colouring, per row (each gate has its own polarity)
    for i, (_n, _f, op, *_x) in enumerate(gate_defs):
        r = 5 + i
        delta_scale(ws, f"P{r}:P{r}", lower_is_better=False)   # narrow pp: higher better
        delta_scale(ws, f"Q{r}:R{r}", lower_is_better=(op == "<="))
    finish_table(ws, 4, last, len(gcols), "per_gate_tbl")
    set_widths(ws, {1: 11, 2: 46, 3: 11}, 11)

    # ---------------------------------------------------------- per_object ---
    ws = wb.create_sheet("per_object")
    ocols = ["object_key", "n", "可出 per-object 结论"]
    for arm in C.ARMS:
        ocols += [f"{arm}_L3", f"{arm}_L1", f"{arm}_12门",
                  f"{arm}_legpen均", f"{arm}_legpen最差",
                  f"{arm}_contact均", f"{arm}_contact最差"]
    title(ws, "E206 · 逐物体（两 arm）",
          "n < 3 的物体整行置灰：样本量不足，只并入总体统计，"
          "不得读作 per-object arm 推荐（plan236 R1d / F3）。", len(ocols))
    ws.append([])
    ws.append(ocols)
    header(ws, 4, len(ocols))
    for obj in sorted({c.split("_")[0] for c in paired}):
        cs = [c for c in paired if c.split("_")[0] == obj]
        rec: list[Any] = [obj, len(cs), "是" if len(cs) >= 3 else "否 —— 样本不足"]
        for arm in C.ARMS:
            rs = [by[(arm, c)] for c in cs]
            lp = [f(x["leg_penetration_frac"]) for x in rs]
            ct = [f(x["hand_object_physics_contact_in_mask_frac"]) for x in rs]
            rec += [sum(1 for x in rs if x["layer"] == "L3_auto"),
                    sum(1 for x in rs if x["layer"] == "L1_reject"),
                    sum(1 for x in rs if truthy(x["gate12_pass"])),
                    round(statistics.mean(lp), 4), round(max(lp), 4),
                    round(statistics.mean(ct), 4), round(min(ct), 4)]
        ws.append(rec)
    last = ws.max_row
    body_font(ws, 5, last)
    for r in range(5, last + 1):
        small = int(ws.cell(r, 2).value) < 3
        for c in range(1, len(ocols) + 1):
            if c >= 7:
                ws.cell(r, c).number_format = "0.000"
            if small:
                ws.cell(r, c).fill = PatternFill("solid", fgColor=LIGHT_GRAY)
                ws.cell(r, c).font = Font(name=ARIAL, size=9, italic=True, color="808080")
        # desk020-style outliers: leg penetration an order of magnitude off
        for c in (7, 14):
            if f(ws.cell(r, c).value) > 0.30:
                ws.cell(r, c).fill = PatternFill("solid", fgColor=LIGHT_RED)
                ws.cell(r, c).font = Font(name=ARIAL, size=9, bold=True, color=RED_TXT)
    finish_table(ws, 4, last, len(ocols), "per_object_tbl")
    set_widths(ws, {1: 12, 2: 6, 3: 20}, 12)

    # --------------------------------------------------------- paired_delta ---
    ws = wb.create_sheet("paired_delta")
    pcols = ["case_id", "object_key", "noPRG layer", "PRG layer"] + \
            [f"Δ {x}" for x in DECISION_FIELDS]
    title(ws, "E206 · 逐 case 配对 delta（PRG − noPRG）",
          "同一 case 同一片段的两 arm 之差。着色按各指标方向：绿=PRG 更好。", len(pcols))
    ws.append([])
    ws.append(pcols)
    header(ws, 4, len(pcols))
    for c in paired:
        rec: list[Any] = [c, c.split("_")[0],
                          by[("noprg", c)]["layer"], by[("prg", c)]["layer"]]
        for field in DECISION_FIELDS:
            rec.append(round(f(by[("prg", c)][field]) - f(by[("noprg", c)][field]), 4))
        ws.append(rec)
    last = ws.max_row
    body_font(ws, 5, last, bold_first_col=False)
    for i, field in enumerate(DECISION_FIELDS):
        L = get_column_letter(5 + i)
        for r in range(5, last + 1):
            ws.cell(r, 5 + i).number_format = "0.000"
        delta_scale(ws, f"{L}5:{L}{last}", lower_is_better=LOWER_BETTER.get(field, True))
    for r in range(5, last + 1):
        for c in (3, 4):
            v = str(ws.cell(r, c).value)
            ws.cell(r, c).fill = PatternFill("solid", fgColor={
                "L3_auto": LIGHT_GREEN, "L2_review": LIGHT_AMBER,
                "L1_reject": LIGHT_RED}.get(v, WHITE))
            ws.cell(r, c).alignment = Alignment(horizontal="center")
    finish_table(ws, 4, last, len(pcols), "paired_tbl")
    set_widths(ws, {1: 30, 2: 11, 3: 12, 4: 12}, 15)

    # ------------------------------------------------------ vs_E174_desk007 ---
    c5a_tsv = EVAL / "c5a_vs_e174_desk007.tsv"
    if c5a_tsv.is_file():
        ws = wb.create_sheet("vs_E174_desk007")
        crows = C.read_tsv(c5a_tsv)
        ccols = list(crows[0].keys())
        title(ws, "C5a · 与 E174 desk007 同尺对比（9 例，3cm 掩码）",
              "E174 时代理 41 box 但只有 2 条 robot↔object pair（F7），40/41 个盒子对机器人"
              "不可见。红底 = E174 该例接触恰为 0.0000。", len(ccols))
        ws.append([])
        ws.append(ccols)
        header(ws, 4, len(ccols))
        for r in crows:
            ws.append([f(r[c]) if c != "case_id" else r[c] for c in ccols])
        last = ws.max_row
        body_font(ws, 5, last)
        for r in range(5, last + 1):
            for c in range(2, len(ccols) + 1):
                ws.cell(r, c).number_format = "0.000"
            if f(ws.cell(r, 2).value) == 0.0:
                for c in range(1, len(ccols) + 1):
                    ws.cell(r, c).fill = PatternFill("solid", fgColor=LIGHT_RED)
                ws.cell(r, 2).font = Font(name=ARIAL, size=9, bold=True, color=RED_TXT)
            ws.cell(r, 4).fill = PatternFill(
                "solid", fgColor=LIGHT_GREEN if f(ws.cell(r, 4).value) > 0 else LIGHT_RED)
        summary_row = last + 2
        ws.cell(summary_row, 1, "均值").font = Font(name=ARIAL, size=9, bold=True)
        for c in (2, 3, 4, 5):
            L = get_column_letter(c)
            ws.cell(summary_row, c, f"=AVERAGE({L}5:{L}{last})")
            ws.cell(summary_row, c).number_format = "0.000"
            ws.cell(summary_row, c).font = Font(name=ARIAL, size=9, bold=True)
            ws.cell(summary_row, c).fill = PatternFill("solid", fgColor=LIGHT_BLUE)
        finish_table(ws, 4, last, len(ccols), "c5a_tbl")
        set_widths(ws, {1: 30}, 15)

    # -------------------------------------------------------------- 说明 ---
    ws = wb.create_sheet("说明", 0)
    title(ws, "E206 · desk+chair noPRG vs PRG 双 arm 评测",
          "65 case × 2 arm = 130 条 CEM，两 arm 均由新鲜 rollout 经同一代码路径打分。", 3)
    guide = [
        ("页", "内容", "读法"),
        ("funnel_summary", "两 arm 的 L1/L2/L3 与 12 门计数，含 C5a/C5b 判决",
         "计数是对 rollout 页的 COUNTIFS 公式；Δ 行蓝底"),
        ("per_gate", "逐门 mean/median/std/worst + narrow 通过率",
         "只看 mean 会被离群主导 —— root_pos 的 mean 46.9cm 是两个摔倒 case 拉的"),
        ("per_object", "逐物体 × arm", "n<3 整行置灰，不得当作 per-object 推荐"),
        ("paired_delta", "逐 case 的 PRG − noPRG", "绿=PRG 更好，红=更差，按指标方向着色"),
        ("vs_E174_desk007", "C5a 同尺对比", "红底行 = E174 接触恰为 0（F7 指纹）"),
        ("rollout", "130 行原始打分", "其余各页公式的数据源；P/F 绿红，layer 三色"),
    ]
    r = 4
    for a, b, c in guide:
        ws.cell(r, 1, a); ws.cell(r, 2, b); ws.cell(r, 3, c)
        for col in range(1, 4):
            cell = ws.cell(r, col)
            cell.alignment = Alignment(vertical="center", wrap_text=True)
            if r == 4:
                cell.font = Font(name=ARIAL, size=9, bold=True, color=WHITE)
                cell.fill = PatternFill("solid", fgColor=BLUE)
            else:
                cell.font = Font(name=ARIAL, size=9, bold=col == 1)
                cell.fill = PatternFill("solid", fgColor=LIGHT_BLUE if r % 2 else WHITE)
            cell.border = Border(bottom=THIN)
        ws.row_dimensions[r].height = 30
        r += 1
    set_widths(ws, {1: 20, 2: 46, 3: 62})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(json.dumps({"workbook": str(OUT.relative_to(REPO)),
                      "sheets": wb.sheetnames, "paired_cases": len(paired),
                      "C5b": {"leg_pen_narrow_pct": {k: round(v, 1) for k, v in legpen.items()},
                              "delta_pp": round(dpp, 1), "L3_delta": l3d,
                              "verdict": verdict}},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
