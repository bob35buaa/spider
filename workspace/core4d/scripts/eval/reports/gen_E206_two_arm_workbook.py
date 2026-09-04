#!/usr/bin/env python3
"""E206 P9 report: the two-arm workbook, formatted for reading rather than parsing.

Sheets:
  说明             what each sheet is + the C5a / C5b verdicts up front
  funnel_summary   per-arm L1/L2/L3 and 12-gate counts (COUNTIFS over `rollout`)
  人审汇总          C6: the 14-gate funnel scored against the human verdict
  人审明细          all 65 reviewed PRG rows, gate result beside human verdict
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
  * the human review covers the PRG arm only (65/65 rows). noPRG manual cells are
    left blank rather than inferred; every count that mixes the two filters on
    arm="prg" explicitly.
  * gate/human disagreements are listed case by case, not summarised away. An
    L3_auto row the human calls UNUSABLE is an auto-accept escape and is the most
    expensive error the funnel can make (plan236 C6).

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
REVIEW = EVAL / "user_manual_review_filled.tsv"

# The human reviewed one arm; `case_id` there carries a `<case>#<ARM>` suffix.
REVIEW_ARM = "prg"

NAVY = "17365D"
BLUE = "4472C4"
LIGHT_BLUE = "D9EAF7"
LIGHT_GREEN = "E2F0D9"
LIGHT_RED = "FCE4D6"
LIGHT_AMBER = "FFF2CC"
LIGHT_GRAY = "E7E6E6"
LIGHT_PURPLE = "E4DFEC"   # gate/human disagreement
WHITE = "FFFFFF"
GREEN_TXT = "1E7145"
RED_TXT = "9C0006"
PURPLE_TXT = "5F497A"

USE_FILL = {"USE": LIGHT_GREEN, "DO_NOT_USE": LIGHT_RED}
QUALITY_FILL = {"CLEAN": LIGHT_GREEN, "MINOR_ACCEPTABLE": LIGHT_AMBER,
                "UNUSABLE": LIGHT_RED}
LAYER_FILL = {"L3_auto": LIGHT_GREEN, "L2_review": LIGHT_AMBER,
              "L1_reject": LIGHT_RED}
THIN = Side(style="thin", color="B7B7B7")
ARIAL = "Arial"

# metric -> True when a LOWER value is better
LOWER_BETTER = {g[1]: (g[2] == "<=") for g in FC.BANDED_GATES}
LOWER_BETTER["body_z_err_p95_m"] = True

def load_review() -> dict[str, dict[str, str]]:
    """`case_id` -> filled review row, for the one arm the human scored.

    Rows are keyed `<case_id>#<ARM>`; anything for another arm is ignored so a
    later second-arm pass cannot silently overwrite this one.
    """
    if not REVIEW.is_file():
        return {}
    out: dict[str, dict[str, str]] = {}
    for r in C.read_tsv(REVIEW):
        raw = r.get("case_id", "")
        if "#" not in raw:
            continue
        case_id, arm = raw.rsplit("#", 1)
        if arm.lower() != REVIEW_ARM:
            continue
        out[case_id] = r
    return out


def conflict_kind(layer: str, decision: str) -> str:
    """Classify a gate/human disagreement, or "" when they agree.

    L3 is auto-accept, so an UNUSABLE L3 row ships unreviewed -- the expensive
    direction. An L1 row the human keeps is only a lost sample.
    """
    if layer == "L3_auto" and decision == "DO_NOT_USE":
        return "漏网 · L3自动接受但人审否决"
    if layer == "L1_reject" and decision == "USE":
        return "误杀 · L1拒绝但人审可用"
    return ""


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
    review = load_review()
    # Carry the verdict onto the scored rows so every count below is a formula
    # over one sheet rather than a second, silently diverging source of truth.
    for r in rows:
        rv = review.get(r["case_id"], {}) if r["arm"] == REVIEW_ARM else {}
        r["manual_use_decision"] = rv.get("manual_use_decision", "")
        r["manual_quality_label"] = rv.get("manual_quality_label", "")
        r["manual_conflict"] = conflict_kind(r["layer"], r["manual_use_decision"])
    reviewed = sorted(c for c in cases if c in review)
    c5a = json.loads((EVAL / "c5a_vs_e174_desk007.json").read_text(encoding="utf-8")) \
        if (EVAL / "c5a_vs_e174_desk007.json").is_file() else {}

    wb = Workbook()
    wb.remove(wb.active)

    # ------------------------------------------------------------ rollout ---
    cols = (["object_key", "case_id", "arm", "layer", "gate12_pass",
             "manual_use_decision", "manual_quality_label",
             "physics6_pass", "tracking6_pass", "narrow_pass"]
            + [g[1] for g in FC.BANDED_GATES]
            + ["body_z_err_p95_m", "fall_flag", "narrow_failed", "wide_failed"])
    ws = wb.create_sheet("rollout")
    title(ws, "E206 · 130 条 rollout 逐行结果",
          "两 arm 均由新鲜 rollout 经同一代码路径打分。P/F = 该门通过与否；"
          "layer 为 E201 14 门漏斗判定；本页是其余各页公式的数据源。"
          "人审两列仅 PRG 行有值（人只审了 PRG arm），noPRG 留空而非推断。", len(cols))
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
                cell.fill = PatternFill(
                    "solid", fgColor=LAYER_FILL.get(str(cell.value), WHITE))
                cell.alignment = Alignment(horizontal="center")
            elif name in ("manual_use_decision", "manual_quality_label"):
                v = str(cell.value or "")
                table = USE_FILL if name == "manual_use_decision" else QUALITY_FILL
                if v:
                    cell.fill = PatternFill("solid", fgColor=table.get(v, WHITE))
                    cell.font = Font(name=ARIAL, size=9, bold=True,
                                     color=RED_TXT if "UN" in v or v == "DO_NOT_USE"
                                     else GREEN_TXT)
                cell.alignment = Alignment(horizontal="center")
            elif name in LOWER_BETTER:
                cell.value = f(cell.value) if math.isfinite(f(cell.value)) else None
                cell.number_format = "0.000"
            elif name == "fall_flag":
                if truthy(cell.value):
                    cell.fill = PatternFill("solid", fgColor=LIGHT_RED)
                    cell.font = Font(name=ARIAL, size=9, bold=True, color=RED_TXT)
    finish_table(ws, 4, last, len(cols), "rollout_tbl")
    set_widths(ws, {1: 11, 2: 30, 3: 8, 4: 11, 6: 15, 7: 18,
                    len(cols) - 1: 26, len(cols): 26}, 11)
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

    # ------------------------------------------------------------ 人审汇总 ---
    conflicts: list[dict[str, str]] = []
    if reviewed:
        arm_q = f'"{REVIEW_ARM}"'
        counts = {L: {d: sum(1 for c in reviewed
                             if by[(REVIEW_ARM, c)]["layer"] == L
                             and review[c]["manual_use_decision"] == d)
                      for d in ("USE", "DO_NOT_USE")}
                  for L in ("L3_auto", "L2_review", "L1_reject")}
        conflicts = [
            {"case_id": c, "kind": conflict_kind(by[(REVIEW_ARM, c)]["layer"],
                                                 review[c]["manual_use_decision"])}
            for c in reviewed
            if conflict_kind(by[(REVIEW_ARM, c)]["layer"],
                             review[c]["manual_use_decision"])
        ]

        ws = wb.create_sheet("人审汇总", 1)
        mcols = ["漏斗层", "n", "人审 USE", "人审 DO_NOT_USE", "USE 率", "判读"]
        title(ws, "C6 · 14 门漏斗 vs 人工目视判定（PRG arm，65/65 全审）",
              "计数为对 rollout 页的 COUNTIFS 公式（已按 arm=\"prg\" 过滤）。"
              "人审是选片权威，门是排序信号 —— 两者不一致的每一例都在下方逐条列出，"
              "不做汇总掩盖（plan236 C6 / rule 5）。", len(mcols))
        ws.append([])
        ws.append(mcols)
        header(ws, 4, len(mcols))
        verdicts = {
            "L3_auto": "自动接受层：此处的 DO_NOT_USE 是漏网，代价最高",
            "L2_review": "待人工层：本就要人看，USE 率居中属预期",
            "L1_reject": "拒绝层：此处的 USE 是误杀，代价是样本损失",
        }
        for i, L in enumerate(("L3_auto", "L2_review", "L1_reject")):
            r = 5 + i
            ws.cell(r, 1, L).fill = PatternFill("solid", fgColor=LAYER_FILL[L])
            ws.cell(r, 2, f'=COUNTIFS({rng("arm")},{arm_q},{rng("layer")},"{L}")')
            for j, d in enumerate(("USE", "DO_NOT_USE")):
                ws.cell(r, 3 + j, f'=COUNTIFS({rng("arm")},{arm_q},'
                                  f'{rng("layer")},"{L}",'
                                  f'{rng("manual_use_decision")},"{d}")')
            ws.cell(r, 5, f"=IF(B{r}=0,\"\",C{r}/B{r})").number_format = "0.0%"
            ws.cell(r, 6, verdicts[L])
        total = 8
        ws.cell(total, 1, "合计")
        for c in (2, 3, 4):
            L_ = get_column_letter(c)
            ws.cell(total, c, f"=SUM({L_}5:{L_}7)")
        ws.cell(total, 5, f'=IF(B{total}=0,"",C{total}/B{total})').number_format = "0.0%"
        ws.cell(total, 6, "人审只覆盖 PRG arm；noPRG 未审，不得据此推断")
        body_font(ws, 5, total)
        for c in range(1, len(mcols) + 1):
            ws.cell(total, c).fill = PatternFill("solid", fgColor=LIGHT_BLUE)
            ws.cell(total, c).font = Font(name=ARIAL, size=9, bold=True)
        for r in range(5, total + 1):
            ws.cell(r, 3).font = Font(name=ARIAL, size=9, bold=True, color=GREEN_TXT)
            ws.cell(r, 4).font = Font(name=ARIAL, size=9, bold=True, color=RED_TXT)
            ws.cell(r, 6).alignment = Alignment(horizontal="left", vertical="center")

        use_rate = {L: counts[L]["USE"] / max(1, sum(counts[L].values()))
                    for L in counts}
        monotone = use_rate["L3_auto"] > use_rate["L2_review"] > use_rate["L1_reject"]
        escapes = counts["L3_auto"]["DO_NOT_USE"]
        overkill = counts["L1_reject"]["USE"]
        banner = [
            ("单调性", f"USE 率 L3 {use_rate['L3_auto']:.1%} > L2 "
                      f"{use_rate['L2_review']:.1%} > L1 {use_rate['L1_reject']:.1%}"
                      f"  ⇒  {'单调，门作为排序信号有效' if monotone else '非单调，门与人审不同向'}",
             LIGHT_GREEN if monotone else LIGHT_RED),
            ("漏网（最贵）", f"{escapes} 例 L3_auto 被人审判为 DO_NOT_USE —— 这些在自动流程里"
                          f"会直接放行。L3 自动接受不能单独作为出片依据。",
             LIGHT_RED if escapes else LIGHT_GREEN),
            ("误杀", f"{overkill} 例 L1_reject 被人审判为 USE —— 门偏严，损失的是样本量而非质量。",
             LIGHT_AMBER if overkill else LIGHT_GREEN),
            ("出片口径", f"人审 USE {sum(c['USE'] for c in counts.values())} 例进入 RL 导出；"
                       f"数值 12 门在 USE 例上多为 False，二者按 E187 惯例不做 AND —— "
                       f"人审是权威，数值门留作 provenance。", LIGHT_BLUE),
        ]
        r = total + 2
        for label, text, fill in banner:
            ws.cell(r, 1, label).font = Font(name=ARIAL, size=10, bold=True)
            ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=len(mcols))
            cell = ws.cell(r, 2, text)
            cell.font = Font(name=ARIAL, size=9)
            cell.alignment = Alignment(wrap_text=True, vertical="center")
            for c in range(1, len(mcols) + 1):
                ws.cell(r, c).fill = PatternFill("solid", fgColor=fill)
                ws.cell(r, c).border = Border(bottom=THIN)
            ws.row_dimensions[r].height = 32
            r += 1

        r += 1
        ws.cell(r, 1, f"逐例冲突清单（{len(conflicts)} 例）").font = Font(
            name=ARIAL, size=11, bold=True, color=PURPLE_TXT)
        r += 1
        ccols2 = ["case_id", "layer", "人审", "质量标签", "冲突类型", "关键指标"]
        for i, name in enumerate(ccols2):
            cell = ws.cell(r, i + 1, name)
            cell.font = Font(name=ARIAL, bold=True, color=WHITE, size=9)
            cell.fill = PatternFill("solid", fgColor=BLUE)
            cell.alignment = Alignment(horizontal="center", wrap_text=True)
        head_r = r
        for item in sorted(conflicts, key=lambda x: (x["kind"], x["case_id"])):
            r += 1
            c = item["case_id"]
            src, rv = by[(REVIEW_ARM, c)], review[c]
            detail = (f"legpen {f(src['leg_penetration_frac']):.3f} · "
                      f"contact {f(src['hand_object_physics_contact_in_mask_frac']):.3f} · "
                      f"未过窄门 {src.get('narrow_failed') or '无'}")
            for i, v in enumerate([c, src["layer"], rv["manual_use_decision"],
                                   rv["manual_quality_label"], item["kind"], detail]):
                cell = ws.cell(r, i + 1, v)
                cell.font = Font(name=ARIAL, size=9)
                cell.fill = PatternFill("solid", fgColor=LIGHT_PURPLE)
                cell.border = Border(bottom=THIN)
            ws.cell(r, 2).fill = PatternFill("solid", fgColor=LAYER_FILL[src["layer"]])
            ws.cell(r, 3).fill = PatternFill(
                "solid", fgColor=USE_FILL[rv["manual_use_decision"]])
            ws.cell(r, 4).fill = PatternFill(
                "solid", fgColor=QUALITY_FILL[rv["manual_quality_label"]])
            ws.cell(r, 5).font = Font(name=ARIAL, size=9, bold=True, color=PURPLE_TXT)
        ws.freeze_panes = f"A{head_r + 1}"
        set_widths(ws, {1: 34, 2: 12, 3: 14, 4: 19, 5: 30, 6: 56})

        # ------------------------------------------------------- 人审明细 ---
        ws = wb.create_sheet("人审明细", 2)
        dcols = ["object_key", "case_id", "layer", "gate12_pass", "人审", "质量标签",
                 "一致性", "leg_penetration_frac",
                 "hand_object_physics_contact_in_mask_frac",
                 "hand_object_physics_penetration_3mm_frame_frac",
                 "track_root_pos_err_cm_mean", "track_obj_pos_err_cm_mean",
                 "narrow_failed", "reviewer", "reviewed_at"]
        title(ws, f"E206 · 人工目视复审明细（PRG arm，{len(reviewed)} 例）",
              "选片权威表。紫底 = 门与人审判定相反；绿/红 = 人审 USE / DO_NOT_USE。"
              "指标列保留在旁，便于核对人眼看到的问题是否有对应数值证据。", len(dcols))
        ws.append([])
        ws.append(dcols)
        header(ws, 4, len(dcols))
        for c in sorted(reviewed, key=lambda x: (x.split("_")[0], x)):
            src, rv = by[(REVIEW_ARM, c)], review[c]
            ws.append([
                src["object_key"], c, src["layer"],
                "P" if truthy(src["gate12_pass"]) else "F",
                rv["manual_use_decision"], rv["manual_quality_label"],
                conflict_kind(src["layer"], rv["manual_use_decision"]) or "一致",
                f(src["leg_penetration_frac"]),
                f(src["hand_object_physics_contact_in_mask_frac"]),
                f(src["hand_object_physics_penetration_3mm_frame_frac"]),
                f(src["track_root_pos_err_cm_mean"]),
                f(src["track_obj_pos_err_cm_mean"]),
                src.get("narrow_failed", ""),
                rv.get("manual_reviewer", ""), rv.get("manual_reviewed_at", ""),
            ])
        last = ws.max_row
        for r in range(5, last + 1):
            for c in range(1, len(dcols) + 1):
                ws.cell(r, c).font = Font(name=ARIAL, size=9)
                ws.cell(r, c).alignment = Alignment(
                    vertical="center", horizontal="left" if c <= 3 else "right")
            for c in range(8, 13):
                ws.cell(r, c).number_format = "0.000"
            gate_ok = ws.cell(r, 4).value == "P"
            ws.cell(r, 4).fill = PatternFill(
                "solid", fgColor=LIGHT_GREEN if gate_ok else LIGHT_RED)
            ws.cell(r, 4).font = Font(name=ARIAL, size=9, bold=True,
                                      color=GREEN_TXT if gate_ok else RED_TXT)
            ws.cell(r, 3).fill = PatternFill(
                "solid", fgColor=LAYER_FILL.get(str(ws.cell(r, 3).value), WHITE))
            ws.cell(r, 5).fill = PatternFill(
                "solid", fgColor=USE_FILL.get(str(ws.cell(r, 5).value), WHITE))
            ws.cell(r, 6).fill = PatternFill(
                "solid", fgColor=QUALITY_FILL.get(str(ws.cell(r, 6).value), WHITE))
            for c in (3, 4, 5, 6):
                ws.cell(r, c).alignment = Alignment(horizontal="center")
            if ws.cell(r, 7).value != "一致":
                ws.cell(r, 7).fill = PatternFill("solid", fgColor=LIGHT_PURPLE)
                ws.cell(r, 7).font = Font(name=ARIAL, size=9, bold=True,
                                          color=PURPLE_TXT)
                ws.cell(r, 2).font = Font(name=ARIAL, size=9, bold=True,
                                          color=PURPLE_TXT)
        finish_table(ws, 4, last, len(dcols), "review_tbl")
        set_widths(ws, {1: 11, 2: 32, 3: 11, 4: 7, 5: 13, 6: 19, 7: 28,
                        13: 24, 14: 11, 15: 22}, 14)

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
            ws.cell(r, c).fill = PatternFill(
                "solid", fgColor=LAYER_FILL.get(str(ws.cell(r, c).value), WHITE))
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
          "65 case × 2 arm = 130 条 CEM，两 arm 均由新鲜 rollout 经同一代码路径打分。"
          f"人工目视复审已完成 {len(reviewed)}/65（PRG arm），"
          "出片名单以人审为权威。", 3)
    guide = [
        ("页", "内容", "读法"),
        ("funnel_summary", "两 arm 的 L1/L2/L3 与 12 门计数，含 C5a/C5b 判决",
         "计数是对 rollout 页的 COUNTIFS 公式；Δ 行蓝底"),
        ("人审汇总", "C6：14 门漏斗判定 vs 人工目视判定（仅 PRG arm）",
         "看「漏网」一行 —— L3 自动接受里被人否决的例数；紫底为逐例冲突清单"),
        ("人审明细", "65 条 PRG 人审逐行结果，指标列并排",
         "紫底行 = 门与人相反；出片名单以「人审」列为准，不与数值门取交集"),
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
    payload: dict[str, Any] = {
        "workbook": str(OUT.relative_to(REPO)),
        "sheets": wb.sheetnames, "paired_cases": len(paired),
        "C5b": {"leg_pen_narrow_pct": {k: round(v, 1) for k, v in legpen.items()},
                "delta_pp": round(dpp, 1), "L3_delta": l3d, "verdict": verdict},
    }
    if reviewed:
        payload["C6"] = {
            "reviewed_arm": REVIEW_ARM, "reviewed": len(reviewed),
            "use": sum(1 for c in reviewed
                       if review[c]["manual_use_decision"] == "USE"),
            "layer_x_manual": {L: counts[L] for L in counts},
            "use_rate": {L: round(v, 3) for L, v in use_rate.items()},
            "monotone": monotone,
            "conflicts": {"n": len(conflicts),
                          "l3_auto_rejected_by_human": escapes,
                          "l1_reject_kept_by_human": overkill,
                          "cases": [x["case_id"] for x in conflicts]},
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
