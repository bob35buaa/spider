#!/usr/bin/env python3
"""E208 workbook: aug-vs-orig, stratified, with under-powered cells made obvious.

The one rule this file exists to enforce: **a cell with n < 3 never shows an
inferential number.**  It is greyed, its point estimate is prefixed with `~`, and
its p-value / std / worst columns are blanked with a note.  Without that, a
20-cell (object x variant) grid invites reading a p=0.5 from n=1 as a result.

Sheet order follows the questions in the order they can invalidate each other:

  README        what was run, what the three different "pass rates" mean
  C4            the pre-registered verdict, pooled -- the headline
  ByOffsetBand  the new question: at what amplitude does tracking start to cost?
  ByVariant     is one augmentation direction worse than the others?
  ByObjectVariant  the 20-cell grid, mostly indicative by construction
  PerCase       every pair, so nothing is hidden behind an aggregate
  F15Watch      the two cases whose retarget was known non-reproducible
  Gates         which gate actually moved, orig vs aug

Usage:
    .venv/bin/python .../eval/reports/gen_E208_aug_workbook.py
"""

from __future__ import annotations

import argparse
import json
import math
import numpy as np
import sys
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment, Font, PatternFill

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E208"))
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E201"))

import e208_common as C  # noqa: E402
import funnel_config as FC  # noqa: E402

# The full 14-gate funnel caliber (4 hard + 10 banded), taken BY VALUE from
# funnel_config so E208 reads on the same scale as E204/E205/E206.  This is a
# superset of both the 6-gate C4 set and the 12-gate review view; all three are
# shown so the reader can see exactly which caliber produced which number.
GATE14 = ([(n, f, op, thr, None) for n, f, op, thr in FC.HARD_GATES]
          + [(n, f, op, nar, wide) for n, f, op, nar, wide in FC.BANDED_GATES])
GATE14_NAMES = [g[0] for g in GATE14]
C4_GATES = {"fall", "obj_pos", "obj_ori", "contact", "hand_pen", "leg_pen"}

EVAL_DIR = C.EVAL_DIR
SUMMARY_JSON = EVAL_DIR / "e208_aug_eval_summary.json"
DELTAS_TSV = EVAL_DIR / "e208_aug_deltas.tsv"
OUT_XLSX = EVAL_DIR / "E208_aug_vs_orig.xlsx"

NAVY, BLUE, GREEN, RED, GRAY, WHITE, AMBER = (
    "1F3864", "2E5496", "C6EFCE", "FFC7CE", "D9D9D9", "FFFFFF", "FFE699")
# Delta text colours -- dark enough to stay readable on the light pass/fail fills.
# 8-digit aRGB: a 6-digit value is normalised to alpha 00 (fully transparent),
# which can render as invisible text in Excel.
TEXT_BETTER, TEXT_WORSE, TEXT_SAME = "FF008000", "FFC00000", "FF808080"

INDICATIVE_MIN_N = 3
PRIMARY = "track_obj_pos_err_cm_mean"
HEADLINE = [
    ("track_obj_pos_err_cm_mean", True), ("track_obj_ori_err_deg_mean", True),
    ("hand_object_physics_contact_in_mask_frac", False),
    ("hand_object_physics_penetration_3mm_frame_frac", True),
    ("leg_penetration_frac", True), ("body_z_err_p95_m", True),
]


def finite(v: Any) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else math.nan
    except (TypeError, ValueError):
        return math.nan


def _title(ws, text: str, subtitle: str, ncol: int) -> None:
    ws["A1"] = text
    ws["A1"].font = Font(size=14, bold=True, color=WHITE)
    ws["A1"].fill = PatternFill("solid", fgColor=NAVY)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max(ncol, 2))
    ws["A2"] = subtitle
    ws["A2"].font = Font(size=9, italic=True, color="666666")
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=max(ncol, 2))


def _hdr(ws, row: int, ncol: int) -> None:
    for cell in ws[row][:ncol]:
        cell.font = Font(bold=True, color=WHITE)
        cell.fill = PatternFill("solid", fgColor=BLUE)
        cell.alignment = Alignment(horizontal="center", wrap_text=True)


def _verdict_fill(ws, row: int, col: int, ok: bool) -> None:
    ws.cell(row, col).fill = PatternFill("solid", fgColor=GREEN if ok else RED)


def _mark_indicative(ws, row: int, ncol: int, n: int) -> None:
    """Grey the row and blank its inferential columns. `n` is a CASE count."""
    for c in range(1, ncol + 1):
        ws.cell(row, c).fill = PatternFill("solid", fgColor=GRAY)
    ws.cell(row, ncol).value = (
        f"indicative ({n} independent cases < {INDICATIVE_MIN_N}); point estimate only")
    ws.cell(row, ncol).font = Font(size=8, italic=True, color="666666")


def stratum_sheet(wb: Workbook, name: str, title: str, subtitle: str,
                  cells: dict[str, Any]) -> None:
    ws = wb.create_sheet(name)
    cols = ["stratum", "n", "cases", "orig pass", "aug pass", "pass drop", "McNemar p",
            f"HL {PRIMARY}", "mean Δ", "std Δ", "worst Δ", "test", "p", "note"]
    _title(ws, title, subtitle, len(cols))
    ws.append([])
    ws.append(cols)
    _hdr(ws, ws.max_row, len(cols))

    for key, cell in sorted(cells.items(), key=lambda kv: -kv[1]["n"]):
        g, m = cell["gates"], cell["metrics"][PRIMARY]
        n = cell["n"]
        # independent units are CASES, not rows: each case contributes 5 variants
        # sharing one orig, and all 5 land in the same offset band (0/21 cross).
        indicative = cell.get("n_cases", n) < INDICATIVE_MIN_N
        hl = m.get("hl_shift")
        row = [
            key, n, cell.get("n_cases", ""),
            round(g["orig_pass_rate"], 3) if g["orig_pass_rate"] is not None else "",
            round(g["aug_pass_rate"], 3) if g["aug_pass_rate"] is not None else "",
            round(g["pass_rate_drop"], 3) if g["pass_rate_drop"] is not None else "",
            "" if indicative else round(g.get("exact_p", float("nan")), 4),
            ("~" if indicative else "") + (f"{hl:+.3f}" if hl is not None
                                           and math.isfinite(hl) else ""),
            "" if indicative else round(m.get("mean_delta", float("nan")), 3),
            "" if indicative else round(m.get("std_delta", float("nan")), 3),
            "" if indicative else round(m.get("worst_delta", float("nan")), 3),
            "" if indicative else m.get("primary_test", ""),
            "" if indicative else round(
                m.get("wilcoxon_p", m.get("sign_test_p", float("nan"))), 4),
            "",
        ]
        ws.append(row)
        r = ws.max_row
        if indicative:
            _mark_indicative(ws, r, len(cols), cell.get("n_cases", n))
        else:
            _verdict_fill(ws, r, 6, (g["pass_rate_drop"] or 0) <= 0.15)
            _verdict_fill(ws, r, 8, hl is not None and hl <= 5.0)
    for i, w in enumerate([24, 6, 7, 10, 10, 10, 11, 16, 10, 10, 10, 16, 9, 46], 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def sheet_readme(wb: Workbook, summary: dict[str, Any]) -> None:
    # An older eval_summary.json predates the clustering block; degrade instead of
    # crashing, so a stale summary still produces a readable workbook.
    cl = summary.get("clustering") or {}
    ws = wb.active
    ws.title = "README"
    _title(ws, "E208 · desk/chair 物体增强 vs orig（R294）",
           "21 case × 5 变体（trans0/1/2 + rot0/1），PRG arm，CEM 1024×32 seed0。"
           "orig 基线复用 E206 的 PRG rollout，不重跑。chair005 已排除。", 4)
    lines = [
        ("范围", f"{summary['n_pairs']} 对配对（aug {summary['n_aug_scored']} / "
                 f"orig {summary['n_orig']}），排除 {summary['excluded_objects']}"
                 f"（{summary['excluded_reason']}）"),
        ("C4 判据（预注册）", "① 配对通过率下降 ≤ 0.15（McNemar）；"
                            "② track_obj_pos_err_cm_mean 的 HL 中位差 ≤ +5.0 cm"),
        ("C4 粒度", summary["C4"]["granularity"] + " —— " + summary["C4"]["granularity_basis"]),
        ("点估计为何用 HL", "n 小且 obj_pos 重尾，均值差会被单个失败 run 拖走；"
                          "HL 是中位数的稳健配对版本"),
        ("n<3 的格", summary["caveats"]["indicative_cells"]),
        ("⚠ 配对是聚类的",
         f"{cl.get('n_rows', '?')} 对来自 {cl.get('n_cases', '?')} 个 case（每例 "
         f"{cl.get('rows_per_case', '?')} 条，共享同一条 orig），且每例的 5 个变体落在"
         "同一 offset 档。Wilcoxon/McNemar 假设配对独立，故**行级 p 值偏小**；"
         "点估计不受影响。分层的独立单元数看 cases 列，不是 n 列。"),
        ("  case 级对照",
         (f"把每例先塌缩成一个数再检验（{cl.get('n_cases', '?')} 个独立单元）：平均通过率下降 "
          f"{cl['case_level_mean_pass_rate_drop']:+.3f}，变差/变好/持平 = "
          f"{cl['case_level_n_cases_worse']}/{cl['case_level_n_cases_better']}/"
          f"{cl['case_level_n_cases_unchanged']} 例")
         if cl else "（本次 summary 尚无 clustering 块，重跑 eval 后生成）"),
        ("", ""),
        ("⚠ 三个不同的通过率", "同一批 case 有三个互不相等的数，都对，口径不同："),
        ("  E206 人审", "22/22 USE"),
        ("  E206 发布口径", "12 门 numeric_release_pass"),
        ("  本表口径", "6 门增强线标准（E199/E202 逐字沿用），同时施加于 orig 与 aug"),
        ("  差异根源", "E206 的 lower_body 门在 leg_penetration_frac≈0.20 放行，"
                      "本表用 0.10（更严）。逐例分歧见 eval_summary.json 的 "
                      "gate_criterion_vs_e206"),
        ("", ""),
        ("⚠ chair006 接触盲区",
         f"blind3cm={summary['caveats']['chair006_blind3cm']['value']} —— "
         + summary["caveats"]["chair006_blind3cm"]["note"]),
        ("rot 变体", "rot_* 不是纯旋转：每档带 ±45° yaw + 0.2 m 侧移，且 yaw 用 tau=25 "
                    "衰减而位移用 tau=50。这是本仓库第一次真正跑出旋转增强"
                    "（此前上游 scipy 形状崩溃，从未执行过 IK）"),
        ("rescore_orig", str(summary["rescore_orig"].get("verdict", "skipped"))
         + " —— 检验的是打分可复现性（同一 npz 打两次），不是 retarget 可复现性"),
    ]
    r = 4
    for k, v in lines:
        ws.cell(r, 1, k).font = Font(bold=True)
        ws.cell(r, 2, v).alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=4)
        r += 1
    ws.column_dimensions["A"].width = 22
    for col in ("B", "C", "D"):
        ws.column_dimensions[col].width = 40


def sheet_c4(wb: Workbook, summary: dict[str, Any]) -> None:
    ws = wb.create_sheet("C4")
    c4 = summary["C4"]
    _title(ws, "C4 · aug 数值不劣于 orig（预注册判据）",
           "pooled；判据在任何 aug 结果被打分之前就已冻结", 5)
    ws.append([])
    ws.append(["判据", "实测", "门", "结论", "备注"])
    _hdr(ws, ws.max_row, 5)
    for key, chk in c4["checks"].items():
        ws.append([key, round(chk["value"], 4) if chk["value"] is not None else "",
                   chk["bar"], chk["verdict"], ""])
        _verdict_fill(ws, ws.max_row, 4, chk["verdict"] == "pass")
    ws.append([])
    ws.append(["总判定", "", "", c4["verdict"], c4["granularity_basis"]])
    _verdict_fill(ws, ws.max_row, 4, c4["verdict"] == "pass")

    pooled = summary["strata"]["pooled"]["all"]
    ws.append([])
    ws.append(["全部 KEY_METRICS 的配对结果（pooled）"])
    ws.cell(ws.max_row, 1).font = Font(bold=True)
    ws.append(["metric", "n", "HL 中位差", "mean Δ", "std Δ", "worst Δ", "主检验", "p"])
    _hdr(ws, ws.max_row, 8)
    for metric, lower_better in HEADLINE:
        m = pooled["metrics"][metric]
        ws.append([metric, m["n"],
                   round(m.get("hl_shift", float("nan")), 4),
                   round(m.get("mean_delta", float("nan")), 4),
                   round(m.get("std_delta", float("nan")), 4),
                   round(m.get("worst_delta", float("nan")), 4),
                   m.get("primary_test", ""),
                   round(m.get("wilcoxon_p", m.get("sign_test_p", float("nan"))), 4)])
        hl = m.get("hl_shift", float("nan"))
        if math.isfinite(hl):
            good = (hl <= 0) if lower_better else (hl >= 0)
            ws.cell(ws.max_row, 3).fill = PatternFill(
                "solid", fgColor=GREEN if good else AMBER)
    for i, w in enumerate([48, 6, 14, 12, 12, 12, 18, 10], 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def sheet_per_case(wb: Workbook, deltas: list[dict[str, str]]) -> None:
    ws = wb.create_sheet("PerCase")
    cols = ["object", "case_id", "variant", "band", "offset(m)", "retarget",
            "orig pass", "aug pass", "aug 失败门",
            f"orig {PRIMARY}", f"aug {PRIMARY}", "Δ", "Δ%",
            "Δ contact", "Δ leg_pen", "F15"]
    _title(ws, "PerCase · 每一对 aug-vs-orig",
           "全量列出，不做筛选。band 是有效位移分档（full≥0.18 / partial 0.10–0.18 / weak<0.10）", len(cols))
    ws.append([])
    ws.append(cols)
    _hdr(ws, ws.max_row, len(cols))
    for d in sorted(deltas, key=lambda r: (r["object_key"], r["case_id"], r["aug_variant"])):
        ws.append([
            d["object_key"], d["case_id"], d["aug_variant"], d.get("offset_band", ""),
            round(finite(d.get("approach_trans_offset_m_max")), 4),
            d.get("effective_retarget_variant", ""),
            d["orig_all_gates_pass"], d["aug_all_gates_pass"], d.get("aug_failure_modes", ""),
            round(finite(d[f"orig_{PRIMARY}"]), 3), round(finite(d[f"aug_{PRIMARY}"]), 3),
            round(finite(d[f"delta_{PRIMARY}"]), 3), round(finite(d[f"pct_{PRIMARY}"]), 1),
            round(finite(d["delta_hand_object_physics_contact_in_mask_frac"]), 4),
            round(finite(d["delta_leg_penetration_frac"]), 4),
            "Y" if str(d.get("f15_divergent", "0")) == "1" else "",
        ])
        r = ws.max_row
        dv = finite(d[f"delta_{PRIMARY}"])
        if math.isfinite(dv):
            ws.cell(r, 12).fill = PatternFill("solid", fgColor=GREEN if dv <= 0 else
                                              (AMBER if dv <= 5.0 else RED))
        if C.truth(d["orig_all_gates_pass"]) and not C.truth(d["aug_all_gates_pass"]):
            ws.cell(r, 8).fill = PatternFill("solid", fgColor=RED)
    ws.freeze_panes = "A4"
    for i, w in enumerate([10, 32, 8, 9, 10, 12, 10, 10, 26, 14, 14, 10, 9, 12, 12, 5], 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def sheet_f15(wb: Workbook, deltas: list[dict[str, str]], summary: dict[str, Any]) -> None:
    ws = wb.create_sheet("F15Watch")
    watch = set(summary["caveats"]["f15_watch"])
    rows = [d for d in deltas if d["case_id"] in watch]
    _title(ws, "F15 watch · retarget 曾被测出不可复现的 case",
           "E208 通过从 E206 逐字节播种 `_original` 从构造上规避了 F15，"
           "这里单列只为可追溯。另：G4 实测 aug IK 跨进程逐字节确定，"
           "说明 F15 的发散不来自求解器随机性。", 6)
    ws.append([])
    ws.append(["case_id", "variant", "band", f"Δ {PRIMARY}", "orig pass", "aug pass"])
    _hdr(ws, ws.max_row, 6)
    for d in sorted(rows, key=lambda r: (r["case_id"], r["aug_variant"])):
        ws.append([d["case_id"], d["aug_variant"], d.get("offset_band", ""),
                   round(finite(d[f"delta_{PRIMARY}"]), 3),
                   d["orig_all_gates_pass"], d["aug_all_gates_pass"]])
    if not rows:
        ws.append(["（本 registry 内的 F15 发散例只有 chair005 与 desk023；"
                   "chair005 已排除）"])
    for i, w in enumerate([32, 9, 9, 16, 10, 10], 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def sheet_gates(wb: Workbook, deltas: list[dict[str, str]]) -> None:
    ws = wb.create_sheet("Gates")
    _title(ws, "Gates · 哪一道门真的动了", "统计 aug 相对 orig 的门翻转方向", 5)
    ws.append([])
    ws.append(["门", "orig 过 → aug 挂", "orig 挂 → aug 过", "净变化", "aug 失败计数"])
    _hdr(ws, ws.max_row, 5)
    gates = ["fall", "object_pos", "object_ori", "contact", "hand_penetration", "lower_body"]
    for gate in gates:
        lost = sum(1 for d in deltas
                   if C.truth(d["orig_all_gates_pass"])
                   and gate in (d.get("aug_failure_modes") or ""))
        aug_fail = sum(1 for d in deltas if gate in (d.get("aug_failure_modes") or ""))
        ws.append([gate, lost, "", -lost, aug_fail])
        if lost:
            ws.cell(ws.max_row, 2).fill = PatternFill("solid", fgColor=RED)
    ws.append([])
    ws.append(["注", "「orig 挂 → aug 过」需要 orig 的逐门明细，当前 deltas 表只记 "
                     "orig 的总判定，故留空；如需该列，在 eval runner 里把 orig 的 "
                     "numeric_failure_modes 一并写出。"])
    ws.cell(ws.max_row, 2).alignment = Alignment(wrap_text=True)
    for i, w in enumerate([22, 18, 18, 12, 14], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.column_dimensions["B"].width = 18



def gate_eval(row: dict[str, str], name: str, field: str, op: str,
              thr: float, wide: float | None) -> dict[str, Any]:
    """One gate's value plus narrow/wide verdicts, with n/a kept distinct from fail."""
    if op == "fall":
        fell = str(row.get(field, "")).strip().lower() in ("true", "1")
        return {"value": 1.0 if fell else 0.0, "narrow": not fell,
                "wide": not fell, "na": False}
    value = finite(row.get(field, ""))
    na = math.isnan(value)
    return {
        "value": value,
        # Convention kept identical to E206/E204 (FC.passes fails a non-finite
        # value) so the pass counts stay comparable; `na` is carried separately
        # so a not-applicable gate is visible rather than posing as a failure.
        "narrow": FC.passes(op, value, thr),
        "wide": FC.passes(op, value, wide) if wide is not None else FC.passes(op, value, thr),
        "na": na,
    }



def better_is_lower(op: str) -> bool:
    """Gate direction, read off its own operator -- only `contact` is >=."""
    return op != ">="


def gate_cell_value(name: str, op: str, cur: dict[str, Any],
                    ref: dict[str, Any] | None):
    """`value(delta)` with the delta coloured by whether it beat orig.

    The sign alone does not say better-or-worse: `contact` is a >= gate, so a
    positive delta is an improvement there and a regression everywhere else.
    Colour is therefore derived from the gate's own operator, never from the sign.
    """
    if cur["na"]:
        return "n/a"
    value = f"{cur['value']:.4f}"
    if ref is None or ref["na"]:
        return value
    delta = cur["value"] - ref["value"]
    if abs(delta) < 5e-5:
        colour = TEXT_SAME
    else:
        improved = (delta < 0) if better_is_lower(op) else (delta > 0)
        colour = TEXT_BETTER if improved else TEXT_WORSE
    return CellRichText(
        TextBlock(InlineFont(), value),
        TextBlock(InlineFont(color=colour), f"({delta:+.4f})"),
    )


def sheet_gate14_values(wb: Workbook, scored: list[dict[str, str]]) -> None:
    """Every row's raw value for all 14 gates -- orig and aug side by side."""
    ws = wb.create_sheet("Gate14Values")
    cols = (["object", "case_id", "variant", "band"]
            + GATE14_NAMES
            + ["narrow 过", "wide 过", "14门全过(narrow)", "narrow 失败门", "n/a 门"])
    _title(ws, "Gate14Values · 14 门完整数值（4 硬门 + 10 带门）",
           "格式 值(Δ)，Δ = aug − orig；orig 行无 Δ。**Δ 的颜色已按每个门自己的方向判定**"
           "（contact 越大越好，其余越小越好）：绿=比 orig 好，红=比 orig 差，灰=持平。"
           "单元格底色是另一件事：粉=该门本身挂，灰=n/a。"
           "阈值取自 funnel_config，与 E204/E205/E206 同一把尺。", len(cols))
    ws.append([])
    ws.append(cols)
    _hdr(ws, ws.max_row, len(cols))

    by_case = {r["case_id"]: r for r in scored if r["aug_variant"] == "orig"}
    order = {"orig": 0, "trans0": 1, "trans1": 2, "trans2": 3, "rot0": 4, "rot1": 5}
    for row in sorted(scored, key=lambda r: (r["object_key"], r["case_id"],
                                             order.get(r["aug_variant"], 9))):
        res = {n: gate_eval(row, n, f, op, thr, wide) for n, f, op, thr, wide in GATE14}
        failed = [n for n in GATE14_NAMES if not res[n]["narrow"]]
        nas = [n for n in GATE14_NAMES if res[n]["na"]]
        is_orig = row["aug_variant"] == "orig"
        base = by_case.get(row["case_id"]) if not is_orig else None
        ref = ({n: gate_eval(base, n, f, op, thr, wide)
                for n, f, op, thr, wide in GATE14} if base is not None else None)

        ws.append(
            [row["object_key"], row["case_id"], row["aug_variant"],
             row.get("offset_band", "")]
            + [None] * len(GATE14_NAMES)
            + [sum(1 for n in GATE14_NAMES if res[n]["narrow"]),
               sum(1 for n in GATE14_NAMES if res[n]["wide"]),
               "TRUE" if not failed else "FALSE",
               ",".join(failed), ",".join(nas)])
        r = ws.max_row
        for i, (n, _f, op, _thr, _wide) in enumerate(GATE14, 5):
            cell = ws.cell(r, i)
            cell.value = gate_cell_value(n, op, res[n], ref[n] if ref else None)
            cell.alignment = Alignment(horizontal="right")
            if res[n]["na"]:
                cell.fill = PatternFill("solid", fgColor=GRAY)
            elif not res[n]["narrow"]:
                cell.fill = PatternFill("solid", fgColor=RED)
        if is_orig:
            for i in range(1, 5):
                ws.cell(r, i).font = Font(bold=True)
    ws.freeze_panes = "E4"
    for i, w in enumerate([10, 32, 8, 9] + [19] * len(GATE14_NAMES)
                          + [10, 9, 16, 34, 16], 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def sheet_gate14_compare(wb: Workbook, scored: list[dict[str, str]]) -> None:
    """Per pair, every gate's orig value, aug value and delta."""
    ws = wb.create_sheet("Gate14Compare")
    cols = ["object", "case_id", "variant", "band"]
    for n in GATE14_NAMES:
        cols += [f"{n} orig", f"{n} aug", f"{n} Δ"]
    _title(ws, "Gate14Compare · 每一对在 14 门上的 orig / aug / Δ",
           "红=aug 挂而 orig 过（该门被增强弄挂了）；黄=两侧都挂；绿=aug 过而 orig 挂。"
           "Δ 的符号按原始指标方向，未按「越小越好」归一化。", len(cols))
    ws.append([])
    ws.append(cols)
    _hdr(ws, ws.max_row, len(cols))

    by_key = {(r["case_id"], r["aug_variant"]): r for r in scored}
    order = {"trans0": 1, "trans1": 2, "trans2": 3, "rot0": 4, "rot1": 5}
    pairs = [(k, v) for k, v in by_key.items() if k[1] != "orig"]
    for (case_id, variant), aug_row in sorted(
            pairs, key=lambda kv: (kv[1]["object_key"], kv[0][0], order.get(kv[0][1], 9))):
        orig_row = by_key.get((case_id, "orig"))
        if orig_row is None:
            continue
        line = [aug_row["object_key"], case_id, variant, aug_row.get("offset_band", "")]
        marks = []
        for n, f, op, thr, wide in GATE14:
            a = gate_eval(aug_row, n, f, op, thr, wide)
            o = gate_eval(orig_row, n, f, op, thr, wide)
            av = "n/a" if a["na"] else round(a["value"], 4)
            ov = "n/a" if o["na"] else round(o["value"], 4)
            dv = ("" if (a["na"] or o["na"])
                  else round(a["value"] - o["value"], 4))
            line += [ov, av, dv]
            marks.append((o["narrow"], a["narrow"]))
        ws.append(line)
        r = ws.max_row
        for i, (o_ok, a_ok) in enumerate(marks):
            col = 5 + i * 3 + 1          # the "aug" column of this gate
            if o_ok and not a_ok:
                ws.cell(r, col).fill = PatternFill("solid", fgColor=RED)
            elif not o_ok and not a_ok:
                ws.cell(r, col).fill = PatternFill("solid", fgColor=AMBER)
            elif not o_ok and a_ok:
                ws.cell(r, col).fill = PatternFill("solid", fgColor=GREEN)
    ws.freeze_panes = "E4"
    for i, w in enumerate([10, 32, 8, 9] + [10] * (3 * len(GATE14_NAMES)), 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def sheet_gate14_summary(wb: Workbook, scored: list[dict[str, str]]) -> None:
    ws = wb.create_sheet("Gate14Summary")
    cols = ["门", "指标", "op", "narrow", "wide", "在 C4 的 6 门内",
            "orig 过/21", "aug 过/105", "orig 过率", "aug 过率", "过率差",
            "orig过→aug挂", "orig挂→aug过", "Δ 中位", "Δ p90", "n/a 数"]
    _title(ws, "Gate14Summary · 逐门汇总（哪一道门真的动了）",
           "orig 分母 21（每例一条），aug 分母 105。「过率差」= orig 过率 − aug 过率，正值表示增强后变差。", len(cols))
    ws.append([])
    ws.append(cols)
    _hdr(ws, ws.max_row, len(cols))

    by_key = {(r["case_id"], r["aug_variant"]): r for r in scored}
    origs = [r for r in scored if r["aug_variant"] == "orig"]
    augs = [r for r in scored if r["aug_variant"] != "orig"]
    for n, f, op, thr, wide in GATE14:
        o_pass = sum(1 for r in origs if gate_eval(r, n, f, op, thr, wide)["narrow"])
        a_pass = sum(1 for r in augs if gate_eval(r, n, f, op, thr, wide)["narrow"])
        lost = gained = 0
        deltas: list[float] = []
        for a in augs:
            o = by_key.get((a["case_id"], "orig"))
            if o is None:
                continue
            ra, ro = gate_eval(a, n, f, op, thr, wide), gate_eval(o, n, f, op, thr, wide)
            if ro["narrow"] and not ra["narrow"]:
                lost += 1
            if not ro["narrow"] and ra["narrow"]:
                gained += 1
            if not (ra["na"] or ro["na"]):
                deltas.append(ra["value"] - ro["value"])
        na = sum(1 for r in scored if gate_eval(r, n, f, op, thr, wide)["na"])
        o_rate = o_pass / len(origs) if origs else math.nan
        a_rate = a_pass / len(augs) if augs else math.nan
        arr = np.array(deltas) if deltas else np.array([math.nan])
        ws.append([n, f, op, thr, wide if wide is not None else "",
                   "✓" if n in C4_GATES else "",
                   o_pass, a_pass, round(o_rate, 3), round(a_rate, 3),
                   round(o_rate - a_rate, 3), lost, gained,
                   round(float(np.nanmedian(arr)), 4),
                   round(float(np.nanpercentile(arr, 90)), 4), na])
        r = ws.max_row
        if lost:
            ws.cell(r, 12).fill = PatternFill("solid", fgColor=RED)
        _verdict_fill(ws, r, 11, (o_rate - a_rate) <= 0.15)
    for i, w in enumerate([12, 46, 5, 9, 8, 14, 11, 11, 10, 10, 9, 13, 13, 10, 10, 7], 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_XLSX)
    args = ap.parse_args()

    if not SUMMARY_JSON.is_file():
        raise SystemExit(f"missing {C.rel(SUMMARY_JSON)} -- run eval_E208_aug.py first")
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    deltas = C.read_tsv(DELTAS_TSV)

    wb = Workbook()
    sheet_readme(wb, summary)
    sheet_c4(wb, summary)
    strata = summary["strata"]
    stratum_sheet(wb, "ByOffsetBand", "按有效位移分档",
                  "E208 新增的分层维度：增强幅度多大时才开始付出跟踪代价。"
                  "full ≥0.18 m / partial 0.10–0.18 / weak <0.10",
                  strata["by_offset_band"])
    stratum_sheet(wb, "ByVariant", "按增强变体",
                  "trans0/1/2 = 0.2 m 前/左/右；rot0/1 = ±45° yaw + 0.2 m 侧移",
                  strata["by_variant"])
    stratum_sheet(wb, "ByObject", "按物体", "chair005 已排除（衰减退化）",
                  strata["by_object"])
    stratum_sheet(wb, "ByObjectVariant", "按 物体 × 变体（20 格）",
                  "多数格 n 很小，按规则强制 indicative —— 灰底、数值带 ~、不出 p 值",
                  strata["by_object_variant"])
    scored = C.read_tsv(EVAL_DIR / "e208_aug_rollout.tsv")
    sheet_gate14_summary(wb, scored)
    sheet_gate14_values(wb, scored)
    sheet_gate14_compare(wb, scored)
    sheet_per_case(wb, deltas)
    sheet_f15(wb, deltas, summary)

    out = C.repo_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    n_ind = sum(1 for level in strata.values() for c in level.values()
                if c["n"] < INDICATIVE_MIN_N)
    print(f"[done] {C.rel(out)}  sheets={len(wb.sheetnames)} {wb.sheetnames}")
    print(f"  pairs={summary['n_pairs']}  indicative cells={n_ind}")
    print(f"  C4 verdict={summary['C4']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
