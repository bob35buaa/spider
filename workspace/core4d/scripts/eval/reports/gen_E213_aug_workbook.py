#!/usr/bin/env python3
"""E213 workbook: selected-arm aug-vs-orig, stratified, under-powered cells greyed.

Same discipline as gen_E208_aug_workbook.py (n<3 cells never show an inferential
number), but E213's orig baseline is each case's OWN selected-arm rollout (E209
G1 / E211 gc / E212 gc / E206 PRG) rather than a single PRG arm, so a new headline
sheet **ByArm** replaces E208's PRG-only view.

Sheets:
  README         what was run; orig baseline = per-case selected arm; row vs case level
  C4             pre-registered verdict (pooled + case-level companion)
  ByArm          the headline: does augmenting cost more on some arms than others?
  ByOffsetBand   at what effective amplitude does tracking start to cost?
  ByVariant      is one augmentation direction worse?
  ByObjectVariant the fine grid (mostly indicative)
  PerCase        every pair, nothing hidden
  Gate14Summary/Values/Compare  full 14-gate funnel caliber, orig vs aug
  Gates          which gate moved

Usage:
    .venv/bin/python workspace/core4d/scripts/eval/reports/gen_E213_aug_workbook.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from openpyxl import Workbook
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment, Font, PatternFill

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E213"))
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E201"))

import e213_common as C  # noqa: E402
import funnel_config as FC  # noqa: E402

GATE14 = ([(n, f, op, thr, None) for n, f, op, thr in FC.HARD_GATES]
          + [(n, f, op, nar, wide) for n, f, op, nar, wide in FC.BANDED_GATES])
GATE14_NAMES = [g[0] for g in GATE14]
C4_GATES = {"fall", "obj_pos", "obj_ori", "contact", "hand_pen", "leg_pen"}

EVAL_DIR = C.EVAL_DIR / "aug"
SUMMARY_JSON = EVAL_DIR / "e213_aug_eval_summary.json"
DELTAS_TSV = EVAL_DIR / "e213_aug_deltas.tsv"
ROLLOUT_TSV = EVAL_DIR / "e213_aug_rollout.tsv"
OUT_XLSX = EVAL_DIR / "E213_aug_vs_orig.xlsx"

NAVY, BLUE, GREEN, RED, GRAY, WHITE, AMBER = (
    "1F3864", "2E5496", "C6EFCE", "FFC7CE", "D9D9D9", "FFFFFF", "FFE699")
TEXT_BETTER, TEXT_WORSE, TEXT_SAME = "FF008000", "FFC00000", "FF808080"
INDICATIVE_MIN_N = 3
PRIMARY = "track_obj_pos_err_cm_mean"
HEADLINE = [
    ("track_obj_pos_err_cm_mean", True), ("track_obj_ori_err_deg_mean", True),
    ("track_obj_z_abs_err_cm_mean", True), ("track_eef_ori_err_deg_mean", True),
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
    for c in range(1, ncol + 1):
        ws.cell(row, c).fill = PatternFill("solid", fgColor=GRAY)
    ws.cell(row, ncol).value = f"indicative ({n} independent cases < {INDICATIVE_MIN_N}); point estimate only"
    ws.cell(row, ncol).font = Font(size=8, italic=True, color="666666")


def stratum_sheet(wb: Workbook, name: str, title: str, subtitle: str, cells: dict[str, Any]) -> None:
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
        indicative = cell.get("n_cases", n) < INDICATIVE_MIN_N
        hl = m.get("hl_shift")
        ws.append([
            key, n, cell.get("n_cases", ""),
            round(g["orig_pass_rate"], 3) if g["orig_pass_rate"] is not None else "",
            round(g["aug_pass_rate"], 3) if g["aug_pass_rate"] is not None else "",
            round(g["pass_rate_drop"], 3) if g["pass_rate_drop"] is not None else "",
            "" if indicative else round(g.get("exact_p", float("nan")), 4),
            ("~" if indicative else "") + (f"{hl:+.3f}" if hl is not None and math.isfinite(hl) else ""),
            "" if indicative else round(m.get("mean_delta", float("nan")), 3),
            "" if indicative else round(m.get("std_delta", float("nan")), 3),
            "" if indicative else round(m.get("worst_delta", float("nan")), 3),
            "" if indicative else m.get("primary_test", ""),
            "" if indicative else round(m.get("wilcoxon_p", m.get("sign_test_p", float("nan"))), 4),
            "",
        ])
        r = ws.max_row
        if indicative:
            _mark_indicative(ws, r, len(cols), cell.get("n_cases", n))
        else:
            _verdict_fill(ws, r, 6, (g["pass_rate_drop"] or 0) <= 0.15)
            _verdict_fill(ws, r, 8, hl is not None and math.isfinite(hl) and hl <= 5.0)
    for i, w in enumerate([26, 6, 7, 10, 10, 10, 11, 16, 10, 10, 10, 16, 9, 46], 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def sheet_readme(wb: Workbook, summary: dict[str, Any]) -> None:
    cl = summary.get("clustering") or {}
    ws = wb.active
    ws.title = "README"
    _title(ws, "E213 · paired-export 选定臂 物体增强 vs orig（R299）",
           "21 case × trans0/1/2+rot0/1，每 case 用其 E212 选定臂（G1/G0.8/G0.6/G0.4/PRG）跑 CEM 1024×32 seed0。"
           "orig 基线 = 各 case 同一选定臂的原始 rollout（非 E206 PRG），同一 evaluator 现打。", 4)
    lines = [
        ("范围", f"{summary['n_pairs']} 对配对（aug {summary['n_aug_scored']} / orig {summary['n_orig_scored']}）"),
        ("orig 基线", summary.get("orig_baseline", "")),
        ("C4 判据（预注册）", "① 配对通过率下降 ≤ 0.15（McNemar，行级+case 级）；② obj_pos HL 中位差 ≤ +5.0 cm"),
        ("点估计为何用 HL", "n 小且 obj_pos 重尾，均值差会被单个失败 run 拖走；HL 是中位数的稳健配对版本"),
        ("n<3 的格", f"任何独立 case 数 < {INDICATIVE_MIN_N} 的格：灰底、点估计带 ~、不出 p 值/std/worst"),
        ("⚠ 配对是聚类的",
         f"{cl.get('n_rows', '?')} 对来自 {cl.get('n_cases', '?')} 个 case（每例约 5 条共享同一 orig）。"
         "Wilcoxon/McNemar 假设独立，故行级 p 偏小；点估计不受影响。分层的独立单元看 cases 列。"),
        ("  case 级对照",
         (f"每例塌缩成一个数再检验（{cl.get('n_cases', '?')} 个独立单元）：平均通过率下降 "
          f"{cl.get('case_level_mean_pass_rate_drop', float('nan')):+.3f}，变差/变好/持平 = "
          f"{cl.get('case_level_n_cases_worse', '?')}/{cl.get('case_level_n_cases_better', '?')}/"
          f"{cl.get('case_level_n_cases_unchanged', '?')} 例") if cl else "（无 clustering 块）"),
        ("每 case 选定臂", f"arm 分布 {summary.get('arm_counts', {})}（见 ByArm 表）"),
        ("PRG case", "desk007_034_p1 选定臂=PRG，其 aug rollout 直接复用 E208 的 PRG aug（不重跑）"),
        ("rot 变体", "rot_* 不是纯旋转：每档带 ±45° yaw + 0.2 m 侧移，yaw 用 tau=25 衰减、位移用 tau=50"),
        ("跨视频比较无效", "自动机位每帧按 sim∪ref 包围盒重算（E209 F6/E210 F3）；只比单视频内 sim-vs-ref，"
                        "或用 viser_replay_E213.py 的共享机位 A/B"),
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
    cl = summary.get("clustering", {})
    _title(ws, "C4 · aug 数值不劣于 orig（预注册判据）",
           "pooled 行级 + case 级对照；判据在任何 aug 结果被打分之前冻结", 5)
    ws.append([])
    ws.append(["判据", "实测", "门", "结论", "备注"])
    _hdr(ws, ws.max_row, 5)
    for key, chk in c4["checks"].items():
        ws.append([key, round(chk["value"], 4) if chk["value"] is not None and math.isfinite(finite(chk["value"])) else "",
                   chk["bar"], chk["verdict"], ""])
        _verdict_fill(ws, ws.max_row, 4, chk["verdict"] == "pass")
    ws.append(["case_level_pass_rate_drop", round(finite(cl.get("case_level_mean_pass_rate_drop")), 4),
               0.15, "pass" if finite(cl.get("case_level_mean_pass_rate_drop")) <= 0.15 else "fail",
               "case 级（独立单元）的诚实检验"])
    _verdict_fill(ws, ws.max_row, 4, finite(cl.get("case_level_mean_pass_rate_drop")) <= 0.15)
    ws.append([])
    ws.append(["总判定", "", "", c4["verdict"], c4.get("note", "")])
    _verdict_fill(ws, ws.max_row, 4, c4["verdict"] == "pass")

    pooled = summary["strata"]["pooled"]["all"]
    ws.append([])
    ws.append(["全部 KEY_METRICS 的配对结果（pooled）"])
    ws.cell(ws.max_row, 1).font = Font(bold=True)
    ws.append(["metric", "n", "HL 中位差", "mean Δ", "std Δ", "worst Δ", "主检验", "p"])
    _hdr(ws, ws.max_row, 8)
    for metric, lower_better in HEADLINE:
        m = pooled["metrics"][metric]
        ws.append([metric, m["n"], round(m.get("hl_shift", float("nan")), 4),
                   round(m.get("mean_delta", float("nan")), 4), round(m.get("std_delta", float("nan")), 4),
                   round(m.get("worst_delta", float("nan")), 4), m.get("primary_test", ""),
                   round(m.get("wilcoxon_p", m.get("sign_test_p", float("nan"))), 4)])
        hl = m.get("hl_shift", float("nan"))
        if math.isfinite(hl):
            good = (hl <= 0) if lower_better else (hl >= 0)
            ws.cell(ws.max_row, 3).fill = PatternFill("solid", fgColor=GREEN if good else AMBER)
    for i, w in enumerate([34, 6, 14, 12, 12, 12, 18, 10], 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def sheet_per_case(wb: Workbook, deltas: list[dict[str, str]]) -> None:
    ws = wb.create_sheet("PerCase")
    cols = ["object", "case_id", "arm", "variant", "band", "offset(m)",
            "orig pass", "aug pass", "aug 失败门",
            f"orig {PRIMARY}", f"aug {PRIMARY}", "Δ obj_pos", "Δ obj_ori", "Δ eef_ori",
            "Δ contact", "Δ hand_pen", "Δ leg_pen"]
    _title(ws, "PerCase · 每一对 aug-vs-orig（选定臂）", "全量列出，不筛选。", len(cols))
    ws.append([])
    ws.append(cols)
    _hdr(ws, ws.max_row, len(cols))
    for d in sorted(deltas, key=lambda r: (r["object_key"], r["case_id"], r["aug_variant"])):
        ws.append([
            d["object_key"], d["case_id"], d.get("arm", ""), d["aug_variant"], d.get("offset_band", ""),
            round(finite(d.get("approach_trans_offset_m_max")), 4),
            d["orig_all_gates_pass"], d["aug_all_gates_pass"], d.get("aug_failure_modes", ""),
            round(finite(d[f"orig_{PRIMARY}"]), 3), round(finite(d[f"aug_{PRIMARY}"]), 3),
            round(finite(d[f"delta_{PRIMARY}"]), 3),
            round(finite(d.get("delta_track_obj_ori_err_deg_mean")), 3),
            round(finite(d.get("delta_track_eef_ori_err_deg_mean")), 3),
            round(finite(d.get("delta_hand_object_physics_contact_in_mask_frac")), 4),
            round(finite(d.get("delta_hand_object_physics_penetration_3mm_frame_frac")), 4),
            round(finite(d.get("delta_leg_penetration_frac")), 4),
        ])
        r = ws.max_row
        dv = finite(d[f"delta_{PRIMARY}"])
        if math.isfinite(dv):
            ws.cell(r, 12).fill = PatternFill("solid", fgColor=GREEN if dv <= 0 else (AMBER if dv <= 5.0 else RED))
        if C.truth(d["orig_all_gates_pass"]) and not C.truth(d["aug_all_gates_pass"]):
            ws.cell(r, 8).fill = PatternFill("solid", fgColor=RED)
    ws.freeze_panes = "A4"
    for i, w in enumerate([10, 32, 6, 8, 9, 10, 10, 10, 24, 14, 14, 11, 11, 11, 11, 11, 11], 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def gate_eval(row: dict[str, str], name: str, field: str, op: str, thr: float, wide: float | None) -> dict[str, Any]:
    if op == "fall":
        fell = str(row.get(field, "")).strip().lower() in ("true", "1")
        return {"value": 1.0 if fell else 0.0, "narrow": not fell, "wide": not fell, "na": False}
    value = finite(row.get(field, ""))
    na = math.isnan(value)
    return {"value": value, "narrow": FC.passes(op, value, thr),
            "wide": FC.passes(op, value, wide) if wide is not None else FC.passes(op, value, thr), "na": na}


def better_is_lower(op: str) -> bool:
    return op != ">="


def gate_cell_value(op: str, cur: dict[str, Any], ref: dict[str, Any] | None):
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
    return CellRichText(TextBlock(InlineFont(), value), TextBlock(InlineFont(color=colour), f"({delta:+.4f})"))


def sheet_gate14_summary(wb: Workbook, scored: list[dict[str, str]]) -> None:
    ws = wb.create_sheet("Gate14Summary")
    cols = ["门", "指标", "op", "narrow", "wide", "在 C4 的 6 门内",
            "orig 过", "aug 过", "orig 过率", "aug 过率", "过率差",
            "orig过→aug挂", "orig挂→aug过", "Δ 中位", "Δ p90", "n/a 数"]
    _title(ws, "Gate14Summary · 逐门汇总（哪一道门真的动了）",
           "14 门 funnel caliber（4 硬 + 10 带，阈值取自 funnel_config，与 E204/205/206 同尺）。"
           "「过率差」= orig 过率 − aug 过率，正值=增强后变差。", len(cols))
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
        ws.append([n, f, op, thr, wide if wide is not None else "", "✓" if n in C4_GATES else "",
                   o_pass, a_pass, round(o_rate, 3), round(a_rate, 3), round(o_rate - a_rate, 3),
                   lost, gained, round(float(np.nanmedian(arr)), 4),
                   round(float(np.nanpercentile(arr, 90)), 4), na])
        r = ws.max_row
        if lost:
            ws.cell(r, 12).fill = PatternFill("solid", fgColor=RED)
        _verdict_fill(ws, r, 11, (o_rate - a_rate) <= 0.15)
    for i, w in enumerate([12, 46, 5, 9, 8, 14, 9, 9, 10, 10, 9, 13, 13, 10, 10, 7], 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def sheet_gate14_values(wb: Workbook, scored: list[dict[str, str]]) -> None:
    ws = wb.create_sheet("Gate14Values")
    cols = (["object", "case_id", "arm", "variant", "band"] + GATE14_NAMES
            + ["narrow 过", "14门全过", "narrow 失败门", "n/a 门"])
    _title(ws, "Gate14Values · 14 门完整数值",
           "格式 值(Δ)，Δ = aug − orig；orig 行无 Δ。Δ 颜色按每门方向（contact 越大越好，其余越小越好）："
           "绿=比 orig 好，红=差，灰=持平。底色：粉=该门挂，灰=n/a。", len(cols))
    ws.append([])
    ws.append(cols)
    _hdr(ws, ws.max_row, len(cols))
    by_case = {r["case_id"]: r for r in scored if r["aug_variant"] == "orig"}
    order = {"orig": 0, "trans0": 1, "trans1": 2, "trans2": 3, "rot0": 4, "rot1": 5}
    for row in sorted(scored, key=lambda r: (r["object_key"], r["case_id"], order.get(r["aug_variant"], 9))):
        res = {n: gate_eval(row, n, f, op, thr, wide) for n, f, op, thr, wide in GATE14}
        failed = [n for n in GATE14_NAMES if not res[n]["narrow"]]
        nas = [n for n in GATE14_NAMES if res[n]["na"]]
        is_orig = row["aug_variant"] == "orig"
        base = by_case.get(row["case_id"]) if not is_orig else None
        ref = ({n: gate_eval(base, n, f, op, thr, wide) for n, f, op, thr, wide in GATE14}
               if base is not None else None)
        ws.append([row["object_key"], row["case_id"], row.get("arm", ""), row["aug_variant"],
                   row.get("offset_band", "")] + [None] * len(GATE14_NAMES)
                  + [len(GATE14_NAMES) - len(failed), "TRUE" if not failed else "FALSE",
                     ",".join(failed), ",".join(nas)])
        r = ws.max_row
        for i, (n, _f, op, _thr, _wide) in enumerate(GATE14, 6):
            cell = ws.cell(r, i)
            cell.value = gate_cell_value(op, res[n], ref[n] if ref else None)
            cell.alignment = Alignment(horizontal="right")
            if res[n]["na"]:
                cell.fill = PatternFill("solid", fgColor=GRAY)
            elif not res[n]["narrow"]:
                cell.fill = PatternFill("solid", fgColor=RED)
        if is_orig:
            for i in range(1, 6):
                ws.cell(r, i).font = Font(bold=True)
    ws.freeze_panes = "F4"
    for i, w in enumerate([10, 32, 6, 8, 9] + [19] * len(GATE14_NAMES) + [10, 12, 34, 16], 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def sheet_gates(wb: Workbook, deltas: list[dict[str, str]]) -> None:
    ws = wb.create_sheet("Gates")
    _title(ws, "Gates · 哪一道门真的动了（6 门增强线口径）", "统计 aug 相对 orig 的门翻转", 4)
    ws.append([])
    ws.append(["门", "orig 过 → aug 挂", "aug 失败计数", "净"])
    _hdr(ws, ws.max_row, 4)
    for gate in ["fall", "object_pos", "object_ori", "contact", "hand_penetration", "lower_body"]:
        lost = sum(1 for d in deltas if C.truth(d["orig_all_gates_pass"]) and gate in (d.get("aug_failure_modes") or ""))
        aug_fail = sum(1 for d in deltas if gate in (d.get("aug_failure_modes") or ""))
        ws.append([gate, lost, aug_fail, -lost])
        if lost:
            ws.cell(ws.max_row, 2).fill = PatternFill("solid", fgColor=RED)
    for i, w in enumerate([22, 18, 16, 8], 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_XLSX)
    args = ap.parse_args()
    if not SUMMARY_JSON.is_file():
        raise SystemExit(f"missing {C.rel(SUMMARY_JSON)} -- run eval_E213_aug.py first")
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    deltas = C.read_tsv(DELTAS_TSV)
    scored = C.read_tsv(ROLLOUT_TSV)

    wb = Workbook()
    sheet_readme(wb, summary)
    sheet_c4(wb, summary)
    strata = summary["strata"]
    stratum_sheet(wb, "ByArm", "按选定臂（headline）",
                  "每 case 用其 E212 选定臂：G1=chair006+desk021 / G08+G06+G04=desk007 / G06=desk023 / PRG=1 例。"
                  "这是 E213 相对 E208（仅 PRG）的新维度：增强代价是否因臂而异。", strata["by_arm"])
    stratum_sheet(wb, "ByOffsetBand", "按有效位移分档",
                  "full ≥0.18 m / partial 0.10–0.18 / weak <0.10：增强幅度多大才开始付跟踪代价", strata["by_offset_band"])
    stratum_sheet(wb, "ByVariant", "按增强变体",
                  "trans0/1/2 = 0.2 m 前/左/右；rot0/1 = ±45° yaw + 0.2 m 侧移", strata["by_variant"])
    stratum_sheet(wb, "ByObjectVariant", "按 物体 × 变体", "多数格 n 小，按规则 indicative", strata["by_object_variant"])
    sheet_gate14_summary(wb, scored)
    sheet_gate14_values(wb, scored)
    sheet_gates(wb, deltas)
    sheet_per_case(wb, deltas)

    out = C.repo_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print(f"[done] {C.rel(out)}  sheets={wb.sheetnames}")
    print(f"  pairs={summary['n_pairs']}  C4 verdict={summary['C4']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
