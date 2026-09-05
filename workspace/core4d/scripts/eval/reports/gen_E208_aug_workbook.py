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
import sys
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment, Font, PatternFill

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E208"))

import e208_common as C  # noqa: E402

EVAL_DIR = C.EVAL_DIR
SUMMARY_JSON = EVAL_DIR / "e208_aug_eval_summary.json"
DELTAS_TSV = EVAL_DIR / "e208_aug_deltas.tsv"
OUT_XLSX = EVAL_DIR / "E208_aug_vs_orig.xlsx"

NAVY, BLUE, GREEN, RED, GRAY, WHITE, AMBER = (
    "1F3864", "2E5496", "C6EFCE", "FFC7CE", "D9D9D9", "FFFFFF", "FFE699")

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
    for c in range(1, ncol + 1):
        ws.cell(row, c).fill = PatternFill("solid", fgColor=GRAY)
    ws.cell(row, ncol).value = f"indicative (n={n} < {INDICATIVE_MIN_N}); point estimate only"
    ws.cell(row, ncol).font = Font(size=8, italic=True, color="666666")


def stratum_sheet(wb: Workbook, name: str, title: str, subtitle: str,
                  cells: dict[str, Any]) -> None:
    ws = wb.create_sheet(name)
    cols = ["stratum", "n", "orig pass", "aug pass", "pass drop", "McNemar p",
            f"HL {PRIMARY}", "mean Δ", "std Δ", "worst Δ", "test", "p", "note"]
    _title(ws, title, subtitle, len(cols))
    ws.append([])
    ws.append(cols)
    _hdr(ws, ws.max_row, len(cols))

    for key, cell in sorted(cells.items(), key=lambda kv: -kv[1]["n"]):
        g, m = cell["gates"], cell["metrics"][PRIMARY]
        n = cell["n"]
        indicative = n < INDICATIVE_MIN_N
        hl = m.get("hl_shift")
        row = [
            key, n,
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
            _mark_indicative(ws, r, len(cols), n)
        else:
            _verdict_fill(ws, r, 5, (g["pass_rate_drop"] or 0) <= 0.15)
            _verdict_fill(ws, r, 7, hl is not None and hl <= 5.0)
    for i, w in enumerate([24, 6, 10, 10, 10, 11, 16, 10, 10, 10, 16, 9, 46], 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def sheet_readme(wb: Workbook, summary: dict[str, Any]) -> None:
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
    sheet_per_case(wb, deltas)
    sheet_f15(wb, deltas, summary)
    sheet_gates(wb, deltas)

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
