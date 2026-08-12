#!/usr/bin/env python3
"""Build an auditable E192 Full vs PRG/A0 comparison workbook."""
from __future__ import annotations

import csv
import hashlib
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo


ROOT = Path("/home/ubuntu/Workspace/spider")
RUN = ROOT / "workspace/core4d/results/E192/s6_downstream"
OUT_DIR = RUN / "eval/full"
OUT = OUT_DIR / "E192_full_vs_PRG_12gate_human_review.xlsx"
E192_METRICS = OUT_DIR / "e192_case_metrics.tsv"
REVIEW_E172 = ROOT / "workspace/core4d/results/E172/s6_downstream/rl_export/box004_user_approved/box004_manual_review_snapshot.tsv"
REVIEW_E173 = ROOT / "workspace/core4d/results/E173/s6_downstream/eval/full/user_manual_review_filled.tsv"
PRG_SOURCE_E172 = ROOT / "workspace/core4d/results/E172/s6_downstream/rl_export/box004_user_approved/box004_user_approved_source_rows.tsv"
PRG_SOURCE_E173 = ROOT / "workspace/core4d/results/E173/s6_downstream/rl_export/box024_user_approved/box024_user_approved_source_rows.tsv"
CLAIMS = OUT_DIR / "e192_claims.json"
SUMMARY = OUT_DIR / "e192_eval_summary.json"
PAIRED = OUT_DIR / "e192_paired_deltas.tsv"
VISUAL = RUN / "render/keyframes/visual_review.tsv"

GATES = ["fall", "body_z", "contact", "release", "hand_penetration", "lower_body",
         "root_pos", "root_ori", "hand_pos", "hand_ori", "object_pos", "object_ori"]
KEY_METRICS = [
    ("fall_flag", "fall flag (1=fall)", "lower"),
    ("body_z_err_p95_m", "body-z error p95 (m)", "lower"),
    ("hand_object_physics_contact_in_mask_frac", "in-mask physics contact fraction", "higher"),
    ("hand_object_release_false_contact_3mm_frac", "release false-contact 3mm fraction", "lower"),
    ("hand_object_physics_penetration_3mm_frame_frac", "3mm penetration frame fraction", "lower"),
    ("leg_penetration_frac", "leg penetration fraction", "lower"),
    ("track_root_pos_err_cm_mean", "root position error (cm)", "lower"),
    ("track_root_ori_err_deg_mean", "root orientation error (deg)", "lower"),
    ("track_eef_pos_err_cm_mean", "EEF position error (cm)", "lower"),
    ("track_eef_ori_err_deg_mean", "EEF orientation error (deg)", "lower"),
    ("track_obj_pos_err_cm_mean", "object position error (cm)", "lower"),
    ("track_obj_ori_err_deg_mean", "object orientation error (deg)", "lower"),
]
GATE_METRIC_META = {
    "fall": ("fall_flag", "lower", 0.0, "flag"),
    "body_z": ("body_z_err_p95_m", "lower", 0.20, "m"),
    "contact": ("hand_object_physics_contact_in_mask_frac", "higher", 0.50, "frac"),
    "release": ("hand_object_release_false_contact_3mm_frac", "lower", 0.30, "frac"),
    "hand_penetration": ("hand_object_physics_penetration_3mm_frame_frac", "lower", 0.30, "frac"),
    "lower_body": ("leg_penetration_frac", "lower", 0.10, "frac"),
    "root_pos": ("track_root_pos_err_cm_mean", "lower", 20.0, "cm"),
    "root_ori": ("track_root_ori_err_deg_mean", "lower", 20.0, "deg"),
    "hand_pos": ("track_eef_pos_err_cm_mean", "lower", 20.0, "cm"),
    "hand_ori": ("track_eef_ori_err_deg_mean", "lower", 20.0, "deg"),
    "object_pos": ("track_obj_pos_err_cm_mean", "lower", 20.0, "cm"),
    "object_ori": ("track_obj_ori_err_deg_mean", "lower", 10.0, "deg"),
}

NAVY = "17365D"; BLUE = "D9EAF7"; LIGHT = "F4F7FB"; GREEN = "E2F0D9"; RED = "FCE4D6"; AMBER = "FFF2CC"; GREY = "666666"
thin = Side(style="thin", color="D9E1F2")


def read_tsv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bval(v):
    return str(v).strip().lower() in {"true", "1", "pass", "yes"}


def num(v):
    if str(v).strip().lower() in {"true", "yes"}: return 1.0
    if str(v).strip().lower() in {"false", "no"}: return 0.0
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def style_sheet(ws, freeze="A2", widths=None):
    ws.freeze_panes = freeze
    ws.sheet_view.showGridLines = False
    ws.auto_filter.ref = ws.dimensions
    if widths:
        for c, w in widths.items(): ws.column_dimensions[c].width = w
    for row in ws.iter_rows():
        for cell in row:
            cell.font = Font(name="Arial", size=10, color="000000")
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = Border(bottom=thin)
    for cell in ws[1]:
        cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 32


def add_table(ws, name):
    ref = ws.dimensions
    tab = Table(displayName=name, ref=ref)
    tab.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showFirstColumn=False, showLastColumn=False, showRowStripes=True, showColumnStripes=False)
    ws.add_table(tab)


def status(v):
    return "PASS" if bval(v) else "FAIL"


def human_decision(v):
    v = str(v or "").strip().upper()
    if v in {"USE", "DNU"}: return v
    if v in {"DO_NOT_USE", "DO NOT USE", "DONOTUSE"}: return "DNU"
    return v or "NOT_REVIEWED"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_tsv(E192_METRICS)
    by_arm = defaultdict(dict)
    for r in rows:
        if r.get("arm") in {"A0-history", "A2"}:
            by_arm[r["case_id"]][r["arm"]] = r
    paired = []
    for case_id, arms in by_arm.items():
        if "A0-history" in arms and "A2" in arms:
            paired.append((case_id, arms["A0-history"], arms["A2"]))
    paired.sort(key=lambda x: (x[1].get("object_key", ""), x[0]))

    review = {}
    for r in read_tsv(REVIEW_E172) + read_tsv(REVIEW_E173):
        if r.get("case_id"):
            review[r["case_id"]] = r
    # Prefer the explicit approved source rows when available; these encode the PRG authority.
    for p in (PRG_SOURCE_E172, PRG_SOURCE_E173):
        for r in read_tsv(p):
            # The source-row export is authoritative for decision/quality/numeric fields;
            # retain reviewer/date/note payload from the filled review table when present.
            old = review.get(r["case_id"], {})
            merged = dict(old)
            merged.update({k: r.get(k, "") for k in ("manual_use_decision", "manual_quality_label", "manual_failure_taxonomy", "numeric_release_pass", "numeric_failure_modes")})
            review[r["case_id"]] = merged
    visual = {r["case_id"]: r for r in read_tsv(VISUAL)} if VISUAL.exists() else {}

    wb = Workbook()
    ws = wb.active; ws.title = "README"
    ws.append(["E192 Full vs PRG/A0：12-gate 与人审汇总"])
    ws.append(["范围", "E192 Full 的 15 条 A2 与同 case 的历史 PRG/A0（A0-history）配对；不把 A2 空白人审字段当作 PRG 人审。"])
    ws.append(["正式判决", "INCONCLUSIVE_GATE_COLLAPSE"])
    ws.append(["Full-only 诊断", "THRESHOLD_POLICY_NOT_EFFECTIVE"])
    ws.append(["关键口径", "12-gate 为 fall/body_z/contact/release/hand_penetration/lower_body/root_pos/root_ori/hand_pos/hand_ori/object_pos/object_ori。"])
    ws.append(["A2 阈值范围", "E192 Full 的 15/15 A2 manifest 与每个 A2 config_act.yaml 均为 min_sdf=-0.010 m、max_violation_pct=0.05、hard_floor=-0.015 m；PRG/A0-history 是历史对照，不是用这组三阈值重跑。"])
    ws.append(["PRG 人审权威", "box004：E172 box004_manual_review_snapshot.tsv；box024：E173 user_manual_review_filled.tsv，并以 *_user_approved_source_rows.tsv 的已导出行优先。"])
    ws.append(["生成时间", datetime.now().astimezone().isoformat(timespec="seconds")])
    ws.append(["输出", str(OUT)])
    ws.append(["说明", "所有派生差值、PASS 计数和失败模式汇总使用 Excel 公式；源指标与标签为硬编码快照。"])
    ws.column_dimensions["A"].width = 22; ws.column_dimensions["B"].width = 120
    for c in ws[1]: c.font = Font(name="Arial", size=14, bold=True, color="FFFFFF"); c.fill = PatternFill("solid", fgColor=NAVY)
    for row in ws.iter_rows(min_row=2):
        row[0].font = Font(name="Arial", bold=True, color=NAVY)
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
    ws.sheet_view.showGridLines = False

    # Case comparison sheet.
    ws = wb.create_sheet("Case Comparison")
    headers = ["case_id", "object_key", "PRG variant", "E192 variant", "PRG human review", "PRG quality", "PRG numeric pass", "E192 numeric pass", "PRG 12-gate", "E192 12-gate"]
    headers += [f"PRG {g}" for g in GATES] + [f"E192 {g}" for g in GATES]
    headers += [f"PRG {label}" for _, label, _ in KEY_METRICS] + [f"E192 {label}" for _, label, _ in KEY_METRICS] + [f"Δ E192−PRG {label}" for _, label, _ in KEY_METRICS]
    headers += ["PRG numeric failure modes", "E192 numeric failure modes", "E192-only failure modes", "PRG-only failure modes", "shared failure modes", "visual note"]
    ws.append(headers)
    for case_id, prg, e192 in paired:
        rv = review.get(case_id, {})
        row = [case_id, prg.get("object_key", ""), prg.get("variant", ""), e192.get("variant", ""), human_decision(rv.get("manual_use_decision", "DNU" if rv else "NOT_REVIEWED")), rv.get("manual_quality_label", ""), status(prg.get("numeric_release_pass_12gate", prg.get("numeric_release_pass", "false"))), status(e192.get("numeric_release_pass_12gate", e192.get("numeric_release_pass", "false"))), status(prg.get("numeric_release_pass_12gate", "false")), status(e192.get("numeric_release_pass_12gate", "false"))]
        row += [status(prg.get(g+"_gate_pass", "false")) for g in GATES] + [status(e192.get(g+"_gate_pass", "false")) for g in GATES]
        for key, _, _ in KEY_METRICS: row.append(num(prg.get(key))); 
        for key, _, _ in KEY_METRICS: row.append(num(e192.get(key)))
        # Formula columns are appended after source values.
        source_len = len(row)
        for i in range(len(KEY_METRICS)):
            prg_col = 1 + 10 + len(GATES)*2 + i
            e_col = prg_col + len(KEY_METRICS)
            row.append(f"={get_column_letter(e_col)}{ws.max_row+1}-{get_column_letter(prg_col)}{ws.max_row+1}")
        pf = prg.get("numeric_failure_modes", "")
        ef = e192.get("numeric_failure_modes", "")
        pset, eset = set(x.strip() for x in pf.split(",") if x.strip()), set(x.strip() for x in ef.split(",") if x.strip())
        row += [pf, ef, ", ".join(sorted(eset-pset)), ", ".join(sorted(pset-eset)), ", ".join(sorted(pset&eset)), visual.get(case_id, {}).get("observation", "")]
        ws.append(row)
    # Fill failure-mode formula columns (their positions are known).
    # 10 metadata + 24 gate statuses + 7 PRG metrics + 7 E192 metrics + 7 deltas.
    base_cols = 10 + 2*len(GATES) + 3*len(KEY_METRICS)
    eonly_col, prgonly_col, shared_col = base_cols + 3, base_cols + 4, base_cols + 5
    pf_col, ef_col = get_column_letter(base_cols + 1), get_column_letter(base_cols + 2)
    widths = {"A": 33, "B": 11, "C": 38, "D": 38, "E": 16, "F": 18, "G": 14, "H": 14, "I": 12, "J": 12}
    for c in range(11, ws.max_column+1): widths[get_column_letter(c)] = 13
    style_sheet(ws, freeze="A2", widths=widths); add_table(ws, "CaseComparison")
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            if cell.value == "PASS": cell.fill = PatternFill("solid", fgColor=GREEN)
            elif cell.value == "FAIL": cell.fill = PatternFill("solid", fgColor=RED)
            elif cell.value in {"USE"}: cell.fill = PatternFill("solid", fgColor=GREEN)
            elif cell.value in {"DNU", "DO_NOT_USE"}: cell.fill = PatternFill("solid", fgColor=RED)
    # Number formats for numeric metrics and delta columns.
    for c in range(11 + 2*len(GATES), ws.max_column-6+1):
        for cell in ws[get_column_letter(c)][1:]: cell.number_format = "0.0000"

    # Aggregate numerical statistics by object and overall. Formulas point to the
    # paired case sheet so the workbook remains updateable when source rows change.
    ws = wb.create_sheet("Metric Summary")
    ws.append(["group", "n cases"] + [f"PRG {label}" for _, label, _ in KEY_METRICS] + [f"E192 {label}" for _, label, _ in KEY_METRICS] + [f"Δ E192−PRG {label}" for _, label, _ in KEY_METRICS])
    groups = [("box004", "box004"), ("box024", "box024"), ("ALL", "*")]
    cc = wb["Case Comparison"]
    # source metric columns in Case Comparison
    metric_start = 11 + 2*len(GATES)
    for label, criterion in groups:
        r = ws.max_row + 1
        ws.cell(r, 1).value = label
        ws.cell(r, 2).value = f'=COUNTIF(\'Case Comparison\'!$B$2:$B${cc.max_row},"{criterion if criterion != "*" else "<>"}")' if criterion != "*" else f'=COUNTA(\'Case Comparison\'!$A$2:$A${cc.max_row})'
        for i in range(len(KEY_METRICS)):
            pcol = get_column_letter(metric_start+i); ecol = get_column_letter(metric_start+len(KEY_METRICS)+i); dcol = get_column_letter(metric_start+2*len(KEY_METRICS)+i)
            if criterion == "*":
                ws.cell(r,3+i).value = f'=IFERROR(AVERAGE(\'Case Comparison\'!${pcol}$2:${pcol}${cc.max_row}),"")'
                ws.cell(r,3+len(KEY_METRICS)+i).value = f'=IFERROR(AVERAGE(\'Case Comparison\'!${ecol}$2:${ecol}${cc.max_row}),"")'
            else:
                ws.cell(r,3+i).value = f'=IFERROR(AVERAGEIF(\'Case Comparison\'!$B$2:$B${cc.max_row},"{criterion}",\'Case Comparison\'!${pcol}$2:${pcol}${cc.max_row}),"")'
                ws.cell(r,3+len(KEY_METRICS)+i).value = f'=IFERROR(AVERAGEIF(\'Case Comparison\'!$B$2:$B${cc.max_row},"{criterion}",\'Case Comparison\'!${ecol}$2:${ecol}${cc.max_row}),"")'
            ws.cell(r,3+2*len(KEY_METRICS)+i).value = f'={get_column_letter(3+len(KEY_METRICS)+i)}{r}-{get_column_letter(3+i)}{r}'
    style_sheet(ws, widths={"A": 16, "B": 10, **{get_column_letter(c): 18 for c in range(3, ws.max_column+1)}}); add_table(ws, "MetricSummary")
    for row in ws.iter_rows(min_row=2):
        for cell in row[2:]: cell.number_format = "0.0000"

    # 12-gate summary
    ws = wb.create_sheet("12Gate Summary")
    ws.append(["Gate", "Metric", "Direction", "Threshold", "Unit", "PRG mean", "E192 mean", "Δ E192−PRG", "PRG PASS", "PRG FAIL", "E192 PASS", "E192 FAIL", "PASS→PASS", "PASS→FAIL", "FAIL→PASS", "FAIL→FAIL", "E192 pass rate", "PRG pass rate"])
    cc = wb["Case Comparison"]
    for i, g in enumerate(GATES):
        prg_col = get_column_letter(11+i); e_col = get_column_letter(11+len(GATES)+i)
        metric_key, direction, threshold, unit = GATE_METRIC_META[g]
        metric_i = next(j for j, (k, _, _) in enumerate(KEY_METRICS) if k == metric_key)
        pmetric_col = get_column_letter(11 + 2*len(GATES) + metric_i)
        emetric_col = get_column_letter(11 + 2*len(GATES) + len(KEY_METRICS) + metric_i)
        ws.append([g, metric_key, direction, threshold, unit,
                   f'=IFERROR(AVERAGE(\'Case Comparison\'!${pmetric_col}$2:${pmetric_col}${cc.max_row}),"")',
                   f'=IFERROR(AVERAGE(\'Case Comparison\'!${emetric_col}$2:${emetric_col}${cc.max_row}),"")',
                   f'=G{i+2}-F{i+2}',
                   f'=COUNTIF(\'Case Comparison\'!${prg_col}$2:${prg_col}${cc.max_row},"PASS")',
                   f'=COUNTIF(\'Case Comparison\'!${prg_col}$2:${prg_col}${cc.max_row},"FAIL")',
                   f'=COUNTIF(\'Case Comparison\'!${e_col}$2:${e_col}${cc.max_row},"PASS")',
                   f'=COUNTIF(\'Case Comparison\'!${e_col}$2:${e_col}${cc.max_row},"FAIL")',
                   f'=COUNTIFS(\'Case Comparison\'!${prg_col}$2:${prg_col}${cc.max_row},"PASS",\'Case Comparison\'!${e_col}$2:${e_col}${cc.max_row},"PASS")',
                   f'=COUNTIFS(\'Case Comparison\'!${prg_col}$2:${prg_col}${cc.max_row},"PASS",\'Case Comparison\'!${e_col}$2:${e_col}${cc.max_row},"FAIL")',
                   f'=COUNTIFS(\'Case Comparison\'!${prg_col}$2:${prg_col}${cc.max_row},"FAIL",\'Case Comparison\'!${e_col}$2:${e_col}${cc.max_row},"PASS")',
                   f'=COUNTIFS(\'Case Comparison\'!${prg_col}$2:${prg_col}${cc.max_row},"FAIL",\'Case Comparison\'!${e_col}$2:${e_col}${cc.max_row},"FAIL")',
                   f'=K{i+2}/(K{i+2}+L{i+2})', f'=I{i+2}/(I{i+2}+J{i+2})'])
    ws.append(["ALL 12 gates", "—", "—", "—", "—", "—", "—", "—", "=SUM(I2:I13)", "=SUM(J2:J13)", "=SUM(K2:K13)", "=SUM(L2:L13)", "=SUM(M2:M13)", "=SUM(N2:N13)", "=SUM(O2:O13)", "=SUM(P2:P13)", "=K14/(K14+L14)", "=I14/(I14+J14)"])
    style_sheet(ws, widths={"A": 20, "B": 44, "C": 12, "D": 12, "E": 10, **{get_column_letter(c): 13 for c in range(6,19)}}); add_table(ws, "GateSummary")
    for c in (17,18):
        for cell in ws[get_column_letter(c)][1:]: cell.number_format = "0.0%"
    for c in (6,7,8):
        for cell in ws[get_column_letter(c)][1:]: cell.number_format = "0.0000"

    # Failure mode summary, with formulas referring to paired comparison.
    ws = wb.create_sheet("Failure Modes")
    ws.append(["Failure mode", "PRG cases", "E192 cases", "PRG-only", "E192-only", "shared", "PRG case rate", "E192 case rate", "E192 cases (IDs)", "PRG cases (IDs)"])
    for g in GATES:
        ws.append([g, None, None, None, None, None, None, None, None, None])
    for r in range(2, 14):
        g = ws.cell(r, 1).value
        # Case Comparison columns: PRG fail string is after all metrics; E192 fail string next.
        pf_col = get_column_letter(base_cols+1); ef_col = get_column_letter(base_cols+2)
        ws.cell(r,2).value = f'=COUNTIF(\'Case Comparison\'!${pf_col}$2:${pf_col}${cc.max_row},"*"&A{r}&"*")'
        ws.cell(r,3).value = f'=COUNTIF(\'Case Comparison\'!${ef_col}$2:${ef_col}${cc.max_row},"*"&A{r}&"*")'
        ws.cell(r,4).value = f'=COUNTIFS(\'Case Comparison\'!${pf_col}$2:${pf_col}${cc.max_row},"*"&A{r}&"*",\'Case Comparison\'!${ef_col}$2:${ef_col}${cc.max_row},"<>*"&A{r}&"*")'
        ws.cell(r,5).value = f'=COUNTIFS(\'Case Comparison\'!${ef_col}$2:${ef_col}${cc.max_row},"*"&A{r}&"*",\'Case Comparison\'!${pf_col}$2:${pf_col}${cc.max_row},"<>*"&A{r}&"*")'
        ws.cell(r,6).value = f'=COUNTIFS(\'Case Comparison\'!${pf_col}$2:${pf_col}${cc.max_row},"*"&A{r}&"*",\'Case Comparison\'!${ef_col}$2:${ef_col}${cc.max_row},"*"&A{r}&"*")'
        ws.cell(r,7).value = f'=B{r}/COUNTA(\'Case Comparison\'!$A$2:$A${cc.max_row})'
        ws.cell(r,8).value = f'=C{r}/COUNTA(\'Case Comparison\'!$A$2:$A${cc.max_row})'
        # IDs are recorded as an auditable text snapshot; counts/rates above remain formulas.
        eids = [case_id for case_id, prg, e192 in paired if g in (e192.get("numeric_failure_modes", "") or "").split(",")]
        pids = [case_id for case_id, prg, e192 in paired if g in (prg.get("numeric_failure_modes", "") or "").split(",")]
        ws.cell(r,9).value = ", ".join(eids)
        ws.cell(r,10).value = ", ".join(pids)
    style_sheet(ws, widths={"A": 20, "B": 12, "C": 12, "D": 12, "E": 12, "F": 12, "G": 14, "H": 14, "I": 72, "J": 72}); add_table(ws, "FailureModeSummary")
    for c in (7,8):
        for cell in ws[get_column_letter(c)][1:]: cell.number_format = "0.0%"

    # Human review sheet.
    ws = wb.create_sheet("PRG Human Review")
    ws.append(["case_id", "object_key", "PRG human decision", "quality", "failure taxonomy", "review note", "reviewer", "reviewed at", "authority source"])
    for case_id, prg, e192 in paired:
        rv = review.get(case_id, {})
        source = str(PRG_SOURCE_E172 if prg.get("object_key") == "box004" else REVIEW_E173)
        ws.append([case_id, prg.get("object_key", ""), human_decision(rv.get("manual_use_decision", "DNU" if rv else "NOT_REVIEWED")), rv.get("manual_quality_label", ""), rv.get("manual_failure_taxonomy", ""), rv.get("manual_review_note", ""), rv.get("manual_reviewer", ""), rv.get("manual_reviewed_at", ""), source])
    ws.append([])
    ws.append(["汇总（公式）", "", '=COUNTIF(C2:C16,"USE")', '=COUNTIF(C2:C16,"DNU")'])
    style_sheet(ws, widths={"A": 33, "B": 11, "C": 18, "D": 18, "E": 26, "F": 55, "G": 15, "H": 24, "I": 90}); add_table(ws, "PrgHumanReview")
    for row in ws.iter_rows(min_row=2, max_row=16):
        if row[2].value == "USE": row[2].fill = PatternFill("solid", fgColor=GREEN)
        elif row[2].value in {"DNU", "DO_NOT_USE"}: row[2].fill = PatternFill("solid", fgColor=RED)

    # Source manifest.
    ws = wb.create_sheet("Sources")
    ws.append(["Source", "Role", "Rows/notes", "SHA256"])
    srcs = [(RUN/"manifests/cem_full_manifest.tsv","E192 Full A2 threshold manifest","15 rows"),(E192_METRICS,"E192 Full case metrics (A0-history + A2)","30 rows"),(PAIRED,"E192 paired deltas","15 rows"),(CLAIMS,"E192 claims and governance decision","JSON"),(SUMMARY,"E192 evaluator summary","15/15 evaluated"),(VISUAL,"E192 visual review observations","8 reviewed cases"),(PRG_SOURCE_E172,"PRG human-review authority, box004","4 USE source rows"),(REVIEW_E172,"PRG review snapshot, box004","6 reviewed rows"),(PRG_SOURCE_E173,"PRG human-review authority, box024","3 USE source rows"),(REVIEW_E173,"PRG review table, box024","9 reviewed rows")]
    for p, role, note in srcs:
        if p.exists(): ws.append([str(p), role, note, sha256(p)])
    style_sheet(ws, widths={"A": 110, "B": 48, "C": 24, "D": 68}); add_table(ws, "SourceManifest")

    # workbook-wide polish
    for sh in wb.worksheets:
        sh.sheet_properties.pageSetUpPr.fitToPage = True
        sh.page_setup.fitToWidth = 1; sh.page_setup.fitToHeight = 0
        sh.sheet_properties.outlinePr.summaryBelow = True
    wb.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
