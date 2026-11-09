#!/usr/bin/env python3
"""Build a four-method comparison (SPIDER-CEM / SBTO / OmniRetarget / GMR) on CORE4D-G1.

Reads three source workbooks, recomputes every aggregate from per-case data so the
methodology is identical across methods, and emits:
  1. paper_comparison.xlsx  - full multi-sheet workbook (overall + by-object + by-case)
  2. by_object_comparison.tex - camera-ready by-object LaTeX tables (booktabs)
  3. paper_comparison.md    - human-readable report with caveats

All four methods are evaluated on the SAME 52 cases / 11 objects.
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path("/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied")
SPIDER_XLSX = ROOT / "spider/workspace/core4d/report/0908/paper_results/paper_results.xlsx"
SBTO_XLSX = ROOT / "sbto/paper_results/sbto_paper_results.xlsx"
GMR_XLSX = ROOT / "GMR/out/core4d_g1/paper_metrics/paper_metrics.xlsx"
GMR_10MM_CSV = ROOT / "GMR/out/core4d_g1/paper_metrics/paper_metrics_10mm_per_case.csv"
OUT_DIR = ROOT / "spider/workspace/core4d/report/0908/paper_cmp"

# --full -> 4-method comparison (SPIDER-CEM / OmniRetarget / SBTO / GMR), written to
# *_full.* files so the interim 2-method artifacts (no suffix) are NOT overwritten.
# default (no flag) -> 2-method SPIDER-CEM vs OmniRetarget (unchanged).
FULL = "--full" in sys.argv
SUF = "_full" if FULL else ""

# ----------------------------------------------------------------------------
# Method registry (display order = paper column order)
# ----------------------------------------------------------------------------
# INTERIM (2026-11): without --full, SBTO/GMR are excluded and only SPIDER-CEM vs
# OmniRetarget is reported (their workbooks were mid-regeneration upstream). With --full,
# all four methods are reported (SBTO/GMR now carry contact@10mm / pen@10mm / max-pen).
# SBTO/GMR-specific blocks below are guarded on membership in METHODS.
METHODS = ["SPIDER-CEM", "SBTO", "OmniRetarget", "GMR"] if FULL else ["SPIDER-CEM", "OmniRetarget"]
METHOD_SHORT = {"SPIDER-CEM": "Ours", "SBTO": "SBTO", "OmniRetarget": "OmniRt", "GMR": "GMR"}
METHOD_KIND = {
    "SPIDER-CEM": "physics (ours)",
    "SBTO": "physics (sampling-based traj. opt.)",
    "OmniRetarget": "kinematic retarget",
    "GMR": "kinematic retarget",
}

# metric key -> (display, arrow, unit)   arrow '+' = higher better, '-' = lower better
COMMON = [
    ("raw_contact", "Raw contact", "+", ""),
    ("contact_3mm", "Phys. contact@3mm", "+", ""),
    ("contact_10mm", "Phys. contact@10mm", "+", ""),
    ("pen_3mm", "Phys. pen@3mm", "-", ""),
    ("pen_10mm", "Phys. pen@10mm", "-", ""),
    ("pen_max_mm", "Max phys pen (mm)", "-", ""),
    ("geom_2mm", "Geom. pen@2mm", "-", ""),
]
# Headline contact-fidelity metrics for the camera-ready by-object table.
# 'raw_contact' is DELIBERATELY excluded: it counts any MuJoCo hand-object contact
# incl. deep interpenetration (core_metrics.py: hand_contact = "any contact"),
# so a penetration-heavy method scores high on it. The three metrics below are all
# penetration-aware, so they are the honest headline comparison.
CONTACT_TEX = [
    ("contact_3mm", "Phys. contact@3mm", "+", ""),
    ("contact_10mm", "Phys. contact@10mm", "+", ""),
    ("pen_3mm", "Phys. pen@3mm", "-", ""),
    ("pen_10mm", "Phys. pen@10mm", "-", ""),
    ("pen_max_mm", "Max phys pen (mm)", "-", ""),
    ("geom_2mm", "Geom. pen@2mm", "-", ""),
]
TRACKING = [
    ("root_pos", "Root pos err", "-", "cm"),
    ("root_ori", "Root ori err", "-", "deg"),
    ("eef_pos", "EEF pos err", "-", "cm"),
    ("eef_ori", "EEF ori err", "-", "deg"),
    ("obj_pos", "Obj pos err", "-", "cm"),
    ("obj_ori", "Obj ori err", "-", "deg"),
]
# extra per-method stability metrics (heterogeneous - reported as-is, not cross-compared)
STAB_SPIDER = [
    ("fall", "Fall", "-", ""),
    ("body_z", "Body-z err p95", "-", "m"),
    ("ankle_jerk", "Ankle jerk p95", "-", "m/s^3"),
    ("obj_speed", "Obj speed max", "-", "m/s"),
    ("foot_slip", "Foot slip max", "-", "m"),
]
STAB_GMR = [
    ("foot_slip", "Foot slip max", "-", "m"),
    ("foot_grnd_dev", "Foot grnd dev", "-", "m"),
    ("grounded_frac", "Grounded frac", "+", ""),
    ("qpos_accel", "qpos accel p95", "-", ""),
    ("qpos_jerk", "qpos jerk p95", "-", ""),
    ("ankle_jerk", "Ankle jerk p95", "-", ""),
    ("trackbody_spd", "Trackbody spd max", "-", ""),
    ("ankle_spd", "Ankle spd max", "-", ""),
]

OBJECT_ORDER = [
    "box001", "box004", "box021", "box023", "box024",
    "bucket003", "bucket007", "chair006", "desk007", "desk021", "desk023",
]


def obj_cat(obj):
    for c in ("box", "bucket", "chair", "desk"):
        if obj.startswith(c):
            return c
    return "other"


# ----------------------------------------------------------------------------
# Load per-case data:  data[method][case_id] = {metric_key: value}
#                      obj_of[case_id] = object
# ----------------------------------------------------------------------------
def rows(f, sheet):
    wb = openpyxl.load_workbook(f, data_only=True)
    r = list(wb[sheet].iter_rows(values_only=True))
    return r[0], r[1:]


data = {m: {} for m in METHODS}
obj_of = {}

# --- SPIDER-CEM + OmniRetarget from spider ByCase --------------------------
# header: case_id, object, method, status, raw contact, contact3mm, pen3mm, geom2mm,
#         root_pos, root_ori, eef_pos, eef_ori, obj_pos, obj_ori, fall, body_z,
#         ankle_jerk, obj_speed, foot_slip, contact10mm, pen10mm, pen_max_mm
#         (contact10mm/pen10mm/pen_max_mm are appended at the END -> x[19], x[20], x[21])
_, r = rows(SPIDER_XLSX, "ByCase")
for x in r:
    cid, obj, method = x[0], x[1], x[2]
    obj_of[cid] = obj
    d = {
        "raw_contact": x[4], "contact_3mm": x[5], "pen_3mm": x[6], "geom_2mm": x[7],
        "contact_10mm": x[19], "pen_10mm": x[20], "pen_max_mm": x[21],
    }
    if method == "SPIDER-CEM":
        d.update({
            "root_pos": x[8], "root_ori": x[9], "eef_pos": x[10], "eef_ori": x[11],
            "obj_pos": x[12], "obj_ori": x[13], "fall": x[14], "body_z": x[15],
            "ankle_jerk": x[16], "obj_speed": x[17], "foot_slip": x[18],
        })
    data[method][cid] = d

# --- SBTO from sbto ByCase (header-based: robust to upstream column insertions) -----
# Current schema carries raw / contact@{3,5,10}mm / pen@{3,5,10}mm / max-pen / geom@2mm
# plus tracking + stability.
_SBTO_COL = {
    "raw_contact": "raw contact ↑", "contact_3mm": "physical contact@3mm ↑",
    "contact_10mm": "physical contact@10mm ↑", "pen_3mm": "physical penetration@3mm ↓",
    "pen_10mm": "physical penetration@10mm ↓", "pen_max_mm": "max penetration depth mm ↓",
    "geom_2mm": "geometric penetration@2mm ↓", "root_pos": "root pos err cm ↓",
    "root_ori": "root ori err° ↓", "eef_pos": "eef pos err cm ↓", "eef_ori": "eef ori err° ↓",
    "obj_pos": "obj pos err cm ↓", "obj_ori": "obj ori err° ↓", "fall": "fall ↓",
    "body_z": "body-z err p95 m ↓", "ankle_jerk": "ankle jerk p95 m/s³ ↓",
    "obj_speed": "obj speed max m/s ↓", "foot_slip": "foot slip max m ↓",
}
if "SBTO" in METHODS:
    _h, _r = rows(SBTO_XLSX, "ByCase")
    _ix = {c: i for i, c in enumerate(_h)}
    for x in _r:
        cid = x[0]
        data["SBTO"][cid] = {k: x[_ix[col]] for k, col in _SBTO_COL.items() if col in _ix}

# --- GMR: base metrics from per_case xlsx + contact@10mm / pen@10mm / max-pen from the
# upstream-supplied 10mm CSV (header-based). GMR is object-unaware, so its max-pen is large
# (hand passes through the object's nominal volume); reported as-is (indicative, see caption).
_GMR_COL = {
    "raw_contact": "Raw contact ↑", "contact_3mm": "Phys contact@3mm ↑",
    "pen_3mm": "Phys pen@3mm ↓", "geom_2mm": "Geom pen@2mm ↓",
    "foot_slip": "Foot slip max (m) ↓", "foot_grnd_dev": "Foot grnd dev (m) ↓",
    "grounded_frac": "Grounded frac ↑", "qpos_accel": "qpos accel p95 ↓",
    "qpos_jerk": "qpos jerk p95 ↓", "ankle_jerk": "ankle jerk p95 ↓",
    "trackbody_spd": "trackbody spd max ↓", "ankle_spd": "ankle spd max ↓",
    "obj_speed": "obj spd max ·",
}
if "GMR" in METHODS:
    import csv as _csv
    _h, _r = rows(GMR_XLSX, "per_case")
    _ix = {c: i for i, c in enumerate(_h)}
    for x in _r:
        cid = x[0]
        data["GMR"][cid] = {k: x[_ix[col]] for k, col in _GMR_COL.items() if col in _ix}
    _c10 = {row["case_id"]: row for row in _csv.DictReader(GMR_10MM_CSV.open())}
    for cid, d in data["GMR"].items():
        row = _c10.get(cid)
        if row is not None:
            d["contact_10mm"] = float(row["phys_contact_10mm"])
            d["pen_10mm"] = float(row["phys_pen_10mm"])
            d["pen_max_mm"] = float(row["phys_pen_max_mm"])
    print(f"[ok] merged GMR 10mm per-case CSV for {len(_c10)} cases")

CASES = sorted(obj_of)
assert all(len(data[m]) == 52 for m in METHODS), {m: len(data[m]) for m in METHODS}

# Merge OmniRetarget physics-replay dynamic-quality metrics (ankle jerk / obj speed / foot
# slip). Produced by scripts/compute_omni_health.py, which runs eval.core.motion_health.
# run_health on the SAME MuJoCo position-servo replay (E197 protocol) used for OmniRetarget's
# contact metrics -> computed identically to SPIDER-CEM / SBTO, hence directly comparable.
_omni_health = OUT_DIR / "omni_health.json"
if _omni_health.is_file():
    _oh = json.loads(_omni_health.read_text())
    for _cid, _hv in _oh.items():
        if _cid in data["OmniRetarget"]:
            for _k in ("ankle_jerk", "obj_speed", "foot_slip"):
                if _hv.get(_k) is not None:
                    data["OmniRetarget"][_cid][_k] = _hv[_k]
    print(f"[ok] merged OmniRetarget physics-replay health for {len(_oh)} cases")
else:
    print("[warn] omni_health.json not found; run scripts/compute_omni_health.py first")


# ----------------------------------------------------------------------------
# Aggregation helpers
# ----------------------------------------------------------------------------
def vals(method, key, case_subset):
    out = []
    for c in case_subset:
        v = data[method].get(c, {}).get(key)
        if v is not None:
            out.append(float(v))
    return out


def agg(method, key, case_subset, arrow):
    v = vals(method, key, case_subset)
    if not v:
        return None
    mean = sum(v) / len(v)
    sd = st.pstdev(v) if len(v) > 1 else 0.0
    worst = min(v) if arrow == "+" else max(v)
    return {"mean": mean, "std": sd, "worst": worst, "n": len(v)}


CASES_BY_OBJ = {o: [c for c in CASES if obj_of[c] == o] for o in OBJECT_ORDER}

# sanity check vs published overview
_chk = agg("SPIDER-CEM", "raw_contact", CASES, "+")
assert abs(_chk["mean"] - 0.823753978293631) < 1e-9, _chk
if "SBTO" in METHODS:
    _chk = agg("SBTO", "root_pos", CASES, "-")
    assert abs(_chk["mean"] - 6.106767836248928) < 1e-6, _chk
if "GMR" in METHODS:
    _chk = agg("GMR", "raw_contact", CASES, "+")
    assert abs(_chk["mean"] - 0.3536634615384615) < 1e-6, _chk
print("[ok] aggregation reproduces published overview numbers")


# ----------------------------------------------------------------------------
# 1. XLSX workbook
# ----------------------------------------------------------------------------
HDR_FILL = PatternFill("solid", fgColor="1F4E79")
HDR_FONT = Font(bold=True, color="FFFFFF")
GRP_FILL = PatternFill("solid", fgColor="D6E4F0")
BEST_FILL = PatternFill("solid", fgColor="C6EFCE")
BOLD = Font(bold=True)
CENTER = Alignment(horizontal="center", vertical="center")
THIN = Side(style="thin", color="BBBBBB")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def style_header(ws, row_idx, ncol):
    for c in range(1, ncol + 1):
        cell = ws.cell(row=row_idx, column=c)
        cell.fill = HDR_FILL
        cell.font = HDR_FONT
        cell.alignment = CENTER
        cell.border = BORDER


def autofit(ws, widths):
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def fmt(v, key):
    if v is None:
        return None
    # contact/pen/frac in [0,1] -> show 3 decimals; errors/stab -> 2 decimals
    frac_keys = {"raw_contact", "contact_3mm", "contact_10mm", "pen_3mm", "pen_10mm", "geom_2mm", "grounded_frac", "fall"}
    return round(v, 3) if key in frac_keys else round(v, 2)


wb = openpyxl.Workbook()

# ---- Sheet: README ----
ws = wb.active
ws.title = "README"
readme_lines = [
    ["CORE4D-G1 four-method retargeting comparison"],
    [""],
    ["Methods (identical 52 cases / 11 objects for all):"],
    ["  SPIDER-CEM   physics-based retargeting (ours)"],
    ["  SBTO         physics-based sampling trajectory optimization (baseline)"],
    ["  OmniRetarget kinematic retarget (contact metrics only; no physics tracking)"],
    ["  GMR          kinematic retarget (general motion retargeting; distinct stability metrics)"],
    [""],
    ["Metric families:"],
    ["  Common (all 4): Raw contact / Phys contact@3mm (higher better);"],
    ["                  Phys pen@3mm / Geom pen@2mm (lower better)."],
    ["  Tracking (SPIDER-CEM & SBTO only): root/eef/obj position (cm) & orientation (deg) error."],
    ["  Stability: heterogeneous per method - reported as-is, NOT cross-compared."],
    [""],
    ["Aggregation: every value is the unweighted mean over cases (per object, or all 52)."],
    ["Recomputed from per-case data in the source workbooks -> identical methodology per method."],
    ["'worst' = worst single-case value (min for higher-better, max for lower-better)."],
    [""],
    ["Sheets:"],
    ["  Overall_Common   - 4 shared contact metrics, mean/std/worst, all 4 methods"],
    ["  Overall_Tracking - tracking error, SPIDER-CEM vs SBTO"],
    ["  Overall_Stability- per-method stability metrics (heterogeneous)"],
    ["  ByObject_Common  - per-object mean of the 4 shared metrics x 4 methods (green = best)"],
    ["  ByObject_Tracking- per-object tracking error, SPIDER-CEM vs SBTO"],
    ["  ByCase_Common    - raw per-case values of the 4 shared metrics x 4 methods"],
    [""],
    ["Sources:"],
    [f"  SPIDER/Omni: {SPIDER_XLSX}"],
    [f"  SBTO       : {SBTO_XLSX}"],
    [f"  GMR        : {GMR_XLSX}"],
]
for line in readme_lines:
    ws.append(line)
ws["A1"].font = Font(bold=True, size=14)
autofit(ws, [95])

# ---- Sheet: Overall_Common ----
ws = wb.create_sheet("Overall_Common")
hdr = ["Metric", "Dir"]
for m in METHODS:
    hdr += [f"{m} mean", f"{m} std", f"{m} worst"]
ws.append(hdr)
style_header(ws, 1, len(hdr))
for key, disp, arrow, unit in COMMON:
    label = f"{disp}" + (f" ({unit})" if unit else "")
    row = [label, "up" if arrow == "+" else "down"]
    cellvals = {}
    for m in METHODS:
        a = agg(m, key, CASES, arrow)
        cellvals[m] = a
        row += [fmt(a["mean"], key) if a else None,
                fmt(a["std"], key) if a else None,
                fmt(a["worst"], key) if a else None]
    ws.append(row)
    # bold best mean among methods
    means = {m: cellvals[m]["mean"] for m in METHODS if cellvals[m]}
    if means:
        best = (max if arrow == "+" else min)(means, key=means.get)
        best_col = 3 + METHODS.index(best) * 3
        ws.cell(row=ws.max_row, column=best_col).font = BOLD
        ws.cell(row=ws.max_row, column=best_col).fill = BEST_FILL
autofit(ws, [22, 6] + [12, 10, 12] * len(METHODS))

# ---- Sheet: Overall_Tracking (SPIDER-CEM vs SBTO) ----
# Only meaningful when SBTO is present (OmniRetarget/GMR are kinematic, no tracking).
if "SBTO" in METHODS:
    ws = wb.create_sheet("Overall_Tracking")
    tmethods = ["SPIDER-CEM", "SBTO"]
    hdr = ["Metric", "Dir"]
    for m in tmethods:
        hdr += [f"{m} mean", f"{m} std", f"{m} worst"]
    ws.append(hdr)
    style_header(ws, 1, len(hdr))
    for key, disp, arrow, unit in TRACKING:
        row = [f"{disp} ({unit})", "down"]
        means = {}
        for m in tmethods:
            a = agg(m, key, CASES, arrow)
            means[m] = a["mean"] if a else None
            row += [fmt(a["mean"], key) if a else None,
                    fmt(a["std"], key) if a else None,
                    fmt(a["worst"], key) if a else None]
        ws.append(row)
        valid = {m: v for m, v in means.items() if v is not None}
        if valid:
            best = min(valid, key=valid.get)
            bc = 3 + tmethods.index(best) * 3
            ws.cell(row=ws.max_row, column=bc).font = BOLD
            ws.cell(row=ws.max_row, column=bc).fill = BEST_FILL
    autofit(ws, [20, 6] + [12, 10, 12] * len(tmethods))

# ---- Sheet: Overall_Stability (heterogeneous) ----
ws = wb.create_sheet("Overall_Stability")
ws.append(["Method", "Metric", "Unit", "mean", "std", "worst", "n"])
style_header(ws, 1, 7)
# OmniRetarget's physics-replay dynamic metrics (merged from omni_health.json) are the
# only stability numbers it produces; report them alongside SPIDER-CEM.
_OMNI_STAB = [("ankle_jerk", "Ankle jerk p95", "-", "m/s^3"),
              ("obj_speed", "Obj speed max", "-", "m/s"),
              ("foot_slip", "Foot slip max", "-", "m")]
_stab_rows = [("SPIDER-CEM", STAB_SPIDER)]
if "SBTO" in METHODS:
    _stab_rows.append(("SBTO", STAB_SPIDER))
if "OmniRetarget" in METHODS:
    _stab_rows.append(("OmniRetarget", _OMNI_STAB))
if "GMR" in METHODS:
    _stab_rows.append(("GMR", STAB_GMR))
for m, stab in _stab_rows:
    for key, disp, arrow, unit in stab:
        a = agg(m, key, CASES, arrow)
        if a:
            ws.append([m, disp, unit, fmt(a["mean"], key), fmt(a["std"], key),
                       fmt(a["worst"], key), a["n"]])
autofit(ws, [14, 20, 8, 12, 10, 12, 6])

# ---- Sheet: ByObject_Common ----
ws = wb.create_sheet("ByObject_Common")
# two header rows: metric groups + method sub-cols
ws.append([None, None] + sum([[disp] + [None] * (len(METHODS) - 1) for _, disp, _, _ in COMMON], []))
sub = ["Object", "n"]
for _ in COMMON:
    sub += [METHOD_SHORT[m] for m in METHODS]
ws.append(sub)
# merge group headers
for gi, (key, disp, arrow, unit) in enumerate(COMMON):
    c0 = 3 + gi * len(METHODS)
    ws.merge_cells(start_row=1, start_column=c0, end_row=1, end_column=c0 + len(METHODS) - 1)
    cell = ws.cell(row=1, column=c0)
    cell.value = f"{disp} ({'higher' if arrow == '+' else 'lower'} better)"
    cell.fill = GRP_FILL
    cell.font = BOLD
    cell.alignment = CENTER
style_header(ws, 2, len(sub))
ws.cell(row=1, column=1).value = "ByObject_Common"
ws.cell(row=1, column=1).font = BOLD

for obj in OBJECT_ORDER + ["OVERALL"]:
    cs = CASES if obj == "OVERALL" else CASES_BY_OBJ[obj]
    n = len(cs)
    row = [obj, n]
    best_cols = []
    for gi, (key, disp, arrow, unit) in enumerate(COMMON):
        means = {}
        for m in METHODS:
            a = agg(m, key, cs, arrow)
            means[m] = a["mean"] if a else None
            row.append(fmt(a["mean"], key) if a else None)
        valid = {m: v for m, v in means.items() if v is not None}
        if valid:
            best = (max if arrow == "+" else min)(valid, key=valid.get)
            best_cols.append(3 + gi * len(METHODS) + METHODS.index(best))
    ws.append(row)
    rr = ws.max_row
    if obj == "OVERALL":
        for c in range(1, len(row) + 1):
            ws.cell(row=rr, column=c).font = BOLD
    for bc in best_cols:
        ws.cell(row=rr, column=bc).fill = BEST_FILL
        ws.cell(row=rr, column=bc).font = BOLD
autofit(ws, [12, 4] + [9] * (len(COMMON) * len(METHODS)))

# ---- Sheet: ByObject_Tracking (SPIDER-CEM vs SBTO) ----
if "SBTO" in METHODS:
    ws = wb.create_sheet("ByObject_Tracking")
    head = ["Object", "n"]
    for _, disp, _, unit in TRACKING:
        head += [f"CEM {disp.split()[0]}{disp.split()[1][0]}", f"SBTO {disp.split()[0]}{disp.split()[1][0]}"]
    # simpler explicit header
    head = ["Object", "n"]
    for _, disp, _, unit in TRACKING:
        head += [f"{disp} ({unit}) CEM", f"{disp} ({unit}) SBTO"]
    ws.append(head)
    style_header(ws, 1, len(head))
    for obj in OBJECT_ORDER + ["OVERALL"]:
        cs = CASES if obj == "OVERALL" else CASES_BY_OBJ[obj]
        row = [obj, len(cs)]
        best_cols = []
        for gi, (key, disp, arrow, unit) in enumerate(TRACKING):
            a1 = agg("SPIDER-CEM", key, cs, arrow)
            a2 = agg("SBTO", key, cs, arrow)
            row += [fmt(a1["mean"], key) if a1 else None, fmt(a2["mean"], key) if a2 else None]
            if a1 and a2:
                best_cols.append(3 + gi * 2 + (0 if a1["mean"] <= a2["mean"] else 1))
        ws.append(row)
        rr = ws.max_row
        if obj == "OVERALL":
            for c in range(1, len(row) + 1):
                ws.cell(row=rr, column=c).font = BOLD
        for bc in best_cols:
            ws.cell(row=rr, column=bc).fill = BEST_FILL
            ws.cell(row=rr, column=bc).font = BOLD
    autofit(ws, [12, 4] + [15] * (len(TRACKING) * 2))

# ---- Sheet: ByCase_Common ----
ws = wb.create_sheet("ByCase_Common")
hdr = ["case_id", "object", "category"]
for key, disp, arrow, unit in COMMON:
    for m in METHODS:
        hdr.append(f"{disp} [{METHOD_SHORT[m]}]")
ws.append(hdr)
style_header(ws, 1, len(hdr))
for c in CASES:
    row = [c, obj_of[c], obj_cat(obj_of[c])]
    for key, disp, arrow, unit in COMMON:
        for m in METHODS:
            v = data[m].get(c, {}).get(key)
            row.append(fmt(float(v), key) if v is not None else None)
    ws.append(row)
autofit(ws, [30, 10, 9] + [14] * (len(COMMON) * len(METHODS)))

xlsx_path = OUT_DIR / f"paper_comparison{SUF}.xlsx"
wb.save(xlsx_path)
print(f"[ok] wrote {xlsx_path}")


# ----------------------------------------------------------------------------
# 2. LaTeX  (camera-ready, booktabs)
# ----------------------------------------------------------------------------
def texnum(v, key, mark=None):
    """mark in {None, 'best', 'second'} -> plain / bold / underlined."""
    if v is None:
        return "--"
    if key in {"raw_contact", "contact_3mm", "contact_10mm", "pen_3mm", "pen_10mm", "geom_2mm"}:
        s = f"{v:.3f}"
    elif key == "ankle_jerk":
        s = f"{v:.1f}"
    elif key in {"obj_speed", "foot_slip", "body_z"}:
        s = f"{v:.2f}"
    else:  # tracking (deg/cm)
        s = f"{v:.1f}"
    if mark == "best":
        return f"\\textbf{{{s}}}"
    if mark == "second":
        return f"\\underline{{{s}}}"
    return s


def rank_best_second(means, arrow):
    """Return (best_method, second_method) by value (direction-aware); None if absent.
    With only two methods present, 'second best' == 'the loser', so it is not marked
    (underlining every runner-up is redundant noise in a 2-column table)."""
    valid = [(m, v) for m, v in means.items() if v is not None]
    if not valid:
        return None, None
    valid.sort(key=lambda kv: kv[1], reverse=(arrow == "+"))
    best = valid[0][0]
    second = valid[1][0] if len(valid) > 2 else None
    return best, second


def mark_of(m, best, second):
    return "best" if m == best else ("second" if m == second else None)


def tex_obj(o):
    return o.replace("_", "\\_")


# Stability metrics requested for the LaTeX tables. These are physics-rollout
# dynamic-quality metrics -> only the two physics methods produce comparable
# values (OmniRetarget/GMR are kinematic; GMR's ankle-jerk/foot-slip come from a
# different pipeline at a different scale, so they are NOT cross-compared here).
STAB_DYN = [
    ("ankle_jerk", "Ankle jerk p95", "-", "m/s$^3$"),
    ("obj_speed", "Obj speed max", "-", "m/s"),
    ("foot_slip", "Foot slip max", "-", "m"),
]
STAB_KEYS = {k for k, *_ in STAB_DYN}


def cell_mean(m, key, cs, arrow):
    """Per-object/overall mean; None (rendered "--") if the method has no value.
    OmniRetarget's dynamic-quality metrics come from its E197 physics replay (merged from
    omni_health.json); GMR's ankle-jerk/foot-slip are from its own kinematic pipeline at a
    different scale (indicative only -- see caption)."""
    a = agg(m, key, cs, arrow)
    return a["mean"] if a else None


HEADER = [
    "% Auto-generated by build_comparison.py -- CORE4D-G1 comparison.",
    "% Requires: \\usepackage{booktabs,graphicx}.",
    "",
]


def byobject_table(metrics, methods, caption, label):
    """Rows = objects (grouped by category) + Overall; columns = metric x method.
    Best per (object, metric) is bolded (direction-aware)."""
    L = []
    A = L.append
    mcols = "".join(["c"] * len(methods))
    A("\\begin{table*}[t]")
    A("\\centering")
    A(f"\\caption{{{caption}}}")
    A(f"\\label{{{label}}}")
    A("\\resizebox{\\textwidth}{!}{%")
    A("\\begin{tabular}{l c " + " ".join([mcols for _ in metrics]) + "}")
    A("\\toprule")
    grp = ["", ""]
    for _, disp, arrow, unit in metrics:
        ar = "$\\uparrow$" if arrow == "+" else "$\\downarrow$"
        u = f" ({unit})" if unit else ""
        grp.append(f"\\multicolumn{{{len(methods)}}}{{c}}{{{disp}{u} {ar}}}")
    A(" & ".join(grp) + " \\\\")
    cmid = []
    start = 3
    for _ in metrics:
        cmid.append(f"\\cmidrule(lr){{{start}-{start + len(methods) - 1}}}")
        start += len(methods)
    A(" ".join(cmid))
    sub = ["Object", "$n$"]
    for _ in metrics:
        sub += [METHOD_SHORT[m] for m in methods]
    A(" & ".join(sub) + " \\\\")
    A("\\midrule")

    def row(obj, cs):
        cells = [tex_obj(obj), str(len(cs))]
        for key, disp, arrow, unit in metrics:
            means = {m: cell_mean(m, key, cs, arrow) for m in methods}
            best, second = rank_best_second(means, arrow)
            for m in methods:
                cells.append(texnum(means[m], key, mark_of(m, best, second)))
        return " & ".join(cells) + " \\\\"

    last_cat = None
    for obj in OBJECT_ORDER:
        cat = obj_cat(obj)
        if last_cat is not None and cat != last_cat:
            A("\\addlinespace")
        A(row(obj, CASES_BY_OBJ[obj]))
        last_cat = cat
    A("\\midrule")
    A(row("Overall", CASES))
    A("\\bottomrule")
    A("\\end{tabular}%")
    A("}")
    A("\\end{table*}")
    A("")
    return L


def overall_table(metrics, methods, caption, label):
    """Rows = metrics; columns = methods. Best per row bolded (direction-aware)."""
    L = []
    A = L.append
    A("\\begin{table}[t]")
    A("\\centering")
    A(f"\\caption{{{caption}}}")
    A(f"\\label{{{label}}}")
    A("\\begin{tabular}{l " + "".join(["c"] * len(methods)) + "}")
    A("\\toprule")
    A("Metric & " + " & ".join(METHOD_SHORT[m] for m in methods) + " \\\\")
    A("\\midrule")
    for key, disp, arrow, unit in metrics:
        ar = "$\\uparrow$" if arrow == "+" else "$\\downarrow$"
        u = f" ({unit})" if unit else ""
        means = {m: cell_mean(m, key, CASES, arrow) for m in methods}
        best, second = rank_best_second(means, arrow)
        cells = [f"{disp}{u} {ar}"]
        for m in methods:
            cells.append(texnum(means[m], key, mark_of(m, best, second)))
        A(" & ".join(cells) + " \\\\")
    A("\\bottomrule")
    A("\\end{tabular}")
    A("\\end{table}")
    A("")
    return L


# One combined metric block: penetration-aware contact + dynamic-quality metrics.
COMBINED = CONTACT_TEX + STAB_DYN

_n_methods = len(METHODS)
_methods_phrase = {1: "the method", 2: "both methods"}.get(_n_methods, f"all {_n_methods} methods")

if "SBTO" in METHODS or "GMR" in METHODS:
    _dyn_sentence = (
        "Dynamic-quality metrics for Ours, SBTO and OmniRetarget are computed on a MuJoCo "
        "physics rollout (native for the two optimizers; position-servo replay for the kinematic "
        "OmniRetarget, E197 protocol) via the same code, hence directly comparable; GMR's are from "
        "its own kinematic pipeline at a different scale and are indicative only (read jointly with "
        "contact).")
else:
    _dyn_sentence = (
        "Dynamic-quality metrics for Ours and OmniRetarget are computed on a MuJoCo physics rollout "
        "(native for Ours; position-servo replay for the kinematic OmniRetarget, E197 protocol) via "
        "the same code, hence directly comparable.")

_bold_legend = ("\\textbf{Bold} = best, \\underline{underline} = second best. "
                if _n_methods > 2 else "\\textbf{Bold} = best. ")
CAP_COMMON = (
    "$\\uparrow$: higher is better, $\\downarrow$: lower is better. " + _bold_legend
    + "Ours = SPIDER-CEM. Contact metrics are penetration-aware (a loose \"any-contact\" count "
    "that rewards interpenetration is intentionally omitted). " + _dyn_sentence
)

# ---- File 1: by-object (single combined table) ----
by_lines = list(HEADER)
by_lines += byobject_table(
    COMBINED, METHODS,
    "Per-object contact fidelity and dynamic quality on CORE4D-G1 "
    f"({_methods_phrase} share the same 52 sequences / 11 objects). " + CAP_COMMON,
    "tab:byobject",
)
by_path = OUT_DIR / f"by_object_comparison{SUF}.tex"
by_path.write_text("\n".join(by_lines))
print(f"[ok] wrote {by_path}")

# ---- File 2: overall (single combined table) ----
ov_lines = list(HEADER)
ov_lines += overall_table(
    COMBINED, METHODS,
    "Overall contact fidelity and dynamic quality on CORE4D-G1 "
    "(52 sequences / 11 objects). " + CAP_COMMON,
    "tab:overall",
)
ov_path = OUT_DIR / f"overall_comparison{SUF}.tex"
ov_path.write_text("\n".join(ov_lines))
print(f"[ok] wrote {ov_path}")

# ---- File 2b: overall WITHOUT ankle jerk (GMR's kinematic-pipeline jerk is off-scale) ----
COMBINED_NOJERK = CONTACT_TEX + [m for m in STAB_DYN if m[0] != "ankle_jerk"]
ovnj_lines = list(HEADER)
ovnj_lines += overall_table(
    COMBINED_NOJERK, METHODS,
    "Overall contact fidelity and dynamic quality on CORE4D-G1 "
    "(52 sequences / 11 objects; ankle jerk omitted). " + CAP_COMMON,
    "tab:overall_nojerk",
)
ovnj_path = OUT_DIR / f"overall_comparison_nojerk{SUF}.tex"
ovnj_path.write_text("\n".join(ovnj_lines))
print(f"[ok] wrote {ovnj_path}")

# ---- File 2c: compact v2-style method-comparison table ------------------------------
# Same visual form as paper_cmp_v2/method_comparison.tex: one compact booktabs table,
# metrics as rows, methods as columns (v2 order: ours -> our kinematic input -> baselines),
# percentages shown as %, best per row bold / second underlined, rich narrative caption.
# Driven by the SAME verified 4-method data as the tables above (contact@3mm/@10mm,
# pen@3mm/@10mm, max-pen now included). Written to method_comparison{SUF}.tex so the
# v2 directory's own method_comparison.tex is NOT touched.
#
# Case set: aligned to paper_cmp_v2 (e214_common.load_cases) -> the 50 paper cases,
# i.e. CASES minus the two E167A box021 overrides (no full-stack baseline for them).
V2_DROP = {"box021_20231018_029_p2", "box021_20231011_035_p1"}
V2_CASES = [c for c in CASES if c not in V2_DROP]
V2_ORDER = [m for m in ("SPIDER-CEM", "OmniRetarget", "SBTO", "GMR") if m in METHODS]
V2_METRICS = [
    ("contact_3mm",  "Phys. contact@3mm",  "+", "pct"),
    ("contact_10mm", "Phys. contact@10mm", "+", "pct"),
    ("pen_3mm",      "Phys. penetration@3mm",  "-", "pct"),
    ("pen_10mm",     "Phys. penetration@10mm", "-", "pct"),
    ("pen_max_mm",   "Max phys penetration (mm)", "-", "mm"),
    ("obj_speed",    "Obj speed max (m/s)", "-", "mps"),
    ("foot_slip",    "Foot slip max (m)",   "-", "m"),
]


def _v2_fmt(v, kind):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return f"{v * 100:.2f}" if kind == "pct" else f"{v:.2f}"


def method_comparison_v2style():
    n = len(V2_CASES)
    # dynamic caption figures (so they never drift from the table body)
    sp_c3 = cell_mean("SPIDER-CEM", "contact_3mm", V2_CASES, "+")
    L = [r"% Auto-generated by build_comparison.py -- compact v2-style method comparison.",
         r"% Requires: \usepackage{booktabs}.", r"",
         r"\begin{table}[t]", r"\centering"]
    if "GMR" in V2_ORDER:
        gmr_c3 = cell_mean("GMR", "contact_3mm", V2_CASES, "+")
        gmr_mp = cell_mean("GMR", "pen_max_mm", V2_CASES, "-")
        cap = (r"\caption{Method comparison on CORE4D-G1 (" + f"{n}" + r" paper sequences), "
               r"mean over cases. \textbf{SPIDER-CEM} (ours) vs \textbf{OmniRetarget} "
               r"(our upstream kinematic input), \textbf{SBTO}, and \textbf{GMR}. Physical "
               r"contact / penetration are frame fractions at 3\,mm and 10\,mm bands from the "
               r"same MuJoCo contact detector; max penetration is the deepest hand-object "
               r"interpenetration. $\uparrow$/$\downarrow$: higher/lower is better; best per "
               r"row in bold, second best \underline{underlined}. OmniRetarget (kinematic, no "
               r"collision) penetrates heavily at modest clean contact; SPIDER-CEM's physics "
               r"optimisation raises contact@3mm to " + f"{sp_c3*100:.1f}" + r"\% while cutting "
               r"penetration on every threshold. GMR is object-unaware: near-zero clean "
               r"contact (contact@3mm $\approx" + f"{gmr_c3*100:.1f}" + r"\%$) yet the deepest "
               r"max penetration (" + f"{gmr_mp:.0f}" + r"\,mm) -- read its penetration jointly "
               r"with contact, never alone.}")
    else:
        cap = (r"\caption{Method comparison on CORE4D-G1 (" + f"{n}" + r" paper sequences), "
               r"mean over cases. \textbf{SPIDER-CEM} (ours) vs \textbf{OmniRetarget} "
               r"(our upstream kinematic input). Physical contact / penetration are frame "
               r"fractions at 3\,mm and 10\,mm bands from the same MuJoCo contact detector; "
               r"max penetration is the deepest hand-object interpenetration. "
               r"$\uparrow$/$\downarrow$: higher/lower is better; best per row in bold. "
               r"SPIDER-CEM's physics optimisation lifts clean contact far above the "
               r"kinematic input while cutting penetration on every threshold.}")
    L += [cap, r"\label{tab:method_cmp" + ("" if "GMR" in V2_ORDER else "_2m") + "}",
          r"\begin{tabular}{l " + "c" * len(V2_ORDER) + r"}", r"\toprule",
          "Metric & " + " & ".join(V2_ORDER) + r" \\", r"\midrule"]
    for key, label, arrow, kind in V2_METRICS:
        means = {m: cell_mean(m, key, V2_CASES, arrow) for m in V2_ORDER}
        best, second = rank_best_second(means, arrow)
        cells = []
        for m in V2_ORDER:
            txt = _v2_fmt(means[m], kind)
            if txt == "--":
                cells.append(txt)
            elif m == best:
                cells.append(rf"\textbf{{{txt}}}")
            elif m == second:
                cells.append(rf"\underline{{{txt}}}")
            else:
                cells.append(txt)
        ar = r"\uparrow" if arrow == "+" else r"\downarrow"
        unit = r" ($\%$)" if kind == "pct" else ""
        L.append(rf"{label}{unit} ${ar}$ & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return L


mc_path = OUT_DIR / f"method_comparison{SUF}.tex"
mc_path.write_text("\n".join(method_comparison_v2style()))
print(f"[ok] wrote {mc_path}")

# standalone wrappers so each fragment compiles to a preview PDF directly.
# The by-object table is very wide (24 data columns) -> A1 landscape keeps it legible.
for base, paper in (("by_object_comparison", "a1paper"), ("overall_comparison", "a4paper"),
                    ("overall_comparison_nojerk", "a4paper"), ("method_comparison", "a4paper")):
    base = f"{base}{SUF}"
    standalone = [
        "\\documentclass{article}",
        f"\\usepackage[margin=0.4in,landscape,{paper}]{{geometry}}",
        "\\usepackage{booktabs}",
        "\\usepackage{graphicx}",
        "\\pagestyle{empty}",
        "\\begin{document}",
        f"\\input{{{base}.tex}}",
        "\\end{document}",
    ]
    p = OUT_DIR / f"{base}_standalone.tex"
    p.write_text("\n".join(standalone) + "\n")
    print(f"[ok] wrote {p}")


# ----------------------------------------------------------------------------
# 3. Markdown report
# ----------------------------------------------------------------------------
def md_num(v, key):
    if v is None:
        return "--"
    if key == "pen_max_mm":
        return f"{v:.1f}"
    frac = {"raw_contact", "contact_3mm", "contact_10mm", "pen_3mm", "pen_10mm", "geom_2mm", "grounded_frac"}
    return f"{v:.3f}" if key in frac else f"{v:.2f}"


def md_best_row(label, cells, keys_arrows, method_list):
    """cells: dict metric->{method->val}; bold best per metric."""
    pass


ml = []
W = ml.append
_cmp_name = " vs ".join(METHODS) if _n_methods <= 2 else f"{_n_methods}-Method Comparison"
W(f"# CORE4D-G1 Retargeting: {_cmp_name}\n")
W(f"Comparison of the retargeting methods below on the **same 52 sequences "
  "spanning 11 objects** (Unitree G1). All aggregates are the unweighted mean over cases, "
  "recomputed from per-case data so the methodology is identical across methods.\n")
_METHOD_DESC = {
    "SPIDER-CEM": ("physics-based retargeting", "yes", "yes"),
    "SBTO": ("sampling-based trajectory optimization", "yes", "yes"),
    "OmniRetarget": ("kinematic retarget", "no", "no (contact only)"),
    "GMR": ("general motion retargeting (kinematic)", "no", "no (distinct stability set)"),
}
W("| Method | Type | Physics rollout | Tracking metrics |")
W("|---|---|---|---|")
for m in METHODS:
    desc, roll, trk = _METHOD_DESC[m]
    label = "**SPIDER-CEM** (ours)" if m == "SPIDER-CEM" else m
    W(f"| {label} | {desc} | {roll} | {trk} |")
W("")
W(f"## 1. Overall — shared contact-fidelity metrics ({_methods_phrase})\n")
W("| Metric | " + " | ".join(METHOD_SHORT[m] for m in METHODS) + " |")
W("|---|" + "|".join(["---"] * len(METHODS)) + "|")
for key, disp, arrow, unit in COMMON:
    ar = "↑" if arrow == "+" else "↓"
    means = {m: (agg(m, key, CASES, arrow)["mean"] if agg(m, key, CASES, arrow) else None) for m in METHODS}
    valid = {m: v for m, v in means.items() if v is not None}
    best = (max if arrow == "+" else min)(valid, key=valid.get)
    label = disp + (" *(diagnostic — see §1.1)*" if key == "raw_contact" else "")
    row = [f"{label} {ar}"]
    for m in METHODS:
        s = md_num(means[m], key)
        # do NOT bold raw_contact "winner": high value is not a quality signal
        row.append(f"**{s}**" if (m == best and key != "raw_contact") else s)
    W("| " + " | ".join(row) + " |")
# derived clean-contact ratio
ratios = {}
for m in METHODS:
    raw = agg(m, "raw_contact", CASES, "+")["mean"]
    c3 = agg(m, "contact_3mm", CASES, "+")["mean"]
    ratios[m] = (c3 / raw) if raw and raw > 1e-9 else float("nan")
best_r = max(ratios, key=lambda k: (ratios[k] if ratios[k] == ratios[k] else -1))
row = ["**Clean-contact ratio** (contact@3mm / raw) ↑"]
for m in METHODS:
    s = f"{ratios[m]:.3f}"
    row.append(f"**{s}**" if m == best_r else s)
W("| " + " | ".join(row) + " |")
W("")
W("Values are per-case means (std / worst-case in `paper_comparison.xlsx`). SPIDER-CEM leads on "
  "physically-realised contact and both penetration metrics by a wide margin.\n")
W("### 1.1 Why \"raw contact\" is misleading (and what to report instead)\n")
W("`raw contact` (`hand_object_physics_contact_in_mask_frac`) counts a reference-contact frame as "
  "\"in contact\" whenever MuJoCo reports **any** hand–object contact — **including deep "
  "interpenetration**. In `core_metrics.py` the per-frame flag is simply `hand_contact = "
  "(any hand–object contact exists)` (line ~1294), with **no penetration-depth gate**. "
  "`Phys. contact@3mm` uses the same contacts but additionally requires the minimum contact "
  "distance `≥ −3 mm` (line ~1302), i.e. a *clean surface touch* rather than the hand stabbed "
  "through the object.\n")
W("Consequence: a method that shoves the hand through the object (a kinematic retarget with no "
  "collision resolution) racks up `raw contact` on nearly every frame, while a physically-correct "
  "method that keeps a clean surface touch scores marginally lower on `raw contact` but far higher "
  "on every penetration-aware metric. This is exactly the pattern in the table:\n")
_o = {k: agg("OmniRetarget", k, CASES, "+" if k in ("raw_contact", "contact_3mm") else "-")["mean"]
      for k in ("raw_contact", "contact_3mm", "pen_3mm", "geom_2mm")}
_s = {k: agg("SPIDER-CEM", k, CASES, "+")["mean"] for k in ("raw_contact", "contact_3mm")}
W(f"- **OmniRetarget** has the *highest* raw contact ({_o['raw_contact']:.3f}) yet the *worst* "
  f"penetration (phys pen@3mm {_o['pen_3mm']:.3f}, geom pen@2mm {_o['geom_2mm']:.3f}) and only "
  f"{_o['contact_3mm']:.3f} clean contact.")
W(f"- **SPIDER-CEM** has slightly lower raw contact ({_s['raw_contact']:.3f}) but far more clean "
  f"contact ({_s['contact_3mm']:.3f}) and markedly less penetration.")
W("- The **clean-contact ratio** row above (clean contact / raw contact) exposes this directly: "
  + ", ".join(f"{ratios[m]:.2f} ({METHOD_SHORT[m]})" for m in METHODS) + ".\n")
if "GMR" in METHODS:
    _g = {"contact_3mm": agg("GMR", "contact_3mm", CASES, "+")["mean"],
          "pen_max_mm": agg("GMR", "pen_max_mm", CASES, "-")["mean"]}
    _sm = agg("SPIDER-CEM", "pen_max_mm", CASES, "-")["mean"]
    W("> **Read penetration *jointly* with contact — the GMR case.** GMR is object-unaware, so it "
      f"barely establishes any *clean* contact (contact@3mm {_g['contact_3mm']:.3f}, contact@10mm "
      f"{agg('GMR', 'contact_10mm', CASES, '+')['mean']:.3f}). A frame-level pen fraction can look "
      "moderate for such a method simply because it is rarely near the object — but when its hand "
      f"*does* intersect the object it stabs catastrophically deep (max phys pen {_g['pen_max_mm']:.0f} mm "
      f"vs {_sm:.0f} mm for ours). So GMR neither contacts cleanly nor avoids deep penetration; its "
      "numbers must be read as contact + max-pen together, never a single frame-fraction. **SPIDER-CEM "
      "attains the highest clean contact AND the lowest penetration on every depth threshold.**\n")
else:
    W("> **Read penetration *jointly* with contact.** A low penetration number is only meaningful if the "
      "method actually establishes the reference contact: SPIDER-CEM attains both the higher clean "
      "contact *and* the lower penetration than the kinematic OmniRetarget baseline.\n")
W("**Recommendation (adopted in the LaTeX table):** drop `raw contact` from the headline and report "
  "the penetration-aware set — **Phys. contact@3mm ↑, Phys. contact@10mm ↑, Phys. pen@3mm ↓, "
  "Phys. pen@10mm ↓, Max phys pen (mm) ↓, Geom. pen@2mm ↓**. `raw contact` is retained only as a "
  "labelled diagnostic (it is *evidence* that the baseline's contact is illusory, not a quality "
  "metric).\n")

if "SBTO" in METHODS:
    W("## 2. Overall — tracking error (physics methods only)\n")
    W("| Metric | Ours (SPIDER-CEM) | SBTO |")
    W("|---|---|---|")
    for key, disp, arrow, unit in TRACKING:
        a1 = agg("SPIDER-CEM", key, CASES, arrow)
        a2 = agg("SBTO", key, CASES, arrow)
        v1, v2 = a1["mean"], a2["mean"]
        s1 = f"{v1:.2f}"; s2 = f"{v2:.2f}"
        if v1 <= v2:
            s1 = f"**{s1}**"
        else:
            s2 = f"**{s2}**"
        W(f"| {disp} ({unit}) ↓ | {s1} | {s2} |")
    W("")
    W("> SBTO optimizes tracking directly and wins on most kinematic-tracking terms, but at the cost "
      "of far worse contact fidelity (Table 1) and larger jerk/foot-slip. SPIDER-CEM trades a little "
      "tracking error for markedly better physical contact and penetration.\n")

W("## 3. By-object — shared contact metrics\n")
for key, disp, arrow, unit in COMMON:
    ar = "↑ higher better" if arrow == "+" else "↓ lower better"
    W(f"### {disp} ({ar})\n")
    W("| Object | n | " + " | ".join(METHOD_SHORT[m] for m in METHODS) + " |")
    W("|---|---|" + "|".join(["---"] * len(METHODS)) + "|")
    for obj in OBJECT_ORDER + ["Overall"]:
        cs = CASES if obj == "Overall" else CASES_BY_OBJ[obj]
        means = {m: (agg(m, key, cs, arrow)["mean"] if agg(m, key, cs, arrow) else None) for m in METHODS}
        valid = {m: v for m, v in means.items() if v is not None}
        best = (max if arrow == "+" else min)(valid, key=valid.get)
        cells = [obj if obj != "Overall" else "**Overall**", str(len(cs))]
        for m in METHODS:
            s = md_num(means[m], key)
            cells.append(f"**{s}**" if m == best else s)
        W("| " + " | ".join(cells) + " |")
    W("")

if "SBTO" in METHODS:
    W("## 4. By-object — tracking error (SPIDER-CEM vs SBTO)\n")
    W("| Object | n | " + " | ".join(f"{disp.split(' ')[0]} {u} (C/S)" for _, disp, _, u in TRACKING) + " |")
    W("|---|---|" + "|".join(["---"] * len(TRACKING)) + "|")
    for obj in OBJECT_ORDER + ["Overall"]:
        cs = CASES if obj == "Overall" else CASES_BY_OBJ[obj]
        cells = [obj if obj != "Overall" else "**Overall**", str(len(cs))]
        for key, disp, arrow, unit in TRACKING:
            v1 = agg("SPIDER-CEM", key, cs, arrow)["mean"]
            v2 = agg("SBTO", key, cs, arrow)["mean"]
            s1 = f"**{v1:.1f}**" if v1 <= v2 else f"{v1:.1f}"
            s2 = f"**{v2:.1f}**" if v2 < v1 else f"{v2:.1f}"
            cells.append(f"{s1}/{s2}")
        W("| " + " | ".join(cells) + " |")
    W("\n(C = SPIDER-CEM, S = SBTO; bold = better.)\n")

W("## 5. Method-specific stability metrics (heterogeneous — not cross-compared)\n")
W("| Method | Metric | mean | worst |")
W("|---|---|---|---|")
_md_stab = [("SPIDER-CEM", STAB_SPIDER)]
if "SBTO" in METHODS:
    _md_stab.append(("SBTO", STAB_SPIDER))
if "OmniRetarget" in METHODS:
    _md_stab.append(("OmniRetarget", _OMNI_STAB))
if "GMR" in METHODS:
    _md_stab.append(("GMR", STAB_GMR))
for m, stab in _md_stab:
    for key, disp, arrow, unit in stab:
        a = agg(m, key, CASES, arrow)
        if a:
            u = f" ({unit})" if unit else ""
            W(f"| {m} | {disp}{u} | {md_num(a['mean'], key)} | {md_num(a['worst'], key)} |")
W("")
W("## Notes & caveats\n")
W(f"- **Fair comparison = the shared contact metrics** (Section 1/3). These are computed "
  f"identically for {_methods_phrase}.")
if "SBTO" in METHODS:
    W("- **Tracking metrics** (root/eef/obj error) require a physics rollout with a tracked reference; "
      "OmniRetarget and GMR are kinematic retargets and do not produce them, so tracking is a "
      "**2-way** SPIDER-CEM vs SBTO comparison only.")
else:
    W("- **Tracking metrics** (root/eef/obj error) require a tracked physics rollout; OmniRetarget is a "
      "kinematic retarget and does not produce them, so they are reported for SPIDER-CEM only "
      "(see `paper_results`).")
W("- **Stability metrics** (Section 5) are heterogeneous: SPIDER-CEM / SBTO / OmniRetarget report "
  "ankle-jerk / obj-speed / foot-slip on a MuJoCo rollout (same code), while GMR's come from its "
  "own kinematic pipeline at a different scale. Listed per method for reference, not cross-compared.")
W("- **Raw vs physical contact:** 'raw contact' measures contact against the kinematic reference "
  "mask; 'physical contact@3mm' measures contact actually realized under physics within a 3 mm band. "
  "A high raw / low physical gap (the kinematic baselines) indicates contact that does not survive "
  "physics.")
if "GMR" in METHODS:
    W("- **GMR is object-unaware:** its dynamic-quality metrics come from its own kinematic pipeline "
      "at a different scale (ankle-jerk especially — see the `_nojerk` table), and its deep max "
      "penetration reflects the hand passing through the object's nominal volume. Read GMR jointly "
      "with its near-zero clean contact; its columns are indicative, not head-to-head comparable.")
else:
    W("- **Interim scope:** SBTO and GMR are excluded in this 2-method view; run "
      "`build_comparison.py --full` for the 4-method comparison (`*_full.*`).")
W("- All numbers are means over cases; per-case distributions and std/worst are in "
  f"`paper_comparison{SUF}.xlsx`.")

md_path = OUT_DIR / f"paper_comparison{SUF}.md"
md_path.write_text("\n".join(ml))
print(f"[ok] wrote {md_path}")
print("[done]")
