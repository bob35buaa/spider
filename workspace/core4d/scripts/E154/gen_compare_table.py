"""
生成 E154 评测修正后的方法对比表格
输出: workspace/core4d/results/E154/e154_masked_tracking_compare.xlsx

对比维度:
  Sheet1 "E153 阈值扫描": 6 个 gate 配置 + baseline + b1 (3-case 聚合)
  Sheet2 "E152 gateA_b1 vs ref": gateA / b1 / gateA_b1 / baseline (3-case 聚合)
  Sheet3 "per-case 明细": 18 行 E153 per-case (含 tracked 判定)
  Sheet4 "说明": 指标来源与颜色编码

新增指标(E154):
  - success_tracked: body tracking 门控 (pz_term < 0.08)
  - inmaskC: 真实 3cm mask 内的物理接触比例
  - release_false: 放手窗口内仍保持接触的比例 (诊断, 不进门控)
  - pz_term: track_pelvis_z_err_terminal (结尾姿态偏差)
"""
import pathlib
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

OUT = pathlib.Path(__file__).parents[2] / "results" / "E154" / "e154_masked_tracking_compare.xlsx"

# ── Style helpers ──────────────────────────────────────────────────────────────
THIN = Side(style="thin", color="FF000000")
MEDIUM = Side(style="medium", color="FF000000")

CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT = Alignment(horizontal="left", vertical="center", wrap_text=True)


def fill(hex6):
    return PatternFill("solid", fgColor="FF" + hex6)


def set_cell(ws, row, col, value="", bold=False, fg="000000", bg=None,
             align=CENTER, size=10, italic=False):
    c = ws.cell(row=row, column=col, value=value)
    c.font = Font(bold=bold, color="FF" + fg, size=size, italic=italic)
    c.alignment = align
    if bg:
        c.fill = fill(bg)
    c.border = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
    return c


def merge_cell(ws, r1, c1, r2, c2, value="", bold=False, fg="000000", bg=None, size=11):
    ws.merge_cells(start_row=r1, start_column=c1, end_row=r2, end_column=c2)
    c = ws.cell(row=r1, column=c1, value=value)
    c.font = Font(bold=bold, color="FF" + fg, size=size)
    c.alignment = CENTER
    if bg:
        c.fill = fill(bg)
    c.border = Border(left=MEDIUM, right=MEDIUM, top=MEDIUM, bottom=MEDIUM)
    return c


def pct(v):
    if v is None:
        return "—"
    return f"{v:.1%}" if abs(v) < 10 else f"{v:.3f}"


def pct3(v):
    """3 decimal places for small values like obj_err."""
    if v is None:
        return "—"
    return f"{v:.4f}"


# ── Color coding ──────────────────────────────────────────────────────────────
HDR_BG = "1F4E79"
SUB_BG = "BDD7EE"
GOOD_BG = "C6EFCE"
GOOD_FG = "276221"
BAD_BG = "FFC7CE"
BAD_FG = "9C0006"
REF_BG = "FFF2CC"
REF_FG = "7F6000"
BEST_BG = "92D050"
METHOD_BG = ["F2F2F2", "E2EFDA"]

# ══════════════════════════════════════════════════════════════════════════════
# Sheet 1: E153 阈值扫描 (6 grid + baseline + b1)
# ══════════════════════════════════════════════════════════════════════════════

# Data from e153_combo_summary.tsv + e153_method_metrics.tsv (3-case means)
# baseline/b1 from method_metrics (mean across 3 cases)
# Grid combos from combo_summary (already delta vs b1, we use raw values from method_metrics)

# Raw absolute values extracted from e153_method_metrics.tsv (3-case means)
BASELINE_RAW = {
    "5cm": (0.492647 + 0.619048 + 0.733333) / 3,  # hand_geom_near_5cm_frac
    "2mm_pen": (0.367647 + 0.314286 + 0.2) / 3,
    "5mm_pen": (0.294118 + 0.161905 + 0.16) / 3,
    "phys_contact": (0.433824 + 0.333333 + 0.28) / 3,
    "phys_pen5mm": (0.698925 + 0.326923 + 0.633333) / 3,
    "leg_pen": (0.007353 + 0.047619 + 0.08) / 3,
    "obj_err": (0.007185 + 0.007183 + 0.010551) / 3,
}

B1_RAW = {
    "5cm": (0.580882 + 0.714286 + 0.786667) / 3,
    "2mm_pen": (0.470588 + 0.542857 + 0.4) / 3,
    "5mm_pen": (0.286765 + 0.304762 + 0.226667) / 3,
    "phys_contact": (0.544118 + 0.590476 + 0.56) / 3,
    "phys_pen5mm": (0.598214 + 0.465909 + 0.583333) / 3,
    "leg_pen": (0.0 + 0.019048 + 0.093333) / 3,
    "obj_err": (0.007362 + 0.007942 + 0.011290) / 3,
}

# E153 grid absolute values (from method_metrics per-case, aggregated)
# Using data from e153_method_metrics.tsv, compute 3-case means for each grid combo
E153_GRID_RAW = {
    # (min_sdf, max_viol): {metric: mean_across_3_cases}
    (-0.005, 0.05): {
        "5cm": (0.786667 + 0.742857 + 0.588235) / 3,
        "2mm_pen": (0.293333 + 0.304762 + 0.279412) / 3,
        "5mm_pen": (0.0266667 + 0.0666667 + 0.0220588) / 3,
        "phys_contact": (0.506667 + 0.571429 + 0.492647) / 3,
        "phys_pen5mm": (0.215686 + 0.139241 + 0.131868) / 3,
        "leg_pen": (0.0666667 + 0.00952381 + 0.0) / 3,
        "obj_err": (0.0108579 + 0.00747149 + 0.00691036) / 3,
    },
    (-0.005, 0.10): {
        "5cm": (0.786667 + 0.771429 + 0.580882) / 3,
        "2mm_pen": (0.213333 + 0.380952 + 0.242647) / 3,
        "5mm_pen": (0.04 + 0.0666667 + 0.0220588) / 3,
        "phys_contact": (0.48 + 0.561905 + 0.463235) / 3,
        "phys_pen5mm": (0.208333 + 0.0933333 + 0.120879) / 3,
        "leg_pen": (0.0533333 + 0.047619 + 0.0) / 3,
        "obj_err": (0.0108527 + 0.00736173 + 0.00707433) / 3,
    },
    (-0.010, 0.05): {
        "5cm": (0.786667 + 0.771429 + 0.580882) / 3,
        "2mm_pen": (0.466667 + 0.4 + 0.433824) / 3,
        "5mm_pen": (0.253333 + 0.152381 + 0.227941) / 3,
        "phys_contact": (0.546667 + 0.609524 + 0.5) / 3,
        "phys_pen5mm": (0.457627 + 0.192771 + 0.413462) / 3,
        "leg_pen": (0.0533333 + 0.0 + 0.0) / 3,
        "obj_err": (0.010669 + 0.00746371 + 0.00717593) / 3,
    },
    (-0.010, 0.10): {
        "5cm": (0.786667 + 0.714286 + 0.580882) / 3,
        "2mm_pen": (0.386667 + 0.285714 + 0.419118) / 3,
        "5mm_pen": (0.253333 + 0.180952 + 0.220588) / 3,
        "phys_contact": (0.56 + 0.561905 + 0.522059) / 3,
        "phys_pen5mm": (0.438596 + 0.273973 + 0.417476) / 3,
        "leg_pen": (0.0666667 + 0.0 + 0.0) / 3,
        "obj_err": (0.0115301 + 0.00723415 + 0.00738985) / 3,
    },
    (-0.015, 0.05): {
        "5cm": (0.786667 + 0.771429 + 0.580882) / 3,
        "2mm_pen": (0.466667 + 0.361905 + 0.419118) / 3,
        "5mm_pen": (0.253333 + 0.171429 + 0.286765) / 3,
        "phys_contact": (0.56 + 0.561905 + 0.5) / 3,
        "phys_pen5mm": (0.439394 + 0.309524 + 0.59434) / 3,
        "leg_pen": (0.106667 + 0.0 + 0.0220588) / 3,
        "obj_err": (0.0110387 + 0.00746466 + 0.00716656) / 3,
    },
    (-0.015, 0.10): {
        "5cm": (0.786667 + 0.771429 + 0.573529) / 3,
        "2mm_pen": (0.48 + 0.466667 + 0.433824) / 3,
        "5mm_pen": (0.28 + 0.247619 + 0.227941) / 3,
        "phys_contact": (0.613333 + 0.6 + 0.485294) / 3,
        "phys_pen5mm": (0.439394 + 0.348837 + 0.470588) / 3,
        "leg_pen": (0.0533333 + 0.0 + 0.0220588) / 3,
        "obj_err": (0.0106362 + 0.00809384 + 0.007083) / 3,
    },
}

# E154 new metrics from combo_summary (already aggregated)
E154_TRACKING = {
    # (min_sdf, max_viol): (succ_tracked, succ_pen2mm, pz_term_mean, pz_term_worst,
    #                        inmaskC_mean, release_false_mean, gate_valid, fallback)
    (-0.005, 0.05): ("2/3", "3/3", 0.045, 0.101, 0.813, 0.187, 0.700, 0.005),
    (-0.005, 0.10): ("3/3", "3/3", 0.028, 0.048, 0.750, 0.510, 0.762, 0.000),
    (-0.010, 0.05): ("1/3", "2/3", 0.038, 0.081, 0.846, 0.335, 0.853, 0.003),
    (-0.010, 0.10): ("3/3", "3/3", 0.024, 0.035, 0.813, 0.499, 0.895, 0.001),
    (-0.015, 0.05): ("2/3", "2/3", 0.031, 0.049, 0.829, 0.377, 0.930, 0.000),
    (-0.015, 0.10): ("1/3", "2/3", 0.045, 0.105, 0.866, 0.281, 0.933, 0.000),
}


# ══════════════════════════════════════════════════════════════════════════════
wb = openpyxl.Workbook()

# ── Sheet 1: E153 阈值扫描 (新版含 tracking) ─────────────────────────────────
ws = wb.active
ws.title = "E153阈值扫描(E154修正)"

COLS_S1 = [
    "方法",
    "succ_tracked\n(新门控)",
    "succ_pen2mm\n(旧门控)",
    "pz_term\nmean",
    "pz_term\nworst",
    "inmaskC\n(真实mask内\n物理接触↑)",
    "release_false\n(诊断↓)",
    "5cm接触\n(↑)",
    "2mm穿透\n(↓)",
    "5mm穿透\n(↓)",
    "物理接触\n(↑)",
    "物理穿透>5mm\n(↓)",
    "腿穿透\n(↓)",
    "gate有效率",
    "fallback",
]

# Title
merge_cell(ws, 1, 1, 1, len(COLS_S1),
           value="E154 评测修正: E153 阈值扫描 (3-case 均值: box021/box004/box023, 真实 3cm mask + body tracking)",
           bold=True, fg="FFFFFF", bg="2F2F2F", size=12)
ws.row_dimensions[1].height = 28

# Headers
for i, label in enumerate(COLS_S1):
    set_cell(ws, 2, 1 + i, label, bold=True, bg=SUB_BG, size=9)
ws.row_dimensions[2].height = 55

# Baseline/b1 new tracking metrics (from e153_method_metrics.tsv, 3-case means)
BASELINE_TRACK = {
    "pz_term_mean": (0.0242349 + 0.0134119 + 0.00337829) / 3,
    "pz_term_worst": max(0.0242349, 0.0134119, 0.00337829),
    "inmaskC": (0.381818 + 0.532258 + 0.907692) / 3,
    "release_false": (0.0 + 0.0 + 0.0) / 3,
}
B1_TRACK = {
    "pz_term_mean": (0.0337855 + 0.043004 + 0.00356455) / 3,
    "pz_term_worst": max(0.0337855, 0.043004, 0.00356455),
    "inmaskC": (0.727273 + 0.887097 + 0.938462) / 3,
    "release_false": (0.5 + 0.4375 + 0.240741) / 3,
}

# Baseline row
r = 3
set_cell(ws, r, 1, "baseline", bold=True, bg=REF_BG, align=LEFT)
set_cell(ws, r, 2, "3/3", bg=REF_BG, fg=REF_FG)
set_cell(ws, r, 3, "3/3", bg=REF_BG, fg=REF_FG)
set_cell(ws, r, 4, f"{BASELINE_TRACK['pz_term_mean']:.3f}", bg=REF_BG, fg=REF_FG)
set_cell(ws, r, 5, f"{BASELINE_TRACK['pz_term_worst']:.3f}", bg=REF_BG, fg=REF_FG)
set_cell(ws, r, 6, pct(BASELINE_TRACK["inmaskC"]), bg=REF_BG, fg=REF_FG)
set_cell(ws, r, 7, pct(BASELINE_TRACK["release_false"]), bg=REF_BG, fg=REF_FG)
set_cell(ws, r, 8, pct(BASELINE_RAW["5cm"]), bg=REF_BG, fg=REF_FG)
set_cell(ws, r, 9, pct(BASELINE_RAW["2mm_pen"]), bg=REF_BG, fg=REF_FG)
set_cell(ws, r, 10, pct(BASELINE_RAW["5mm_pen"]), bg=REF_BG, fg=REF_FG)
set_cell(ws, r, 11, pct(BASELINE_RAW["phys_contact"]), bg=REF_BG, fg=REF_FG)
set_cell(ws, r, 12, pct(BASELINE_RAW["phys_pen5mm"]), bg=REF_BG, fg=REF_FG)
set_cell(ws, r, 13, pct(BASELINE_RAW["leg_pen"]), bg=REF_BG, fg=REF_FG)
set_cell(ws, r, 14, "—", bg=REF_BG, fg=REF_FG)
set_cell(ws, r, 15, "—", bg=REF_BG, fg=REF_FG)
ws.row_dimensions[r].height = 28

# b1 row
r = 4
set_cell(ws, r, 1, "b1 (hand surface reward)", bold=True, bg=REF_BG, align=LEFT)
set_cell(ws, r, 2, "3/3", bg=REF_BG, fg=REF_FG)
set_cell(ws, r, 3, "3/3", bg=REF_BG, fg=REF_FG)
set_cell(ws, r, 4, f"{B1_TRACK['pz_term_mean']:.3f}", bg=REF_BG, fg=REF_FG)
set_cell(ws, r, 5, f"{B1_TRACK['pz_term_worst']:.3f}", bg=REF_BG, fg=REF_FG)
set_cell(ws, r, 6, pct(B1_TRACK["inmaskC"]), bg=REF_BG, fg=REF_FG)
set_cell(ws, r, 7, pct(B1_TRACK["release_false"]), bg=REF_BG, fg=REF_FG)
set_cell(ws, r, 8, pct(B1_RAW["5cm"]), bg=REF_BG, fg=REF_FG)
set_cell(ws, r, 9, pct(B1_RAW["2mm_pen"]), bg=REF_BG, fg=REF_FG)
set_cell(ws, r, 10, pct(B1_RAW["5mm_pen"]), bg=REF_BG, fg=REF_FG)
set_cell(ws, r, 11, pct(B1_RAW["phys_contact"]), bg=REF_BG, fg=REF_FG)
set_cell(ws, r, 12, pct(B1_RAW["phys_pen5mm"]), bg=REF_BG, fg=REF_FG)
set_cell(ws, r, 13, pct(B1_RAW["leg_pen"]), bg=REF_BG, fg=REF_FG)
set_cell(ws, r, 14, "—", bg=REF_BG, fg=REF_FG)
set_cell(ws, r, 15, "—", bg=REF_BG, fg=REF_FG)
ws.row_dimensions[r].height = 28

# Grid rows
GRID_ORDER = [
    (-0.005, 0.05), (-0.005, 0.10),
    (-0.010, 0.05), (-0.010, 0.10),
    (-0.015, 0.05), (-0.015, 0.10),
]

for gi, (msdf, mviol) in enumerate(GRID_ORDER):
    r = 5 + gi
    raw = E153_GRID_RAW[(msdf, mviol)]
    trk = E154_TRACKING[(msdf, mviol)]
    label = f"gateA+b1\n(sdf={msdf:.3f}, viol={mviol:.2f})"

    # Highlight best config
    is_best = (msdf == -0.010 and mviol == 0.10)
    mbg = METHOD_BG[gi % 2]

    set_cell(ws, r, 1, label, bold=is_best, bg=BEST_BG if is_best else mbg, align=LEFT)

    # succ_tracked (color: 3/3=green, 2/3=yellow, 1/3=red)
    st = trk[0]
    if st == "3/3":
        set_cell(ws, r, 2, st, bold=True, bg=GOOD_BG, fg=GOOD_FG)
    elif st == "2/3":
        set_cell(ws, r, 2, st, bg=REF_BG, fg=REF_FG)
    else:
        set_cell(ws, r, 2, st, bg=BAD_BG, fg=BAD_FG)

    # succ_pen2mm
    sp = trk[1]
    if sp == "3/3":
        set_cell(ws, r, 3, sp, bold=True, bg=GOOD_BG, fg=GOOD_FG)
    elif sp == "2/3":
        set_cell(ws, r, 3, sp, bg=REF_BG, fg=REF_FG)
    else:
        set_cell(ws, r, 3, sp, bg=BAD_BG, fg=BAD_FG)

    # pz_term mean/worst
    pz_m, pz_w = trk[2], trk[3]
    set_cell(ws, r, 4, f"{pz_m:.3f}",
             bg=GOOD_BG if pz_m < 0.04 else (BAD_BG if pz_m > 0.06 else mbg),
             fg=GOOD_FG if pz_m < 0.04 else (BAD_FG if pz_m > 0.06 else "000000"))
    set_cell(ws, r, 5, f"{pz_w:.3f}",
             bg=GOOD_BG if pz_w < 0.08 else BAD_BG,
             fg=GOOD_FG if pz_w < 0.08 else BAD_FG)

    # inmaskC
    imc = trk[4]
    set_cell(ws, r, 6, pct(imc),
             bg=GOOD_BG if imc > 0.82 else mbg,
             fg=GOOD_FG if imc > 0.82 else "000000")

    # release_false (diagnostic, no color coding for pass/fail, just intensity)
    rf = trk[5]
    set_cell(ws, r, 7, pct(rf),
             bg=BAD_BG if rf > 0.4 else mbg,
             fg=BAD_FG if rf > 0.4 else "000000")

    # Standard metrics (vs b1 as reference for coloring)
    metrics_order = ["5cm", "2mm_pen", "5mm_pen", "phys_contact", "phys_pen5mm", "leg_pen"]
    directions = [+1, -1, -1, +1, -1, -1]
    for ci, (mk, direction) in enumerate(zip(metrics_order, directions)):
        col = 8 + ci
        val = raw[mk]
        ref = B1_RAW[mk]
        if direction == +1:
            bg_c = GOOD_BG if val >= ref else (BAD_BG if val < ref * 0.9 else mbg)
            fg_c = GOOD_FG if val >= ref else (BAD_FG if val < ref * 0.9 else "000000")
        else:
            bg_c = GOOD_BG if val <= ref else (BAD_BG if val > ref * 1.1 else mbg)
            fg_c = GOOD_FG if val <= ref else (BAD_FG if val > ref * 1.1 else "000000")
        set_cell(ws, r, col, pct(val), bg=bg_c, fg=fg_c)

    # gate_valid, fallback
    gv, fb = trk[6], trk[7]
    set_cell(ws, r, 14, pct(gv), bg=GOOD_BG if gv > 0.85 else mbg,
             fg=GOOD_FG if gv > 0.85 else "000000")
    set_cell(ws, r, 15, f"{fb:.3f}", bg=mbg)

    ws.row_dimensions[r].height = 36

# Footnote
fn_row = 5 + len(GRID_ORDER)
merge_cell(ws, fn_row, 1, fn_row, len(COLS_S1),
           value="★ (-0.010, 0.10) 为最优配置: tracking 3/3 + gate有效率89.5% + fallback仅0.1%。"
                 "  颜色: 绿=优于b1参考 / 红=显著劣化 / 黄=参考行。"
                 "  pz_term阈值=0.08m(2×b1最差值)。release_false在所有run含b1都高→训练侧bug。",
           bold=False, fg="595959", bg="F7F7F7", size=9)

# Column widths
ws.column_dimensions["A"].width = 26
for c in range(2, len(COLS_S1) + 1):
    ws.column_dimensions[get_column_letter(c)].width = 13
ws.freeze_panes = "B3"

# ══════════════════════════════════════════════════════════════════════════════
# Sheet 2: E152 对比 (含 E154 tracking 修正)
# ══════════════════════════════════════════════════════════════════════════════
ws2 = wb.create_sheet("E152方法对比(E154修正)")

COLS_S2 = [
    "方法",
    "succ_tracked\n(E154新)",
    "succ (0mm|2mm)\n(旧)",
    "pz_term\nmean",
    "pz_term\nworst",
    "inmaskC\n(真实mask内\n物理接触↑)",
    "5cm接触\n(↑)",
    "2mm穿透\n(↓)",
    "5mm穿透\n(↓)",
    "物理接触\n(↑)",
    "物理穿透>5mm\n(↓)",
    "腿穿透\n(↓)",
    "fall",
    "fallback",
    "来源",
]

# E152 tracking metrics from e152_method_metrics.tsv (per-method, 3-case)
E152_TRACK = {
    # method: (pz_term_mean, pz_term_worst, inmaskC_mean)
    "baseline": (
        (0.024234898 + 0.013411949 + 0.0033782865) / 3,
        max(0.024234898, 0.013411949, 0.0033782865),
        (0.38181818 + 0.53225806 + 0.90769231) / 3,
    ),
    "gateA": (
        (0.027555159 + 0.0071307247 + 0.0037406837) / 3,
        max(0.027555159, 0.0071307247, 0.0037406837),
        (0.30909091 + 0.51612903 + 0.8) / 3,
    ),
    "b1": (
        (0.033785491 + 0.043003962 + 0.003564549) / 3,
        max(0.033785491, 0.043003962, 0.003564549),
        (0.72727273 + 0.88709677 + 0.93846154) / 3,
    ),
    "gateA_b1_050": (
        (0.024136984 + 0.046760041 + 0.0035077815) / 3,
        max(0.024136984, 0.046760041, 0.0035077815),
        (0.76363636 + 0.72580645 + 0.92307692) / 3,
    ),
    "gateA_b1_100": (
        # E153 data for (-0.010, 0.10) combo
        (0.0354336 + 0.0313934 + 0.0047944) / 3,
        max(0.0354336, 0.0313934, 0.0047944),
        (0.709091 + 0.790323 + 0.938462) / 3,
    ),
}

# E152 data from e152_method_metrics.tsv (3-case means)
# (label, 5cm, 2mm, 5mm, phys_c, phys_p5mm, leg, succ_old, succ_tracked, fall, fallback, source, track_key)
E152_ROWS = [
    ("baseline",
     (0.733333 + 0.619048 + 0.492647) / 3,
     (0.2 + 0.314286 + 0.367647) / 3,
     (0.16 + 0.161905 + 0.294118) / 3,
     (0.28 + 0.333333 + 0.433824) / 3,
     (0.633333 + 0.326923 + 0.698925) / 3,
     (0.08 + 0.047619 + 0.007353) / 3,
     "ref", "3/3", "0/3", "0.000", "reuse E148", "baseline"),

    ("gateA\n(sdf=-0.010, viol=0.050)",
     (0.666667 + 0.6 + 0.485294) / 3,
     (0.213333 + 0.247619 + 0.345588) / 3,
     (0.146667 + 0.104762 + 0.183824) / 3,
     (0.226667 + 0.323810 + 0.382353) / 3,
     (0.392857 + 0.256410 + 0.390805) / 3,
     (0.093333 + 0.0 + 0.014706) / 3,
     "0/3 | 0/3", "3/3", "0/3", "0.052", "E152 新跑", "gateA"),

    ("b1 (hand surface reward)",
     (0.786667 + 0.714286 + 0.580882) / 3,
     (0.4 + 0.542857 + 0.470588) / 3,
     (0.226667 + 0.304762 + 0.286765) / 3,
     (0.56 + 0.590476 + 0.544118) / 3,
     (0.583333 + 0.465909 + 0.598214) / 3,
     (0.093333 + 0.019048 + 0.0) / 3,
     "ref", "3/3", "0/3", "0.000", "reuse E151", "b1"),

    ("gateA+b1\n(sdf=-0.010, viol=0.050)",
     (0.786667 + 0.790476 + 0.580882) / 3,
     (0.36 + 0.457143 + 0.382353) / 3,
     (0.186667 + 0.276190 + 0.191176) / 3,
     (0.586667 + 0.542857 + 0.470588) / 3,
     (0.285714 + 0.453333 + 0.405941) / 3,
     (0.066667 + 0.0 + 0.0) / 3,
     "2/3 | 3/3", "3/3", "0/3", "0.032", "E152 新跑", "gateA_b1_050"),

    ("gateA+b1 ★\n(sdf=-0.010, viol=0.100)\n[E153最优]",
     (0.786667 + 0.714286 + 0.580882) / 3,
     (0.386667 + 0.285714 + 0.419118) / 3,
     (0.253333 + 0.180952 + 0.220588) / 3,
     (0.56 + 0.561905 + 0.522059) / 3,
     (0.438596 + 0.273973 + 0.417476) / 3,
     (0.066667 + 0.0 + 0.0) / 3,
     "3/3 | 3/3", "3/3", "0/3", "0.001", "E153 新跑", "gateA_b1_100"),
]

# Title
merge_cell(ws2, 1, 1, 1, len(COLS_S2),
           value="E154 评测修正: E152/E153 方法对比 (3-case 均值, 含 body tracking + 真实mask指标)",
           bold=True, fg="FFFFFF", bg="2F2F2F", size=12)
ws2.row_dimensions[1].height = 28

# Headers
for i, label in enumerate(COLS_S2):
    set_cell(ws2, 2, 1 + i, label, bold=True, bg=SUB_BG, size=9)
ws2.row_dimensions[2].height = 55

# Baseline reference for color coding
S2_BASELINE_REF = {
    "5cm": E152_ROWS[0][1],
    "2mm": E152_ROWS[0][2],
    "5mm": E152_ROWS[0][3],
    "phys_c": E152_ROWS[0][4],
    "phys_p5mm": E152_ROWS[0][5],
    "leg": E152_ROWS[0][6],
}

for ri, row_data in enumerate(E152_ROWS):
    r = 3 + ri
    (label, v5cm, v2mm, v5mm, vphys_c, vphys_p5mm, vleg,
     succ_old, succ_tracked, fall, fallback, source, track_key) = row_data

    mbg = METHOD_BG[ri % 2]
    is_ref = (succ_old == "ref")
    is_best = ("★" in label)

    set_cell(ws2, r, 1, label, bold=True,
             bg=BEST_BG if is_best else (REF_BG if is_ref else mbg), align=LEFT)

    # succ_tracked
    if succ_tracked == "3/3":
        set_cell(ws2, r, 2, succ_tracked, bold=True,
                 bg=REF_BG if is_ref else GOOD_BG,
                 fg=REF_FG if is_ref else GOOD_FG)
    elif succ_tracked == "0/3":
        set_cell(ws2, r, 2, succ_tracked, bg=BAD_BG, fg=BAD_FG)
    else:
        set_cell(ws2, r, 2, succ_tracked, bg=mbg)

    # succ old
    if succ_old == "ref":
        set_cell(ws2, r, 3, succ_old, bg=REF_BG, fg=REF_FG)
    else:
        set_cell(ws2, r, 3, succ_old, bg=mbg)

    # pz_term mean / worst / inmaskC (from E152_TRACK)
    trk = E152_TRACK[track_key]
    pz_m, pz_w, imc = trk

    if is_ref:
        set_cell(ws2, r, 4, f"{pz_m:.3f}", bg=REF_BG, fg=REF_FG)
        set_cell(ws2, r, 5, f"{pz_w:.3f}", bg=REF_BG, fg=REF_FG)
        set_cell(ws2, r, 6, pct(imc), bg=REF_BG, fg=REF_FG)
    else:
        set_cell(ws2, r, 4, f"{pz_m:.3f}",
                 bg=GOOD_BG if pz_m < 0.04 else (BAD_BG if pz_m > 0.06 else mbg),
                 fg=GOOD_FG if pz_m < 0.04 else (BAD_FG if pz_m > 0.06 else "000000"))
        set_cell(ws2, r, 5, f"{pz_w:.3f}",
                 bg=GOOD_BG if pz_w < 0.08 else BAD_BG,
                 fg=GOOD_FG if pz_w < 0.08 else BAD_FG)
        set_cell(ws2, r, 6, pct(imc),
                 bg=GOOD_BG if imc > 0.7 else (BAD_BG if imc < 0.5 else mbg),
                 fg=GOOD_FG if imc > 0.7 else (BAD_FG if imc < 0.5 else "000000"))

    # Numeric metrics (cols 7-12)
    vals = [v5cm, v2mm, v5mm, vphys_c, vphys_p5mm, vleg]
    refs = [S2_BASELINE_REF["5cm"], S2_BASELINE_REF["2mm"], S2_BASELINE_REF["5mm"],
            S2_BASELINE_REF["phys_c"], S2_BASELINE_REF["phys_p5mm"], S2_BASELINE_REF["leg"]]
    directions = [+1, -1, -1, +1, -1, -1]

    for ci, (val, ref, direction) in enumerate(zip(vals, refs, directions)):
        col = 7 + ci
        if is_ref:
            set_cell(ws2, r, col, pct(val), bg=REF_BG, fg=REF_FG)
        else:
            if direction == +1:
                bg_c = GOOD_BG if val >= ref else (BAD_BG if val < ref * 0.85 else mbg)
                fg_c = GOOD_FG if val >= ref else (BAD_FG if val < ref * 0.85 else "000000")
            else:
                bg_c = GOOD_BG if val <= ref else (BAD_BG if val > ref * 1.15 else mbg)
                fg_c = GOOD_FG if val <= ref else (BAD_FG if val > ref * 1.15 else "000000")
            set_cell(ws2, r, col, pct(val), bg=bg_c, fg=fg_c)

    # fall, fallback, source
    set_cell(ws2, r, 13, fall, bg=REF_BG if is_ref else mbg,
             fg=REF_FG if is_ref else "000000")
    set_cell(ws2, r, 14, fallback, bg=REF_BG if is_ref else mbg,
             fg=REF_FG if is_ref else "000000")
    set_cell(ws2, r, 15, source, bg=mbg, size=9, italic=True)

    ws2.row_dimensions[r].height = 44

# Column widths
ws2.column_dimensions["A"].width = 28
for c in range(2, len(COLS_S2) + 1):
    ws2.column_dimensions[get_column_letter(c)].width = 14
ws2.freeze_panes = "B3"

# ══════════════════════════════════════════════════════════════════════════════
# Sheet 3: Per-case 明细 (E153 18行)
# ══════════════════════════════════════════════════════════════════════════════
ws3 = wb.create_sheet("per-case明细")

COLS_S3 = [
    "case", "combo\n(sdf_viol)", "tracked", "pz_term", "inmaskC",
    "release_false", "5cm", "2mm_pen", "5mm_pen", "phys_contact",
    "phys_pen>5mm", "leg_pen", "fail原因",
]

# Title
merge_cell(ws3, 1, 1, 1, len(COLS_S3),
           value="E153 per-case 明细 (E154 tracking 门控, 阈值 pz_term < 0.08m)",
           bold=True, fg="FFFFFF", bg="2F2F2F", size=12)
ws3.row_dimensions[1].height = 26

for i, label in enumerate(COLS_S3):
    set_cell(ws3, 2, 1 + i, label, bold=True, bg=SUB_BG, size=9)
ws3.row_dimensions[2].height = 40

# Per-case data: (case, combo, tracked, pz_term, inmaskC, release_false,
#                 5cm, 2mm_pen, 5mm_pen, phys_c, phys_p5mm, leg_pen, fail_reason)
PERCASE = [
    ("box021", "sdf005_v05", True, 0.032, 0.691, 0.000, 0.787, 0.293, 0.027, 0.507, 0.216, 0.067, "—"),
    ("box021", "sdf005_v10", True, 0.031, 0.600, 0.750, 0.787, 0.213, 0.040, 0.480, 0.208, 0.053, "—"),
    ("box021", "sdf010_v05", False, 0.028, 0.727, 0.250, 0.787, 0.467, 0.253, 0.547, 0.458, 0.053, "pen2mm(起身OK)"),
    ("box021", "sdf010_v10", True, 0.035, 0.709, 0.750, 0.787, 0.387, 0.253, 0.560, 0.439, 0.067, "—"),
    ("box021", "sdf015_v05", False, 0.040, 0.727, 0.500, 0.787, 0.467, 0.253, 0.560, 0.439, 0.107, "pen2mm(起身OK)"),
    ("box021", "sdf015_v10", False, 0.030, 0.836, 0.000, 0.787, 0.480, 0.280, 0.613, 0.439, 0.053, "pen2mm(起身OK)"),
    ("box004", "sdf005_v05", False, 0.101, 0.871, 0.375, 0.743, 0.305, 0.067, 0.571, 0.139, 0.010, "弯腰(tracking)"),
    ("box004", "sdf005_v10", True, 0.048, 0.758, 0.688, 0.771, 0.381, 0.067, 0.562, 0.093, 0.048, "—"),
    ("box004", "sdf010_v05", False, 0.081, 0.871, 0.625, 0.771, 0.400, 0.152, 0.610, 0.193, 0.000, "弯腰(tracking)"),
    ("box004", "sdf010_v10", True, 0.031, 0.790, 0.562, 0.714, 0.286, 0.181, 0.562, 0.274, 0.000, "—"),
    ("box004", "sdf015_v05", True, 0.049, 0.823, 0.500, 0.771, 0.362, 0.171, 0.562, 0.310, 0.000, "—"),
    ("box004", "sdf015_v10", False, 0.105, 0.823, 0.750, 0.771, 0.467, 0.248, 0.600, 0.349, 0.000, "弯腰(tracking)"),
    ("box023", "sdf005_v05", True, 0.003, 0.877, 0.185, 0.588, 0.279, 0.022, 0.493, 0.132, 0.000, "—"),
    ("box023", "sdf005_v10", True, 0.004, 0.892, 0.093, 0.581, 0.243, 0.022, 0.463, 0.121, 0.000, "—"),
    ("box023", "sdf010_v05", True, 0.004, 0.938, 0.130, 0.581, 0.434, 0.228, 0.500, 0.413, 0.000, "—"),
    ("box023", "sdf010_v10", True, 0.005, 0.938, 0.185, 0.581, 0.419, 0.221, 0.522, 0.417, 0.000, "—"),
    ("box023", "sdf015_v05", True, 0.003, 0.938, 0.130, 0.581, 0.419, 0.287, 0.500, 0.594, 0.022, "—"),
    ("box023", "sdf015_v10", True, 0.002, 0.938, 0.093, 0.574, 0.434, 0.228, 0.485, 0.471, 0.022, "—"),
]

for ri, pc in enumerate(PERCASE):
    r = 3 + ri
    (case, combo, tracked, pz_t, imc, rf, v5cm, v2mm, v5mm, vpc, vpp, vleg, reason) = pc
    mbg = METHOD_BG[ri % 2]

    set_cell(ws3, r, 1, case, bold=True, bg=mbg, align=LEFT)
    set_cell(ws3, r, 2, combo, bg=mbg, size=9)

    # tracked
    if tracked:
        set_cell(ws3, r, 3, "✅", bg=GOOD_BG, fg=GOOD_FG)
    else:
        set_cell(ws3, r, 3, "❌", bg=BAD_BG, fg=BAD_FG)

    # pz_term
    set_cell(ws3, r, 4, f"{pz_t:.3f}",
             bg=BAD_BG if pz_t >= 0.08 else (GOOD_BG if pz_t < 0.04 else mbg),
             fg=BAD_FG if pz_t >= 0.08 else (GOOD_FG if pz_t < 0.04 else "000000"))

    # inmaskC
    set_cell(ws3, r, 5, pct(imc), bg=GOOD_BG if imc > 0.85 else mbg,
             fg=GOOD_FG if imc > 0.85 else "000000")

    # release_false
    set_cell(ws3, r, 6, pct(rf), bg=BAD_BG if rf > 0.5 else mbg,
             fg=BAD_FG if rf > 0.5 else "000000")

    # Standard metrics
    for ci, val in enumerate([v5cm, v2mm, v5mm, vpc, vpp, vleg]):
        set_cell(ws3, r, 7 + ci, pct(val), bg=mbg)

    # fail reason
    is_fail = (reason != "—")
    set_cell(ws3, r, 13, reason, bg=BAD_BG if is_fail else mbg,
             fg=BAD_FG if is_fail else "000000", size=9)

    ws3.row_dimensions[r].height = 22

# Column widths
ws3.column_dimensions["A"].width = 10
ws3.column_dimensions["B"].width = 14
for c in range(3, len(COLS_S3) + 1):
    ws3.column_dimensions[get_column_letter(c)].width = 12
ws3.column_dimensions[get_column_letter(13)].width = 18
ws3.freeze_panes = "C3"

# ══════════════════════════════════════════════════════════════════════════════
# Sheet 4: 说明
# ══════════════════════════════════════════════════════════════════════════════
ws4 = wb.create_sheet("说明")
notes = [
    ("指标", "来源", "说明"),
    ("", "", ""),
    ("—— E154 新增指标 ——", "", ""),
    ("succ_tracked", "track_pelvis_z_err_terminal_m < 0.08",
     "body tracking门控: 结尾帧pelvis高度误差<8cm(≈2×b1最差值), 识别弯腰不起身"),
    ("pz_term", "track_pelvis_z_err_terminal_m",
     "结尾15%帧的pelvis高度跟踪误差(对kinematic reference的qpos), 单位m"),
    ("inmaskC", "hand_object_physics_contact_in_mask_frac",
     "真实3cm contact mask窗口内的物理接触比例(↑更好)"),
    ("release_false", "hand_object_release_false_contact_frac",
     "放手窗口内仍保持物理接触的帧比例(诊断用, 不进门控; 在所有run含b1都高→训练侧all-1 mask bug)"),
    ("", "", ""),
    ("—— 沿用指标 ——", "", ""),
    ("5cm接触", "hand_geom_near_5cm_frac", "手部几何体距物体表面≤5cm的帧比例 (↑更好)"),
    ("2mm穿透", "hand_geom_penetration_2mm_frac", "手部几何体与物体重叠≥2mm的帧比例 (↓更好)"),
    ("5mm穿透", "hand_geom_penetration_5mm_frac", "手部几何体与物体重叠≥5mm的帧比例 (↓更好)"),
    ("物理接触", "hand_object_physics_contact_frac", "手与物体存在MuJoCo物理接触的帧比例 (↑更好)"),
    ("物理穿透>5mm", "hand_object_con_dist_frac_lt_neg5mm", "物理接触距离<-5mm的帧比例 (↓更好)"),
    ("腿穿透", "leg_penetration_frac", "腿部与物体穿透的帧比例 (↓更好)"),
    ("gate有效率", "hand_gate_valid_frac", "CEM sample中通过gate约束的比例(越高=gate越容易满足)"),
    ("fallback", "gate_fallback_used", "本轮CEM所有sample均违反gate时退化到无gate采样的帧比例"),
    ("", "", ""),
    ("—— 颜色编码 ——", "", ""),
    ("绿色", "", "改善(高于参考或低于参考, 取决于方向)"),
    ("红色", "", "劣化(低于参考或高于参考)"),
    ("黄色", "", "参考行(baseline/b1), 不做颜色对比"),
    ("亮绿底", "", "推荐最优配置"),
    ("", "", ""),
    ("—— 关键结论 ——", "", ""),
    ("1", "", "success_tracked取代旧全序列pen2mm门控; 3个box004 combo因弯腰被正确判fail(旧门控全pass)"),
    ("2", "", "(-0.010,0.10)存活3/3; (-0.005,0.05)降级为2/3"),
    ("3", "", "release_false普遍高(0.19-0.51)→训练侧all-1 mask使机器人从未被激励放手"),
    ("4", "", "真正可区分行为的指标是body tracking, 不是接触绝对值"),
    ("5", "", "下一步: 用真实3cm mask替换训练侧contact target重训"),
    ("", "", ""),
    ("—— 数据来源 ——", "", ""),
    ("baseline", "reuse E148", "3 cases: box021_029_p2, box004_083_p2, box023_person2"),
    ("b1", "reuse E151", "hand surface contact reward (b1_mesh variant)"),
    ("gateA+b1 (E152)", "E152 新跑", "gate(min_sdf=-0.010,max_viol=0.050) + b1 reward"),
    ("gateA+b1 (E153)", "E153 阈值扫描", "6个(min_sdf×max_viol)配置 + b1 reward"),
    ("E154 修正", "本实验", "纯评测修订: 真实3cm mask + body tracking, 无重训"),
]

for i, (a, b, c) in enumerate(notes, 1):
    is_section = a.startswith("——")
    ws4.cell(row=i, column=1, value=a).font = Font(
        bold=(i == 1 or is_section), size=10,
        color="FF1F4E79" if is_section else "FF000000")
    ws4.cell(row=i, column=2, value=b).font = Font(size=9)
    cl = ws4.cell(row=i, column=3, value=c)
    cl.alignment = Alignment(wrap_text=True)
    cl.font = Font(size=9)

ws4.column_dimensions["A"].width = 22
ws4.column_dimensions["B"].width = 38
ws4.column_dimensions["C"].width = 65

# ── Save ──────────────────────────────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
wb.save(OUT)
print(f"saved → {OUT}")
