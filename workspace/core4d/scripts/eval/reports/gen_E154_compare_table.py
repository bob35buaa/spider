"""
Generate the E154 masked/tracking comparison workbook from local result TSVs.

Output:
  workspace/core4d/results/E154/e154_masked_tracking_compare.xlsx

The workbook intentionally reads metrics from local experiment artifacts instead
of embedding hand-copied numbers in this script.
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys
from copy import copy
from collections import defaultdict
from dataclasses import dataclass

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


CORE4D = pathlib.Path(__file__).parents[3]
sys.path.insert(0, str(CORE4D / "scripts"))
from eval.core.core_metrics import (  # noqa: E402
    EVAL_METRIC_STANDARD_ID,
    STANDARD_TABLE_METRIC_DIRECTIONS,
    STANDARD_TABLE_METRIC_FIELDS,
    STANDARD_TABLE_METRIC_ORDER,
)

RESULTS = CORE4D / "results"
OUT = RESULTS / "E154" / "e154_masked_tracking_compare.xlsx"

DEFAULT_E152_METHOD_METRICS = (
    RESULTS
    / "E152/axis1_hand_object_physics_gate/eval/full/e152_method_metrics.tsv"
)
DEFAULT_E153_METHOD_METRICS = (
    RESULTS / "E153/gate_threshold_sweep/eval/full/e153_method_metrics.tsv"
)
DEFAULT_E153_GRID_DELTA = (
    RESULTS / "E153/gate_threshold_sweep/eval/full/e153_grid_delta_vs_b1.tsv"
)
DEFAULT_OMNIRETARGET_METRICS = (
    RESULTS / "E154/omniretarget_eval/e154_omniretarget_method_metrics.tsv"
)

CASE_ORDER = ["box021_029_p2", "box004_083_p2", "box023_person2"]
CASE_ALIASES = {
    "box021_029_p2": "box021_029_p2",
    "d003_box021_20231018_029_p2": "box021_029_p2",
    "box004_083_p2": "box004_083_p2",
    "e091_box004_20231003_2_083_p2": "box004_083_p2",
    "box023_person2": "box023_person2",
}

METRIC_FIELDS = dict(STANDARD_TABLE_METRIC_FIELDS)

GRID_RUN_FIELDS = {
    "5cm": "hand_geom_near_5cm_frac_run",
    "2mm_pen": "hand_geom_penetration_2mm_frac_run",
    "5mm_pen": "hand_geom_penetration_5mm_frac_run",
    "phys_contact3": "hand_object_physics_contact_3mm_frac_run",
    "phys_pen3": "hand_object_physics_penetration_3mm_frame_frac_run",
    "phys_contact5": "hand_object_physics_contact_5mm_frac_run",
    "phys_pen5": "hand_object_physics_penetration_5mm_frame_frac_run",
    "leg_pen": "leg_penetration_frac_run",
    "obj_err": "obj_err_mean_m_run",
}

METRIC_ORDER = list(STANDARD_TABLE_METRIC_ORDER)
METRIC_DIRECTIONS = list(STANDARD_TABLE_METRIC_DIRECTIONS)
METRIC_START_COL = 10
TAIL_START_COL = METRIC_START_COL + len(METRIC_ORDER)


@dataclass
class Summary:
    label: str
    source: str
    n_cases: int
    metrics: dict[str, float | None]
    pz_term_mean: float | None = None
    pz_term_worst: float | None = None
    inmaskC: float | None = None
    inmaskC5: float | None = None
    release_false: float | None = None
    release_false5: float | None = None
    success_tracked: str | None = None
    success_pen2mm: str | None = None
    fall: str | None = None
    fallback: float | None = None
    gate_valid: float | None = None


# Style helpers
THIN = Side(style="thin", color="FF000000")
MEDIUM = Side(style="medium", color="FF000000")
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT = Alignment(horizontal="left", vertical="center", wrap_text=True)


def fill(hex6: str) -> PatternFill:
    return PatternFill("solid", fgColor="FF" + hex6)


def set_cell(
    ws,
    row: int,
    col: int,
    value="",
    bold: bool = False,
    fg: str = "000000",
    bg: str | None = None,
    align=CENTER,
    size: int = 10,
    italic: bool = False,
):
    c = ws.cell(row=row, column=col, value=value)
    c.font = Font(bold=bold, color="FF" + fg, size=size, italic=italic)
    c.alignment = align
    if bg:
        c.fill = fill(bg)
    c.border = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
    return c


def merge_cell(
    ws,
    r1: int,
    c1: int,
    r2: int,
    c2: int,
    value="",
    bold: bool = False,
    fg: str = "000000",
    bg: str | None = None,
    size: int = 11,
):
    ws.merge_cells(start_row=r1, start_column=c1, end_row=r2, end_column=c2)
    c = ws.cell(row=r1, column=c1, value=value)
    c.font = Font(bold=bold, color="FF" + fg, size=size)
    c.alignment = CENTER
    if bg:
        c.fill = fill(bg)
    c.border = Border(left=MEDIUM, right=MEDIUM, top=MEDIUM, bottom=MEDIUM)
    return c


def pct(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:.1%}" if abs(v) < 10 else f"{v:.3f}"


def fixed(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def read_tsv(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() in {"nan", "none"}:
        return None
    try:
        v = float(value)
    except ValueError:
        return None
    if math.isnan(v):
        return None
    return v


def parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    return None


def mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def canonical_case_id(case_id: str) -> str | None:
    return CASE_ALIASES.get(case_id)


def select_cases(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = []
    for row in rows:
        case = canonical_case_id(row.get("short_case_id", "") or row.get("case_id", ""))
        if case in CASE_ORDER:
            row = dict(row)
            row["_case"] = case
            selected.append(row)
    return selected


def count_str(count: int | None, total: int) -> str | None:
    if count is None:
        return None
    return f"{count}/{total}"


def summarize_metric_rows(
    rows: list[dict[str, str]],
    label: str,
    source: str,
    metric_fields: dict[str, str] = METRIC_FIELDS,
) -> Summary:
    metrics = {k: mean([parse_float(r.get(field)) for r in rows]) for k, field in metric_fields.items()}
    pz_values = [parse_float(r.get("track_pelvis_z_err_terminal_m")) for r in rows]
    inmask_values = [parse_float(r.get("hand_object_physics_contact_3mm_in_mask_frac")) for r in rows]
    inmask5_values = [parse_float(r.get("hand_object_physics_contact_5mm_in_mask_frac")) for r in rows]
    release_values = [parse_float(r.get("hand_object_release_false_contact_3mm_frac")) for r in rows]
    release5_values = [parse_float(r.get("hand_object_release_false_contact_5mm_frac")) for r in rows]
    fallback_values = [
        parse_float(r.get("gate_fallback_used"))
        for r in rows
        if parse_float(r.get("gate_fallback_used")) is not None
    ]
    gate_values = [
        parse_float(r.get("hand_gate_valid_frac") or r.get("cem_hand_gate_valid_frac_mean"))
        for r in rows
        if parse_float(r.get("hand_gate_valid_frac") or r.get("cem_hand_gate_valid_frac_mean")) is not None
    ]
    tracked = [v for v in pz_values if v is not None]
    fall_count = sum(1 for r in rows if parse_bool(r.get("fall_flag")) is True)
    return Summary(
        label=label,
        source=source,
        n_cases=len(rows),
        metrics=metrics,
        pz_term_mean=mean(pz_values),
        pz_term_worst=max(tracked) if tracked else None,
        inmaskC=mean(inmask_values),
        inmaskC5=mean(inmask5_values),
        release_false=mean(release_values),
        release_false5=mean(release5_values),
        success_tracked=count_str(sum(1 for v in tracked if v < 0.08), len(rows)) if tracked else None,
        success_pen2mm=None,
        fall=count_str(fall_count, len(rows)),
        fallback=mean(fallback_values),
        gate_valid=mean(gate_values),
    )


def summarize_grid_rows(rows: list[dict[str, str]], label: str, source: str) -> Summary:
    metrics = {k: mean([parse_float(r.get(field)) for r in rows]) for k, field in GRID_RUN_FIELDS.items()}
    pz_values = [parse_float(r.get("track_pelvis_z_err_terminal_m")) for r in rows]
    tracked_flags = [parse_bool(r.get("success_tracked")) for r in rows]
    pen2_flags = [parse_bool(r.get("success_pen2mm")) for r in rows]
    return Summary(
        label=label,
        source=source,
        n_cases=len(rows),
        metrics=metrics,
        pz_term_mean=mean(pz_values),
        pz_term_worst=max([v for v in pz_values if v is not None], default=None),
        inmaskC=mean([parse_float(r.get("hand_object_physics_contact_3mm_in_mask_frac")) for r in rows]),
        inmaskC5=mean([parse_float(r.get("hand_object_physics_contact_5mm_in_mask_frac")) for r in rows]),
        release_false=mean([parse_float(r.get("hand_object_release_false_contact_3mm_frac")) for r in rows]),
        release_false5=mean([parse_float(r.get("hand_object_release_false_contact_5mm_frac")) for r in rows]),
        success_tracked=count_str(sum(1 for v in tracked_flags if v is True), len(rows)),
        success_pen2mm=count_str(sum(1 for v in pen2_flags if v is True), len(rows)),
        fall=count_str(sum(1 for r in rows if parse_bool(r.get("fall_flag")) is True), len(rows)),
        fallback=mean([parse_float(r.get("gate_fallback_used")) for r in rows]),
        gate_valid=mean([parse_float(r.get("hand_gate_valid_frac")) for r in rows]),
    )


def load_inputs(args) -> tuple[dict[str, Summary], dict[tuple[float, float], Summary], list[dict[str, str]]]:
    e152 = select_cases(read_tsv(args.e152_method_metrics))
    e153 = select_cases(read_tsv(args.e153_method_metrics))
    e153_grid = select_cases(read_tsv(args.e153_grid_delta))
    omni = select_cases(read_tsv(args.omniretarget_metrics))
    omni = [r for r in omni if is_omniretarget_row(r)]

    summaries: dict[str, Summary] = {}
    summaries["omniretarget"] = summarize_metric_rows(
        omni,
        label="OmniRetarget",
        source=rel(args.omniretarget_metrics),
        metric_fields={
            "5cm": "hand_geom_near_5cm_frac",
            "2mm_pen": "hand_geom_penetration_2mm_frac",
            "5mm_pen": "hand_geom_penetration_5mm_frac",
            "phys_contact3": "hand_object_physics_contact_3mm_frac",
            "phys_pen3": "hand_object_physics_penetration_3mm_frame_frac",
            "phys_contact5": "hand_object_physics_contact_5mm_frac",
            "phys_pen5": "hand_object_physics_penetration_5mm_frame_frac",
            "leg_pen": "leg_penetration_frac",
            "obj_err": "obj_err_mean_m",
        },
    )

    for method, label in [
        ("baseline", "spider-rubberhand"),
        ("b1", "b1 (hand surface reward)"),
    ]:
        rows = [r for r in e153 if r.get("method") == method]
        summaries[f"e153_{method}"] = summarize_metric_rows(rows, label, rel(args.e153_method_metrics))

    for method, label in [
        ("baseline", "spider-rubberhand"),
        ("gateA", "gateA\n(sdf=-0.010, viol=0.050)"),
        ("b1", "b1 (hand surface reward)"),
        ("gateA_b1", "gateA+b1\n(sdf=-0.010, viol=0.050)"),
    ]:
        rows = [r for r in e152 if r.get("method") == method]
        summaries[f"e152_{method}"] = summarize_metric_rows(rows, label, rel(args.e152_method_metrics))

    grid: dict[tuple[float, float], Summary] = {}
    grouped: dict[tuple[float, float], list[dict[str, str]]] = defaultdict(list)
    for row in e153_grid:
        msdf = parse_float(row.get("min_sdf_m"))
        mviol = parse_float(row.get("max_viol"))
        if msdf is not None and mviol is not None:
            grouped[(round(msdf, 3), round(mviol, 2))].append(row)
    for key, rows in grouped.items():
        msdf, mviol = key
        label = f"gateA+b1\n(sdf={msdf:.3f}, viol={mviol:.2f})"
        grid[key] = summarize_grid_rows(rows, label, rel(args.e153_grid_delta))

    return summaries, grid, e153_grid


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(CORE4D.parents[1]))
    except ValueError:
        return str(path)


def is_omniretarget_row(row: dict[str, str]) -> bool:
    return (
        row.get("method_key") == "OmniRetarget"
        or row.get("hand_collision_variant_id") == "OmniRetarget"
        or row.get("method") == "OmniRetarget"
        or row.get("method") == "OmniRetarget kinematic replay"
    )


def success_fill(value: str | None, bg_default: str) -> tuple[str, str, str, bool]:
    if value is None:
        return "-", bg_default, "000000", False
    if value.startswith("3/"):
        return value, GOOD_BG, GOOD_FG, True
    if value.startswith("2/"):
        return value, REF_BG, REF_FG, False
    if value.startswith("1/") or value.startswith("0/"):
        return value, BAD_BG, BAD_FG, False
    return value, bg_default, "000000", False


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


def write_summary_row(
    ws,
    r: int,
    summary: Summary,
    ref: Summary,
    mbg: str,
    is_ref: bool = False,
    is_best: bool = False,
    include_gate: bool = True,
    source_col: int | None = None,
):
    set_cell(ws, r, 1, summary.label, bold=True, bg=BEST_BG if is_best else (REF_BG if is_ref else mbg), align=LEFT)
    st, bg, fg, bold = success_fill(summary.success_tracked, REF_BG if is_ref else mbg)
    set_cell(ws, r, 2, st, bold=bold, bg=REF_BG if is_ref else bg, fg=REF_FG if is_ref else fg)
    sp, bg, fg, bold = success_fill(summary.success_pen2mm, REF_BG if is_ref else mbg)
    set_cell(ws, r, 3, sp, bold=bold, bg=REF_BG if is_ref else bg, fg=REF_FG if is_ref else fg)

    if is_ref:
        metric_bg = REF_BG
        metric_fg = REF_FG
        set_cell(ws, r, 4, fixed(summary.pz_term_mean), bg=metric_bg, fg=metric_fg)
        set_cell(ws, r, 5, fixed(summary.pz_term_worst), bg=metric_bg, fg=metric_fg)
        set_cell(ws, r, 6, pct(summary.inmaskC), bg=metric_bg, fg=metric_fg)
        set_cell(ws, r, 7, pct(summary.inmaskC5), bg=metric_bg, fg=metric_fg)
        set_cell(ws, r, 8, pct(summary.release_false), bg=metric_bg, fg=metric_fg)
        set_cell(ws, r, 9, pct(summary.release_false5), bg=metric_bg, fg=metric_fg)
    else:
        set_cell(ws, r, 4, fixed(summary.pz_term_mean), bg=value_bg(summary.pz_term_mean, low_good=True, good=0.04, bad=0.06, default=mbg))
        set_cell(ws, r, 5, fixed(summary.pz_term_worst), bg=value_bg(summary.pz_term_worst, low_good=True, good=0.08, bad=0.08, default=mbg))
        set_cell(ws, r, 6, pct(summary.inmaskC), bg=value_bg(summary.inmaskC, low_good=False, good=0.82, bad=0.5, default=mbg))
        set_cell(ws, r, 7, pct(summary.inmaskC5), bg=value_bg(summary.inmaskC5, low_good=False, good=0.82, bad=0.5, default=mbg))
        set_cell(ws, r, 8, pct(summary.release_false), bg=value_bg(summary.release_false, low_good=True, good=0.25, bad=0.4, default=mbg))
        set_cell(ws, r, 9, pct(summary.release_false5), bg=value_bg(summary.release_false5, low_good=True, good=0.25, bad=0.4, default=mbg))

    for ci, (key, direction) in enumerate(zip(METRIC_ORDER, METRIC_DIRECTIONS)):
        val = summary.metrics.get(key)
        ref_val = ref.metrics.get(key)
        col = METRIC_START_COL + ci
        if is_ref:
            set_cell(ws, r, col, pct(val), bg=REF_BG, fg=REF_FG)
        else:
            bg_c, fg_c = compare_colors(val, ref_val, direction, mbg)
            set_cell(ws, r, col, pct(val), bg=bg_c, fg=fg_c)

    if include_gate:
        set_cell(ws, r, TAIL_START_COL, pct(summary.gate_valid), bg=value_bg(summary.gate_valid, low_good=False, good=0.85, bad=0.5, default=REF_BG if is_ref else mbg), fg=REF_FG if is_ref else "000000")
        set_cell(ws, r, TAIL_START_COL + 1, fixed(summary.fallback), bg=REF_BG if is_ref else mbg, fg=REF_FG if is_ref else "000000")
    else:
        set_cell(ws, r, TAIL_START_COL, summary.fall or "-", bg=REF_BG if is_ref else mbg, fg=REF_FG if is_ref else "000000")
        set_cell(ws, r, TAIL_START_COL + 1, fixed(summary.fallback), bg=REF_BG if is_ref else mbg, fg=REF_FG if is_ref else "000000")
        if source_col:
            set_cell(ws, r, source_col, summary.source, bg=mbg, size=9, italic=True)


def value_bg(v: float | None, low_good: bool, good: float, bad: float, default: str) -> str:
    if v is None:
        return default
    if low_good:
        if v <= good:
            return GOOD_BG
        if v >= bad:
            return BAD_BG
    else:
        if v >= good:
            return GOOD_BG
        if v <= bad:
            return BAD_BG
    return default


def compare_colors(val: float | None, ref: float | None, direction: int, default: str) -> tuple[str, str]:
    if val is None or ref is None:
        return default, "000000"
    if direction == +1:
        if val >= ref:
            return GOOD_BG, GOOD_FG
        if val < ref * 0.85:
            return BAD_BG, BAD_FG
    else:
        if val <= ref:
            return GOOD_BG, GOOD_FG
        if ref == 0 or val > ref * 1.15:
            return BAD_BG, BAD_FG
    return default, "000000"


def ratio_count(value: str | None) -> float | None:
    if not value or "/" not in value:
        return None
    num, den = str(value).split("/", 1)
    try:
        den_v = float(den)
        return float(num) / den_v if den_v else None
    except ValueError:
        return None


def rankable_values(summary: Summary, *, include_gate: bool) -> list[tuple[int, float | None, int]]:
    values = [
        (2, ratio_count(summary.success_tracked), +1),
        (3, ratio_count(summary.success_pen2mm), +1),
        (4, summary.pz_term_mean, -1),
        (5, summary.pz_term_worst, -1),
        (6, summary.inmaskC, +1),
        (7, summary.inmaskC5, +1),
        (8, summary.release_false, -1),
        (9, summary.release_false5, -1),
    ]
    values.extend(
        (METRIC_START_COL + i, summary.metrics.get(key), direction)
        for i, (key, direction) in enumerate(zip(METRIC_ORDER, METRIC_DIRECTIONS))
    )
    if include_gate:
        values.extend([(TAIL_START_COL, summary.gate_valid, +1), (TAIL_START_COL + 1, summary.fallback, -1)])
    else:
        values.extend([(TAIL_START_COL, ratio_count(summary.fall), -1), (TAIL_START_COL + 1, summary.fallback, -1)])
    return values


def apply_rank_marks(ws, ranked_rows: list[tuple[int, Summary]], *, include_gate: bool) -> None:
    by_col: dict[int, list[tuple[int, float, int]]] = defaultdict(list)
    for row_idx, summary in ranked_rows:
        for col, value, direction in rankable_values(summary, include_gate=include_gate):
            if value is None or not math.isfinite(value):
                continue
            by_col[col].append((row_idx, value, direction))

    for col, entries in by_col.items():
        if not entries:
            continue
        direction = entries[0][2]
        unique = sorted({value for _, value, _ in entries}, reverse=(direction > 0))
        if not unique:
            continue
        best = unique[0]
        second = unique[1] if len(unique) > 1 else None
        for row_idx, value, _direction in entries:
            cell = ws.cell(row=row_idx, column=col)
            if value == best:
                font = copy(cell.font)
                font.bold = True
                font.color = "FF000000"
                cell.font = font
            elif second is not None and value == second:
                font = copy(cell.font)
                font.underline = "single"
                cell.font = font


def build_workbook(summaries: dict[str, Summary], grid: dict[tuple[float, float], Summary], percase: list[dict[str, str]]):
    wb = openpyxl.Workbook()
    write_sheet_threshold_scan(wb.active, summaries, grid)
    write_sheet_method_compare(wb, summaries, grid)
    write_sheet_percase(wb, percase)
    write_sheet_notes(wb, summaries)
    return wb


def write_sheet_threshold_scan(ws, summaries: dict[str, Summary], grid: dict[tuple[float, float], Summary]):
    ws.title = "E153阈值扫描(E154修正)"
    cols = [
        "方法",
        "succ_tracked\n(新门控)",
        "succ_pen2mm\n(旧门控)",
        "pz_term\nmean",
        "pz_term\nworst",
        "inmaskC3\n(真实mask内\n物理接触↑)",
        "inmaskC5\n(真实mask内\n物理接触↑)",
        "releaseF3\n(诊断↓)",
        "releaseF5\n(诊断↓)",
        "5cm接触\n(↑)",
        "2mm穿透\n(↓)",
        "5mm穿透\n(↓)",
        "物理接触3\n(↑)",
        "物理穿透>3mm\n(↓)",
        "物理接触5\n(↑)",
        "物理穿透>5mm\n(↓)",
        "腿穿透\n(↓)",
        "gate有效率",
        "fallback",
    ]
    merge_cell(
        ws,
        1,
        1,
        1,
        len(cols),
        value="E154 评测修正: E153 阈值扫描 (3-case 均值, OmniRetarget 为 baseline)",
        bold=True,
        fg="FFFFFF",
        bg="2F2F2F",
        size=12,
    )
    ws.row_dimensions[1].height = 28
    for i, label in enumerate(cols):
        set_cell(ws, 2, 1 + i, label, bold=True, bg=SUB_BG, size=9)
    ws.row_dimensions[2].height = 55

    ref = summaries["omniretarget"]
    rows = [
        summaries["omniretarget"],
        summaries["e153_baseline"],
        summaries["e153_b1"],
    ]
    ranked_rows: list[tuple[int, Summary]] = []
    for i, summary in enumerate(rows, 3):
        write_summary_row(ws, i, summary, ref, METHOD_BG[i % 2], is_ref=(summary is ref), include_gate=True)
        ranked_rows.append((i, summary))
        ws.row_dimensions[i].height = 30

    grid_order = [
        (-0.005, 0.05),
        (-0.005, 0.10),
        (-0.010, 0.05),
        (-0.010, 0.10),
        (-0.015, 0.05),
        (-0.015, 0.10),
    ]
    grid_ref = summaries["e153_b1"]
    for gi, key in enumerate(grid_order):
        r = 6 + gi
        summary = grid.get((round(key[0], 3), round(key[1], 2)))
        if summary is None:
            continue
        write_summary_row(
            ws,
            r,
            summary,
            grid_ref,
            METHOD_BG[gi % 2],
            is_best=(round(key[0], 3), round(key[1], 2)) == (-0.01, 0.10),
            include_gate=True,
        )
        ranked_rows.append((r, summary))
        ws.row_dimensions[r].height = 36

    apply_rank_marks(ws, ranked_rows, include_gate=True)

    fn_row = 6 + len(grid_order)
    merge_cell(
        ws,
        fn_row,
        1,
        fn_row,
        len(cols),
        value="数据从本地 TSV 聚合生成；OmniRetarget 为 baseline，spider-rubberhand 为旧 rubber-hand SPIDER/CEM 对照。"
        "  OmniRetarget 行来自 E154 对每个 CEM 输入参考 trajectory_kinematic.npz 的 kinematic replay 评测。",
        fg="595959",
        bg="F7F7F7",
        size=9,
    )
    ws.column_dimensions["A"].width = 28
    for c in range(2, len(cols) + 1):
        ws.column_dimensions[get_column_letter(c)].width = 13
    ws.freeze_panes = "B3"


def write_sheet_method_compare(wb, summaries: dict[str, Summary], grid: dict[tuple[float, float], Summary]):
    ws = wb.create_sheet("E152方法对比(E154修正)")
    cols = [
        "方法",
        "succ_tracked\n(E154新)",
        "succ_pen2mm\n(旧)",
        "pz_term\nmean",
        "pz_term\nworst",
        "inmaskC3\n(真实mask内\n物理接触↑)",
        "inmaskC5\n(真实mask内\n物理接触↑)",
        "releaseF3\n(诊断↓)",
        "releaseF5\n(诊断↓)",
        "5cm接触\n(↑)",
        "2mm穿透\n(↓)",
        "5mm穿透\n(↓)",
        "物理接触3\n(↑)",
        "物理穿透>3mm\n(↓)",
        "物理接触5\n(↑)",
        "物理穿透>5mm\n(↓)",
        "腿穿透\n(↓)",
        "fall",
        "fallback",
        "来源",
    ]
    merge_cell(
        ws,
        1,
        1,
        1,
        len(cols),
        value="E154 评测修正: E152/E153 方法对比 (3-case 均值, OmniRetarget 为 baseline)",
        bold=True,
        fg="FFFFFF",
        bg="2F2F2F",
        size=12,
    )
    ws.row_dimensions[1].height = 28
    for i, label in enumerate(cols):
        set_cell(ws, 2, 1 + i, label, bold=True, bg=SUB_BG, size=9)
    ws.row_dimensions[2].height = 55

    rows = [
        summaries["omniretarget"],
        summaries["e152_baseline"],
        summaries["e152_gateA"],
        summaries["e152_b1"],
        summaries["e152_gateA_b1"],
        grid[(-0.01, 0.10)],
    ]
    rows[-1].label = "gateA+b1 *\n(sdf=-0.010, viol=0.100)\n[E153最优]"
    ref = summaries["omniretarget"]
    ranked_rows: list[tuple[int, Summary]] = []
    for ri, summary in enumerate(rows):
        r = 3 + ri
        write_summary_row(
            ws,
            r,
            summary,
            ref,
            METHOD_BG[ri % 2],
            is_ref=(summary is ref),
            is_best=summary is rows[-1],
            include_gate=False,
            source_col=TAIL_START_COL + 2,
        )
        ranked_rows.append((r, summary))
        ws.row_dimensions[r].height = 46

    apply_rank_marks(ws, ranked_rows, include_gate=False)

    ws.column_dimensions["A"].width = 30
    for c in range(2, len(cols) + 1):
        ws.column_dimensions[get_column_letter(c)].width = 14
    ws.column_dimensions[get_column_letter(TAIL_START_COL + 2)].width = 44
    ws.freeze_panes = "B3"


def write_sheet_percase(wb, percase: list[dict[str, str]]):
    ws = wb.create_sheet("per-case明细")
    cols = [
        "case",
        "combo\n(sdf_viol)",
        "tracked",
        "pen2mm_ok",
        "pz_term",
        "inmaskC3",
        "inmaskC5",
        "releaseF3",
        "releaseF5",
        "5cm",
        "2mm_pen",
        "5mm_pen",
        "physC3",
        "physP3",
        "physC5",
        "physP5",
        "leg_pen",
        "fail原因",
    ]
    merge_cell(
        ws,
        1,
        1,
        1,
        len(cols),
        value="E153 per-case 明细 (从 e153_grid_delta_vs_b1.tsv 读取)",
        bold=True,
        fg="FFFFFF",
        bg="2F2F2F",
        size=12,
    )
    for i, label in enumerate(cols):
        set_cell(ws, 2, 1 + i, label, bold=True, bg=SUB_BG, size=9)
    ws.row_dimensions[2].height = 40

    def sort_key(row):
        return (CASE_ORDER.index(row["_case"]), parse_float(row.get("min_sdf_m")) or 0, parse_float(row.get("max_viol")) or 0)

    for ri, row in enumerate(sorted(percase, key=sort_key)):
        r = 3 + ri
        mbg = METHOD_BG[ri % 2]
        msdf = parse_float(row.get("min_sdf_m"))
        mviol = parse_float(row.get("max_viol"))
        combo = f"sdf{abs(msdf or 0):.3f}_v{mviol or 0:.2f}".replace("0.", "")
        tracked = parse_bool(row.get("success_tracked"))
        pen2 = parse_bool(row.get("success_pen2mm"))
        pz = parse_float(row.get("track_pelvis_z_err_terminal_m"))
        reason = fail_reason(tracked, pen2, pz)

        set_cell(ws, r, 1, row["_case"], bold=True, bg=mbg, align=LEFT)
        set_cell(ws, r, 2, combo, bg=mbg, size=9)
        set_cell(ws, r, 3, "PASS" if tracked else "FAIL", bg=GOOD_BG if tracked else BAD_BG, fg=GOOD_FG if tracked else BAD_FG)
        set_cell(ws, r, 4, "PASS" if pen2 else "FAIL", bg=GOOD_BG if pen2 else BAD_BG, fg=GOOD_FG if pen2 else BAD_FG)
        set_cell(ws, r, 5, fixed(pz), bg=BAD_BG if (pz is not None and pz >= 0.08) else (GOOD_BG if (pz is not None and pz < 0.04) else mbg))
        set_cell(ws, r, 6, pct(parse_float(row.get("hand_object_physics_contact_3mm_in_mask_frac"))), bg=mbg)
        set_cell(ws, r, 7, pct(parse_float(row.get("hand_object_physics_contact_5mm_in_mask_frac"))), bg=mbg)
        set_cell(ws, r, 8, pct(parse_float(row.get("hand_object_release_false_contact_3mm_frac"))), bg=mbg)
        set_cell(ws, r, 9, pct(parse_float(row.get("hand_object_release_false_contact_5mm_frac"))), bg=mbg)
        for ci, key in enumerate(METRIC_ORDER):
            set_cell(ws, r, METRIC_START_COL + ci, pct(parse_float(row.get(GRID_RUN_FIELDS[key]))), bg=mbg)
        set_cell(ws, r, METRIC_START_COL + len(METRIC_ORDER), reason, bg=BAD_BG if reason != "-" else mbg, fg=BAD_FG if reason != "-" else "000000", size=9)
        ws.row_dimensions[r].height = 22

    ws.column_dimensions["A"].width = 14
    ws.column_dimensions["B"].width = 14
    for c in range(3, len(cols) + 1):
        ws.column_dimensions[get_column_letter(c)].width = 12
    ws.column_dimensions[get_column_letter(METRIC_START_COL + len(METRIC_ORDER))].width = 20
    ws.freeze_panes = "C3"


def fail_reason(tracked: bool | None, pen2: bool | None, pz: float | None) -> str:
    reasons = []
    if tracked is False:
        reasons.append("tracking")
    if pen2 is False:
        reasons.append("pen2mm")
    if pz is not None and pz >= 0.08 and "tracking" not in reasons:
        reasons.append("pz>=0.08")
    return "+".join(reasons) if reasons else "-"


def write_sheet_notes(wb, summaries: dict[str, Summary]):
    ws = wb.create_sheet("说明")
    notes = [
        ("指标", "来源", "说明"),
        ("", "", ""),
        ("-- 数据来源 --", "", ""),
        ("metric standard", EVAL_METRIC_STANDARD_ID, "后续 E154+ 实验应引用 workspace/core4d/scripts/eval/lib/core_metrics.py 中的标准字段集合。"),
        ("OmniRetarget", summaries["omniretarget"].source, "本表 baseline；来自 E154 对每个 CEM case 的 OmniRetarget/kinematic reference 输入回放评测，包含 tracking、真实 3cm mask 和穿透分层指标。"),
        ("spider-rubberhand", summaries["e153_baseline"].source, "旧 baseline 行重命名为 spider-rubberhand，表示 rubber hand collision 版本的 SPIDER/CEM 对照。"),
        ("b1 / gateA / gateA+b1", summaries["e152_gateA_b1"].source, "E152 方法对比输入。"),
        ("E153 threshold sweep", summaries["e153_b1"].source, "E153 baseline/b1 聚合输入。"),
        ("E153 grid per-case", "workspace/core4d/results/E153/gate_threshold_sweep/eval/full/e153_grid_delta_vs_b1.tsv", "E153 6 个 gate 配置的 per-case 明细、success_tracked、success_pen2mm 与 gate 统计。"),
        ("", "", ""),
        ("-- 指标 --", "", ""),
        ("succ_tracked", "track_pelvis_z_err_terminal_m < 0.08", "body tracking 门控。"),
        ("pz_term", "track_pelvis_z_err_terminal_m", "结尾帧 pelvis 高度跟踪误差，单位 m。"),
        ("inmaskC3/C5", "hand_object_physics_contact_3mm_in_mask_frac / hand_object_physics_contact_5mm_in_mask_frac", "真实 3cm contact mask 窗口内，按 3mm/5mm 阈值过滤穿透后的物理接触比例。"),
        ("releaseF3/F5", "hand_object_release_false_contact_3mm_frac / hand_object_release_false_contact_5mm_frac", "放手窗口内仍保持 3mm/5mm clean 物理接触的帧比例，诊断用。"),
        ("5cm接触", "hand_geom_near_5cm_frac", "手部几何体距物体表面 <= 5cm 的帧比例。"),
        ("2mm/5mm穿透", "hand_geom_penetration_2mm_frac / hand_geom_penetration_5mm_frac", "手部几何体与物体重叠超过阈值的帧比例。"),
        ("物理接触3/5", "hand_object_physics_contact_3mm_frac / hand_object_physics_contact_5mm_frac", "同一阈值下存在 MuJoCo 手-物体 contact，且该帧最小 contact dist >= -3mm / -5mm。"),
        ("物理穿透>3/5mm", "hand_object_physics_penetration_3mm_frame_frac / hand_object_physics_penetration_5mm_frame_frac", "同一阈值下存在 MuJoCo 手-物体 contact，且该帧最小 contact dist < -3mm / -5mm；与对应物理接触列互斥。"),
        ("腿穿透", "leg_penetration_frac", "腿部与物体穿透的帧比例。"),
        ("gate有效率", "hand_gate_valid_frac", "CEM sample 中通过 hand gate 约束的比例。"),
        ("fallback", "gate_fallback_used", "本轮 CEM 所有 sample 均违反 gate 时退化到无 gate 采样的帧比例。"),
        ("", "", ""),
        ("-- 样式 --", "", ""),
        ("黑色加粗", "", "该列最优结果。"),
        ("下划线", "", "该列次优结果。"),
    ]
    for i, (a, b, c) in enumerate(notes, 1):
        is_section = a.startswith("--")
        ws.cell(row=i, column=1, value=a).font = Font(
            bold=(i == 1 or is_section),
            size=10,
            color="FF1F4E79" if is_section else "FF000000",
        )
        ws.cell(row=i, column=2, value=b).font = Font(size=9)
        cl = ws.cell(row=i, column=3, value=c)
        cl.alignment = Alignment(wrap_text=True)
        cl.font = Font(size=9)
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 72
    ws.column_dimensions["C"].width = 72


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    parser.add_argument("--e152-method-metrics", type=pathlib.Path, default=DEFAULT_E152_METHOD_METRICS)
    parser.add_argument("--e153-method-metrics", type=pathlib.Path, default=DEFAULT_E153_METHOD_METRICS)
    parser.add_argument("--e153-grid-delta", type=pathlib.Path, default=DEFAULT_E153_GRID_DELTA)
    parser.add_argument("--omniretarget-metrics", type=pathlib.Path, default=DEFAULT_OMNIRETARGET_METRICS)
    return parser.parse_args()


def main():
    args = parse_args()
    summaries, grid, percase = load_inputs(args)
    wb = build_workbook(summaries, grid, percase)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
