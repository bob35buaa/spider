#!/usr/bin/env python3
"""50-case method comparison: SPIDER-CEM vs SBTO vs GMR.

Modelled on report/0908/code/{gen_paper_results,paper_report}.py, but restricted
to the five user-selected metrics and three methods, over the E214 50-case paper
set (the authority is ``e214_common.load_cases()``; the two E167A cases that GMR/
SBTO also carry are excluded so all three methods aggregate over identical cases).

Metrics (all already on the same scale across the three sources):
  Phys. contact@5mm       physical hand-object contact within 5mm (frame frac)  higher
  Phys. penetration@5mm   physical hand-object penetration deeper than 5mm       lower
  Max phys penetration    deepest hand-object penetration, mm                    lower
  Obj speed max           max object speed, m/s                                  lower
  Foot slip max           max stance-foot slip, m                                lower

Sources:
  SPIDER-CEM  gen_E214_ablation_table.load_full()  -- the ablation "full" column
              (paper-cache SPIDER-CEM values + the recomputed 5mm/max-pen metrics),
              identical to the SPIDER-CEM method in gen_paper_results.
  SBTO        sbto/paper_results/sbto_paper_results.xlsx  (ByCase sheet)
  GMR         GMR/out/core4d_g1/paper_metrics/paper_metrics.xlsx  (per_case sheet)

Outputs (report/0908/paper_cmp_v2/): method_comparison.{md,tex,tsv,xlsx}
Overall + per-object mean +/- std (+ worst); best method per metric is bold.
"""

from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path
from typing import Any

import openpyxl

REPO = Path(__file__).resolve().parents[5]
for _p in ("workspace/core4d/report/0908/code", "workspace/core4d/scripts",
           "workspace/core4d/scripts/eval/reports",
           "workspace/core4d/scripts/experiments/E214"):
    sys.path.insert(0, str(REPO / _p))

import e214_common as C  # noqa: E402
import gen_E214_ablation_table as A  # noqa: E402

OUT = Path(__file__).resolve().parent
GMR_XLSX = Path("/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/GMR/out/core4d_g1/paper_metrics/paper_metrics.xlsx")
SBTO_XLSX = Path("/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/sbto/paper_results/sbto_paper_results.xlsx")

METHODS = ["SPIDER-CEM", "SBTO", "GMR"]  # ours first, then baselines

# report key, label, direction, kind
METRICS = [
    ("contact_5mm", "Phys. contact@5mm", "higher", "pct"),
    ("pen_5mm",     "Phys. penetration@5mm", "lower", "pct"),
    ("pen_max_mm",  "Max phys penetration (mm)", "lower", "mm"),
    ("obj_speed",   "Obj speed max (m/s)", "lower", "mps"),
    ("foot_slip",   "Foot slip max (m)", "lower", "m"),
]
MK = [m[0] for m in METRICS]

# Per-source column headers -> report key (exact header strings; verified on disk).
GMR_COLS = {
    "contact_5mm": "Phys contact@5mm ↑",
    "pen_5mm": "Phys pen@5mm ↓",
    "pen_max_mm": "Max pen depth (mm) ↓",
    "obj_speed": "obj spd max ·",
    "foot_slip": "Foot slip max (m) ↓",
}
SBTO_COLS = {
    "contact_5mm": "physical contact@5mm ↑",
    "pen_5mm": "physical penetration@5mm ↓",
    "pen_max_mm": "max penetration depth mm ↓",
    "obj_speed": "obj speed max m/s ↓",
    "foot_slip": "foot slip max m ↓",
}

OBJECT_ORDER = ("box001", "box004", "box021", "box023", "box024", "bucket003",
                "bucket007", "chair006", "desk007", "desk021", "desk023")


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _num(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def object_of(case_id: str) -> str:
    return case_id.split("_", 1)[0]


def _xlsx_by_case(path: Path, sheet: str, cols: dict[str, str],
                  cases: set[str]) -> dict[str, dict[str, float]]:
    """Read a per-case sheet into {case_id: {metric_key: value}} for `cases`."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    header = list(rows[0])
    col_idx = {}
    for key, colname in cols.items():
        if colname not in header:
            raise KeyError(f"{path.name}/{sheet}: column {colname!r} not found; have {header}")
        col_idx[key] = header.index(colname)
    out: dict[str, dict[str, float]] = {}
    for r in rows[1:]:
        cid = r[0]
        if cid not in cases:
            continue
        out[cid] = {k: _num(r[col_idx[k]]) for k in cols}
    return out


def load_methods(cases: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    caseset = set(cases)
    full = A.load_full()  # {case: merged SPIDER-CEM metrics}
    spider = {c: {k: _num(full.get(c, {}).get(k)) for k in MK} for c in cases}
    gmr = _xlsx_by_case(GMR_XLSX, "per_case", GMR_COLS, caseset)
    sbto = _xlsx_by_case(SBTO_XLSX, "ByCase", SBTO_COLS, caseset)
    return {"SPIDER-CEM": spider, "SBTO": sbto, "GMR": gmr}


def agg(values: list[float], direction: str) -> dict[str, float]:
    vals = [v for v in values if _finite(v)]
    if not vals:
        return {"mean": math.nan, "std": math.nan, "worst": math.nan, "n": 0}
    worst = min(vals) if direction == "higher" else max(vals)
    return {"mean": statistics.fmean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "worst": worst, "n": len(vals)}


def summarize(method_data: dict[str, dict[str, float]], case_ids: list[str]) -> dict[str, dict[str, float]]:
    out = {}
    for key, _label, direction, _kind in METRICS:
        vals = [method_data[c][key] for c in case_ids
                if c in method_data and key in method_data[c] and _finite(method_data[c][key])]
        out[key] = agg(vals, direction)
    return out


def fmt(v: float, kind: str) -> str:
    if not _finite(v):
        return "--"
    if kind == "pct":
        return f"{v * 100:.1f}"
    if kind == "mm":
        return f"{v:.1f}"
    if kind in ("mps", "m"):
        return f"{v:.3f}"
    return f"{v:.3g}"


def best_method(summ: dict[str, dict[str, dict[str, float]]], key: str, direction: str) -> str | None:
    """Method with the best mean for a metric (over the given per-object summaries)."""
    cand = [(m, summ[m][key]["mean"]) for m in METHODS if _finite(summ[m][key]["mean"])]
    if not cand:
        return None
    return (max if direction == "higher" else min)(cand, key=lambda t: t[1])[0]


# --------------------------------------------------------------------------- md
def write_markdown(methods, cases, objects) -> None:
    L = ["# 50-case method comparison — SPIDER-CEM vs SBTO vs GMR", "",
         f"_{len(cases)} cases (E214 paper set). All three methods aggregate over "
         "the identical case set; mean ± std, worst in parentheses. Best mean per "
         "metric in **bold**._", ""]
    # overall
    summ = {m: summarize(methods[m], cases) for m in METHODS}
    L += ["## Overall (mean ± std, worst)", "",
          "| metric | dir | " + " | ".join(METHODS) + " |",
          "| --- | :---: | " + " | ".join("---:" for _ in METHODS) + " |"]
    for key, label, direction, kind in METRICS:
        b = best_method(summ, key, direction)
        cells = []
        for m in METHODS:
            s = summ[m][key]
            txt = f"{fmt(s['mean'], kind)} ± {fmt(s['std'], kind)} ({fmt(s['worst'], kind)})" if s["n"] else "--"
            cells.append(f"**{txt}**" if m == b and s["n"] else txt)
        arrow = "↑" if direction == "higher" else "↓"
        L.append(f"| {label} | {arrow} | " + " | ".join(cells) + " |")
    # per-object (means only)
    L += ["", "## Per-object (mean)", ""]
    for key, label, direction, kind in METRICS:
        L += [f"### {label} ({'↑' if direction=='higher' else '↓'})", "",
              "| object | n | " + " | ".join(METHODS) + " |",
              "| --- | ---: | " + " | ".join("---:" for _ in METHODS) + " |"]
        for o in objects:
            ocases = [c for c in cases if object_of(c) == o]
            osum = {m: summarize(methods[m], ocases) for m in METHODS}
            b = best_method(osum, key, direction)
            cells = []
            for m in METHODS:
                s = osum[m][key]
                txt = fmt(s["mean"], kind) if s["n"] else "--"
                cells.append(f"**{txt}**" if m == b and s["n"] else txt)
            L.append(f"| {o} | {len(ocases)} | " + " | ".join(cells) + " |")
        L.append("")
    # per-case
    L += ["## Per-case detail", "",
          "| case | object | " + " | ".join(f"{lbl} ({mth})" for mth in METHODS for _, lbl, _, _ in [] ) + "" ]
    # build per-case wide header: metric x method
    hdr = ["case", "object"] + [f"{lbl}·{mth}" for _, lbl, _, _ in METRICS for mth in METHODS]
    L[-1] = "| " + " | ".join(hdr) + " |"
    L.append("| " + " | ".join("---" for _ in hdr) + " |")
    for c in cases:
        row = [c, object_of(c)]
        for key, _lbl, _d, kind in METRICS:
            for m in METHODS:
                v = methods[m].get(c, {}).get(key, math.nan)
                row.append(fmt(v, kind))
        L.append("| " + " | ".join(row) + " |")
    L += ["", "## Caveats", "",
          "- 全部 3 方法在**同一 50 case**上聚合（覆盖 50/50）。contact@5mm / pen@5mm 为帧占比（%），"
          "max pen 为 mm，obj speed m/s，foot slip m。",
          "- **GMR 的低 pen@5mm 是假象**：GMR 几乎不接触物体（contact@5mm≈0.2%），不接触自然不穿透；"
          "其 max pen 23.5mm 反而比 SPIDER 12.0mm 更深。评价接触质量应看 contact@5mm + max pen，不能单看 pen@5mm。",
          "- SPIDER-CEM 取自 E214 ablation 的 `full` 列（论文缓存 + 5mm/max-pen 现算），与 gen_paper_results "
          "的 SPIDER-CEM 一致；GMR/SBTO 取各自 paper xlsx 的 per-case 表。",
          "- obj speed / foot slip 属健康类指标，三方法各自 pipeline 计算，绝对值口径可能略有差异；"
          "接触/穿透用同一 MuJoCo 接触检测器，最可比。", ""]
    (OUT / "method_comparison.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# -------------------------------------------------------------------------- tex
def write_latex(methods, cases) -> None:
    summ = {m: summarize(methods[m], cases) for m in METHODS}
    L = [r"% Auto-generated by build_method_comparison_v2.py (50-case, 3 methods).",
         r"% Requires: \usepackage{booktabs}.", r"",
         r"\begin{table}[t]", r"\centering",
         r"\caption{Method comparison on CORE4D-G1 (all 50 paper sequences), "
         r"mean over cases. \textbf{SPIDER-CEM} (ours) vs \textbf{SBTO} and "
         r"\textbf{GMR}. Physical contact/penetration are frame fractions from the "
         r"same MuJoCo contact detector; max penetration is the deepest hand-object "
         r"interpenetration. $\uparrow$/$\downarrow$: higher/lower is better; best "
         r"per row in bold. Note GMR's low penetration@5mm is an artifact of barely "
         r"contacting the object (contact@5mm $\approx0.2\%$), not clean contact: "
         r"its max penetration (23.5\,mm) still exceeds ours (12.0\,mm).}",
         r"\label{tab:method_cmp50}",
         r"\begin{tabular}{l ccc}", r"\toprule",
         r"Metric & SPIDER-CEM & SBTO & GMR \\", r"\midrule"]
    for key, label, direction, kind in METRICS:
        b = best_method(summ, key, direction)
        cells = []
        for m in METHODS:
            s = summ[m][key]
            txt = fmt(s["mean"], kind) if s["n"] else "--"
            cells.append(rf"\textbf{{{txt}}}" if m == b and s["n"] else txt)
        arrow = r"\uparrow" if direction == "higher" else r"\downarrow"
        unit = r" ($\%$)" if kind == "pct" else ""
        L.append(rf"{label}{unit} ${arrow}$ & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (OUT / "method_comparison.tex").write_text("\n".join(L), encoding="utf-8")


# -------------------------------------------------------------------------- tsv
def write_tsv(methods, cases) -> None:
    summ = {m: summarize(methods[m], cases) for m in METHODS}
    lines = ["metric\tmethod\tmean\tstd\tworst\tn"]
    for key, _label, _direction, _kind in METRICS:
        for m in METHODS:
            s = summ[m][key]
            lines.append(f"{key}\t{m}\t{s['mean']:.6g}\t{s['std']:.6g}\t{s['worst']:.6g}\t{s['n']}")
    (OUT / "method_comparison_overall.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------- xlsx
def write_xlsx(methods, cases, objects) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    wb = Workbook()
    hdr = Font(bold=True, color="FFFFFF")
    fill = PatternFill("solid", fgColor="2F75B5")

    def style(ws, ncol):
        for cell in ws[1][:ncol]:
            cell.font = hdr; cell.fill = fill
            cell.alignment = Alignment(horizontal="center", wrap_text=True)

    summ = {m: summarize(methods[m], cases) for m in METHODS}
    ws = wb.active; ws.title = "Overall"
    ws.append(["metric", "dir"] + [f"{m} {s}" for m in METHODS for s in ("mean", "std", "worst", "n")])
    for key, label, direction, _kind in METRICS:
        row = [label, "higher" if direction == "higher" else "lower"]
        for m in METHODS:
            s = summ[m][key]
            row += [_num(s["mean"]), _num(s["std"]), _num(s["worst"]), s["n"]]
        ws.append(row)
    style(ws, 2 + 4 * len(METHODS))

    ws = wb.create_sheet("ByObject")
    ws.append(["object", "n"] + [f"{lbl}·{m}" for _, lbl, _, _ in METRICS for m in METHODS])
    for o in objects:
        ocases = [c for c in cases if object_of(c) == o]
        osum = {m: summarize(methods[m], ocases) for m in METHODS}
        row = [o, len(ocases)]
        for key, _lbl, _d, _k in METRICS:
            for m in METHODS:
                row.append(_num(osum[m][key]["mean"]))
        ws.append(row)
    style(ws, 2 + len(METRICS) * len(METHODS))

    ws = wb.create_sheet("ByCase")
    ws.append(["case_id", "object"] + [f"{lbl}·{m}" for _, lbl, _, _ in METRICS for m in METHODS])
    for c in cases:
        row = [c, object_of(c)]
        for key, _lbl, _d, _k in METRICS:
            for m in METHODS:
                row.append(_num(methods[m].get(c, {}).get(key)))
        ws.append(row)
    style(ws, 2 + len(METRICS) * len(METHODS))

    for w in wb.worksheets:
        w.freeze_panes = "A2"
    wb.save(OUT / "method_comparison.xlsx")


def main() -> int:
    cases = C.load_cases()
    objects = [o for o in OBJECT_ORDER if any(object_of(c) == o for c in cases)]
    methods = load_methods(cases)
    # coverage check
    for m in METHODS:
        n = sum(1 for c in cases if c in methods[m])
        print(f"{m}: {n}/{len(cases)} cases covered")
    write_markdown(methods, cases, objects)
    write_latex(methods, cases)
    write_tsv(methods, cases)
    write_xlsx(methods, cases, objects)
    print("wrote:")
    for f in ("method_comparison.md", "method_comparison.tex",
              "method_comparison_overall.tsv", "method_comparison.xlsx"):
        print("  ", OUT / f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
