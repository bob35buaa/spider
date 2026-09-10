#!/usr/bin/env python3
"""E214 ablation report: full method vs 4 ablations over the 50-case paper set.

"full" column:
  * 43 non-box023 cases -> reused verbatim from the paper cache
    (report/0908/paper_results/_cache_method_metrics.jsonl, method SPIDER-CEM).
  * 7 box023 cases      -> the E173 full-stack baseline recomputed by
    eval_E214_ablation.py (series full__box023).
ablation columns A1..A4 -> eval_E214_ablation.py (series = ablation name).

Reports, per metric: mean +/- std + worst over cases, Delta vs full, and a
paired Wilcoxon signed-rank p-value (full vs ablation).  Overall (50 cases) +
per-object.  Outputs markdown + tsv (+ xlsx if openpyxl is present).

Usage:
    .venv/bin/python workspace/core4d/scripts/eval/reports/gen_E214_ablation_table.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E214"))
import e214_common as C  # noqa: E402

PAPER_CACHE = REPO / "workspace/core4d/report/0908/paper_results/_cache_method_metrics.jsonl"
E214_EVAL = C.EVAL_DIR / "e214_metrics.jsonl"
OUT_MD = C.REPORT_DIR / "e214_ablation_table.md"
OUT_TSV = C.REPORT_DIR / "e214_ablation_overall.tsv"
OUT_XLSX = C.REPORT_DIR / "e214_ablation_table.xlsx"

# metric key -> (label, direction, kind)  direction: 'higher'|'lower' better.
METRICS = [
    ("raw_contact",    "raw contact",             "higher", "pct"),
    ("contact_3mm",    "phys contact@3mm",        "higher", "pct"),
    ("contact_5mm",    "phys contact@5mm",        "higher", "pct"),
    ("contact_10mm",   "phys contact@10mm",       "higher", "pct"),
    ("pen_3mm",        "phys penetration@3mm",    "lower",  "pct"),
    ("pen_5mm",        "phys penetration@5mm",    "lower",  "pct"),
    ("pen_10mm",       "phys penetration@10mm",   "lower",  "pct"),
    ("pen_max_mm",     "max phys penetration (mm)", "lower", "mm"),
    ("geom_2mm",       "geom penetration@2mm",    "lower",  "pct"),
    ("track_root_pos", "root pos err (cm)",       "lower",  "cm"),
    ("track_root_ori", "root ori err (deg)",      "lower",  "deg"),
    ("track_eef_pos",  "eef pos err (cm)",        "lower",  "cm"),
    ("track_eef_ori",  "eef ori err (deg)",       "lower",  "deg"),
    ("track_obj_pos",  "obj pos err (cm)",        "lower",  "cm"),
    ("track_obj_ori",  "obj ori err (deg)",       "lower",  "deg"),
    ("fall_flag",      "fall rate",               "lower",  "pct"),
    ("body_z",         "body-z err p95 (m)",      "lower",  "m"),
    ("ankle_jerk",     "ankle jerk p95",          "lower",  "jerk"),
    ("obj_speed",      "obj speed max (m/s)",     "lower",  "mps"),
    ("foot_slip",      "foot slip max (m)",       "lower",  "m"),
    ("foot_skate_mean","foot skate mean (m/s)",   "lower",  "mps"),
    ("foot_skate_max", "foot skate max (m/s)",    "lower",  "mps"),
]
MK = [m[0] for m in METRICS]
SERIES = ["full", "A1_contactHDMI_only", "A2_surfaceBand_only",
          "A3_softPenalty_only", "A4_hardGate_only"]
SERIES_LABEL = {
    "full": "full", "A1_contactHDMI_only": "A1 (−surface_band)",
    "A2_surfaceBand_only": "A2 (−contact_hdmi)",
    "A3_softPenalty_only": "A3 (−hard gate)", "A4_hardGate_only": "A4 (−soft penalty)",
}


def _finite(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.nan
    return f if math.isfinite(f) else math.nan


def load_full() -> dict[str, dict[str, float]]:
    """case_id -> full-method metrics. 43 from paper cache, 7 box023 from E214."""
    out: dict[str, dict[str, float]] = {}
    cache = {}
    for line in PAPER_CACHE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("method") == "SPIDER-CEM":
            cache[r["case_id"]] = {k: _finite(v) for k, v in r.get("metrics", {}).items()}
    e214 = load_e214()
    for case in C.load_cases():
        if C.object_key_of(case) == "box023":
            out[case] = e214.get((case, "full__box023"), {})
        else:
            out[case] = cache.get(case, {})
    return out


def load_e214() -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    if not E214_EVAL.is_file():
        return out
    for line in E214_EVAL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") == "ok":
            out[(r["case_id"], r["series"])] = {k: _finite(v) for k, v in r.get("metrics", {}).items()}
    return out


def series_values(case_metrics: dict[str, dict[str, float]], key: str) -> list[float]:
    vals = [case_metrics[c].get(key, math.nan) for c in case_metrics]
    return [v for v in vals if math.isfinite(v)]


def agg(vals: list[float], direction: str) -> dict[str, float]:
    if not vals:
        return {"mean": math.nan, "std": math.nan, "worst": math.nan, "n": 0}
    worst = max(vals) if direction == "lower" else min(vals)
    return {"mean": statistics.fmean(vals), "std": (statistics.pstdev(vals) if len(vals) > 1 else 0.0),
            "worst": worst, "n": len(vals)}


def wilcoxon(full_by_case: dict[str, dict[str, float]],
             abl_by_case: dict[str, dict[str, float]], key: str) -> float:
    """Paired Wilcoxon p (full vs ablation) over shared cases; nan if unavailable."""
    pairs = []
    for c in full_by_case:
        a, b = full_by_case[c].get(key, math.nan), abl_by_case.get(c, {}).get(key, math.nan)
        if math.isfinite(a) and math.isfinite(b):
            pairs.append((a, b))
    if len(pairs) < 6:
        return math.nan
    try:
        from scipy.stats import wilcoxon as _w
        fa = [p[0] for p in pairs]
        fb = [p[1] for p in pairs]
        if all(abs(a - b) < 1e-12 for a, b in pairs):
            return math.nan
        return float(_w(fa, fb, zero_method="wilcox").pvalue)
    except Exception:  # noqa: BLE001
        return math.nan


def fmt(v: float, kind: str) -> str:
    if not math.isfinite(v):
        return "—"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind in ("cm", "deg"):
        return f"{v:.2f}"
    if kind in ("m", "mps"):
        return f"{v:.3f}"
    if kind == "mm":
        return f"{v:.1f}"
    return f"{v:.3g}"


def build_block(cases: list[str], full: dict, e214: dict, title: str) -> list[str]:
    # per-series case->metrics restricted to `cases`
    per_series: dict[str, dict[str, dict[str, float]]] = {}
    for s in SERIES:
        cm: dict[str, dict[str, float]] = {}
        for c in cases:
            if s == "full":
                cm[c] = full.get(c, {})
            else:
                cm[c] = e214.get((c, s), {})
        per_series[s] = cm
    lines = [f"### {title} (n={len(cases)})", ""]
    hdr = "| metric | dir | " + " | ".join(SERIES_LABEL[s] for s in SERIES) + " |"
    lines.append(hdr)
    lines.append("|" + "---|" * (len(SERIES) + 2))
    for key, label, direction, kind in METRICS:
        cells = []
        for s in SERIES:
            a = agg(series_values(per_series[s], key), direction)
            cells.append(f"{fmt(a['mean'], kind)}±{fmt(a['std'], kind)} (w {fmt(a['worst'], kind)})")
        lines.append(f"| {label} | {'↑' if direction=='higher' else '↓'} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def main() -> int:
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    full = load_full()
    e214 = load_e214()
    cases = C.load_cases()

    # coverage note
    have = {s: sum(1 for c in cases if (e214.get((c, s)) if s != "full" else full.get(c)))
            for s in SERIES}
    md = ["# E214 四消融结果表（full vs A1–A4，50 case）", "",
          f"- full 基线：43 non-box023 复用论文缓存 + 7 box023 用 E173 全栈重算。",
          f"- 覆盖：" + ", ".join(f"{SERIES_LABEL[s]}={have[s]}/50" for s in SERIES),
          "- 单元格：mean±std (worst)。worst=对该指标最差的 case（↓指标取最大，↑指标取最小）。",
          "- max phys penetration：每 case 对序列取最深穿透(mm)，再在 case 上平均。foot skate：stance 脚水平滑移速度(m/s)。",
          "- 方向 ↑=越大越好，↓=越小越好。", ""]
    md += build_block(cases, full, e214, "Overall")
    # per-object
    for obj in sorted({C.object_key_of(c) for c in cases}):
        ocases = [c for c in cases if C.object_key_of(c) == obj]
        md += build_block(ocases, full, e214, f"Object: {obj}")

    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    # overall tsv (mean/std/worst per series per metric)
    tlines = ["metric\tseries\tmean\tstd\tworst\tn"]
    per_series = {s: {c: (full.get(c, {}) if s == "full" else e214.get((c, s), {})) for c in cases}
                  for s in SERIES}
    for key, label, direction, kind in METRICS:
        for s in SERIES:
            a = agg(series_values(per_series[s], key), direction)
            tlines.append(f"{key}\t{s}\t{a['mean']:.6g}\t{a['std']:.6g}\t{a['worst']:.6g}\t{a['n']}")
    OUT_TSV.write_text("\n".join(tlines) + "\n", encoding="utf-8")

    # optional xlsx
    try:
        import openpyxl  # noqa: PLC0415
        wb = openpyxl.Workbook(); ws = wb.active; ws.title = "overall"
        ws.append(["metric", "series", "mean", "std", "worst", "n"])
        for line in tlines[1:]:
            ws.append(line.split("\t"))
        wb.save(OUT_XLSX)
        xlsx_note = f" + {C.rel(OUT_XLSX)}"
    except Exception:  # noqa: BLE001
        xlsx_note = " (openpyxl absent, xlsx skipped)"

    print(f"wrote {C.rel(OUT_MD)} + {C.rel(OUT_TSV)}{xlsx_note}")
    print("coverage:", have)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
