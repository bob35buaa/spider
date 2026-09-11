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
# Full-method recompute of metrics absent from the paper cache (eval_E214_full_recompute.py).
FULL_RECOMPUTE = C.EVAL_DIR / "e214_full_recompute.jsonl"
# MPJPE (all-joint tracking error) for every series (eval_E214_mpjpe.py).
MPJPE_CACHE = C.EVAL_DIR / "e214_mpjpe.jsonl"
# Holosoma-gauge metrics (eval_E214_holosoma.py) -- reported separately, labelled holosoma.
HOLOSOMA_CACHE = C.EVAL_DIR / "e214_holosoma.jsonl"
HOLOSOMA_FIELD = {  # report short key -> holosoma cache field (threshold sweeps)
    "fs_holo_vel": "foot_sliding_holosoma_vel_mean",
    "fs_holo_5": "foot_sliding_holosoma_frac_5mm",
    "fs_holo_10": "foot_sliding_holosoma_frac_10mm",
    "fs_holo_20": "foot_sliding_holosoma_frac_20mm",
    "pen_holo_5": "penetration_holosoma_frac_5mm",
    "pen_holo_10": "penetration_holosoma_frac_10mm",
    "pen_holo_20": "penetration_holosoma_frac_20mm",
    "pen_holo_max": "penetration_holosoma_depth_max_m",
    "cprec_holo_2": "contact_precision_holosoma_2cm",
    "cprec_holo_5": "contact_precision_holosoma_5cm",
    "cprec_holo_10": "contact_precision_holosoma_10cm",
}
# Metrics the paper cache never stored -> overlaid from FULL_RECOMPUTE for non-box023 cases.
# (box023 full comes wholesale from e214_metrics.jsonl series full__box023, all metrics present.)
RECOMPUTE_OVERLAY_KEYS = ["contact_5mm", "contact_10mm", "pen_5mm", "pen_10mm",
                          "pen_max_mm", "foot_skate_mean", "foot_skate_max"]
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
    ("track_mpjpe_g",  "MPJPE-G (cm)",            "lower",  "cm"),
    ("track_mpjpe_l",  "MPJPE-L (cm)",            "lower",  "cm"),
    ("track_obj_pos",  "obj pos err (cm)",        "lower",  "cm"),
    ("track_obj_ori",  "obj ori err (deg)",       "lower",  "deg"),
    ("fall_flag",      "fall rate",               "lower",  "pct"),
    ("body_z",         "body-z err p95 (m)",      "lower",  "m"),
    ("ankle_jerk",     "ankle jerk p95",          "lower",  "jerk"),
    ("obj_speed",      "obj speed max (m/s)",     "lower",  "mps"),
    ("foot_slip",      "foot slip max (m)",       "lower",  "m"),
    ("foot_skate_mean","foot skate mean (m/s)",   "lower",  "mps"),
    ("foot_skate_max", "foot skate max (m/s)",    "lower",  "mps"),
    # --- holosoma gauge (ported from eval_retargeting.py; separate criteria).
    # foot sliding & contact precision reference the CORE4D SMPLX human GT (scene
    # frame); penetration needs no reference. ---
    ("fs_holo_vel",  "foot sliding vel mean (holosoma SMPLX-GT, m/frame)", "lower", "m"),
    ("fs_holo_5",    "foot sliding frac >5mm/f (holosoma SMPLX-GT)",  "lower",  "pct"),
    ("fs_holo_10",   "foot sliding frac >10mm/f (holosoma SMPLX-GT)", "lower",  "pct"),
    ("fs_holo_20",   "foot sliding frac >20mm/f (holosoma SMPLX-GT)", "lower",  "pct"),
    ("pen_holo_5",   "penetration frac@5mm (holosoma)",      "lower",  "pct"),
    ("pen_holo_10",  "penetration frac@10mm (holosoma)",     "lower",  "pct"),
    ("pen_holo_20",  "penetration frac@20mm (holosoma)",     "lower",  "pct"),
    ("pen_holo_max", "penetration depth max (holosoma, m)",  "lower",  "m"),
    ("cprec_holo_2", "contact precision@2cm (holosoma SMPLX-GT)",     "higher", "pct"),
    ("cprec_holo_5", "contact precision@5cm (holosoma SMPLX-GT)",     "higher", "pct"),
    ("cprec_holo_10","contact precision@10cm (holosoma SMPLX-GT)",    "higher", "pct"),
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


def load_full_recompute() -> dict[str, dict[str, float]]:
    """case_id -> recomputed full metrics (metrics absent from the paper cache)."""
    out: dict[str, dict[str, float]] = {}
    if not FULL_RECOMPUTE.is_file():
        return out
    for line in FULL_RECOMPUTE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") == "ok":
            out[r["case_id"]] = {k: _finite(v) for k, v in r.get("metrics", {}).items()}
    return out


def load_full() -> dict[str, dict[str, float]]:
    """case_id -> full-method metrics.

    43 non-box023 cases: paper-cache values for the published metrics, overlaid
    with the recomputed new metrics (contact@5/10mm, pen@5/10mm, max pen, foot
    skate) that the paper cache never stored -- so the full column covers all 50
    cases, comparable to the ablation columns.
    7 box023 cases: the E173 full-stack recompute (all metrics present).
    """
    out: dict[str, dict[str, float]] = {}
    cache = {}
    for line in PAPER_CACHE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("method") == "SPIDER-CEM":
            cache[r["case_id"]] = {k: _finite(v) for k, v in r.get("metrics", {}).items()}
    e214 = load_e214()
    recompute = load_full_recompute()
    mpjpe = load_mpjpe()
    holo = load_holosoma()
    for case in C.load_cases():
        fseries = "full__box023" if C.object_key_of(case) == "box023" else "full"
        if C.object_key_of(case) == "box023":
            merged = dict(e214.get((case, "full__box023"), {}))
        else:
            merged = dict(cache.get(case, {}))
            rc = recompute.get(case, {})
            for k in RECOMPUTE_OVERLAY_KEYS:
                if k in rc:
                    merged[k] = rc[k]
        merged.update(mpjpe.get((case, fseries), {}))
        merged.update(holo.get((case, fseries), {}))
        out[case] = merged
    return out


def load_mpjpe() -> dict[tuple[str, str], dict[str, float]]:
    """(case_id, series) -> {"track_mpjpe_g", "track_mpjpe_l"} in cm.

    G = global (un-aligned) MPJPE; L = pelvis-aligned (pose-only). Empty if the
    pass has not been run.
    """
    out: dict[tuple[str, str], dict[str, float]] = {}
    if not MPJPE_CACHE.is_file():
        return out
    for line in MPJPE_CACHE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        vals: dict[str, float] = {}
        if r.get("mpjpe") is not None:
            vals["track_mpjpe_g"] = _finite(r["mpjpe"])
        if r.get("mpjpe_local") is not None:
            vals["track_mpjpe_l"] = _finite(r["mpjpe_local"])
        if vals:
            out[(r["case_id"], r["series"])] = vals
    return out


def load_holosoma() -> dict[tuple[str, str], dict[str, float]]:
    """(case_id, series) -> holosoma-gauge metrics under their report short keys."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    if not HOLOSOMA_CACHE.is_file():
        return out
    for line in HOLOSOMA_CACHE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        m = r.get("metrics", {})
        vals = {short: _finite(m[fld]) for short, fld in HOLOSOMA_FIELD.items()
                if m.get(fld) is not None}
        if vals:
            out[(r["case_id"], r["series"])] = vals
    return out


def holosoma_alignment_note() -> str:
    """One-line SMPLX-GT alignment coverage summary from the holosoma cache."""
    if not HOLOSOMA_CACHE.is_file():
        return ""
    per_case: dict[str, tuple[float, str]] = {}
    for line in HOLOSOMA_CACHE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        res = r.get("smplx_gt_align_residual_m")
        st = r.get("smplx_gt_status", "")
        if res is not None and r["case_id"] not in per_case:
            per_case[r["case_id"]] = (float(res), st)
    if not per_case:
        return ""
    res_mm = sorted(v[0] * 1000 for v in per_case.values())
    noalign = [c for c, (_, st) in per_case.items() if not str(st).startswith("ok")]
    med = res_mm[len(res_mm) // 2]
    return (f"- holosoma foot-sliding/contact 参考=CORE4D SMPLX 人体 GT（场景系）；对齐 "
            f"{len(per_case) - len(noalign)}/{len(per_case)} case（物体拟合残差 median "
            f"{med:.1f}mm, max {max(res_mm):.1f}mm）"
            + (f"；NO_GT_ALIGN: {noalign}" if noalign else "；全部通过对齐 gate") + "。")


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
    for src in (load_mpjpe(), load_holosoma()):
        for (case, series), vals in src.items():
            if (case, series) in out:
                out[(case, series)].update(vals)
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


def build_block(cases: list[str], full: dict, e214: dict, title: str,
                mean_only: bool = False) -> list[str]:
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
            if mean_only:
                cells.append(fmt(a["mean"], kind))
            else:
                cells.append(f"{fmt(a['mean'], kind)}±{fmt(a['std'], kind)} (w {fmt(a['worst'], kind)})")
        lines.append(f"| {label} | {'↑' if direction=='higher' else '↓'} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mean-only", action="store_true",
                    help="cells show only the mean (no ±std / worst); writes *_mean.md")
    ap.add_argument("--exclude-cases", default="",
                    help="comma-separated case_ids to drop before aggregating (floor-effect cases)")
    ap.add_argument("--tag", default="",
                    help="output subdir under reports/ (e.g. exclude_floor3); default = canonical reports/")
    args = ap.parse_args()

    full = load_full()
    e214 = load_e214()
    cases = C.load_cases()
    excluded = [c for c in (x.strip() for x in args.exclude_cases.split(",")) if c]
    dropped = [c for c in excluded if c in cases]
    cases = [c for c in cases if c not in set(excluded)]
    n = len(cases)

    out_dir = (C.REPORT_DIR / args.tag) if args.tag else C.REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / ("e214_ablation_table_mean.md" if args.mean_only else "e214_ablation_table.md")
    out_tsv = out_dir / "e214_ablation_overall.tsv"
    out_xlsx = out_dir / "e214_ablation_table.xlsx"

    # coverage note
    have = {s: sum(1 for c in cases if (e214.get((c, s)) if s != "full" else full.get(c)))
            for s in SERIES}
    cell_note = (f"- 单元格：仅均值（over {n} case）。"
                 if args.mean_only else
                 "- 单元格：mean±std (worst)。worst=对该指标最差的 case（↓指标取最大，↑指标取最小）。")
    md = [f"# E214 四消融结果表（full vs A1–A4，{n} case）"
          + ("（均值版）" if args.mean_only else ""), "",
          f"- full 基线：non-box023 复用论文缓存（新指标由已选 rollout 现算补齐）+ box023 用 E173 全栈重算。",
          f"- 覆盖：" + ", ".join(f"{SERIES_LABEL[s]}={have[s]}/{n}" for s in SERIES),
          cell_note,
          "- max phys penetration：每 case 对序列取最深穿透(mm)，再在 case 上平均。foot skate：stance 脚水平滑移速度(m/s)。",
          "- 方向 ↑=越大越好，↓=越小越好。"]
    holo_note = holosoma_alignment_note()
    if holo_note:
        md.append(holo_note)
    if dropped:
        md.append(f"- 已剔除 {len(dropped)} 个 floor-effect case：{', '.join(dropped)}。")
    md.append("")
    md += build_block(cases, full, e214, "Overall", mean_only=args.mean_only)
    # per-object
    for obj in sorted({C.object_key_of(c) for c in cases}):
        ocases = [c for c in cases if C.object_key_of(c) == obj]
        md += build_block(ocases, full, e214, f"Object: {obj}", mean_only=args.mean_only)

    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    if args.mean_only:
        print(f"wrote {C.rel(out_md)} (mean-only, n={n})")
        return 0

    # overall tsv (mean/std/worst per series per metric)
    tlines = ["metric\tseries\tmean\tstd\tworst\tn"]
    per_series = {s: {c: (full.get(c, {}) if s == "full" else e214.get((c, s), {})) for c in cases}
                  for s in SERIES}
    for key, label, direction, kind in METRICS:
        for s in SERIES:
            a = agg(series_values(per_series[s], key), direction)
            tlines.append(f"{key}\t{s}\t{a['mean']:.6g}\t{a['std']:.6g}\t{a['worst']:.6g}\t{a['n']}")
    out_tsv.write_text("\n".join(tlines) + "\n", encoding="utf-8")

    # optional xlsx
    try:
        import openpyxl  # noqa: PLC0415
        wb = openpyxl.Workbook(); ws = wb.active; ws.title = "overall"
        ws.append(["metric", "series", "mean", "std", "worst", "n"])
        for line in tlines[1:]:
            ws.append(line.split("\t"))
        wb.save(out_xlsx)
        xlsx_note = f" + {C.rel(out_xlsx)}"
    except Exception:  # noqa: BLE001
        xlsx_note = " (openpyxl absent, xlsx skipped)"

    print(f"wrote {C.rel(out_md)} + {C.rel(out_tsv)}{xlsx_note}")
    print(f"n={n} coverage:", have)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
