#!/usr/bin/env python3
"""E214c part-wise-threshold ablation report.

Primary table (the part-wise story, claims C1-C4):
    full | A5_uniform_hand | A6_uniform_leg
  over the 50-case set, ALL metrics including PER-PART penetration
  (leg / body / upper-body) + hand contact + tracking + fall.  All three series
  come from E214c's OWN eval (e214c_metrics.jsonl): full is recomputed fresh from
  each case's baseline rollout with the identical evaluator, so the comparison is
  internally consistent (same code version, same metric definitions).

Secondary 6-arm table (paper combined ablation):
    full | A1 | A2 | A3 | A4 | A5 | A6
  full/A5/A6 from E214c eval; A1-A4 pulled from E214's eval cache (standard
  metrics only -- per-part cells show "—" since E214 did not compute them).

Per metric: mean +/- std + worst over cases, and a paired Wilcoxon p (full vs arm).
Overall (50) + per-object.  Outputs markdown + tsv (+ xlsx if openpyxl present).

Usage:
    .venv/bin/python workspace/core4d/scripts/eval/reports/gen_E214c_ablation_table.py
    ... --mean-only       cells show only the mean
    ... --no-6arm         skip the secondary 6-arm table
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E214c"))
import e214c_common as C  # noqa: E402

E214C_EVAL = C.EVAL_DIR / "e214c_metrics.jsonl"
E214_EVAL = C.E214.RESULTS / "eval" / "e214_metrics.jsonl"  # for A1-A4 (6-arm only)
OUT_MD = C.REPORT_DIR / "e214c_partwise_table.md"
OUT_TSV = C.REPORT_DIR / "e214c_partwise_overall.tsv"
OUT_XLSX = C.REPORT_DIR / "e214c_partwise_table.xlsx"
OUT_6ARM_MD = C.REPORT_DIR / "e214c_6arm_table.md"

# metric key -> (label, direction, kind).  direction: 'higher'|'lower' better.
METRICS = [
    # --- hand-object contact (A6 should collapse these) ---
    ("raw_contact",    "raw contact",            "higher", "pct"),
    ("contact_3mm",    "phys contact@3mm",       "higher", "pct"),
    ("contact_5mm",    "phys contact@5mm",       "higher", "pct"),
    ("contact_10mm",   "phys contact@10mm",      "higher", "pct"),
    # --- hand-object penetration ---
    ("pen_3mm",        "phys penetration@3mm",   "lower",  "pct"),
    ("pen_5mm",        "phys penetration@5mm",   "lower",  "pct"),
    ("pen_max_mm",     "max phys penetration (mm)", "lower", "mm"),
    ("geom_2mm",       "geom penetration@2mm (hand)", "lower", "pct"),
    # --- PER-PART penetration (A5 should inflate leg/body) ---
    ("leg_pen",        "LEG penetration frac",        "lower", "pct"),
    ("body_pen",       "BODY penetration frac",       "lower", "pct"),
    ("upper_body_pen", "UPPER-BODY penetration frac", "lower", "pct"),
    ("leg_contact",    "leg-object contact frac",     "lower", "pct"),
    # --- tracking / stability ---
    ("track_root_pos", "root pos err (cm)",      "lower",  "cm"),
    ("track_eef_pos",  "eef pos err (cm)",       "lower",  "cm"),
    ("track_eef_ori",  "eef ori err (deg)",      "lower",  "deg"),
    ("track_obj_pos",  "obj pos err (cm)",       "lower",  "cm"),
    ("track_obj_ori",  "obj ori err (deg)",      "lower",  "deg"),
    ("fall_flag",      "fall rate",              "lower",  "pct"),
    ("body_z",         "body-z err p95 (m)",     "lower",  "m"),
    ("ankle_jerk",     "ankle jerk p95",         "lower",  "jerk"),
    ("obj_speed",      "obj speed max (m/s)",    "lower",  "mps"),
    ("foot_slip",      "foot slip max (m)",      "lower",  "m"),
]
PRIMARY_SERIES = ["full", "A5_uniform_hand", "A6_uniform_leg"]
PRIMARY_LABEL = {
    "full": "full (part-wise)",
    "A5_uniform_hand": "A5 (uniform-hand θ)",
    "A6_uniform_leg": "A6 (uniform-leg θ)",
}
SIXARM_SERIES = ["full", "A1_contactHDMI_only", "A2_surfaceBand_only",
                 "A3_softPenalty_only", "A4_hardGate_only",
                 "A5_uniform_hand", "A6_uniform_leg"]
SIXARM_LABEL = {
    "full": "full", "A1_contactHDMI_only": "A1 (−surf_band)",
    "A2_surfaceBand_only": "A2 (−contact_hdmi)", "A3_softPenalty_only": "A3 (−hard gate)",
    "A4_hardGate_only": "A4 (−soft pen)", "A5_uniform_hand": "A5 (uniform-hand θ)",
    "A6_uniform_leg": "A6 (uniform-leg θ)",
}
# metrics E214's cache lacks (per-part) -> "—" for A1-A4 in the 6-arm table
PERPART_KEYS = {"leg_pen", "body_pen", "upper_body_pen", "leg_contact"}


def _finite(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.nan
    return f if math.isfinite(f) else math.nan


def load_jsonl(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") == "ok":
            out[(r["case_id"], r["series"])] = {k: _finite(v) for k, v in r.get("metrics", {}).items()}
    return out


def series_values(case_metrics: dict[str, dict[str, float]], key: str) -> list[float]:
    return [v for v in (case_metrics[c].get(key, math.nan) for c in case_metrics) if math.isfinite(v)]


def agg(vals: list[float], direction: str) -> dict[str, float]:
    if not vals:
        return {"mean": math.nan, "std": math.nan, "worst": math.nan, "n": 0}
    worst = max(vals) if direction == "lower" else min(vals)
    return {"mean": statistics.fmean(vals),
            "std": (statistics.pstdev(vals) if len(vals) > 1 else 0.0),
            "worst": worst, "n": len(vals)}


def wilcoxon(full_by_case: dict[str, dict[str, float]],
             abl_by_case: dict[str, dict[str, float]], key: str) -> float:
    pairs = []
    for c in full_by_case:
        a, b = full_by_case[c].get(key, math.nan), abl_by_case.get(c, {}).get(key, math.nan)
        if math.isfinite(a) and math.isfinite(b):
            pairs.append((a, b))
    if len(pairs) < 6:
        return math.nan
    try:
        from scipy.stats import wilcoxon as _w
        if all(abs(a - b) < 1e-12 for a, b in pairs):
            return math.nan
        return float(_w([p[0] for p in pairs], [p[1] for p in pairs], zero_method="wilcox").pvalue)
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


def build_block(cases: list[str], per_series: dict[str, dict[str, dict[str, float]]],
                series: list[str], label: dict[str, str], title: str,
                mean_only: bool, wilcox_full: str = "full") -> list[str]:
    lines = [f"### {title} (n={len(cases)})", ""]
    lines.append("| metric | dir | " + " | ".join(label[s] for s in series) + " |")
    lines.append("|" + "---|" * (len(series) + 2))
    full_cm = per_series.get(wilcox_full, {})
    for key, mlabel, direction, kind in METRICS:
        cells = []
        for s in series:
            cm = {c: per_series[s].get(c, {}) for c in cases}
            a = agg(series_values(cm, key), direction)
            if mean_only:
                cell = fmt(a["mean"], kind)
            else:
                cell = f"{fmt(a['mean'], kind)}±{fmt(a['std'], kind)} (w {fmt(a['worst'], kind)})"
            if s != wilcox_full:
                p = wilcoxon({c: full_cm.get(c, {}) for c in cases}, cm, key)
                if math.isfinite(p):
                    cell += f" {'*' if p < 0.05 else ''}p={p:.3g}"
            cells.append(cell)
        lines.append(f"| {mlabel} | {'↑' if direction=='higher' else '↓'} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mean-only", action="store_true")
    ap.add_argument("--no-6arm", action="store_true")
    args = ap.parse_args()

    e214c = load_jsonl(E214C_EVAL)
    e214 = load_jsonl(E214_EVAL)
    cases = C.load_cases()
    n = len(cases)
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    def per_series(series: list[str], src_map: dict[str, dict]) -> dict[str, dict[str, dict[str, float]]]:
        out: dict[str, dict[str, dict[str, float]]] = {}
        for s in series:
            src = src_map[s]
            out[s] = {c: src.get((c, s), {}) for c in cases}
        return out

    # ---- primary table: full | A5 | A6 (all from E214c eval) ----
    src_primary = {s: e214c for s in PRIMARY_SERIES}
    ps = per_series(PRIMARY_SERIES, src_primary)
    have = {s: sum(1 for c in cases if ps[s].get(c)) for s in PRIMARY_SERIES}
    cell_note = ("- 单元格：仅均值。" if args.mean_only else
                 "- 单元格：mean±std (worst)；附 full-vs-臂 配对 Wilcoxon p（* = p<0.05）。"
                 "worst=对该指标最差 case（↓取最大，↑取最小）。")
    md = [f"# E214c Part-wise 阈值统一消融（full vs A5 uniform-hand vs A6 uniform-leg，{n} case）", "",
          "- full 列：各 case 基线 rollout 用**同一评估器现算**（非论文缓存），与 A5/A6 内部一致。",
          "- A5 uniform-hand：body/leg 硬门 ← 该 case 自己的 hand 阈值（最宽松，允许接触级浅穿）。",
          "- A6 uniform-leg：body/hand 硬门 ← leg 阈值（最严格，必须净空 +5mm）。",
          "- 软惩罚/posture/e167/peak 均保持基线（part-wise 差异只在硬门）。",
          f"- 覆盖：" + ", ".join(f"{PRIMARY_LABEL[s]}={have[s]}/{n}" for s in PRIMARY_SERIES),
          cell_note,
          "- **逐部位穿透**（LEG/BODY/UPPER-BODY penetration frac）= part-wise 故事核心：A5 预期腿/身↑，A6 预期手接触↓。",
          "- 方向 ↑=越大越好，↓=越小越好。", ""]
    md += build_block(cases, ps, PRIMARY_SERIES, PRIMARY_LABEL, "Overall", args.mean_only)
    for obj in sorted({C.object_key_of(c) for c in cases}):
        ocases = [c for c in cases if C.object_key_of(c) == obj]
        md += build_block(ocases, ps, PRIMARY_SERIES, PRIMARY_LABEL, f"Object: {obj}", args.mean_only)
    out_md = C.REPORT_DIR / ("e214c_partwise_table_mean.md" if args.mean_only else "e214c_partwise_table.md")
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    # overall tsv
    tlines = ["metric\tseries\tmean\tstd\tworst\tn"]
    for key, mlabel, direction, kind in METRICS:
        for s in PRIMARY_SERIES:
            cm = {c: ps[s].get(c, {}) for c in cases}
            a = agg(series_values(cm, key), direction)
            tlines.append(f"{key}\t{s}\t{a['mean']:.6g}\t{a['std']:.6g}\t{a['worst']:.6g}\t{a['n']}")
    OUT_TSV.write_text("\n".join(tlines) + "\n", encoding="utf-8")
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
    print(f"wrote {C.rel(out_md)} + {C.rel(OUT_TSV)}{xlsx_note}")
    print(f"n={n} coverage(primary):", have)

    # ---- secondary 6-arm table: full|A1-A4|A5|A6 (standard metrics) ----
    if not args.no_6arm:
        src6 = {"full": e214c, "A5_uniform_hand": e214c, "A6_uniform_leg": e214c,
                "A1_contactHDMI_only": e214, "A2_surfaceBand_only": e214,
                "A3_softPenalty_only": e214, "A4_hardGate_only": e214}
        ps6 = per_series(SIXARM_SERIES, src6)
        have6 = {s: sum(1 for c in cases if ps6[s].get(c)) for s in SIXARM_SERIES}
        md6 = [f"# E214c 6 臂合并消融表（full | A1–A4 | A5 | A6，{n} case）", "",
               "- full/A5/A6 来自 E214c 现算；A1–A4 来自 E214 eval 缓存（标准指标）。",
               "- 逐部位穿透(LEG/BODY/UPPER-BODY/leg-contact)仅 full/A5/A6 有，A1–A4 显示 —。",
               f"- 覆盖：" + ", ".join(f"{SIXARM_LABEL[s]}={have6[s]}/{n}" for s in SIXARM_SERIES),
               "- 方向 ↑=越大越好，↓=越小越好。", ""]
        md6 += build_block(cases, ps6, SIXARM_SERIES, SIXARM_LABEL, "Overall", args.mean_only)
        OUT_6ARM_MD.write_text("\n".join(md6) + "\n", encoding="utf-8")
        print(f"wrote {C.rel(OUT_6ARM_MD)}")
        print(f"n={n} coverage(6arm):", have6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
