#!/usr/bin/env python3
"""Unified paper-aligned evaluation table generator (E019 P0).

Reads one or more per-experiment ``comparison.csv`` files produced by
``paper_metrics.add_paper_metrics`` (E014 / E018 / E018b / ...) and emits:

  * ``tables/table_spider_t4.md``       — SPIDER Table 4 alignment
  * ``tables/table_omniretarget_t2.md`` — OmniRetarget Table II alignment
  * ``tables/table_dynaretarget_t5.md`` — DynaRetarget Table V alignment
  * ``tables/table_core4d_collab.md``   — Project-specific stability / contact
  * ``tables/table_paper_all.xlsx``     — multi-sheet workbook (raw / mean±std /
                                          by_object / failures) for cross-method
                                          comparison
  * ``per_case/{method}/{case}.json``   — one JSON per case, for downstream
                                          tools (E020 audit, RL export filter)

Usage:
    .venv/bin/python workspace/core4d_collab_retarget/scripts/eval/unified_eval.py \
        --method spider_E018b \
        --comparison workspace/core4d_collab_retarget/results/E018b/comparison.csv \
        --out workspace/core4d_collab_retarget/results/eval_unified

    # Multiple methods (cross-method xlsx will appear automatically):
    --method spider_E014 --comparison .../E014/comparison.csv \
    --method spider_E018b --comparison .../E018b/comparison.csv

This script is a pure post-processor — it does NOT re-run rollouts or FK.
Run ``eval_E0NN.py --all`` first to populate the per-experiment comparison.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Column groups (SPIDER T4 / OmniRetarget T2 / DynaRetarget T5 / core4d)
# ---------------------------------------------------------------------------

SPIDER_T4_COLS = [
    # All err metrics are lower-is-better (↓)
    ("paper_spider_joint_err_deg", "Joint Err. (°) ↓"),
    ("paper_spider_pos_err_cm", "Pos. Err. (cm) [MPKPE] ↓"),
    ("paper_spider_ori_err_deg", "Ori. Err. (°) ↓"),
    ("paper_spider_root_pos_err_cm", "Root Pos. Err. (cm) ↓"),
    ("paper_spider_root_ori_err_deg", "Root Ori. Err. (°) ↓"),
    ("paper_spider_eef_pos_err_cm", "EEF Pos. Err. (cm) ↓"),
    ("paper_spider_eef_ori_err_deg", "EEF Ori. Err. (°) ↓"),
    ("paper_spider_obj_pos_err_cm", "Obj. Pos. Err. (cm) ↓"),
    ("paper_spider_obj_ori_err_deg", "Obj. Ori. Err. (°) ↓"),
]

OMNIRETARGET_T2_COLS = [
    # Penetration / skating: lower = better (↓); contact preservation: higher (↑)
    ("paper_omniretarget_mj_penetration_duration_pct", "Pen. Duration (%) ↓"),
    ("paper_omniretarget_mj_penetration_max_depth_cm", "Pen. Max Depth (cm) ↓"),
    ("paper_omniretarget_foot_skating_duration_pct", "Foot Skating Duration (%) ↓"),
    ("paper_omniretarget_foot_skating_max_vel_cm_s", "Foot Skating Max Vel (cm/s) ↓"),
    ("paper_omniretarget_contact_preservation_5cm_pct", "Contact Preservation 5cm (%) ↑"),
]

DYNARETARGET_T5_COLS = [
    # Errors / smoothness (mean |q̈|) lower-better; success rate higher-better;
    # relative smoothness lower-better (sim more平滑 vs ref).
    ("paper_object_Epos_case_m", "Obj. Epos (m) ↓"),
    ("paper_object_Erot_case_deg", "Obj. Erot (°) ↓"),
    ("paper_dynaretarget_object_success", "Object Success ↑"),
    ("paper_dynaretarget_smoothness", "Smoothness (rad/s²) ↓"),
    ("paper_dynaretarget_relative_smoothness", "Rel. Smoothness vs Ref ↓"),
]

CORE4D_COLS = [
    # Pelvis min z: higher = upright (↑); robot fall count: ↓; success / progress ratio: ↑;
    # penetration / leg interference: ↓.
    ("case_window_pelvis_z_min_m", "Pelvis Min z (m) ↑"),
    ("E018b_robot_fall_detected", "Robot Fall ↓"),
    ("paper_carry_progress_ratio_case", "Carry Progress Ratio ↑→1"),
    ("paper_transport_success", "Transport Success ↑"),
    ("paper_omniretarget_robot_object_deep_penetration_duration_pct", "Deep Pen. Duration (%) ↓"),
    ("case_window_ref_leg_box_interference_frames_pct", "Leg Intf (ref) (%) ↓"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce(v: str) -> Any:
    if v is None or v == "":
        return None
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        f = float(v)
        if math.isnan(f):
            return None
        return f
    except ValueError:
        return v


def _read_comparison(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for raw in csv.DictReader(f):
            rows.append({k: _coerce(v) for k, v in raw.items()})
    return rows


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if abs(f) >= 100:
            return f"{f:.1f}"
        if abs(f) >= 1:
            return f"{f:.2f}"
        return f"{f:.3f}"
    return str(v)


def _object_class(case_name: str) -> str:
    n = case_name.lower()
    for tag in ("box", "bucket", "desk", "chair", "board", "stick"):
        if tag in n:
            return tag
    return "other"


def _short_case(variant: str, method_prefix: str = "") -> str:
    """Strip leading method/version prefix and trailing canonical/derived
    suffixes. Normalises CORE4D case naming so cross-method comparison can
    match e.g. ``box025_person2_freejoint_legobj_e018b`` with ``box025_p2``.

    Output canonical form: ``{obj}_{p1|p2}`` (e.g. ``box025_p2``,
    ``bucket005_s2_p1``).
    """
    name = variant
    if method_prefix and name.startswith(method_prefix):
        name = name[len(method_prefix):]
    # Strip experiment-specific suffixes
    for suffix in (
        "_canonical_t02",
        "_jointB_t02",
        "_freejoint_legobj_e018b",
        "_freejoint_legobj",
    ):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    # Normalise person1/person2 -> p1/p2
    name = name.replace("_person1", "_p1").replace("_person2", "_p2")
    return name


# ---------------------------------------------------------------------------
# Per-method markdown tables
# ---------------------------------------------------------------------------


def _md_table(
    rows: list[dict[str, Any]],
    cols: list[tuple[str, str]],
    case_key: str = "case",
    method: str = "",
    title: str = "",
) -> str:
    headers = ["Case"] + [label for _, label in cols]
    out = []
    if title:
        out.append(f"### {title}")
        out.append("")
    if method:
        out.append(f"**Method:** `{method}`  ·  **N:** {len(rows)}")
        out.append("")
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        case_short = _short_case(str(row.get(case_key, "")), method_prefix=f"{method}_" if method else "")
        cells = [case_short] + [_fmt(row.get(k)) for k, _ in cols]
        out.append("| " + " | ".join(cells) + " |")
    # Aggregate
    out.append("")
    out.append("**Aggregate (mean ± std over cases):**")
    out.append("")
    out.append("| Statistic | " + " | ".join(label for _, label in cols) + " |")
    out.append("|---|" + "|".join(["---"] * len(cols)) + "|")
    means, stds = [], []
    for k, _ in cols:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
        if vals:
            means.append(f"{float(np.mean(vals)):.3f}")
            stds.append(f"{float(np.std(vals)):.3f}")
        else:
            # Maybe boolean
            bvals = [bool(r[k]) for r in rows if isinstance(r.get(k), bool)]
            if bvals:
                means.append(f"{sum(bvals)}/{len(bvals)}")
                stds.append("—")
            else:
                means.append("—")
                stds.append("—")
    out.append("| Mean | " + " | ".join(means) + " |")
    out.append("| Std  | " + " | ".join(stds) + " |")
    out.append("")
    return "\n".join(out)


def _method_comparison_table(
    methods: list[tuple[str, list[dict[str, Any]]]],
    cols: list[tuple[str, str]],
    title: str,
) -> str:
    """Side-by-side aggregate of N methods across a metric column group.

    Restricts to the *intersection* of cases (matched by short case name) and
    also reports each method's full N so reader knows the gap.
    """
    out = [f"### {title}", ""]
    if len(methods) < 2:
        out.append("_Need ≥2 methods for comparison. Run with multiple `--method`._")
        out.append("")
        return "\n".join(out)

    # Index each method's rows by canonical short case (strip method prefix + suffix)
    method_cases: list[dict[str, dict[str, Any]]] = []
    for method, rows in methods:
        idx: dict[str, dict[str, Any]] = {}
        for r in rows:
            short = _short_case(str(r.get("case", "")), method_prefix=f"{method}_")
            idx[short] = r
        method_cases.append(idx)

    common = set(method_cases[0].keys())
    for m in method_cases[1:]:
        common &= set(m.keys())
    common = sorted(common)
    out.append(f"**Methods:** {', '.join(f'`{m}`' for m, _ in methods)}")
    out.append(f"**Cases used (intersection of all methods):** N={len(common)} → {', '.join(common) if common else '(none)'}")
    out.append("**Full case counts per method:** " + ", ".join(f"`{m}`={len(r)}" for m, r in methods))
    out.append("")
    if not common:
        out.append("_No overlapping cases — can't compare._")
        out.append("")
        return "\n".join(out)

    headers = ["Metric"] + [m for m, _ in methods] + ["Δ (m1 − m0)"]
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for k, label in cols:
        per_method_vals: list[list[float]] = []
        for m_rows in method_cases:
            vals = []
            for c in common:
                v = m_rows[c].get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    vals.append(float(v))
            per_method_vals.append(vals)
        means = [(np.mean(v) if v else None) for v in per_method_vals]
        row_cells = [label] + [
            ("—" if m is None else _fmt(m)) for m in means
        ]
        if means[0] is not None and means[-1] is not None:
            row_cells.append(_fmt(means[-1] - means[0]))
        else:
            row_cells.append("—")
        out.append("| " + " | ".join(row_cells) + " |")
    out.append("")
    return "\n".join(out)


def _table_failure_modes(rows: list[dict[str, Any]], method: str) -> str:
    diag_key = "E018b_diagnostic_class"  # other experiments would use their own
    classes: dict[str, list[str]] = {}
    for r in rows:
        cls = r.get(diag_key) or "(no_diagnostic)"
        cls = str(cls)
        classes.setdefault(cls, []).append(_short_case(str(r.get("case", "")), method_prefix=f"{method}_"))
    out = [f"### 失败模式分布 — `{method}`", ""]
    out.append("| Diagnostic Class | Count | Cases |")
    out.append("|---|---|---|")
    for cls, cases in sorted(classes.items(), key=lambda kv: -len(kv[1])):
        out.append(f"| `{cls}` | {len(cases)} | {', '.join(cases)} |")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Multi-method xlsx (one sheet per table + a cross-method summary)
# ---------------------------------------------------------------------------


def _write_xlsx(
    methods: list[tuple[str, list[dict[str, Any]]]],
    out_path: Path,
) -> None:
    try:
        from openpyxl import Workbook
    except ImportError:
        # Fall back: write CSV bundle
        out_path = out_path.with_suffix(".csv_bundle")
        out_path.mkdir(exist_ok=True)
        for method, rows in methods:
            with (out_path / f"{method}_raw.csv").open("w", newline="") as f:
                if not rows:
                    continue
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        print(f"[xlsx] openpyxl unavailable, wrote CSV bundle to {out_path}", file=sys.stderr)
        return

    wb = Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    # raw sheet per method
    for method, rows in methods:
        ws = wb.create_sheet(title=f"raw_{method}"[:31])
        if not rows:
            continue
        cols = list(rows[0].keys())
        ws.append(cols)
        for r in rows:
            ws.append([r.get(c) for c in cols])

    # SPIDER T4 cross-method summary sheet
    ws = wb.create_sheet(title="spider_t4_mean")
    ws.append(["method", "n"] + [label for _, label in SPIDER_T4_COLS])
    for method, rows in methods:
        means = ["mean"]
        for k, _ in SPIDER_T4_COLS:
            vals = [r[k] for r in rows if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
            means.append(float(np.mean(vals)) if vals else None)
        ws.append([method, len(rows)] + means[1:])

    # OmniRetarget T2 cross-method
    ws = wb.create_sheet(title="omni_t2_mean")
    ws.append(["method", "n"] + [label for _, label in OMNIRETARGET_T2_COLS])
    for method, rows in methods:
        means = []
        for k, _ in OMNIRETARGET_T2_COLS:
            vals = [r[k] for r in rows if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
            means.append(float(np.mean(vals)) if vals else None)
        ws.append([method, len(rows)] + means)

    # by_object sheet for first method
    if methods:
        method, rows = methods[0]
        ws = wb.create_sheet(title=f"by_object_{method}"[:31])
        ws.append(["object_class", "n_cases"] + [label for _, label in SPIDER_T4_COLS])
        by_obj: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            by_obj.setdefault(_object_class(str(r.get("case", ""))), []).append(r)
        for obj_cls, sub in sorted(by_obj.items()):
            means = []
            for k, _ in SPIDER_T4_COLS:
                vals = [r[k] for r in sub if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
                means.append(float(np.mean(vals)) if vals else None)
            ws.append([obj_cls, len(sub)] + means)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--method",
        action="append",
        required=True,
        help="Method label (one per --comparison). Used in tables/xlsx.",
    )
    ap.add_argument(
        "--comparison",
        action="append",
        required=True,
        help="Per-experiment comparison.csv path. Provide one per --method.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory (will create tables/, per_case/).",
    )
    args = ap.parse_args()

    if len(args.method) != len(args.comparison):
        print(
            f"[unified_eval] --method count ({len(args.method)}) must equal --comparison count "
            f"({len(args.comparison)})",
            file=sys.stderr,
        )
        return 2

    out_root: Path = args.out
    tables_dir = out_root / "tables"
    per_case_dir = out_root / "per_case"
    tables_dir.mkdir(parents=True, exist_ok=True)
    per_case_dir.mkdir(parents=True, exist_ok=True)

    methods: list[tuple[str, list[dict[str, Any]]]] = []
    for method, comp_path in zip(args.method, args.comparison):
        p = Path(comp_path)
        if not p.is_file():
            print(f"[unified_eval] comparison file missing: {p}", file=sys.stderr)
            return 2
        rows = _read_comparison(p)
        if not rows:
            print(f"[unified_eval] empty comparison: {p}", file=sys.stderr)
            continue
        methods.append((method, rows))

        # per-case JSON
        method_dir = per_case_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        for r in rows:
            case = str(r.get("case", "unknown"))
            short = _short_case(case, method_prefix=f"{method}_")
            with (method_dir / f"{short}.json").open("w") as f:
                json.dump(r, f, indent=2, default=str)

    # Per-method markdown tables (each method gets its own concatenated tables.md)
    for method, rows in methods:
        md_parts = [f"# Paper-aligned metrics — `{method}` ({len(rows)} cases)", ""]
        md_parts.append(_md_table(rows, SPIDER_T4_COLS, method=method, title="SPIDER Table 4 alignment"))
        md_parts.append(_md_table(rows, OMNIRETARGET_T2_COLS, method=method, title="OmniRetarget Table II alignment"))
        md_parts.append(_md_table(rows, DYNARETARGET_T5_COLS, method=method, title="DynaRetarget Table V alignment"))
        md_parts.append(_md_table(rows, CORE4D_COLS, method=method, title="CORE4D 协作 自定义 (stability / contact)"))
        md_parts.append(_table_failure_modes(rows, method))
        (tables_dir / f"table_{method}.md").write_text("\n".join(md_parts))

    # Standalone tables (default = first method, but include cross-method comparison if >=2)
    if methods:
        first_method, first_rows = methods[0]
        (tables_dir / "table_spider_t4.md").write_text(
            f"# SPIDER Table 4 — Paper-Aligned Metrics\n\n"
            + _md_table(first_rows, SPIDER_T4_COLS, method=first_method, title="")
        )
        (tables_dir / "table_omniretarget_t2.md").write_text(
            f"# OmniRetarget Table II — Paper-Aligned Metrics\n\n"
            + _md_table(first_rows, OMNIRETARGET_T2_COLS, method=first_method, title="")
        )
        (tables_dir / "table_dynaretarget_t5.md").write_text(
            f"# DynaRetarget Table V — Paper-Aligned Metrics\n\n"
            + _md_table(first_rows, DYNARETARGET_T5_COLS, method=first_method, title="")
        )
        (tables_dir / "table_core4d_collab.md").write_text(
            f"# CORE4D 协作 — Stability / Contact 自定义指标\n\n"
            + _md_table(first_rows, CORE4D_COLS, method=first_method, title="")
            + "\n\n"
            + _table_failure_modes(first_rows, first_method)
        )

    # Multi-method xlsx
    _write_xlsx(methods, tables_dir / "table_paper_all.xlsx")

    # E019 P1: Tab.5 cross-method comparison (intersection of cases)
    if len(methods) >= 2:
        tab5 = ["# Tab.5 — 跨方法对比 (E019 P1)", ""]
        tab5.append(
            "本表对比 **同一组 case** 上各方法的 paper-aligned 指标 mean，按指标分组。"
        )
        tab5.append(
            "**说明**：N（交集）= 各方法 case 集合交集 size，自动由 unified_eval 计算。"
            "spider physical 当前 13 case；holosoma v2 kinematic 当前 12 case "
            "(desk021_p1 SOCP infeasible，详见 log/20b)；交集 N=12。"
        )
        tab5.append("")
        tab5.append(
            _method_comparison_table(methods, SPIDER_T4_COLS, "SPIDER Table 4 (Body Tracking)")
        )
        tab5.append(
            _method_comparison_table(
                methods, OMNIRETARGET_T2_COLS, "OmniRetarget Table II (Physical Plausibility)"
            )
        )
        tab5.append(
            _method_comparison_table(
                methods, DYNARETARGET_T5_COLS, "DynaRetarget Table V (Object Tracking + Smoothness)"
            )
        )
        tab5.append(
            _method_comparison_table(methods, CORE4D_COLS, "CORE4D 协作 自定义 (Stability / Contact)")
        )
        (tables_dir / "table_method_comparison.md").write_text("\n".join(tab5))

    # Top-level index
    index_md = ["# Unified paper-aligned evaluation tables", ""]
    index_md.append(f"Generated by `scripts/eval/unified_eval.py` — methods evaluated: {len(methods)}")
    index_md.append("")
    for method, rows in methods:
        index_md.append(f"- `{method}` ({len(rows)} cases) → `tables/table_{method}.md`")
    index_md.extend([
        "",
        "## Standalone (paper-aligned) tables",
        "- SPIDER Table 4 → `tables/table_spider_t4.md`",
        "- OmniRetarget Table II → `tables/table_omniretarget_t2.md`",
        "- DynaRetarget Table V → `tables/table_dynaretarget_t5.md`",
        "- CORE4D 协作 (stability / contact) → `tables/table_core4d_collab.md`",
        "- Multi-method xlsx → `tables/table_paper_all.xlsx`",
        "",
        "## 跨方法对比 (E019 P1)",
        "- Tab.5 method comparison → `tables/table_method_comparison.md`"
        + (" (生成)" if len(methods) >= 2 else " (需 ≥2 method)"),
        "",
        "## Per-case JSON",
        f"- `per_case/{{method}}/{{case}}.json` (共 {sum(len(r) for _, r in methods)} 条)",
        "",
        "See `docs/eval_metrics.md` for the canonical metric definitions (P1 章节含 28cm contact preservation 与 Tab.5 caveat).",
        "",
    ])
    (out_root / "INDEX.md").write_text("\n".join(index_md))
    print(f"[unified_eval] wrote tables to {tables_dir}")
    print(f"[unified_eval] wrote per_case JSON to {per_case_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
