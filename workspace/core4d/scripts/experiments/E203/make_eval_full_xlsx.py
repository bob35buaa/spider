#!/usr/bin/env python3
"""E203 eval_full → xlsx: v1-vs-v2 comparison + 14-gate metrics workbook.

Reads the artifacts produced by ``run_E203_p1_eval.py --object-keys ""`` under
``workspace/core4d/results/E203/s6_downstream/eval_full/`` and emits a single
styled ``.xlsx`` with these sheets:

  1. Summary          run counts, per-object aggregates, funnel layers, numeric-fail modes
  2. vs_v1_summary    per-metric matched E203(v2) vs prior PRG(v1) means + Δ + improved/n
  3. per_case         one row per task: id/object/variant + 14-gate (layer, hard/wide/narrow,
                      per-gate flags) + key metrics + prior-v1 value + Δ + improved (matched)
  4. per_object       per-object obj_pos/eef/fall/leg_pen distribution + gate pass counts

The per_case sheet is the join of e203_case_metrics.tsv (authoritative, carries
metrics AND the 14-gate columns) with the prior-v1 columns from
e203_vs_prg_comparison.tsv (keyed by physical case_id).

Usage:
    .venv/bin/python workspace/core4d/scripts/experiments/E203/make_eval_full_xlsx.py
    # optional: --eval-dir <path>  --out <path.xlsx>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

REPO = Path(__file__).resolve().parents[5]
DEFAULT_EVAL = REPO / "workspace/core4d/results/E203/s6_downstream/eval_full"

# Metrics carried into per_case + their better-direction (mirrors run_E203_p1_eval).
COMPARE_METRICS = {
    "track_root_pos_err_cm_mean": "lower",
    "track_root_ori_err_deg_mean": "lower",
    "track_eef_pos_err_cm_mean": "lower",
    "track_eef_ori_err_deg_mean": "lower",
    "track_obj_pos_err_cm_mean": "lower",
    "track_obj_ori_err_deg_mean": "lower",
    "body_z_err_p95_m": "lower",
    "hand_object_physics_contact_in_mask_frac": "higher",
    "hand_object_physics_penetration_3mm_frame_frac": "lower",
    "hand_object_release_false_contact_3mm_frac": "lower",
    "leg_penetration_frac": "lower",
    "ankle_jerk_p95": "lower",
    "obj_speed_max": "lower",
}

GATE_FLAGS = [
    "fall_gate_pass", "body_z_gate_pass", "contact_gate_pass", "release_gate_pass",
    "hand_penetration_gate_pass", "lower_body_gate_pass",
    "root_pos_gate_pass", "root_ori_gate_pass", "hand_pos_gate_pass",
    "hand_ori_gate_pass", "object_pos_gate_pass", "object_ori_gate_pass",
]

HDR_FILL = PatternFill("solid", fgColor="305496")
HDR_FONT = Font(bold=True, color="FFFFFF")
GOOD_FILL = PatternFill("solid", fgColor="C6EFCE")
BAD_FILL = PatternFill("solid", fgColor="FFC7CE")
SUB_FILL = PatternFill("solid", fgColor="D9E1F2")


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def truthy(x) -> bool:
    return str(x).strip().lower() in ("true", "1", "yes")


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


# --- sheet builders ----------------------------------------------------------

def build_per_case(metrics: pd.DataFrame, cmp: pd.DataFrame) -> pd.DataFrame:
    prior = {}
    for _, r in cmp.iterrows():
        prior[r["case_id"]] = r
    cols_id = ["case_id", "object_key", "retarget_variant_id", "target_task"]
    rows = []
    for _, m in metrics.iterrows():
        cid = m["case_id"]
        p = prior.get(cid)
        row = {
            "case_id": cid,
            "object_key": m.get("object_key", ""),
            "retarget_variant": m.get("retarget_variant_id", ""),
            "matched_v1": bool(p is not None and truthy(p.get("matched", ""))),
            "prior_exp": (p.get("prior_exp", "") if p is not None else ""),
            # 14-gate
            "funnel_layer": m.get("funnel_layer", ""),
            "funnel_14gate_pass": truthy(m.get("funnel_14gate_pass", "")),
            "funnel_hard_pass": truthy(m.get("funnel_hard_pass", "")),
            "funnel_wide_pass": truthy(m.get("funnel_wide_pass", "")),
            "funnel_narrow_pass": truthy(m.get("funnel_narrow_pass", "")),
            "funnel_hard_failed": m.get("funnel_hard_failed", ""),
            "funnel_narrow_failed": m.get("funnel_narrow_failed", ""),
            "numeric_release_pass": truthy(m.get("numeric_release_pass", "")),
        }
        for g in GATE_FLAGS:
            row[g] = truthy(m.get(g, ""))
        for metric in COMPARE_METRICS:
            v2 = num(m.get(metric))
            v1 = num(p.get(f"prior_{metric}")) if p is not None else None
            row[f"{metric}__v2"] = v2
            row[f"{metric}__v1"] = v1
            row[f"{metric}__delta"] = (v2 - v1) if (v2 is not None and v1 is not None) else None
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values(["object_key", "retarget_variant", "case_id"]).reset_index(drop=True)


def build_vs_v1_summary(summary: dict) -> pd.DataFrame:
    cmp = summary["comparison"]
    rows = []
    for metric, st in cmp["per_metric"].items():
        rows.append({
            "metric": metric,
            "direction": st["direction"],
            "E203_v2_mean": round(st["e203_mean"], 4) if st["e203_mean"] == st["e203_mean"] else None,
            "prior_v1_mean": round(st["prior_mean"], 4) if st["prior_mean"] == st["prior_mean"] else None,
            "mean_delta_v2_minus_v1": round(st["mean_delta"], 4) if st["mean_delta"] == st["mean_delta"] else None,
            "improved": st["improved"],
            "regressed": st["regressed"],
            "n": st["n"],
            "improved_pct": round(100 * st["improved"] / st["n"], 1) if st["n"] else None,
        })
    return pd.DataFrame(rows)


def build_per_object(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for obj, g in metrics.groupby("object_key"):
        op = [num(x) for x in g["track_obj_pos_err_cm_mean"]]
        op = [x for x in op if x is not None]
        ef = [num(x) for x in g["track_eef_pos_err_cm_mean"]]
        ef = [x for x in ef if x is not None]
        lp = [num(x) for x in g["leg_penetration_frac"]]
        lp = [x for x in lp if x is not None]
        fall = sum(1 for x in g["fall_flag"] if truthy(x)) if "fall_flag" in g else 0
        npass = sum(1 for x in g["numeric_release_pass"] if truthy(x))
        g14 = sum(1 for x in g["funnel_14gate_pass"] if truthy(x))
        rows.append({
            "object_key": obj,
            "n": len(g),
            "obj_pos_cm_mean": round(sum(op) / len(op), 1) if op else None,
            "obj_pos_cm_worst": round(max(op), 1) if op else None,
            "eef_pos_cm_mean": round(sum(ef) / len(ef), 1) if ef else None,
            "leg_pen_frac_mean": round(sum(lp) / len(lp), 3) if lp else None,
            "fall": fall,
            "numeric_6gate_pass": npass,
            "funnel_14gate_pass": g14,
        })
    df = pd.DataFrame(rows).sort_values("object_key").reset_index(drop=True)
    return df


def build_summary_sheet(summary: dict, per_object: pd.DataFrame) -> list[list]:
    c = summary["counts"]
    cmp = summary["comparison"]
    block = [
        ["E203 eval_full — CORE4D v2 orig retarget + CEM (full 171 tasks)"],
        ["generated_at", summary.get("generated_at", "")],
        ["metric_standard", summary.get("metric_standard_id", "")],
        [],
        ["Coverage"],
        ["evaluated", c["evaluated"], "manifest_rows", c["manifest_rows"],
         "errors", c["errors"], "not_ready", c["not_ready"]],
        ["numeric_6gate_pass", c["numeric_pass"], f"{100*c['numeric_pass']/c['evaluated']:.0f}%"],
        ["retarget_variants", json.dumps(summary["retarget_variant_counts"])],
        ["objects", json.dumps(summary["object_counts"])],
        [],
        ["14-gate funnel (E201)"],
        ["strict_pass_all_narrow", summary["funnel_14gate"]["strict_pass"]],
        ["layers", json.dumps(summary["funnel_14gate"]["layers"])],
        ["numeric_failure_modes", json.dumps(summary["numeric_failure_counts"])],
        [],
        ["vs v1 (matched)"],
        ["matched", cmp["matched_cases"], "of", cmp["e203_cases"],
         "sources", json.dumps(cmp["prior_sources"])],
    ]
    return block


# --- styling -----------------------------------------------------------------

def style_header(ws, ncols: int):
    for col in range(1, ncols + 1):
        cell = ws.cell(row=1, column=col)
        cell.fill = HDR_FILL
        cell.font = HDR_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.freeze_panes = "A2"


def autosize(ws, max_w: int = 42):
    for col in ws.columns:
        letter = get_column_letter(col[0].column)
        width = max((len(str(c.value)) for c in col if c.value is not None), default=8)
        ws.column_dimensions[letter].width = min(max(width + 2, 10), max_w)


def highlight_delta(ws, df: pd.DataFrame):
    """Color __delta columns green(improved)/red(regressed) per metric direction."""
    headers = [c.value for c in ws[1]]
    for metric, direction in COMPARE_METRICS.items():
        col_name = f"{metric}__delta"
        if col_name not in headers:
            continue
        cidx = headers.index(col_name) + 1
        for r in range(2, ws.max_row + 1):
            val = ws.cell(row=r, column=cidx).value
            if val is None or val == "":
                continue
            try:
                d = float(val)
            except (TypeError, ValueError):
                continue
            improved = (d < 0) if direction == "lower" else (d > 0)
            if d == 0:
                continue
            ws.cell(row=r, column=cidx).fill = GOOD_FILL if improved else BAD_FILL


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ed = args.eval_dir
    out = args.out or ed / "e203_eval_full_v1_vs_v2.xlsx"
    metrics = read_tsv(ed / "e203_case_metrics.tsv")
    cmp = read_tsv(ed / "e203_vs_prg_comparison.tsv")
    summary = json.loads((ed / "summary.json").read_text())

    per_case = build_per_case(metrics, cmp)
    vs_v1 = build_vs_v1_summary(summary)
    per_object = build_per_object(metrics)
    summary_block = build_summary_sheet(summary, per_object)

    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        # Sheet 1: Summary (manual block + per_object table below)
        pd.DataFrame(summary_block).to_excel(xl, sheet_name="Summary", index=False, header=False)
        startrow = len(summary_block) + 1
        per_object.to_excel(xl, sheet_name="Summary", index=False, startrow=startrow)
        vs_v1.to_excel(xl, sheet_name="vs_v1_summary", index=False)
        per_case.to_excel(xl, sheet_name="per_case", index=False)
        per_object.to_excel(xl, sheet_name="per_object", index=False)

        wb = xl.book
        # per_case styling
        ws = wb["per_case"]
        style_header(ws, per_case.shape[1])
        autosize(ws)
        highlight_delta(ws, per_case)
        # vs_v1_summary styling + delta highlight by direction
        ws2 = wb["vs_v1_summary"]
        style_header(ws2, vs_v1.shape[1])
        autosize(ws2)
        dcol = list(vs_v1.columns).index("mean_delta_v2_minus_v1") + 1
        for r in range(2, ws2.max_row + 1):
            direction = ws2.cell(row=r, column=list(vs_v1.columns).index("direction") + 1).value
            val = ws2.cell(row=r, column=dcol).value
            if val in (None, ""):
                continue
            improved = (val < 0) if direction == "lower" else (val > 0)
            if val != 0:
                ws2.cell(row=r, column=dcol).fill = GOOD_FILL if improved else BAD_FILL
        # per_object styling
        for name in ("per_object",):
            wsx = wb[name]
            style_header(wsx, per_object.shape[1])
            autosize(wsx)
        # Summary sheet: title + section headers bold
        wss = wb["Summary"]
        wss["A1"].font = Font(bold=True, size=13)
        for row in wss.iter_rows():
            first = row[0]
            if first.value in ("Coverage", "14-gate funnel (E201)", "vs v1 (matched)"):
                first.font = Font(bold=True)
                first.fill = SUB_FILL
        autosize(wss)
        # header on the embedded per_object table inside Summary
        for col in range(1, per_object.shape[1] + 1):
            hc = wss.cell(row=startrow + 1, column=col)
            hc.fill = HDR_FILL
            hc.font = HDR_FONT

    print(f"per_case rows: {len(per_case)}  matched: {int(per_case['matched_v1'].sum())}")
    print(f"objects: {len(per_object)}  vs_v1 metrics: {len(vs_v1)}")
    print(f"xlsx -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
