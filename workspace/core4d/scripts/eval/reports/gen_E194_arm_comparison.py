#!/usr/bin/env python3
"""E194 report: C1-C9 verdicts + the 2x2 interaction table, from the eval outputs.

Reads:
  s6_downstream/eval/<stage>/e194_paired_deltas.tsv
  s6_downstream/eval/<stage>/e194_arm_diff_summary.json
Writes:
  s6_downstream/eval/<stage>/E194_arm_comparison.md

Verdicts follow the pre-registered thresholds in
plan/220_E194_object_gravity_compensation_plan.md (C1-C9, C6/C7 main discriminant).
Baselines that C-claims compare against are the plan's A0 table (box004/box024).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/E194"))
import e194_common as C  # noqa: E402


def finite(v: Any) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else math.nan
    except (TypeError, ValueError):
        return math.nan


def mean(vals: list[float]) -> float:
    vals = [v for v in vals if math.isfinite(v)]
    return statistics.fmean(vals) if vals else math.nan


def by(deltas: list[dict[str, Any]], arm: str, object_key: str | None = None) -> list[dict[str, Any]]:
    return [d for d in deltas if d["arm"] == arm and (object_key is None or d["object_key"] == object_key)]


def fmt(v: float, n: int = 3) -> str:
    return f"{v:.{n}f}" if isinstance(v, float) and math.isfinite(v) else "—"


def verdict(ok: bool | None) -> str:
    return "✅ PASS" if ok is True else ("❌ FAIL" if ok is False else "⚠️ N/A")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", nargs="?", default="full", choices=("canary", "full"))
    args = parser.parse_args()
    eval_dir = C.RESULTS / f"s6_downstream/eval/{args.stage}"
    deltas = C.read_tsv(eval_dir / "e194_paired_deltas.tsv")
    diff = json.loads((eval_dir / "e194_arm_diff_summary.json").read_text())["arm_diff"]

    lines: list[str] = []
    L = lines.append
    L(f"# E194 Gravity-Compensation 2x2 — Arm Comparison ({args.stage})\n")
    L(f"_generated {C.now()} · A0=E172/E173 baseline · G1=gravcomp · G2=kp2500 · G3=both_\n")

    # --- 2x2 interaction table (penetration drop) ---------------------------
    L("## 2×2: 3mm hand→object penetration drop vs A0\n")
    L("| object | pen_drop G1 | pen_drop G2 | pen_drop G3 | C6 (G1−G2) | C7 (G3−G1) |")
    L("|---|---:|---:|---:|---:|---:|")
    for ok in ("box024", "box004"):
        s = diff.get(ok, {})
        L(f"| {ok} | {fmt(finite(s.get('pen_drop_G1_mean')))} | {fmt(finite(s.get('pen_drop_G2_mean')))} "
          f"| {fmt(finite(s.get('pen_drop_G3_mean')))} | {fmt(finite(s.get('C6_G1_minus_G2')))} "
          f"| {fmt(finite(s.get('C7_G3_minus_G1')))} |")
    L("")

    # --- C6 main discriminant ----------------------------------------------
    L("## C6 — main discriminant: penetration from position-error or servo-force?\n")
    for ok in ("box024", "box004"):
        c6 = finite(diff.get(ok, {}).get("C6_G1_minus_G2"))
        g1 = finite(diff.get(ok, {}).get("pen_drop_G1_mean"))
        g2 = finite(diff.get(ok, {}).get("pen_drop_G2_mean"))
        if math.isfinite(c6):
            if c6 >= 0.08:
                tag = "PENETRATION_FROM_SERVO_FORCE"
            elif abs(c6) < 0.03:
                tag = "PENETRATION_FROM_POSITION_ONLY"
            elif max(g1, g2) < 0.05:
                tag = "PENETRATION_NOT_FROM_B"
            else:
                tag = "INDETERMINATE"
        else:
            tag = "NO_DATA"
        L(f"- **{ok}**: C6 = {fmt(c6)} → `{tag}`")
    L("")

    # --- C7 compliance independent contribution ----------------------------
    L("## C7 — compliance (G3 vs G1) independent effect\n")
    for ok in ("box024", "box004"):
        c7 = finite(diff.get(ok, {}).get("C7_G3_minus_G1"))
        tag = "compliance_harmful" if math.isfinite(c7) and c7 <= -0.05 else (
            "kp_irrelevant_under_gravcomp" if math.isfinite(c7) and abs(c7) < 0.05 else "check")
        L(f"- **{ok}**: G3−G1 pen drop diff = {fmt(c7)} → `{tag}`")
    L("")

    # --- per-arm claim means (C1/C2/C3/C4/C5/C8/C9) -------------------------
    def arm_metric_mean(arm: str, ok: str, key: str, use_arm_value: bool = True) -> float:
        col = f"arm_{key}" if use_arm_value else f"delta_{key}"
        return mean([finite(d.get(col)) for d in by(deltas, arm, ok)])

    def a0_mean(ok: str, key: str) -> float:
        return mean([finite(d.get(f"a0_{key}")) for d in by(deltas, "G1", ok)])

    L("## C1–C5, C8, C9 per-arm summary\n")
    L("| claim | metric | object | A0 | G1 | G2 | G3 |")
    L("|---|---|---|---:|---:|---:|---:|")
    rows = [
        ("C1", "track_obj_z_err_m_lifted_mean"),
        ("C2", "track_obj_pos_err_cm_mean"),
        ("C3", "obj_side_z_asym_cm"),
        ("C4", "hand_object_physics_contact_3mm_in_mask_frac"),
        ("C5/C6", "hand_object_physics_penetration_3mm_frame_frac"),
        ("C8", "leg_penetration_frac"),
        ("C9", "qpos_jerk_l2_p95"),
    ]
    for claim, key in rows:
        for ok in ("box024", "box004"):
            L(f"| {claim} | {key} | {ok} | {fmt(a0_mean(ok, key))} "
              f"| {fmt(arm_metric_mean('G1', ok, key))} | {fmt(arm_metric_mean('G2', ok, key))} "
              f"| {fmt(arm_metric_mean('G3', ok, key))} |")
    L("")

    # --- explicit C1/C2/C4 verdicts (pre-registered thresholds) ------------
    L("## Pre-registered verdicts\n")
    for ok in ("box024", "box004"):
        z_g1 = abs(arm_metric_mean("G1", ok, "track_obj_z_err_m_lifted_mean"))
        z_g3 = abs(arm_metric_mean("G3", ok, "track_obj_z_err_m_lifted_mean"))
        z_g2 = abs(arm_metric_mean("G2", ok, "track_obj_z_err_m_lifted_mean"))
        c1 = (z_g1 <= 0.02 and z_g3 <= 0.02 and z_g2 <= 0.04) if math.isfinite(z_g1) else None
        pos_a0 = a0_mean(ok, "track_obj_pos_err_cm_mean")
        pos_g1 = arm_metric_mean("G1", ok, "track_obj_pos_err_cm_mean")
        drop = (pos_a0 - pos_g1) / pos_a0 if math.isfinite(pos_a0) and pos_a0 else math.nan
        c2 = (0.15 <= drop <= 0.45) if math.isfinite(drop) else None
        c4_thr = 0.3079 if ok == "box024" else 0.4507
        c4_g1 = arm_metric_mean("G1", ok, "hand_object_physics_contact_3mm_in_mask_frac")
        c4 = (c4_g1 >= c4_thr) if math.isfinite(c4_g1) else None
        L(f"### {ok}")
        L(f"- **C1** (sag fixed): |z| G1={fmt(z_g1)} G3={fmt(z_g3)} (≤0.02), G2={fmt(z_g2)} (≤0.04) → {verdict(c1)}")
        L(f"- **C2** (sag is small part): pos_err drop = {fmt(drop*100 if math.isfinite(drop) else math.nan,1)}% "
          f"(expect 15–45%) → {verdict(c2)}")
        L(f"- **C4** (load-bearing contact kept): G1 3mm-contact = {fmt(c4_g1)} (≥{c4_thr}) → {verdict(c4)}")
        L("")

    out = eval_dir / "E194_arm_comparison.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {C.rel(out)}")
    print("\n".join(lines[:40]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
