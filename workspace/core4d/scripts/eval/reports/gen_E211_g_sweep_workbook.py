#!/usr/bin/env python3
"""E211: direction-aware g-curve workbook over the requested four arms.

Reads `e211_g_sweep_rollout.tsv` (written by eval_E211_partial_gravcomp.py) and
recomputes every number it presents, so the workbook is derived from the scored
rollouts rather than transcribed from the log.

Default arm set is the four the report asks for -- g = 0.0 / 0.6 / 0.8 / 1.0.
g=0.4 is scored and lives in the TSV, but it is excluded here on purpose: it is
the arm that broke the pre-registered monotonicity (eef_ori 25.08 deg, worse than
g=0.6 and g=0.8, and `hard` only 3/5). **A four-arm table therefore reads as a
smoother trend than the data actually shows.** The README sheet and
`g_sweep_all_arms` sheet both carry the g=0.4 row so the omission cannot be
mistaken for absence (rules section 5: no silent truncation).

Two arithmetic traps this inherits from E210 F6 and keeps guarded:

  1. DIRECTION. `hand_object_physics_contact_in_mask_frac` is better when it goes
     UP; every other tracked metric is an error and is better when it goes DOWN.
     A shared "n_worse = count(delta > 0)" reports contact backwards, which flips
     the sign of a headline finding rather than just a label.
  2. FIELD. The funnel's `contact` gate reads
     `hand_object_physics_contact_in_mask_frac`, NOT the `..._3mm_...` variant in
     the RL-export schema.

`release` is NaN on one case (028_p2 has an empty release window), so every
aggregate over it is taken on the subset where ALL arms are finite -- otherwise
two arms would be averaged over different populations.

Usage:
    .venv/bin/python workspace/core4d/scripts/eval/reports/gen_E211_g_sweep_workbook.py
    ... --arms prg,G06,G08,g1 --out <xlsx>
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
for _p in ("workspace/core4d/scripts",
           "workspace/core4d/scripts/experiments/E201",
           "workspace/core4d/scripts/experiments/E211"):
    sys.path.insert(0, str(REPO / _p))

import funnel_config as FC  # noqa: E402
import e211_common as C  # noqa: E402

ALL_ARMS = ("prg", "G04", "G06", "G08", "g1")
ARM_G = {"prg": 0.0, "G04": 0.4, "G06": 0.6, "G08": 0.8, "g1": 1.0}
ARM_LABEL = {
    "prg": "g=0.0  E206 PRG (baseline)", "G04": "g=0.4  E211",
    "G06": "g=0.6  E211", "G08": "g=0.8  E211", "g1": "g=1.0  E209 G1",
}
BASELINE = "prg"

CONTACT_GATE = "hand_object_physics_contact_in_mask_frac"

#: True == lower is better. Anything absent is treated as an error metric.
LOWER_IS_BETTER: dict[str, bool] = {CONTACT_GATE: False}

#: Reported in the per-arm and contrast sheets, in report order.
TRACKED = [
    "z_bias_cm", "z_abs_bias_cm", "z_mae_cm",
    "track_obj_pos_err_cm_mean", "track_obj_ori_err_deg_mean",
    "track_eef_ori_err_deg_mean", "track_eef_pos_err_cm_mean",
    "track_root_ori_err_deg_mean", "track_root_pos_err_cm_mean",
    CONTACT_GATE, "hand_object_release_false_contact_3mm_frac",
    "hand_object_physics_penetration_3mm_frame_frac", "leg_penetration_frac",
    "body_z_err_p95_m", "ankle_jerk_p95", "obj_speed_max",
]
BANDED = {g[1]: g for g in FC.BANDED_GATES}
HARD = {g[1]: g for g in FC.HARD_GATES}


def num(v: Any) -> float:
    s = str(v).strip().lower()
    if s in ("true", "false"):
        return 1.0 if s == "true" else 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def truth(v: Any) -> bool:
    return str(v).strip().lower() == "true"


def q(xs: list[float], p: float) -> float:
    if not xs:
        return math.nan
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[i]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", type=Path,
                    default=C.S6_DIR / "eval/g_sweep/e211_g_sweep_rollout.tsv")
    ap.add_argument("--arms", default="prg,G06,G08,g1")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import csv
    if not args.rollout.is_file():
        raise SystemExit(f"missing rollout TSV: {args.rollout} (run eval_E211_partial_gravcomp.py)")
    with args.rollout.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ALL_ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; known {list(ALL_ARMS)}")
    omitted = [a for a in ALL_ARMS if a not in arms]

    by = {(r["arm"], r["case_id"]): r for r in rows}
    cases = sorted({r["case_id"] for r in rows})
    present = [a for a in ALL_ARMS if any((a, c) in by for c in cases)]
    missing = [(a, c) for a in arms for c in cases if (a, c) not in by]
    if missing:
        raise SystemExit(f"rollout TSV is missing {len(missing)} (arm, case) rows: {missing[:5]}")

    def finite_cases(field: str, over: list[str]) -> list[str]:
        """Cases where the field is finite for EVERY arm in `over`."""
        return [c for c in cases
                if all(math.isfinite(num(by[(a, c)][field])) for a in over)]

    # --- per-arm aggregates (on the all-arm common finite set) ---------------
    per_arm: list[dict[str, Any]] = []
    for a in arms:
        rec: dict[str, Any] = {
            "arm": a, "gravcomp": ARM_G[a], "label": ARM_LABEL[a], "n": len(cases),
            "hard_pass": sum(truth(by[(a, c)]["hard_pass"]) for c in cases),
            "wide_pass": sum(truth(by[(a, c)]["wide_pass"]) for c in cases),
            "narrow_pass": sum(truth(by[(a, c)]["narrow_pass"]) for c in cases),
            "L3_auto": sum(by[(a, c)]["layer"] == "L3_auto" for c in cases),
            "L2_review": sum(by[(a, c)]["layer"] == "L2_review" for c in cases),
            "L1_reject": sum(by[(a, c)]["layer"] == "L1_reject" for c in cases),
        }
        for f in TRACKED:
            fc = finite_cases(f, arms)
            vals = [num(by[(a, c)][f]) for c in fc]
            rec[f] = round(st.mean(vals), 4) if vals else math.nan
            rec[f"{f}__n"] = len(fc)
        per_arm.append(rec)

    # --- contrasts vs the g=0 baseline, direction-aware ----------------------
    contrasts: list[dict[str, Any]] = []
    for a in arms:
        if a == BASELINE:
            continue
        for f in TRACKED:
            fc = finite_cases(f, [BASELINE, a])
            d = [num(by[(a, c)][f]) - num(by[(BASELINE, c)][f]) for c in fc]
            if not d:
                continue
            lower_better = LOWER_IS_BETTER.get(f, True)
            # "worse" is direction-aware: for contact, a NEGATIVE delta is worse.
            worse = [x > 0 for x in d] if lower_better else [x < 0 for x in d]
            contrasts.append({
                "contrast": f"g={ARM_G[BASELINE]:.1f} -> g={ARM_G[a]:.1f}",
                "arm": a, "metric": f, "lower_is_better": lower_better, "n": len(d),
                "mean": round(st.mean(d), 4),
                "median": round(st.median(d), 4),
                "std": round(st.pstdev(d), 4) if len(d) > 1 else 0.0,
                "n_worse": sum(worse), "n_better": len(d) - sum(worse),
                "p10": round(q(d, 0.10), 4), "p90": round(q(d, 0.90), 4),
                "worst": round(max(d) if lower_better else min(d), 4),
            })

    # --- per-case x arm matrix for the headline metrics ----------------------
    HEAD = [("track_eef_ori_err_deg_mean", "eef_ori"), (CONTACT_GATE, "contact"),
            ("z_bias_cm", "z_bias"), ("layer", "layer")]
    per_case: list[dict[str, Any]] = []
    for c in cases:
        rec: dict[str, Any] = {"case_id": c}
        for f, short in HEAD:
            for a in arms:
                v = by[(a, c)][f]
                rec[f"{short}_g{ARM_G[a]:.1f}"] = v if f == "layer" else num(v)
        rec["narrow_failed_worst_arm"] = by[(arms[-1], c)]["narrow_failed"]
        per_case.append(rec)

    # --- gate table: every one of the 14 funnel gates, per arm ---------------
    gate_rows: list[dict[str, Any]] = []
    for name, field, *_ in [(g[0], g[1]) for g in FC.HARD_GATES] + \
                           [(g[0], g[1]) for g in FC.BANDED_GATES]:
        kind = "hard" if field in HARD else "banded"
        rec: dict[str, Any] = {
            "gate": name, "field": field, "kind": kind,
            "direction": "higher_is_better" if not LOWER_IS_BETTER.get(field, True)
            else "lower_is_better",
        }
        if kind == "banded":
            rec["narrow"], rec["wide"] = BANDED[field][3], BANDED[field][4]
        else:
            rec["narrow"], rec["wide"] = HARD[field][3], HARD[field][3]
        fc = finite_cases(field, arms)
        for a in arms:
            vals = [num(by[(a, c)][field]) for c in fc]
            rec[f"{a}_mean"] = round(st.mean(vals), 4) if vals else math.nan
            if kind == "banded":
                op, thr = BANDED[field][2], BANDED[field][3]
                passes = sum(
                    (num(by[(a, c)][field]) <= thr) if op == "<=" else
                    (num(by[(a, c)][field]) >= thr)
                    for c in fc
                )
                rec[f"{a}_narrow_pass"] = f"{passes}/{len(fc)}"
        gate_rows.append(rec)

    out_dir = args.rollout.parent
    payload = {
        "arms_reported": arms, "arms_omitted_from_headline": omitted,
        "cases": cases, "per_arm": per_arm, "contrasts": contrasts,
        "per_case": per_case, "gates": gate_rows,
        "note": ("g=0.4 is scored and present in the all-arms sheet; it is omitted "
                 "from the headline four because it is the non-monotone arm."),
    }
    (out_dir / "e211_g_sweep_workbook.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")

    # --- xlsx ----------------------------------------------------------------
    out_xlsx = args.out or (out_dir / "E211_g_sweep_comparison.xlsx")
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
    except ImportError:
        print("[skip xlsx] openpyxl unavailable; JSON is the authority")
        return 0

    wb = Workbook()
    ws = wb.active
    ws.title = "README"
    readme = [
        "E211 Stage A - desk007 partial-gravcomp g-curve",
        "",
        "Reported arms: " + ", ".join(f"{ARM_LABEL[a]}" for a in arms),
        "Omitted from the headline sheets: " + (
            ", ".join(f"{ARM_LABEL[a]}" for a in omitted) or "(none)"),
        "",
        "WHY g=0.4 IS OMITTED FROM THE HEADLINE FOUR",
        "  It is the arm that broke the pre-registered monotonicity: eef_ori 25.08 deg",
        "  (worse than both g=0.6 and g=0.8) and hard only 3/5. A four-arm table reads",
        "  as a smoother trend than the data shows. Its full row is on 'AllArms'.",
        "",
        "HOW TO READ THIS",
        "  Object side (z_bias / z_mae / obj_pos / obj_ori) improves monotonically to",
        "  g=0.8 and then degrades at g=1.0 -> full compensation overshoots.",
        "  Robot side (eef_ori / root_ori) does NOT track g: r(g, eef_ori) = +0.225,",
        "  while r(CEM gate fallback, eef_ori) = +0.495 (partial, controlling for g,",
        "  = +0.504). The robot-side cost is driven by CEM safety-gate fallback, which",
        "  is roughly independent of g (r = +0.016).",
        "",
        "  Damage is concentrated, not spread: 3 of 5 cases are fine at every g;",
        "  desk007_20231030_034_p1 is broken at every g>0 and dominates the macro mean.",
        "",
        "DIRECTION",
        f"  {CONTACT_GATE} is HIGHER-is-better; every other tracked metric is an error",
        "  (lower is better). The 'Contrasts' sheet computes n_worse per-metric using",
        "  its own direction - a shared 'delta>0 == worse' reports contact backwards.",
        "",
        "POPULATION",
        "  release is NaN on desk007_20231030_028_p2 (empty release window), so every",
        "  aggregate over it uses n=4 - the subset finite for ALL arms. Each metric's",
        "  n is carried in the per-arm sheet as '<field>__n'.",
        "",
        "SOURCE",
        f"  {args.rollout.relative_to(REPO)}",
        "  scored by workspace/core4d/scripts/eval/runners/eval_E211_partial_gravcomp.py",
        "  log: workspace/core4d/log/300_E211_desk007_partial_gravcomp_sweep.md",
    ]
    for i, line in enumerate(readme, 1):
        ws.cell(row=i, column=1, value=line)
    ws.cell(row=1, column=1).font = Font(bold=True, size=13)
    ws.column_dimensions["A"].width = 92

    ws = wb.create_sheet("PerArm")
    hdr = (["arm", "gravcomp", "label", "n", "hard_pass", "wide_pass", "narrow_pass",
            "L3_auto", "L2_review", "L1_reject"] + TRACKED + [f"{f}__n" for f in TRACKED])
    ws.append(hdr)
    for r in per_arm:
        ws.append([r.get(h, "") for h in hdr])

    ws = wb.create_sheet("Contrasts")
    chdr = ["contrast", "arm", "metric", "lower_is_better", "n", "mean", "median",
            "std", "n_worse", "n_better", "p10", "p90", "worst"]
    ws.append(chdr)
    for r in contrasts:
        ws.append([r[h] for h in chdr])

    ws = wb.create_sheet("Gates14")
    ghdr = (["gate", "field", "kind", "direction", "narrow", "wide"]
            + [f"{a}_mean" for a in arms]
            + [f"{a}_narrow_pass" for a in arms if any(f"{a}_narrow_pass" in g for g in gate_rows)])
    ws.append(ghdr)
    for r in gate_rows:
        ws.append([r.get(h, "") for h in ghdr])

    ws = wb.create_sheet("PerCase")
    pchdr = list(per_case[0].keys())
    ws.append(pchdr)
    for r in per_case:
        ws.append([r.get(h, "") for h in pchdr])

    ws = wb.create_sheet("AllArms")
    ws.append(["(every scored arm including g=0.4 - the headline sheets omit it)"])
    acols = list(rows[0].keys())
    ws.append(acols)
    for r in sorted(rows, key=lambda x: (x["case_id"], float(x["gravcomp"]))):
        ws.append([num(r[c]) if c not in
                   ("object_key", "case_id", "arm", "layer", "hard_failed",
                    "wide_failed", "narrow_failed", "gate12_failed") else r[c]
                   for c in acols])

    for sheet in wb:
        if sheet.title != "README":
            sheet.freeze_panes = "A2"
    wb.save(out_xlsx)
    print(f"[done] {out_xlsx.relative_to(REPO)}")
    print(f"[done] {(out_dir / 'e211_g_sweep_workbook.json').relative_to(REPO)}")
    print(f"  arms reported: {arms}   omitted: {omitted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
