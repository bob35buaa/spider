#!/usr/bin/env python3
"""E206 P9 report: one workbook holding every number the arm comparison rests on.

Sheets (plan236 P9):
  funnel_summary   per-arm L1/L2/L3 + 12-gate counts, and the C5b verdict
  per_gate         per gate: mean / std / worst per arm + paired delta
  per_object       per object x arm, with n and whether n permits a recommendation
  paired_delta     per case: PRG - noPRG on the decision-relevant metrics
  vs_E174_desk007  the C5a like-for-like table
  rollout          the full 130-row scored TSV

Deliberate reporting choices:
  * `worst` is always present -- rule 5 forbids reporting mean alone.
  * root_pos also carries a median, because two L1 fall cases (root error ~15 m)
    dominate its mean; hiding that behind a mean would be metric deception, and
    dropping them would be selective reporting. Both are shown.

Usage:
    .venv/bin/python .../gen_E206_two_arm_workbook.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E206"))
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E201"))

import e206_common as C  # noqa: E402
import funnel_config as FC  # noqa: E402

EVAL = C.S6_DIR / "eval/two_arm"
OUT = EVAL / "e206_two_arm.xlsx"
DECISION_FIELDS = [
    "leg_penetration_frac",
    "hand_object_physics_penetration_3mm_frame_frac",
    "hand_object_physics_contact_in_mask_frac",
    "hand_object_release_false_contact_3mm_frac",
    "body_z_err_p95_m",
    "track_obj_pos_err_cm_mean",
    "track_root_pos_err_cm_mean",
]


def f(v: Any) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else math.nan
    except (TypeError, ValueError):
        return math.nan


def desc(vals: list[float], *, lower_is_better: bool = True) -> dict[str, Any]:
    v = [x for x in vals if math.isfinite(x)]
    if not v:
        return {"n": 0}
    return {"n": len(v), "mean": round(statistics.mean(v), 4),
            "median": round(statistics.median(v), 4),
            "std": round(statistics.pstdev(v), 4) if len(v) > 1 else 0.0,
            "worst": round(max(v) if lower_is_better else min(v), 4)}


def main() -> int:
    rollout = pd.read_csv(EVAL / "e206_two_arm_rollout.tsv", sep="\t")
    rows = rollout.to_dict("records")
    by = {(str(r["arm"]), str(r["case_id"])): r for r in rows}
    cases = sorted({str(r["case_id"]) for r in rows})
    paired = [c for c in cases if all((a, c) in by for a in C.ARMS)]

    # ---- funnel_summary + C5b verdict -------------------------------------
    summary = []
    for arm in C.ARMS:
        rs = [by[(arm, c)] for c in paired]
        summary.append({
            "arm": arm, "n": len(rs),
            "L3_auto": sum(1 for r in rs if r["layer"] == "L3_auto"),
            "L2_review": sum(1 for r in rs if r["layer"] == "L2_review"),
            "L1_reject": sum(1 for r in rs if r["layer"] == "L1_reject"),
            "narrow_pass": sum(1 for r in rs if bool(r["narrow_pass"])),
            "gate12_pass": sum(1 for r in rs if bool(r["gate12_pass"])),
            "physics6_pass": sum(1 for r in rs if bool(r["physics6_pass"])),
            "tracking6_pass": sum(1 for r in rs if bool(r["tracking6_pass"])),
        })
    legpen = {}
    for arm in C.ARMS:
        ok = sum(1 for c in paired
                 if f(by[(arm, c)]["leg_penetration_frac"]) <= 0.20)
        legpen[arm] = 100.0 * ok / len(paired)
    legpen_delta_pp = legpen["prg"] - legpen["noprg"]
    l3_delta = summary[1]["L3_auto"] - summary[0]["L3_auto"]
    if legpen_delta_pp >= 10.0 and l3_delta >= 0:
        verdict = "PRG wins (C5b bar met)"
    elif legpen_delta_pp <= -10.0:
        verdict = "noPRG wins"
    else:
        verdict = "no separation"
    summary.append({
        "arm": "__C5b__", "n": len(paired),
        "L3_auto": l3_delta,
        "L2_review": "", "L1_reject": "",
        "narrow_pass": f"leg_pen narrow {legpen['noprg']:.1f}% -> {legpen['prg']:.1f}% "
                       f"({legpen_delta_pp:+.1f} pp, bar +10.0)",
        "gate12_pass": verdict, "physics6_pass": "", "tracking6_pass": "",
    })

    # ---- per_gate -----------------------------------------------------------
    per_gate = []
    for name, field, op, narrow, wide in FC.BANDED_GATES:
        e: dict[str, Any] = {"gate": name, "field": field, "op": op,
                             "narrow": narrow, "wide": wide}
        lower = op == "<="
        for arm in C.ARMS:
            d = desc([f(by[(arm, c)][field]) for c in paired], lower_is_better=lower)
            for k, v in d.items():
                e[f"{arm}_{k}"] = v
            e[f"{arm}_narrow_pass_pct"] = round(100.0 * sum(
                1 for c in paired if FC.passes(op, f(by[(arm, c)][field]), narrow)
            ) / len(paired), 1)
        deltas = [f(by[("prg", c)][field]) - f(by[("noprg", c)][field]) for c in paired]
        dd = desc(deltas, lower_is_better=lower)
        e["paired_delta_mean"] = dd.get("mean")
        e["paired_delta_median"] = dd.get("median")
        e["paired_delta_std"] = dd.get("std")
        e["narrow_pass_delta_pp"] = round(
            e["prg_narrow_pass_pct"] - e["noprg_narrow_pass_pct"], 1)
        per_gate.append(e)

    # ---- per_object ---------------------------------------------------------
    per_object = []
    for obj in sorted({c.split("_")[0] for c in paired}):
        cs = [c for c in paired if c.split("_")[0] == obj]
        row: dict[str, Any] = {"object_key": obj, "n": len(cs),
                               "per_object_recommendation_allowed": len(cs) >= 3}
        for arm in C.ARMS:
            rs = [by[(arm, c)] for c in cs]
            row[f"{arm}_L3"] = sum(1 for r in rs if r["layer"] == "L3_auto")
            row[f"{arm}_L1"] = sum(1 for r in rs if r["layer"] == "L1_reject")
            row[f"{arm}_gate12"] = sum(1 for r in rs if bool(r["gate12_pass"]))
            lp = desc([f(r["leg_penetration_frac"]) for r in rs])
            row[f"{arm}_legpen_mean"] = lp.get("mean")
            row[f"{arm}_legpen_worst"] = lp.get("worst")
            ct = desc([f(r["hand_object_physics_contact_in_mask_frac"]) for r in rs],
                      lower_is_better=False)
            row[f"{arm}_contact_mean"] = ct.get("mean")
            row[f"{arm}_contact_worst"] = ct.get("worst")
        per_object.append(row)

    # ---- paired_delta -------------------------------------------------------
    paired_rows = []
    for c in paired:
        r: dict[str, Any] = {"case_id": c, "object_key": c.split("_")[0],
                             "noprg_layer": by[("noprg", c)]["layer"],
                             "prg_layer": by[("prg", c)]["layer"]}
        for field in DECISION_FIELDS:
            r[f"d_{field}"] = round(
                f(by[("prg", c)][field]) - f(by[("noprg", c)][field]), 4)
        paired_rows.append(r)

    sheets: dict[str, pd.DataFrame] = {
        "funnel_summary": pd.DataFrame(summary),
        "per_gate": pd.DataFrame(per_gate),
        "per_object": pd.DataFrame(per_object),
        "paired_delta": pd.DataFrame(paired_rows),
        "rollout": rollout,
    }
    c5a = EVAL / "c5a_vs_e174_desk007.tsv"
    if c5a.is_file():
        sheets["vs_E174_desk007"] = pd.read_csv(c5a, sep="\t")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)

    print(json.dumps({
        "workbook": str(OUT.relative_to(REPO)),
        "sheets": {k: len(v) for k, v in sheets.items()},
        "paired_cases": len(paired),
        "C5b": {"leg_pen_narrow_pct": legpen,
                "delta_pp": round(legpen_delta_pp, 1),
                "L3_delta": l3_delta, "verdict": verdict},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
