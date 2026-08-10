#!/usr/bin/env python3
"""Generate the preregistered E192 C1-C7 report from paired A0/A2 deltas."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/E192"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/E189"))
import e192_common as C  # noqa: E402
import e189_common as E189  # noqa: E402


def f(value: Any) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else math.nan
    except (TypeError, ValueError):
        return math.nan


def truth(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [f(row.get(key)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return statistics.fmean(values) if values else math.nan


def bootstrap_did(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    rng = np.random.default_rng(0)
    b24 = [row for row in rows if row["object_key"] == "box024"]
    b04 = [row for row in rows if row["object_key"] == "box004"]
    if not b24 or not b04:
        return {"estimate": math.nan, "ci_low": math.nan, "ci_high": math.nan}

    def sample(group: list[dict[str, Any]]) -> float:
        return mean(group, key)

    estimate = sample(b24) - sample(b04)
    draws = []
    for _ in range(10_000):
        s24 = [b24[i] for i in rng.integers(0, len(b24), len(b24))]
        s04 = [b04[i] for i in rng.integers(0, len(b04), len(b04))]
        a, b = sample(s24), sample(s04)
        if math.isfinite(a) and math.isfinite(b):
            draws.append(a - b)
    lo, hi = np.quantile(draws, [0.025, 0.975]) if draws else (math.nan, math.nan)
    return {"estimate": estimate, "ci_low": float(lo), "ci_high": float(hi)}


def gate_transitions(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for gate in E189.ALL_GATES:
        counts = {"PASS_TO_PASS": 0, "PASS_TO_FAIL": 0, "FAIL_TO_PASS": 0, "FAIL_TO_FAIL": 0}
        for row in rows:
            old = truth(row.get(f"a0_{gate}_gate_pass"))
            new = truth(row.get(f"a2_{gate}_gate_pass"))
            key = "PASS_TO_PASS" if old and new else "PASS_TO_FAIL" if old else "FAIL_TO_PASS" if new else "FAIL_TO_FAIL"
            counts[key] += 1
        result[gate] = counts
    return result


def verdict(ok: bool | None) -> str:
    return "PASS" if ok is True else ("FAIL" if ok is False else "UNVERIFIED")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("canary", "full"), nargs="?", default="full")
    args = parser.parse_args()
    out_dir = C.RESULTS / f"s6_downstream/eval/{args.stage}"
    rows = C.read_tsv(out_dir / "e192_paired_deltas.tsv")
    pen = "delta_hand_object_physics_penetration_3mm_frame_frac"
    for row in rows:
        row["reduction_penetration"] = -f(row.get(pen))
    did = bootstrap_did(rows, "reduction_penetration")
    b24 = [row for row in rows if row["object_key"] == "box024"]
    b04 = [row for row in rows if row["object_key"] == "box004"]

    improved_pen_24 = sum(f(row.get(pen)) < 0 for row in b24)
    c1 = len(b24) == 9 and mean(b24, "a2_hand_object_physics_penetration_3mm_frame_frac") <= 0.20 and improved_pen_24 >= 7
    c2_numeric = len(b24) == 9 and mean(b24, "a2_hand_object_physics_contact_3mm_in_mask_frac") >= 0.3079 and mean(b24, "a2_hand_object_physics_contact_in_mask_frac") >= 0.75
    visual_path = C.RESULTS / "s6_downstream/render/keyframes/visual_review.tsv"
    visual_rows = C.read_tsv(visual_path) if visual_path.is_file() else []
    required_visual_cases = {
        "box024_20231011_026_p1", "box024_20231011_027_p2",
        "box024_20231011_028_p2",
    }
    reviewed_visual_cases = {row.get("case_id", "") for row in visual_rows}
    visual_complete = required_visual_cases <= reviewed_visual_cases
    confirmed_contact_loss = any(
        row.get("case_id", "") in C.CASES["box024"]
        and row.get("load_bearing_contact") == "confirmed_loss"
        for row in visual_rows
    )
    c2 = c2_numeric and visual_complete and not confirmed_contact_loss
    c3 = math.isfinite(did["estimate"]) and did["estimate"] >= 0.10 and did["ci_low"] > 0.0

    transitions_all = gate_transitions(rows)
    transitions_04 = gate_transitions(b04)
    box004_pass_to_fail = sum(item["PASS_TO_FAIL"] for item in transitions_04.values())
    max_leg_delta_24 = max((f(row.get("delta_leg_penetration_frac")) for row in b24), default=math.nan)
    c4 = box004_pass_to_fail == 0 and max_leg_delta_24 <= 0.05

    diagnostic_keys = list(C.GATE_DIAGNOSTIC_KEYS) + [
        "cem_hand_gate_nonfallback_ticks",
        "cem_hand_gate_nonfallback_selected_min_sdf_m",
        "cem_hand_gate_nonfallback_hard_floor_violation_count",
        "cem_hand_gate_nonfallback_hard_floor_violation_frac",
    ]
    missing_diagnostics = [
        f"{row['case_id']}:{key}" for row in rows for key in diagnostic_keys
        if not math.isfinite(f(row.get(f"a2_{key}")))
    ]
    c5 = len(rows) == 15 and not missing_diagnostics

    fixed15_delta = "delta_hand_gate_fixed_depth_15mm_frame_frac"
    fixed15_improved = sum(f(row.get(fixed15_delta)) < 0 for row in b24)
    floor_violation_count = sum(int(f(row.get("a2_cem_hand_gate_nonfallback_hard_floor_violation_count"))) for row in rows)
    c6 = (
        len(b24) == 9
        and mean(b24, "a2_hand_gate_fixed_depth_15mm_frame_frac") < mean(b24, "a0_hand_gate_fixed_depth_15mm_frame_frac")
        and fixed15_improved >= 7
        and floor_violation_count == 0
    )

    gate_health: dict[str, dict[str, float | bool]] = {}
    for object_key, group in (("box024", b24), ("box004", b04)):
        hand_valid = mean(group, "a2_cem_hand_gate_valid_frac")
        fallback_a0 = mean(group, "a0_cem_gate_fallback_used")
        fallback_a2 = mean(group, "a2_cem_gate_fallback_used")
        gate_health[object_key] = {
            "a2_hand_gate_valid_frac": hand_valid,
            "a0_gate_fallback_used": fallback_a0,
            "a2_gate_fallback_used": fallback_a2,
            "fallback_limit": fallback_a0 + 0.15,
            "pass": hand_valid >= 0.60 and fallback_a2 <= fallback_a0 + 0.15,
        }
    c7 = len(rows) == 15 and all(bool(item["pass"]) for item in gate_health.values())
    claims = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "C6": c6, "C7": c7}

    canary_path = C.RESULTS / "s6_downstream/eval/canary/e192_canary_stop_loss.json"
    canary = json.loads(canary_path.read_text(encoding="utf-8")) if canary_path.is_file() else {}
    canary_stop_loss = canary.get("stop_loss", {})
    canary_collapse = (
        bool(canary_stop_loss.get("triggered"))
        and canary_stop_loss.get("decision") == "INCONCLUSIVE_GATE_COLLAPSE"
    )
    if not rows or len(rows) != 15:
        diagnostic_decision = "INCOMPLETE"
    elif not c7:
        diagnostic_decision = "INCONCLUSIVE_GATE_COLLAPSE"
    elif c1 and c2 and not c6:
        diagnostic_decision = "MECHANISM_MISMATCH"
    elif c1 and c2 and c3 and c4 and c6:
        diagnostic_decision = "BOX024_SPECIFIC_THRESHOLD_EFFECT"
    elif c1 and c2 and c4 and c6:
        diagnostic_decision = "GLOBALLY_SUBOPTIMAL"
    elif not c1:
        diagnostic_decision = "THRESHOLD_POLICY_NOT_EFFECTIVE"
    else:
        diagnostic_decision = "INCOMPLETE_CLAIM_SET"
    decision = "INCONCLUSIVE_GATE_COLLAPSE" if canary_collapse else diagnostic_decision

    object_summary = {}
    summary_metrics = (
        "hand_object_physics_penetration_3mm_frame_frac",
        "hand_object_physics_contact_3mm_in_mask_frac",
        "hand_object_physics_contact_in_mask_frac",
        "leg_penetration_frac",
        "track_obj_pos_err_cm_mean",
        "track_obj_ori_err_deg_mean",
        "hand_gate_fixed_depth_10mm_frame_frac",
        "hand_gate_fixed_depth_15mm_frame_frac",
        "hand_gate_fixed_depth_20mm_frame_frac",
    )
    for object_key, group in (("box024", b24), ("box004", b04)):
        object_summary[object_key] = {
            key: {"a0": mean(group, f"a0_{key}"), "a2": mean(group, f"a2_{key}"), "delta": mean(group, f"delta_{key}")}
            for key in summary_metrics
        }

    evidence = {
        "C1": {"box024_a2_penetration": mean(b24, "a2_hand_object_physics_penetration_3mm_frame_frac"), "improved_cases": improved_pen_24, "required_cases": 7},
        "C2": {
            "numeric_pass": c2_numeric,
            "contact_3mm_in_mask": mean(b24, "a2_hand_object_physics_contact_3mm_in_mask_frac"),
            "contact_in_mask": mean(b24, "a2_hand_object_physics_contact_in_mask_frac"),
            "visual_review": C.rel(visual_path),
            "required_visual_cases": sorted(required_visual_cases),
            "reviewed_visual_cases": sorted(reviewed_visual_cases),
            "visual_complete": visual_complete,
            "confirmed_contact_loss": confirmed_contact_loss,
        },
        "C3": did,
        "C4": {"box004_pass_to_fail": box004_pass_to_fail, "box024_max_leg_penetration_delta": max_leg_delta_24},
        "C5": {"missing_diagnostics": missing_diagnostics},
        "C6": {"fixed15_improved_cases": fixed15_improved, "required_cases": 7, "nonfallback_hard_floor_violation_count": floor_violation_count},
        "C7": gate_health,
    }

    lines = [
        f"# E192 hand-gate A0 vs A2 ({args.stage})", "",
        f"Governance decision: **{decision}**", "",
        f"Full-only diagnostic decision: **{diagnostic_decision}**", "",
        "The governance decision retains the preregistered canary collapse. The user waiver authorized Full execution but did not reclassify the canary as PASS.", "",
        "## Claims", "", "| Claim | Status |", "|---|---|",
    ]
    lines.extend(f"| {key} | {verdict(value)} |" for key, value in claims.items())
    lines += [
        "", "## Signed DiD", "",
        f"- penetration reduction(box024) − reduction(box004): estimate `{did['estimate']:.6f}`; bootstrap 95% CI `[{did['ci_low']:.6f}, {did['ci_high']:.6f}]`.",
        "", "## Gate health", "", "| Object | A2 hand valid | A0 fallback | A2 fallback | Limit | Status |", "|---|---:|---:|---:|---:|---|",
    ]
    for object_key, item in gate_health.items():
        lines.append(f"| {object_key} | {item['a2_hand_gate_valid_frac']:.4f} | {item['a0_gate_fallback_used']:.4f} | {item['a2_gate_fallback_used']:.4f} | {item['fallback_limit']:.4f} | {verdict(bool(item['pass']))} |")
    lines += ["", "## Gate transitions", "", "| Gate | PASS→PASS | PASS→FAIL | FAIL→PASS | FAIL→FAIL |", "|---|---:|---:|---:|---:|"]
    for gate, counts in transitions_all.items():
        lines.append(f"| {gate} | {counts['PASS_TO_PASS']} | {counts['PASS_TO_FAIL']} | {counts['FAIL_TO_PASS']} | {counts['FAIL_TO_FAIL']} |")
    lines += [
        "", "No semantic gate aggregation is used for C4/C5/C7.",
        "", "## Visualisation", "",
        f"Visual review complete: `{C.rel(visual_path)}`. Required cases reviewed={visual_complete}; confirmed box024 load-bearing contact loss={confirmed_contact_loss}. Numeric C2 is therefore {'retained' if c2 else 'downgraded'}.",
    ]
    (out_dir / "E192_arm_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    C.write_json(out_dir / "e192_claims.json", {
        "created_at": C.now(), "stage": args.stage, "claims": claims,
        "claim_evidence": evidence, "decision": decision,
        "diagnostic_decision": diagnostic_decision,
        "canary_stop_loss_retained": canary_collapse,
        "signed_did": did, "object_summary": object_summary,
        "gate_transitions": transitions_all,
    })
    print(f"E192 report: decision={decision} diagnostic={diagnostic_decision} claims={claims}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
