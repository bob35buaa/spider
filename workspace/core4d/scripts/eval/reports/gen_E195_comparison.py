#!/usr/bin/env python3
"""Generate the preregistered E195 C1-C7 report from E192/E195 paired deltas."""

from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/E195"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/E189"))
import e189_common as E189  # noqa: E402
import e195_common as C  # noqa: E402


def number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def truth(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [number(row.get(key)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return statistics.fmean(values) if values else math.nan


def bootstrap_improvement(rows: list[dict[str, Any]]) -> dict[str, float]:
    rng = np.random.default_rng(0)
    b24 = [row for row in rows if row["object_key"] == "box024"]
    b04 = [row for row in rows if row["object_key"] == "box004"]

    def improvement(group: list[dict[str, Any]]) -> float:
        return -mean(group, "delta_hand_object_physics_penetration_3mm_frame_frac")

    imp24, imp04 = improvement(b24), improvement(b04)
    draws24, draws04, draws_did = [], [], []
    for _ in range(10_000):
        s24 = [b24[index] for index in rng.integers(0, len(b24), len(b24))]
        s04 = [b04[index] for index in rng.integers(0, len(b04), len(b04))]
        a, b = improvement(s24), improvement(s04)
        draws24.append(a); draws04.append(b); draws_did.append(a - b)

    def summary(estimate: float, draws: list[float]) -> dict[str, float]:
        low, high = np.quantile(draws, [0.025, 0.975])
        return {"estimate": estimate, "ci_low": float(low), "ci_high": float(high)}

    return {
        "box024": summary(imp24, draws24),
        "box004": summary(imp04, draws04),
        "signed_did": summary(imp24 - imp04, draws_did),
    }


def gate_transitions(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for gate in E189.ALL_GATES:
        counts = {"PASS_TO_PASS": 0, "PASS_TO_FAIL": 0, "FAIL_TO_PASS": 0, "FAIL_TO_FAIL": 0}
        for row in rows:
            old = truth(row.get(f"e192_{gate}_gate_pass"))
            new = truth(row.get(f"e195_{gate}_gate_pass"))
            key = "PASS_TO_PASS" if old and new else "PASS_TO_FAIL" if old else "FAIL_TO_PASS" if new else "FAIL_TO_FAIL"
            counts[key] += 1
        output[gate] = counts
    return output


def status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def main() -> int:
    out_dir = C.RESULTS / "s6_downstream/eval/full"
    rows = C.read_tsv(out_dir / "e195_paired_deltas.tsv")
    b24 = [row for row in rows if row["object_key"] == "box024"]
    b04 = [row for row in rows if row["object_key"] == "box004"]
    transitions = gate_transitions(rows)
    transitions04 = gate_transitions(b04)

    penetration_key = "hand_object_physics_penetration_3mm_frame_frac"
    improved24 = sum(number(row.get(f"delta_{penetration_key}")) < 0 for row in b24)
    e192_pen24 = mean(b24, f"e192_{penetration_key}")
    e195_pen24 = mean(b24, f"e195_{penetration_key}")
    c1 = len(b24) == 9 and e195_pen24 <= e192_pen24 - 0.03 and improved24 >= 6

    e192_contact3 = mean(b24, "e192_hand_object_physics_contact_3mm_in_mask_frac")
    e195_contact3 = mean(b24, "e195_hand_object_physics_contact_3mm_in_mask_frac")
    e195_contact = mean(b24, "e195_hand_object_physics_contact_in_mask_frac")
    c2_numeric = e195_contact3 >= e192_contact3 - 0.05 and e195_contact >= 0.75

    fixed12_improved = sum(number(row.get("delta_hand_gate_fixed_depth_12mm_frame_frac")) < 0 for row in b24)
    floor_violations = sum(
        int(number(row.get("e195_cem_hand_gate_nonfallback_hard_floor_violation_count")))
        for row in rows
    )
    c3 = len(b24) == 9 and fixed12_improved >= 6 and floor_violations == 0

    gate_health: dict[str, dict[str, float | bool]] = {}
    for object_key, group in (("box024", b24), ("box004", b04)):
        hand_valid = mean(group, "e195_cem_hand_gate_valid_frac")
        fallback_old = mean(group, "e192_cem_gate_fallback_used")
        fallback_new = mean(group, "e195_cem_gate_fallback_used")
        gate_health[object_key] = {
            "e195_hand_valid": hand_valid,
            "e192_fallback": fallback_old,
            "e195_fallback": fallback_new,
            "limit": fallback_old + 0.15,
            "pass": hand_valid >= 0.60 and fallback_new <= fallback_old + 0.15,
        }
    c4 = len(rows) == 15 and all(bool(item["pass"]) for item in gate_health.values())

    box004_pass_to_fail = sum(item["PASS_TO_FAIL"] for item in transitions04.values())
    max_leg_delta24 = max((number(row.get("delta_leg_penetration_frac")) for row in b24), default=math.nan)
    c5 = box004_pass_to_fail == 0 and max_leg_delta24 <= 0.05
    effects = bootstrap_improvement(rows) if len(rows) == 15 else {}
    c6 = len(rows) == 15

    required_visual = {
        "box024_20231011_026_p1",
        "box024_20231011_027_p2",
    }
    for row in rows:
        gate_migration = any(
            truth(row.get(f"e192_{gate}_gate_pass")) != truth(row.get(f"e195_{gate}_gate_pass"))
            for gate in E189.ALL_GATES
        )
        penetration_reverse = number(row.get(f"delta_{penetration_key}")) >= 0
        leg_increase = number(row.get("delta_leg_penetration_frac")) > 0
        if gate_migration or penetration_reverse or leg_increase:
            required_visual.add(row["case_id"])

    render_manifest = C.RESULTS / "s6_downstream/render/full/render_manifest.tsv"
    render_rows = C.read_tsv(render_manifest) if render_manifest.is_file() else []
    self_complete = len(render_rows) == 15 and all(row.get("e195_status") in {"rendered", "existing"} for row in render_rows)
    pair_complete = len(render_rows) == 15 and all(row.get("paired_status") in {"rendered", "existing"} for row in render_rows)
    visual_path = C.RESULTS / "s6_downstream/render/keyframes/visual_review.tsv"
    visual_rows = C.read_tsv(visual_path) if visual_path.is_file() else []
    reviewed = {row.get("case_id", "") for row in visual_rows if row.get("review_status") == "reviewed"}
    visual_complete = required_visual <= reviewed
    confirmed_loss = any(
        row.get("case_id", "") in C.CASES["box024"]
        and row.get("load_bearing_contact") == "confirmed_loss"
        for row in visual_rows
    )
    c2 = c2_numeric and visual_complete and not confirmed_loss
    c7 = len(rows) == 15 and self_complete and pair_complete and visual_complete
    claims = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "C6": c6, "C7": c7}

    if not c7:
        decision = "INCOMPLETE"
    elif not c5:
        decision = "SAFETY_REGRESSION"
    elif c1 and c3 and not c2:
        decision = "PENETRATION_CONTACT_TRADEOFF"
    elif not c4:
        decision = "GATE_LIMITED"
    elif all(claims[key] for key in ("C1", "C2", "C3", "C4", "C5", "C7")):
        decision = "STRICTER_GATE_EFFECTIVE"
    else:
        decision = "NO_ADDITIONAL_BENEFIT"

    object_summary = {}
    summary_metrics = (
        penetration_key,
        "hand_object_physics_contact_3mm_in_mask_frac",
        "hand_object_physics_contact_in_mask_frac",
        "leg_penetration_frac",
        "hand_gate_fixed_depth_8mm_frame_frac",
        "hand_gate_fixed_depth_12mm_frame_frac",
        "hand_gate_fixed_depth_15mm_frame_frac",
    )
    for object_key, group in (("box024", b24), ("box004", b04)):
        object_summary[object_key] = {
            key: {
                "e192": mean(group, f"e192_{key}"),
                "e195": mean(group, f"e195_{key}"),
                "delta": mean(group, f"delta_{key}"),
            }
            for key in summary_metrics
        }

    evidence = {
        "C1": {"e192_box024_penetration": e192_pen24, "e195_box024_penetration": e195_pen24, "improved_cases": improved24},
        "C2": {"numeric_pass": c2_numeric, "e192_contact3": e192_contact3, "e195_contact3": e195_contact3, "e195_contact": e195_contact, "confirmed_contact_loss": confirmed_loss},
        "C3": {"fixed12_improved_cases": fixed12_improved, "nonfallback_floor_violations": floor_violations},
        "C4": gate_health,
        "C5": {"box004_pass_to_fail": box004_pass_to_fail, "box024_max_leg_delta": max_leg_delta24},
        "C6": effects,
        "C7": {"self_complete": self_complete, "pair_complete": pair_complete, "required_visual_cases": sorted(required_visual), "reviewed_visual_cases": sorted(reviewed), "visual_complete": visual_complete},
    }

    lines = [
        "# E195 stricter hand-gate comparison", "",
        f"Decision: **{decision}**", "",
        "## Claims", "", "| Claim | Status |", "|---|---|",
    ]
    lines.extend(f"| {key} | {status(value)} |" for key, value in claims.items())
    if effects:
        lines += ["", "## Paired penetration effect", "", "| Object/effect | Estimate | 95% CI |", "|---|---:|---:|"]
        for key in ("box024", "box004", "signed_did"):
            item = effects[key]
            lines.append(f"| {key} | {item['estimate']:.6f} | [{item['ci_low']:.6f}, {item['ci_high']:.6f}] |")
    lines += ["", "## Gate health", "", "| Object | E195 hand valid | E192 fallback | E195 fallback | Limit | Status |", "|---|---:|---:|---:|---:|---|"]
    for object_key, item in gate_health.items():
        lines.append(f"| {object_key} | {item['e195_hand_valid']:.4f} | {item['e192_fallback']:.4f} | {item['e195_fallback']:.4f} | {item['limit']:.4f} | {status(bool(item['pass']))} |")
    lines += ["", "## Visual closure", "", f"Required cases: {', '.join(sorted(required_visual))}", "", f"Reviewed={visual_complete}; self videos={self_complete}; paired videos={pair_complete}; confirmed load-bearing contact loss={confirmed_loss}."]
    (out_dir / "E195_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    C.write_json(out_dir / "e195_claims.json", {
        "created_at": C.now(), "claims": claims, "claim_evidence": evidence,
        "decision": decision, "object_summary": object_summary,
        "gate_transitions": transitions,
    })
    print(f"E195 report: decision={decision} claims={claims}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

