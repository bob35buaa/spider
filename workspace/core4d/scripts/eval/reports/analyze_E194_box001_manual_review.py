#!/usr/bin/env python3
"""Re-evaluate completed E194 G1 box001 human review against E173 PRG authority."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


CORE4D = Path(__file__).resolve().parents[3]
RESULTS = CORE4D / "results"
EVAL = RESULTS / "E194/s6_downstream/eval/full_g1_expansion"
PRG_AUTHORITY = RESULTS / "E173/s6_downstream/rl_export/box001_user_approved/box001_user_approved_source_rows.tsv"
G1_REVIEW = EVAL / "user_manual_review_filled.tsv"
PAIRED = EVAL / "e194_three_arm_paired_deltas.tsv"
CASE_OUTPUT = EVAL / "e194_box001_manual_review_case_comparison.tsv"
SUMMARY_OUTPUT = EVAL / "e194_box001_manual_review_summary.json"
EXCLUDED_CASE = "box001_20231023_110_p1"

METRICS = (
    ("track_obj_z_abs_err_cm_mean", "lower"),
    ("track_obj_pos_err_cm_mean", "lower"),
    ("track_obj_ori_err_deg_mean", "lower"),
    ("body_z_err_p95_m", "lower"),
    ("track_pelvis_z_err_terminal_m", "lower"),
    ("track_root_pos_err_cm_mean", "lower"),
    ("track_root_ori_err_deg_mean", "lower"),
    ("track_eef_pos_err_cm_mean", "lower"),
    ("track_eef_ori_err_deg_mean", "lower"),
    ("hand_object_physics_contact_3mm_in_mask_frac", "higher"),
    ("hand_object_physics_contact_in_mask_frac", "higher"),
    ("hand_object_release_false_contact_3mm_frac", "lower"),
    ("hand_object_physics_penetration_3mm_frame_frac", "lower"),
    ("leg_penetration_frac", "lower"),
)
GATES = ("fall", "body_z", "contact", "release", "hand_penetration", "lower_body",
         "root_pos", "root_ori", "hand_pos", "hand_ori", "object_pos", "object_ori")
TRANSITIONS = (
    "USE_TO_USE",
    "USE_TO_DO_NOT_USE",
    "DO_NOT_USE_TO_USE",
    "DO_NOT_USE_TO_DO_NOT_USE",
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_mcnemar(pass_to_fail: int, fail_to_pass: int) -> float:
    total = pass_to_fail + fail_to_pass
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, index) for index in range(min(pass_to_fail, fail_to_pass) + 1))
    return min(1.0, 2.0 * tail / (2**total))


def boolean(value: str) -> bool:
    return value.strip().lower() == "true"


def metric_summary(cases: list[str], paired: dict[str, dict[str, str]]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for metric_index, (metric, direction) in enumerate(METRICS):
        before: list[float] = []
        after: list[float] = []
        for case_id in cases:
            before_raw = paired[case_id][f"before_{metric}"]
            after_raw = paired[case_id][f"after_{metric}"]
            if before_raw == "" or after_raw == "":
                continue
            before.append(float(before_raw))
            after.append(float(after_raw))
        before_array = np.asarray(before, dtype=float)
        after_array = np.asarray(after, dtype=float)
        delta = after_array - before_array
        rng = np.random.default_rng(194 + metric_index)
        bootstrap = rng.choice(delta, size=(20000, len(delta)), replace=True).mean(axis=1)
        better = delta > 0 if direction == "higher" else delta < 0
        worse = delta < 0 if direction == "higher" else delta > 0
        result[metric] = {
            "direction": direction,
            "n": len(delta),
            "prg_mean": float(before_array.mean()),
            "g1_mean": float(after_array.mean()),
            "delta_mean": float(delta.mean()),
            "bootstrap_95ci_low": float(np.quantile(bootstrap, 0.025)),
            "bootstrap_95ci_high": float(np.quantile(bootstrap, 0.975)),
            "case_better": int(better.sum()),
            "case_worse": int(worse.sum()),
            "case_tie": int(len(delta) - better.sum() - worse.sum()),
        }
    return result


def scope_summary(
    cases: list[str],
    prg_use: set[str],
    g1_review: dict[str, dict[str, str]],
    paired: dict[str, dict[str, str]],
) -> dict[str, object]:
    transitions = Counter(
        f"{'USE' if case_id in prg_use else 'DO_NOT_USE'}_TO_{g1_review[case_id]['manual_use_decision']}"
        for case_id in cases
    )
    prg_use_count = sum(case_id in prg_use for case_id in cases)
    g1_use_count = sum(g1_review[case_id]["manual_use_decision"] == "USE" for case_id in cases)
    gate_summary: dict[str, dict[str, object]] = {}
    for gate in GATES:
        before = [boolean(paired[case_id][f"before_{gate}_gate_pass"]) for case_id in cases]
        after = [boolean(paired[case_id][f"after_{gate}_gate_pass"]) for case_id in cases]
        pass_to_fail = sum(left and not right for left, right in zip(before, after))
        fail_to_pass = sum(not left and right for left, right in zip(before, after))
        gate_summary[gate] = {
            "prg_pass": sum(before),
            "g1_pass": sum(after),
            "delta_pp": (sum(after) - sum(before)) / len(cases) * 100.0,
            "pass_to_fail": pass_to_fail,
            "fail_to_pass": fail_to_pass,
            "exact_mcnemar_p": exact_mcnemar(pass_to_fail, fail_to_pass),
        }
    transition_metric_deltas: dict[str, dict[str, object]] = {}
    for transition in TRANSITIONS:
        transition_cases = [
            case_id for case_id in cases
            if f"{'USE' if case_id in prg_use else 'DO_NOT_USE'}_TO_{g1_review[case_id]['manual_use_decision']}"
            == transition
        ]
        deltas: dict[str, float | None] = {}
        for metric, _ in METRICS:
            values = [
                float(paired[case_id][f"after_{metric}"]) - float(paired[case_id][f"before_{metric}"])
                for case_id in transition_cases
                if paired[case_id][f"before_{metric}"] != "" and paired[case_id][f"after_{metric}"] != ""
            ]
            deltas[metric] = float(np.mean(values)) if values else None
        transition_metric_deltas[transition] = {
            "n": len(transition_cases),
            "case_ids": transition_cases,
            "delta_mean": deltas,
        }
    return {
        "n": len(cases),
        "prg_use": prg_use_count,
        "g1_use": g1_use_count,
        "prg_use_rate": prg_use_count / len(cases),
        "g1_use_rate": g1_use_count / len(cases),
        "net_g1_minus_prg_use": g1_use_count - prg_use_count,
        "net_g1_minus_prg_use_pp": (g1_use_count - prg_use_count) / len(cases) * 100.0,
        "transitions": {transition: transitions[transition] for transition in TRANSITIONS},
        "agreement": transitions["USE_TO_USE"] + transitions["DO_NOT_USE_TO_DO_NOT_USE"],
        "churn": transitions["USE_TO_DO_NOT_USE"] + transitions["DO_NOT_USE_TO_USE"],
        "exact_mcnemar_p": exact_mcnemar(
            transitions["USE_TO_DO_NOT_USE"], transitions["DO_NOT_USE_TO_USE"]
        ),
        "metrics": metric_summary(cases, paired),
        "gates": gate_summary,
        "transition_metric_deltas": transition_metric_deltas,
    }


def main() -> int:
    prg_rows = read_tsv(PRG_AUTHORITY)
    g1_rows = read_tsv(G1_REVIEW)
    paired_rows = [
        row for row in read_tsv(PAIRED)
        if row["comparison"] == "PRG_to_G1" and row["object_key"] == "box001"
    ]
    prg_use = {row["case_id"] for row in prg_rows}
    g1_review = {row["case_id"]: row for row in g1_rows}
    paired = {row["case_id"]: row for row in paired_rows}
    if len(prg_rows) != len(prg_use) or len(prg_use) != 13:
        raise SystemExit("invalid PRG authority cardinality")
    if any(
        row["object_key"] != "box001"
        or row["manual_use_decision"] != "USE"
        or row["source_exp_id"] != "E173"
        for row in prg_rows
    ):
        raise SystemExit("invalid PRG authority content")
    if len(g1_rows) != len(g1_review) or len(g1_review) != 28 or set(g1_review) != set(paired):
        raise SystemExit("G1 review does not exactly cover paired box001 cases")
    if any(row["user_manual_review_status"] != "reviewed" for row in g1_rows):
        raise SystemExit("G1 review is incomplete")
    if any(
        row["manual_use_decision"] not in {"USE", "DO_NOT_USE"}
        or row["manual_quality_label"] not in {"CLEAN", "MINOR_ACCEPTABLE", "UNUSABLE"}
        for row in g1_rows
    ):
        raise SystemExit("G1 review contains an invalid decision or quality label")

    case_rows: list[dict[str, object]] = []
    for case_id in sorted(paired):
        prg_decision = "USE" if case_id in prg_use else "DO_NOT_USE"
        g1_decision = g1_review[case_id]["manual_use_decision"]
        row: dict[str, object] = {
            "case_id": case_id,
            "analysis_inclusion": "EXCLUDED_BY_USER" if case_id == EXCLUDED_CASE else "INCLUDED_PRIMARY_27",
            "prg_authoritative_decision": prg_decision,
            "g1_manual_decision": g1_decision,
            "manual_migration": f"{prg_decision}_TO_{g1_decision}",
            "g1_quality_label": g1_review[case_id]["manual_quality_label"],
            "g1_reviewer": g1_review[case_id]["manual_reviewer"],
            "g1_reviewed_at": g1_review[case_id]["manual_reviewed_at"],
            "prg_12gate_pass": paired[case_id]["before_numeric_release_pass_12gate"],
            "g1_12gate_pass": paired[case_id]["after_numeric_release_pass_12gate"],
            "prg_numeric_failure_modes": paired[case_id]["before_numeric_failure_modes"],
            "g1_numeric_failure_modes": paired[case_id]["after_numeric_failure_modes"],
        }
        for metric, _ in METRICS:
            before = paired[case_id][f"before_{metric}"]
            after = paired[case_id][f"after_{metric}"]
            row[f"delta_{metric}"] = "" if before == "" or after == "" else float(after) - float(before)
        case_rows.append(row)
    write_tsv(CASE_OUTPUT, case_rows)

    all_cases = sorted(paired)
    primary_cases = [case_id for case_id in all_cases if case_id != EXCLUDED_CASE]
    summary = {
        "status": "pass",
        "decision": "NOT_COMPREHENSIVE_IMPROVEMENT",
        "recommended_policy": "CASE_LEVEL_PRG_G1_SELECTION",
        "excluded_case": EXCLUDED_CASE,
        "source_sha256": {
            "prg_authority": sha256(PRG_AUTHORITY),
            "g1_review": sha256(G1_REVIEW),
            "paired_metrics": sha256(PAIRED),
        },
        "primary_27": scope_summary(primary_cases, prg_use, g1_review, paired),
        "all_28_sensitivity": scope_summary(all_cases, prg_use, g1_review, paired),
    }
    SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(CASE_OUTPUT.relative_to(CORE4D.parent.parent))
    print(SUMMARY_OUTPUT.relative_to(CORE4D.parent.parent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
