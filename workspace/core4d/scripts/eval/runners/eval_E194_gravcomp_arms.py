#!/usr/bin/env python3
"""E194 four-cell (A0/G1/G2/G3) paired evaluation for the gravity-comp 2x2.

Scores G1/G2/G3 rollouts with the EXACT machinery that produced the A0 baselines
(E172/E173 `evaluate_row`), so arm-vs-arm deltas are methodologically identical:
  evaluate_sequence (core + E191 support fields) + run_health (jerk) +
  gate_health (leg-gate CEM diagnostics) + apply_gates (6 numeric gates).

A0 metrics are read straight from the landed E172/E173 case_metrics.tsv (same
code path, already computed) rather than re-run.

Outputs under workspace/core4d/results/E194/s6_downstream/eval/<stage>/:
  e194_arm_case_metrics.tsv   one row per (arm, case): all metrics
  e194_paired_deltas.tsv      G1/G2/G3 minus A0, per case, key metrics
  e194_arm_diff_summary.json  C6/C7 arm-difference aggregates per object

Usage:
  .venv/bin/python .../eval_E194_gravcomp_arms.py full
  .venv/bin/python .../eval_E194_gravcomp_arms.py canary
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))            # .../scripts
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval/runners"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/E194"))

from eval.core.core_metrics import EvalConfig  # noqa: E402
from eval_E172_box004 import evaluate_row  # noqa: E402  (reuse the frozen scorer)
import e194_common as C  # noqa: E402

# A0 baselines: the E172/E173 case_metrics tables E191 also consumed.
A0_METRICS = {
    "box004": C.REPO / "workspace/core4d/results/E172/s6_downstream/eval/full/e171_case_metrics.tsv",
    "box024": C.REPO / "workspace/core4d/results/E173/s6_downstream/eval/full/e173_case_metrics.tsv",
}

# Metrics carried into the paired-delta table and the C-claim judgments.
KEY_METRICS = [
    "track_obj_pos_err_cm_mean",
    "track_obj_z_err_m_lifted_mean", "track_obj_xy_err_cm_lifted_mean", "track_obj_z_err_share_lifted",
    "obj_side_near_z_err_m", "obj_side_far_z_err_m", "obj_side_z_asym_cm",
    "hand_object_physics_penetration_3mm_frame_frac",
    "hand_object_physics_contact_3mm_in_mask_frac", "hand_object_physics_contact_in_mask_frac",
    "object_guidance_force_z_N_p95", "object_guidance_torque_Nm_p95",
    "leg_penetration_frac", "qpos_jerk_l2_p95", "fall_flag",
]
# The 6 numeric gates apply_gates writes (plan's "12 门" = these 6 numeric gates
# reported per case; the report layer pairs them A0-vs-arm to get PASS/FAIL moves).
GATE_FIELDS = ["fall_gate_pass", "body_z_gate_pass", "contact_gate_pass",
               "release_gate_pass", "hand_penetration_gate_pass", "lower_body_gate_pass",
               "numeric_release_pass", "numeric_failure_modes"]


def finite(value: Any) -> float:
    try:
        f = float(value)
        return f if math.isfinite(f) else math.nan
    except (TypeError, ValueError):
        return math.nan


def score_arm_rows(stage: str) -> list[dict[str, Any]]:
    """Score every completed G1/G2/G3 rollout in the E194 manifest."""
    manifest = C.RESULTS / f"s6_downstream/manifests/cem_{'canary' if stage == 'canary' else 'full'}_manifest.tsv"
    rows = C.read_tsv(manifest)
    cfg = EvalConfig()
    out: list[dict[str, Any]] = []
    for row in rows:
        outdir = C.repo_path(row["outdir_npz"])
        if not outdir.is_file():
            print(f"[skip-incomplete] {row['variant']} (no outdir npz yet)")
            continue
        # evaluate_row needs these keys; inject the constants E194 froze.
        scoring_row = dict(row)
        scoring_row.setdefault("spider_method_id", C.E194_METHOD_ID)
        scoring_row.setdefault("hand_collision_variant_id", C.HAND_COLLISION_VARIANT_ID)
        try:
            item = evaluate_row(scoring_row, cfg, {}, {})
        except Exception as exc:  # noqa: BLE001
            print(f"[eval-error] {row['variant']}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        item["arm"] = row["arm"]
        item["case_id"] = row["case_id"]
        item["object_key"] = row["object_key"]
        out.append(item)
        print(f"[scored] {row['variant']}")
    return out


def load_a0() -> dict[str, dict[str, str]]:
    a0: dict[str, dict[str, str]] = {}
    for object_key, path in A0_METRICS.items():
        for r in C.read_tsv(path):
            if r["case_id"] in C.CASES[object_key]:
                r["arm"] = "A0"
                r["object_key"] = object_key
                a0[r["case_id"]] = r
    return a0


def paired_deltas(arm_rows: list[dict[str, Any]], a0: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for item in arm_rows:
        base = a0.get(item["case_id"])
        if base is None:
            continue
        rec = {"case_id": item["case_id"], "object_key": item["object_key"], "arm": item["arm"]}
        for metric in KEY_METRICS:
            cur, old = finite(item.get(metric)), finite(base.get(metric))
            rec[f"a0_{metric}"] = old
            rec[f"arm_{metric}"] = cur
            rec[f"delta_{metric}"] = cur - old if math.isfinite(cur) and math.isfinite(old) else math.nan
        for g in GATE_FIELDS:
            rec[f"a0_{g}"] = base.get(g, "")
            rec[f"arm_{g}"] = item.get(g, "")
        out.append(rec)
    return out


def arm_diff_summary(deltas: list[dict[str, Any]]) -> dict[str, Any]:
    """C6/C7 aggregates: penetration drop per arm, and G1-vs-G2 / G3-vs-G1 diffs."""
    PEN = "hand_object_physics_penetration_3mm_frame_frac"
    summary: dict[str, Any] = {}
    for object_key in ("box024", "box004"):
        by_arm_case: dict[str, dict[str, float]] = {"G1": {}, "G2": {}, "G3": {}}
        for d in deltas:
            if d["object_key"] != object_key or d["arm"] not in by_arm_case:
                continue
            by_arm_case[d["arm"]][d["case_id"]] = finite(d.get(f"delta_{PEN}"))

        def mean_drop(arm: str) -> float:
            # drop = -delta (penetration going down is a positive "drop")
            vals = [-v for v in by_arm_case[arm].values() if math.isfinite(v)]
            return sum(vals) / len(vals) if vals else math.nan

        g1, g2, g3 = mean_drop("G1"), mean_drop("G2"), mean_drop("G3")
        summary[object_key] = {
            "n_cases": len(by_arm_case["G1"]),
            "pen_drop_G1_mean": g1, "pen_drop_G2_mean": g2, "pen_drop_G3_mean": g3,
            "C6_G1_minus_G2": (g1 - g2) if math.isfinite(g1) and math.isfinite(g2) else math.nan,
            "C7_G3_minus_G1": (g3 - g1) if math.isfinite(g3) and math.isfinite(g1) else math.nan,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", nargs="?", default="full", choices=("canary", "full"))
    args = parser.parse_args()

    arm_rows = score_arm_rows(args.stage)
    a0 = load_a0()
    deltas = paired_deltas(arm_rows, a0)
    diff = arm_diff_summary(deltas)

    out_dir = C.RESULTS / f"s6_downstream/eval/{args.stage}"
    # combined per-arm-case metrics (A0 rows included for reference)
    combined = [dict(v) for v in a0.values()] + arm_rows
    all_fields: list[str] = []
    for r in combined:
        for k in r:
            if k not in all_fields:
                all_fields.append(k)
    for k in ("arm", "case_id", "object_key"):
        if k in all_fields:
            all_fields.remove(k)
    all_fields = ["arm", "case_id", "object_key"] + all_fields
    C.write_tsv(out_dir / "e194_arm_case_metrics.tsv", combined, all_fields)

    delta_fields: list[str] = []
    for r in deltas:
        for k in r:
            if k not in delta_fields:
                delta_fields.append(k)
    C.write_tsv(out_dir / "e194_paired_deltas.tsv", deltas, delta_fields)
    C.write_json(out_dir / "e194_arm_diff_summary.json", {
        "created_at": C.now(), "stage": args.stage,
        "arm_rows_scored": len(arm_rows), "a0_cases": len(a0), "paired": len(deltas),
        "arm_diff": diff,
    })
    print(f"[eval] stage={args.stage} scored={len(arm_rows)} a0={len(a0)} paired={len(deltas)}")
    print(json.dumps(diff, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
