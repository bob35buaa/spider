#!/usr/bin/env python3
"""Evaluate E179 and compare the exact 16 box023 rows with frozen E173."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(SCRIPT_ROOT / "experiments/E179"))

from eval.core.core_metrics import (  # noqa: E402
    EVAL_METRIC_STANDARD_ID,
    EvalConfig,
    evaluate_sequence,
    npz_qpos,
)
from eval_E168_e167a_metrics import (  # noqa: E402
    fixed_reference_z_metrics,
    release_window_info,
)

import e179_common as C  # noqa: E402


PAIR_METRICS = {
    "body_z_err_p95_m": "lower",
    "hand_object_physics_contact_in_mask_frac": "higher",
    "hand_object_release_false_contact_3mm_frac": "lower",
    "hand_object_physics_penetration_3mm_frame_frac": "lower",
    "leg_penetration_frac": "lower",
    "track_root_pos_err_cm_mean": "lower",
    "track_root_ori_err_deg_mean": "lower",
    "track_eef_pos_err_cm_mean": "lower",
    "track_eef_ori_err_deg_mean": "lower",
    "track_obj_pos_err_cm_mean": "lower",
    "track_obj_ori_err_deg_mean": "lower",
}
BOOTSTRAP_SEED = 0
BOOTSTRAP_SAMPLES = 10_000
MIGRATION_ORDER = (
    "PASS_TO_PASS",
    "PASS_TO_FAIL",
    "FAIL_TO_PASS",
    "FAIL_TO_FAIL",
)


def finite(value: Any, default: float = math.nan) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return default
    return output if math.isfinite(output) else default


def person_idx(row: dict[str, str]) -> int:
    value = str(row.get("person_idx", "")).strip()
    if value:
        return int(value)
    return 0 if row["case_id"].endswith("_p1") else 1


def evaluate_row(
    row: dict[str, str], config: EvalConfig
) -> dict[str, Any]:
    qpos_path = C.repo_path(row["outdir_npz"])
    result_npz = C.repo_path(row["result_npz"])
    scene = C.repo_path(row["scene_act"])
    trajectory = C.repo_path(row["trajectory"])
    mask = C.repo_path(row["contact_mask"])
    for label, path in (
        ("outdir_npz", qpos_path),
        ("result_npz", result_npz),
        ("scene", scene),
        ("trajectory", trajectory),
        ("contact_mask", mask),
        ("config_act", C.repo_path(row["config_act"])),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{row['case_id']}:{label}:{path}")

    item = evaluate_sequence(
        row=row,
        method=row["spider_method_id"],
        hand_collision_variant_id=row["hand_collision_variant_id"],
        qpos_path=qpos_path,
        scene_xml=scene,
        config=config,
        kin_ref_path=trajectory,
        contact_mask_path=mask,
        person_idx=person_idx(row),
    )
    sim_qpos, _ = npz_qpos(qpos_path)
    item.update(
        fixed_reference_z_metrics(qpos_path, scene, trajectory)
    )
    item.update(
        release_window_info(mask, person_idx(row), len(sim_qpos))
    )
    for key in (
        "ordinal",
        "case_id",
        "variant",
        "retarget_variant_id",
        "selected_retarget_variant_id",
        "target_variant_id",
        "hand_collision_variant_id",
        "spider_method_id",
        "assigned_worker",
        "assigned_gpu",
        "status",
    ):
        item[key] = row.get(key, "")
    item.update(
        {
            "metric_standard_id": EVAL_METRIC_STANDARD_ID,
            "scoring_contract_id": C.SCORING_CONTRACT_ID,
            "result_npz": C.rel(result_npz),
            "outdir_npz": C.rel(qpos_path),
            "config_act": C.rel(row["config_act"]),
            "video": C.rel(row["video"]),
            "trajectory": C.rel(trajectory),
            "contact_mask": C.rel(mask),
            "scene_xml": C.rel(scene),
        }
    )
    return C.apply_12gate_scoring(item)


def baseline_rows() -> list[dict[str, str]]:
    path = (
        C.RESULTS
        / "input_authority/e173_box023_baseline_12gate.tsv"
    )
    rows = C.read_tsv(path)
    if (
        len(rows) != C.EXPECTED_PAIRED_ROWS
        or len({row["case_id"] for row in rows})
        != C.EXPECTED_PAIRED_ROWS
    ):
        raise ValueError("frozen E173 baseline must contain 16 unique rows")
    return rows


def paired_outputs(
    current: list[dict[str, Any]],
    baseline: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_by_case = {row["case_id"]: row for row in baseline}
    comparison = []
    gate_matrix = []
    for item in current:
        case_id = str(item["case_id"])
        old = baseline_by_case[case_id]
        row: dict[str, Any] = {
            "case_id": case_id,
            "retarget_variant_id": item["retarget_variant_id"],
            "assigned_worker": item["assigned_worker"],
            "e173_physics6_pass": C.boolish(
                old["legacy_physics6_pass"]
            ),
            "e179_physics6_pass": bool(
                item["legacy_physics6_pass"]
            ),
            "e173_12gate_pass": C.boolish(
                old["numeric_release_pass_12gate"]
            ),
            "e179_12gate_pass": bool(
                item["numeric_release_pass_12gate"]
            ),
            "e173_failure_modes": old["numeric_failure_modes"],
            "e179_failure_modes": item["numeric_failure_modes"],
            "e173_result_npz": old.get("result_npz", ""),
            "e179_result_npz": item.get("result_npz", ""),
            "e173_video": old.get("video", ""),
            "e179_video": item.get("video", ""),
        }
        old_pass = row["e173_12gate_pass"]
        new_pass = row["e179_12gate_pass"]
        row["pass_migration"] = (
            "PASS_TO_PASS"
            if old_pass and new_pass
            else (
                "PASS_TO_FAIL"
                if old_pass
                else ("FAIL_TO_PASS" if new_pass else "FAIL_TO_FAIL")
            )
        )
        for metric, direction in PAIR_METRICS.items():
            e173_value = finite(old.get(metric))
            e179_value = finite(item.get(metric))
            delta = (
                e179_value - e173_value
                if math.isfinite(e173_value)
                and math.isfinite(e179_value)
                else math.nan
            )
            row[f"e173_{metric}"] = e173_value
            row[f"e179_{metric}"] = e179_value
            row[f"delta_{metric}"] = delta
            row[f"improvement_{metric}"] = (
                delta if direction == "higher" else -delta
            )
        comparison.append(row)
        for gate in C.ALL_GATES:
            e173_pass = C.boolish(old.get(f"{gate}_gate_pass"))
            e179_pass = bool(item[f"{gate}_gate_pass"])
            gate_matrix.append(
                {
                    "case_id": case_id,
                    "gate": gate,
                    "e173_pass": e173_pass,
                    "e179_pass": e179_pass,
                    "migration": (
                        "PASS_TO_PASS"
                        if e173_pass and e179_pass
                        else (
                            "PASS_TO_FAIL"
                            if e173_pass
                            else (
                                "FAIL_TO_PASS"
                                if e179_pass
                                else "FAIL_TO_FAIL"
                            )
                        )
                    ),
                }
            )
    return comparison, gate_matrix


def aggregate_values(
    comparison: list[dict[str, Any]], key: str
) -> dict[str, Any]:
    values = [
        finite(row[key])
        for row in comparison
    ]
    values = [value for value in values if math.isfinite(value)]
    array = np.asarray(values, dtype=np.float64)
    if not values:
        return {
            "n": 0,
            "mean": math.nan,
            "median": math.nan,
            "q25": math.nan,
            "q75": math.nan,
            "iqr": math.nan,
            "bootstrap_mean_ci95_low": math.nan,
            "bootstrap_mean_ci95_high": math.nan,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        }
    q25, q75 = np.quantile(array, [0.25, 0.75])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    resampled_means = rng.choice(
        array,
        size=(BOOTSTRAP_SAMPLES, len(array)),
        replace=True,
    ).mean(axis=1)
    ci_low, ci_high = np.quantile(
        resampled_means, [0.025, 0.975]
    )
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "bootstrap_mean_ci95_low": float(ci_low),
        "bootstrap_mean_ci95_high": float(ci_high),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }


def migration_summary(migrations: list[str]) -> dict[str, int]:
    counts = Counter(migrations)
    return {key: int(counts.get(key, 0)) for key in MIGRATION_ORDER}


def exact_mcnemar(
    *, pass_to_fail: int, fail_to_pass: int
) -> dict[str, Any]:
    discordant = pass_to_fail + fail_to_pass
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, k)
            for k in range(min(pass_to_fail, fail_to_pass) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "pass_to_fail": pass_to_fail,
        "fail_to_pass": fail_to_pass,
        "discordant_pairs": discordant,
        "exact_two_sided_p": p_value,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", nargs="?", choices=("canary", "full"), default="full"
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()

    manifest = args.manifest or (
        C.RESULTS
        / "s6_downstream/manifests"
        / (
            "cem_canary_local_gpu0.tsv"
            if args.mode == "canary"
            else "cem_full_manifest.tsv"
        )
    )
    output = args.out_dir or (
        C.RESULTS / f"s6_downstream/eval/{args.mode}"
    )
    rows = C.read_tsv(manifest)
    expected = 3 if args.mode == "canary" else 16
    if len(rows) != expected:
        raise ValueError(
            f"{args.mode} manifest rows={len(rows)} expected={expected}"
        )

    metrics = []
    errors = []
    config = EvalConfig()
    for row in rows:
        try:
            metrics.append(evaluate_row(row, config))
        except Exception as exc:  # preserve per-case failure evidence
            errors.append(
                {
                    "case_id": row.get("case_id", ""),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    if args.require_all and errors:
        C.write_json(output / "e179_eval_errors.json", errors)
        raise RuntimeError(f"E179 evaluation failures: {errors}")

    selected_cases = {row["case_id"] for row in rows}
    baseline = [
        row
        for row in baseline_rows()
        if row["case_id"] in selected_cases
    ]
    comparison, gate_matrix = paired_outputs(metrics, baseline)
    if len(gate_matrix) != len(metrics) * len(C.ALL_GATES):
        raise ValueError("paired gate matrix cardinality drift")
    if args.mode == "full" and len(gate_matrix) != 192:
        raise ValueError("Full paired gate matrix must contain 192 rows")

    pass_migrations = migration_summary(
        [str(row["pass_migration"]) for row in comparison]
    )
    gate_transitions = {
        gate: migration_summary(
            [
                str(row["migration"])
                for row in gate_matrix
                if row["gate"] == gate
            ]
        )
        for gate in C.ALL_GATES
    }
    e173_pass = sum(row["e173_12gate_pass"] for row in comparison)
    e179_pass = sum(row["e179_12gate_pass"] for row in comparison)
    numeric_verdict = (
        "NO_PRG_BETTER"
        if e179_pass > e173_pass
        else (
            "PRG_BETTER"
            if e179_pass < e173_pass
            else "NO_PRG_NONINFERIOR_ON_PASS_COUNT"
        )
    )
    summary = {
        "created_at": C.now(),
        "mode": args.mode,
        "status": (
            "pass"
            if len(metrics) == expected and not errors
            else "incomplete"
        ),
        "manifest": C.rel(manifest),
        "manifest_rows": len(rows),
        "evaluated_rows": len(metrics),
        "errors": errors,
        "metric_standard_id": EVAL_METRIC_STANDARD_ID,
        "scoring_contract_id": C.SCORING_CONTRACT_ID,
        "gates": list(C.ALL_GATES),
        "gate_cells": len(gate_matrix),
        "e173_physics6_pass": sum(
            row["e173_physics6_pass"] for row in comparison
        ),
        "e179_physics6_pass": sum(
            row["e179_physics6_pass"] for row in comparison
        ),
        "e173_12gate_pass": e173_pass,
        "e179_12gate_pass": e179_pass,
        "pass_migrations": pass_migrations,
        "mcnemar_12gate": exact_mcnemar(
            pass_to_fail=pass_migrations["PASS_TO_FAIL"],
            fail_to_pass=pass_migrations["FAIL_TO_PASS"],
        ),
        "gate_pass_counts": {
            gate: {
                "e173": sum(
                    row["gate"] == gate and row["e173_pass"]
                    for row in gate_matrix
                ),
                "e179": sum(
                    row["gate"] == gate and row["e179_pass"]
                    for row in gate_matrix
                ),
            }
            for gate in C.ALL_GATES
        },
        "gate_transitions": gate_transitions,
        "gate_mcnemar": {
            gate: exact_mcnemar(
                pass_to_fail=gate_transitions[gate]["PASS_TO_FAIL"],
                fail_to_pass=gate_transitions[gate]["FAIL_TO_PASS"],
            )
            for gate in C.ALL_GATES
        },
        "metric_deltas": {
            metric: aggregate_values(
                comparison, f"delta_{metric}"
            )
            for metric in PAIR_METRICS
        },
        "metric_improvements": {
            metric: aggregate_values(
                comparison, f"improvement_{metric}"
            )
            for metric in PAIR_METRICS
        },
        "paired_statistics_contract": {
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_interval": "percentile_95pct_of_paired_mean",
            "mcnemar": "exact_two_sided_binomial",
        },
        "numeric_verdict": numeric_verdict,
        "visual_verdict": "PENDING_16_PAIRED_VIDEO_REVIEW",
    }
    output.mkdir(parents=True, exist_ok=True)
    C.write_tsv(output / "e179_case_metrics.tsv", metrics)
    C.write_tsv(output / "e179_vs_e173_paired.tsv", comparison)
    C.write_tsv(output / "e179_vs_e173_gate_matrix.tsv", gate_matrix)
    C.write_json(output / "e179_eval_summary.json", summary)
    C.write_json(output / "e179_eval_errors.json", errors)
    print(summary)
    if args.require_all and summary["status"] != "pass":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
