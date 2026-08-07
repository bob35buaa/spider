#!/usr/bin/env python3
"""Evaluate E189 and pair all 43 box004/box024/box001 rows with E172/E173.

Unlike E179 (single object, box023), E189 spans three objects with very
different volumes and historical failure modes. Every aggregate in this
script is computed twice: once per object (box004/box024/box001, the
authoritative denominator per plan/215) and once over the 43-row union
(secondary, for completeness bookkeeping only). Conclusions must never be
drawn from the combined table alone.
"""

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
sys.path.insert(0, str(SCRIPT_ROOT / "experiments/E168"))
sys.path.insert(0, str(SCRIPT_ROOT / "experiments/E189"))

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

import e189_common as C  # noqa: E402


OBJECTS = ("box004", "box024", "box001")
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


def evaluate_row(row: dict[str, str], config: EvalConfig) -> dict[str, Any]:
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
    item.update(fixed_reference_z_metrics(qpos_path, scene, trajectory))
    item.update(release_window_info(mask, person_idx(row), len(sim_qpos)))
    for key in (
        "ordinal",
        "case_id",
        "object_key",
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
    path = C.RESULTS / "input_authority/e172_e173_baseline_12gate.tsv"
    rows = C.read_tsv(path)
    if (
        len(rows) != C.EXPECTED_PAIRED_ROWS
        or len({row["case_id"] for row in rows}) != C.EXPECTED_PAIRED_ROWS
    ):
        raise ValueError(
            f"frozen E172/E173 baseline must contain "
            f"{C.EXPECTED_PAIRED_ROWS} unique rows"
        )
    by_object = Counter(row["object_key"] for row in rows)
    for obj in OBJECTS:
        expected = C.SOURCES[obj]["expected_rows"]
        if by_object.get(obj, 0) != expected:
            raise ValueError(
                f"baseline object_key={obj} rows={by_object.get(obj, 0)} "
                f"expected={expected}"
            )
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
        object_key = str(item["object_key"]) or str(old["object_key"])
        row: dict[str, Any] = {
            "case_id": case_id,
            "object_key": object_key,
            "retarget_variant_id": item["retarget_variant_id"],
            "assigned_worker": item["assigned_worker"],
            "prg_source_experiment_id": old["source_experiment_id"],
            "prg_physics6_pass": C.boolish(old["legacy_physics6_pass"]),
            "e189_physics6_pass": bool(item["legacy_physics6_pass"]),
            "prg_12gate_pass": C.boolish(old["numeric_release_pass_12gate"]),
            "e189_12gate_pass": bool(item["numeric_release_pass_12gate"]),
            "prg_failure_modes": old["numeric_failure_modes"],
            "e189_failure_modes": item["numeric_failure_modes"],
            "prg_result_npz": old.get("source_metrics_path", ""),
            "e189_result_npz": item.get("result_npz", ""),
            "e189_video": item.get("video", ""),
        }
        old_pass = row["prg_12gate_pass"]
        new_pass = row["e189_12gate_pass"]
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
            prg_value = finite(old.get(metric))
            e189_value = finite(item.get(metric))
            delta = (
                e189_value - prg_value
                if math.isfinite(prg_value) and math.isfinite(e189_value)
                else math.nan
            )
            row[f"prg_{metric}"] = prg_value
            row[f"e189_{metric}"] = e189_value
            row[f"delta_{metric}"] = delta
            row[f"improvement_{metric}"] = (
                delta if direction == "higher" else -delta
            )
        comparison.append(row)
        for gate in C.ALL_GATES:
            prg_pass = C.boolish(old.get(f"{gate}_gate_pass"))
            e189_pass = bool(item[f"{gate}_gate_pass"])
            gate_matrix.append(
                {
                    "case_id": case_id,
                    "object_key": object_key,
                    "gate": gate,
                    "prg_pass": prg_pass,
                    "e189_pass": e189_pass,
                    "migration": (
                        "PASS_TO_PASS"
                        if prg_pass and e189_pass
                        else (
                            "PASS_TO_FAIL"
                            if prg_pass
                            else (
                                "FAIL_TO_PASS"
                                if e189_pass
                                else "FAIL_TO_FAIL"
                            )
                        )
                    ),
                }
            )
    return comparison, gate_matrix


def aggregate_values(comparison: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [finite(row[key]) for row in comparison]
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
        array, size=(BOOTSTRAP_SAMPLES, len(array)), replace=True
    ).mean(axis=1)
    ci_low, ci_high = np.quantile(resampled_means, [0.025, 0.975])
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


def exact_mcnemar(*, pass_to_fail: int, fail_to_pass: int) -> dict[str, Any]:
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


def classify_verdict(*, prg_pass: int, e189_pass: int, n: int, new_fall: bool,
                      net_regression: int, net_improvement: int) -> str:
    if new_fall:
        return "PRG_BETTER"
    if e189_pass >= prg_pass + 1 and net_regression == 0:
        return "NO_PRG_BETTER"
    if e189_pass <= prg_pass - 1 and (
        (prg_pass - e189_pass) >= 2 or net_regression - net_improvement >= 2
    ):
        return "PRG_BETTER"
    if abs(e189_pass - prg_pass) <= 1 and (net_regression - net_improvement) <= 1:
        return "NO_PRG_NONINFERIOR"
    return "MIXED"


def per_object_block(
    object_key: str,
    comparison: list[dict[str, Any]],
    gate_matrix: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = [row for row in comparison if row["object_key"] == object_key]
    cells = [row for row in gate_matrix if row["object_key"] == object_key]
    n = len(rows)
    expected = C.SOURCES[object_key]["expected_rows"]
    if n != expected:
        raise ValueError(
            f"{object_key}: comparison rows={n} expected={expected}"
        )
    pass_migrations = migration_summary(
        [str(row["pass_migration"]) for row in rows]
    )
    gate_transitions = {
        gate: migration_summary(
            [
                str(cell["migration"])
                for cell in cells
                if cell["gate"] == gate
            ]
        )
        for gate in C.ALL_GATES
    }
    prg_pass = sum(row["prg_12gate_pass"] for row in rows)
    e189_pass = sum(row["e189_12gate_pass"] for row in rows)
    new_fall = any(
        cell["gate"] == "fall" and cell["prg_pass"] and not cell["e189_pass"]
        for cell in cells
    )
    net_regression = pass_migrations["PASS_TO_FAIL"]
    net_improvement = pass_migrations["FAIL_TO_PASS"]
    verdict = classify_verdict(
        prg_pass=prg_pass,
        e189_pass=e189_pass,
        n=n,
        new_fall=new_fall,
        net_regression=net_regression,
        net_improvement=net_improvement,
    )
    return {
        "object_key": object_key,
        "n": n,
        "prg_physics6_pass": sum(row["prg_physics6_pass"] for row in rows),
        "e189_physics6_pass": sum(row["e189_physics6_pass"] for row in rows),
        "prg_12gate_pass": prg_pass,
        "e189_12gate_pass": e189_pass,
        "new_fall_case": new_fall,
        "pass_migrations": pass_migrations,
        "mcnemar_12gate": exact_mcnemar(
            pass_to_fail=pass_migrations["PASS_TO_FAIL"],
            fail_to_pass=pass_migrations["FAIL_TO_PASS"],
        ),
        "gate_pass_counts": {
            gate: {
                "prg": sum(
                    cell["gate"] == gate and cell["prg_pass"] for cell in cells
                ),
                "e189": sum(
                    cell["gate"] == gate and cell["e189_pass"] for cell in cells
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
            metric: aggregate_values(rows, f"delta_{metric}")
            for metric in PAIR_METRICS
        },
        "metric_improvements": {
            metric: aggregate_values(rows, f"improvement_{metric}")
            for metric in PAIR_METRICS
        },
        "verdict": verdict,
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
            "cem_canary_manifest.tsv"
            if args.mode == "canary"
            else "cem_full_manifest.tsv"
        )
    )
    output = args.out_dir or (C.RESULTS / f"s6_downstream/eval/{args.mode}")
    rows = C.read_tsv(manifest)
    expected = 9 if args.mode == "canary" else C.EXPECTED_PAIRED_ROWS
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
                    "object_key": row.get("object_key", ""),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    if args.require_all and errors:
        C.write_json(output / "e189_eval_errors.json", errors)
        raise RuntimeError(f"E189 evaluation failures: {errors}")

    selected_cases = {row["case_id"] for row in rows}
    baseline = [
        row for row in baseline_rows() if row["case_id"] in selected_cases
    ]
    comparison, gate_matrix = paired_outputs(metrics, baseline)
    if len(gate_matrix) != len(metrics) * len(C.ALL_GATES):
        raise ValueError("paired gate matrix cardinality drift")

    by_object = {}
    if args.mode == "full":
        for obj in OBJECTS:
            by_object[obj] = per_object_block(obj, comparison, gate_matrix)

    combined_pass_migrations = migration_summary(
        [str(row["pass_migration"]) for row in comparison]
    )
    combined_prg_pass = sum(row["prg_12gate_pass"] for row in comparison)
    combined_e189_pass = sum(row["e189_12gate_pass"] for row in comparison)

    summary = {
        "created_at": C.now(),
        "mode": args.mode,
        "status": (
            "pass" if len(metrics) == expected and not errors else "incomplete"
        ),
        "manifest": C.rel(manifest),
        "manifest_rows": len(rows),
        "evaluated_rows": len(metrics),
        "errors": errors,
        "metric_standard_id": EVAL_METRIC_STANDARD_ID,
        "scoring_contract_id": C.SCORING_CONTRACT_ID,
        "gates": list(C.ALL_GATES),
        "gate_cells": len(gate_matrix),
        "paired_expected_by_object": {
            obj: C.SOURCES[obj]["expected_rows"] for obj in OBJECTS
        },
        "paired_evaluated_by_object": {
            obj: sum(1 for row in comparison if row["object_key"] == obj)
            for obj in OBJECTS
        },
        "tracking_gates_enabled": True,
        "by_object": by_object,
        "combined_43_row_secondary_only": {
            "prg_12gate_pass": combined_prg_pass,
            "e189_12gate_pass": combined_e189_pass,
            "pass_migrations": combined_pass_migrations,
            "note": (
                "This union table is a completeness convenience only. "
                "Verdicts must come from by_object, never from this block."
            ),
        },
        "paired_statistics_contract": {
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_interval": "percentile_95pct_of_paired_mean",
            "mcnemar": "exact_two_sided_binomial",
        },
        "verdicts_by_object": {
            obj: by_object[obj]["verdict"] for obj in OBJECTS
        }
        if by_object
        else {},
        "visual_verdict": "PENDING_43_PAIRED_VIDEO_REVIEW",
    }
    output.mkdir(parents=True, exist_ok=True)
    C.write_tsv(output / "e189_case_metrics.tsv", metrics)
    C.write_tsv(output / "e189_vs_prg_paired.tsv", comparison)
    C.write_tsv(output / "e189_vs_prg_gate_matrix.tsv", gate_matrix)
    C.write_json(output / "e189_eval_summary.json", summary)
    C.write_json(output / "e189_eval_errors.json", errors)
    print(summary)
    if args.require_all and summary["status"] != "pass":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
