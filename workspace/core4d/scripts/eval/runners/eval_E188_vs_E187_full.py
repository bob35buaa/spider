#!/usr/bin/env python3
"""Evaluate E188 directly with shared metrics and compare against frozen E187."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.core.core_metrics import EVAL_METRIC_STANDARD_ID, METRIC_FIELDS, EvalConfig, evaluate_sequence, npz_qpos  # noqa: E402
from eval.core.motion_health import HEALTH_AGGS, METRIC_KEYS, run_health  # noqa: E402
from eval.runners.eval_E176_lowgeom import (  # noqa: E402
    PAIR_METRICS,
    TRACKING_GATE_KEYS,
    apply_gates,
    finite,
    fixed_reference_z_metrics,
    gate_health,
    group_summary,
    person_idx,
    release_window_info,
    repo_path,
    sha256,
    write_tsv,
)

REPO = Path(__file__).resolve().parents[5]
MANIFEST = REPO / "workspace/core4d/results/E188/s6_downstream/manifests/e188_full_evaluation_manifest.tsv"
BASELINE = REPO / "workspace/core4d/results/E187/s6_downstream/eval/full/e187_case_metrics.tsv"
OUT = REPO / "workspace/core4d/results/E188/s6_downstream/eval/full"
EXPECTED = 15
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0
TRACKING_THRESHOLDS = {"root_pos": 20.0, "root_ori": 20.0, "hand_pos": 20.0, "hand_ori": 20.0, "object_pos": 20.0, "object_ori": 10.0}
KEY_METRICS = {
    "hand_object_physics_contact_in_mask_frac": "higher",
    "hand_object_physics_penetration_3mm_frame_frac": "lower",
    "leg_penetration_frac": "lower",
    "track_root_pos_err_cm_mean": "lower",
    "track_root_ori_err_deg_mean": "lower",
    "track_eef_pos_err_cm_mean": "lower",
    "track_eef_ori_err_deg_mean": "lower",
    "track_obj_pos_err_cm_mean": "lower",
    "track_obj_ori_err_deg_mean": "lower",
}
GATES = ("fall", "body_z", "contact", "release", "hand_penetration", "lower_body", "root_pos", "root_ori", "hand_pos", "hand_ori", "object_pos", "object_ori")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def logical_rel(value: str | Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        text = str(path)
        for marker in ("workspace/", "example_datasets/", "logs/", "examples/"):
            if marker in text:
                return marker + text.split(marker, 1)[1]
        return text


def evaluate_row(row: dict[str, str]) -> dict[str, Any]:
    result = repo_path(row["result_npz"])
    scene = repo_path(row["scene_act"])
    trajectory = repo_path(row["trajectory"])
    contact = repo_path(row["contact_mask"])
    actor = person_idx(row)
    config = EvalConfig()
    item = evaluate_sequence(row=row, method=row["spider_method_id"], hand_collision_variant_id=row["hand_collision_variant_id"], qpos_path=result, scene_xml=scene, config=config, kin_ref_path=trajectory, contact_mask_path=contact, person_idx=actor)
    qpos, _ = npz_qpos(result)
    item.update(run_health(result, scene, config))
    item.update(fixed_reference_z_metrics(result, scene, trajectory))
    item.update(release_window_info(contact, actor, len(qpos)))
    item.update(gate_health(result))
    for key in ("ordinal", "case_id", "variant", "object_key", "person", "retarget_variant_id", "target_variant_id", "hand_collision_variant_id", "spider_method_id", "status", "e188_worker", "e188_device", "e187_worker", "e187_device", "device_scope", "old_mass_kg", "new_mass_kg", "inertia_scale", "render_mode"):
        item[key] = row.get(key, "")
    video = repo_path(row["video"])
    item.update({"metric_standard_id": EVAL_METRIC_STANDARD_ID, "result_npz": logical_rel(result), "outdir_npz": logical_rel(result), "config_act": logical_rel(row["config_act"]), "video": logical_rel(video), "trajectory": logical_rel(trajectory), "contact_mask": logical_rel(contact), "scene_xml": logical_rel(scene)})
    apply_gates(item, TRACKING_THRESHOLDS)
    return item


def paired_row(item: dict[str, Any], old: dict[str, str]) -> dict[str, Any]:
    row: dict[str, Any] = {key: item.get(key, "") for key in ("case_id", "object_key", "person", "e187_worker", "e188_worker", "e187_device", "e188_device", "device_scope", "old_mass_kg", "new_mass_kg", "inertia_scale")}
    old_pass = boolish(old["numeric_release_pass"])
    new_pass = bool(item["numeric_release_pass"])
    row.update({"e187_numeric_release_pass": old_pass, "e188_numeric_release_pass": new_pass, "gate_transition": f"E187_{'PASS' if old_pass else 'FAIL'}_TO_E188_{'PASS' if new_pass else 'FAIL'}", "e187_numeric_failure_modes": old["numeric_failure_modes"], "e188_numeric_failure_modes": item["numeric_failure_modes"]})
    for metric, direction in PAIR_METRICS.items():
        before, after = finite(old.get(metric)), finite(item.get(metric))
        raw = after - before if math.isfinite(before) and math.isfinite(after) else math.nan
        improvement = raw if direction == "higher" else -raw
        row[f"e187_{metric}"] = before
        row[f"e188_{metric}"] = after
        row[f"delta_e188_minus_e187_{metric}"] = raw
        row[f"improvement_{metric}"] = improvement
    return row


def bootstrap(values: list[float], seed_offset: int) -> tuple[float, float, float]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if not len(array):
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    means = array[rng.integers(0, len(array), size=(BOOTSTRAP_DRAWS, len(array)))].mean(axis=1)
    return float(array.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def strata(paired: list[dict[str, Any]]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    groups = [("pooled", "all15", paired)]
    for field in ("device_scope", "e188_worker", "e187_worker", "object_key"):
        for value in sorted({str(row[field]) for row in paired}):
            groups.append((field, value, [row for row in paired if str(row[field]) == value]))
    return groups


def device_summary(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for group_index, (group_type, group_value, members) in enumerate(strata(paired)):
        for metric_index, (metric, direction) in enumerate(KEY_METRICS.items()):
            improvements = [finite(row[f"improvement_{metric}"]) for row in members]
            point, low, high = bootstrap(improvements, group_index * 100 + metric_index)
            old_values = [finite(row[f"e187_{metric}"]) for row in members]
            new_values = [finite(row[f"e188_{metric}"]) for row in members]
            rows.append({"group_type": group_type, "group_value": group_value, "rows": len(members), "metric": metric, "better_direction": direction, "e187_mean": float(np.nanmean(old_values)), "e188_mean": float(np.nanmean(new_values)), "improvement_mean": point, "bootstrap_ci95_low": low, "bootstrap_ci95_high": high, "bootstrap_draws": BOOTSTRAP_DRAWS, "nondegraded_cases": sum(value >= 0 for value in improvements if math.isfinite(value))})
    return rows


def gate_transitions(metrics: list[dict[str, Any]], baseline: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for gate in GATES:
        counts: Counter[str] = Counter()
        cases: dict[str, list[str]] = {"PASS_TO_PASS": [], "PASS_TO_FAIL": [], "FAIL_TO_PASS": [], "FAIL_TO_FAIL": []}
        for item in metrics:
            old = boolish(baseline[item["case_id"]][f"{gate}_gate_pass"])
            new = bool(item[f"{gate}_gate_pass"])
            label = f"{'PASS' if old else 'FAIL'}_TO_{'PASS' if new else 'FAIL'}"
            counts[label] += 1
            cases[label].append(item["case_id"])
        rows.append({"gate": gate, "rows": len(metrics), "e187_pass": counts["PASS_TO_PASS"] + counts["PASS_TO_FAIL"], "e188_pass": counts["PASS_TO_PASS"] + counts["FAIL_TO_PASS"], **{key.lower(): counts[key] for key in cases}, **{f"{key.lower()}_cases": ",".join(cases[key]) for key in cases}})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run"))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    manifest = repo_path(args.manifest)
    baseline_path = repo_path(args.baseline)
    manifest_rows = read_tsv(manifest)
    baseline_all = {row["case_id"]: row for row in read_tsv(baseline_path)}
    if len(manifest_rows) != EXPECTED or len({row["case_id"] for row in manifest_rows}) != EXPECTED:
        raise ValueError("E188 evaluation manifest must contain 15 unique rows")
    missing_baseline = sorted({row["case_id"] for row in manifest_rows} - set(baseline_all))
    required = ("result_npz", "config_act", "scene_act", "trajectory", "contact_mask")
    not_ready = [{"case_id": row["case_id"], "missing": ",".join(key for key in required if not repo_path(row[key]).is_file())} for row in manifest_rows]
    not_ready = [row for row in not_ready if row["missing"]]
    if missing_baseline or not_ready:
        raise RuntimeError(f"preflight failed: missing_baseline={missing_baseline} not_ready={not_ready}")
    if args.mode == "preflight":
        print(json.dumps({"status": "PASS", "rows": len(manifest_rows), "baseline_rows": len(baseline_all), "same_device_local4": sum(row["device_scope"] == "same_device_local4" for row in manifest_rows)}, sort_keys=True))
        return 0
    metrics, errors = [], []
    for index, row in enumerate(manifest_rows, 1):
        print(f"[{index}/{EXPECTED}] {row['case_id']}", flush=True)
        try:
            metrics.append(evaluate_row(row))
        except Exception as exc:  # noqa: BLE001
            errors.append({"case_id": row["case_id"], "error": f"{type(exc).__name__}: {exc}"})
    baseline = {case_id: baseline_all[case_id] for case_id in {row["case_id"] for row in manifest_rows}}
    paired = [paired_row(item, baseline[item["case_id"]]) for item in metrics]
    device_rows = device_summary(paired)
    gate_rows = gate_transitions(metrics, baseline)
    fields: list[str] = []
    for key in ["case_id", "variant", "object_key", "person", "numeric_release_pass", "numeric_failure_modes", "leg_gate_health_pass", *PAIR_METRICS, *METRIC_FIELDS, *METRIC_KEYS, *HEALTH_AGGS]:
        if key not in fields:
            fields.append(key)
    for row in metrics:
        for key in row:
            if key not in fields:
                fields.append(key)
    out = repo_path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_tsv(out / "e188_case_metrics.tsv", metrics, fields)
    write_tsv(out / "e188_vs_e187_paired_deltas.tsv", paired)
    write_tsv(out / "e188_device_stratified_summary.tsv", device_rows)
    write_tsv(out / "e188_gate_transitions.tsv", gate_rows)
    write_tsv(out / "e188_group_summary.tsv", group_summary(metrics))
    write_tsv(out / "e188_evaluation_errors.tsv", errors, ["case_id", "error"])
    write_tsv(out / "evaluated_manifest_snapshot.tsv", manifest_rows)
    pooled = {(row["metric"]): row for row in device_rows if row["group_type"] == "pooled"}
    transitions = Counter(row["gate_transition"] for row in paired)
    claim_values = {
        "lower_body_pass_e187": sum(boolish(baseline[row["case_id"]]["lower_body_gate_pass"]) for row in paired),
        "lower_body_pass_e188": sum(bool(item["lower_body_gate_pass"]) for item in metrics),
        "leg_improvement_mean": pooled["leg_penetration_frac"]["improvement_mean"],
        "leg_nondegraded": pooled["leg_penetration_frac"]["nondegraded_cases"],
        "contact_regression": -pooled["hand_object_physics_contact_in_mask_frac"]["improvement_mean"],
        "hand_penetration_regression": -pooled["hand_object_physics_penetration_3mm_frame_frac"]["improvement_mean"],
        "numeric_pass_e187": sum(boolish(baseline[row["case_id"]]["numeric_release_pass"]) for row in paired),
        "numeric_pass_e188": sum(bool(row["e188_numeric_release_pass"]) for row in paired),
        "numeric_pass_to_fail": transitions["E187_PASS_TO_E188_FAIL"],
    }
    claims = {
        "C5_lower_body": claim_values["lower_body_pass_e188"] >= 7 and claim_values["leg_improvement_mean"] >= 0.05 and claim_values["leg_nondegraded"] >= 10,
        "C6_contact_hand_pen_noninferior": claim_values["contact_regression"] <= 0.03 and claim_values["hand_penetration_regression"] <= 0.03,
        "C7_tracking_noninferior": all(-pooled[metric]["improvement_mean"] <= (2.0 if "track_" in metric else 2.0) for metric in KEY_METRICS if metric.startswith("track_")),
        "C8_total_gate": claim_values["numeric_pass_e188"] >= 5 and claim_values["numeric_pass_to_fail"] <= 1,
    }
    summary = {"generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "experiment_id": "E188", "metric_standard_id": EVAL_METRIC_STANDARD_ID, "manifest": logical_rel(manifest), "manifest_sha256": sha256(manifest), "baseline": logical_rel(baseline_path), "baseline_sha256": sha256(baseline_path), "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED}, "counts": {"expected_rows": EXPECTED, "evaluated": len(metrics), "errors": len(errors), "paired_rows": len(paired), "numeric_pass": sum(bool(row["numeric_release_pass"]) for row in metrics), "same_device_local4": sum(row["device_scope"] == "same_device_local4" for row in paired), "cross_device11": sum(row["device_scope"] == "cross_device11" for row in paired)}, "paired_numeric_transitions": dict(transitions), "claim_values": claim_values, "claims": claims, "thresholds": {"body_z_err_p95_m_max": 0.20, "contact_in_mask_min": 0.50, "release_false_3mm_max": 0.30, "hand_penetration_3mm_max": 0.30, "leg_penetration_max": 0.10, "track_root_pos_err_cm_mean_max": 20.0, "track_root_ori_err_deg_mean_max": 20.0, "track_eef_pos_err_cm_mean_max": 20.0, "track_eef_ori_err_deg_mean_max": 20.0, "track_obj_pos_err_cm_mean_max": 20.0, "track_obj_ori_err_deg_mean_max": 10.0}, "interpretation_boundary": "same_device_local4 is strongest mass-only evidence; cross_device11 confounds mass with RTX6000 Ada to A100/RTX5090 device changes", "status": "pass" if len(metrics) == EXPECTED and not errors else "incomplete"}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["counts"], sort_keys=True))
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
