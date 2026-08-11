#!/usr/bin/env python3
"""Uniform public-core noPRG / PRG / G1 evaluation for the E194 72-case authority."""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/E194"))

import eval_E194_G1_expansion as base  # noqa: E402
import e194_g1_expansion_common as C  # noqa: E402
from eval.core.core_metrics import EVAL_METRIC_STANDARD_ID, EvalConfig  # noqa: E402

OUT = C.RESULTS / "s6_downstream/eval/full_g1_expansion"
EXISTING_METRICS = OUT / "e194_g1_expansion_case_metrics.tsv"
NOPRG_CACHE = OUT / "e194_noprg_case_metrics_cache.tsv"
SOURCE_SPECS = (
    ("box001", "E189", C.REPO / "workspace/core4d/results/E189/s6_downstream/manifests/cem_full_manifest.tsv"),
    ("box023", "E179", C.REPO / "workspace/core4d/results/E179/s6_downstream/manifests/cem_full_manifest.tsv"),
    ("box021", "E168", C.REPO / "workspace/core4d/results/E168/s6_downstream/cem/manifests/cem_production_manifest.tsv"),
)
COMPARISONS = (("noPRG_to_PRG", "noPRG", "PRG"), ("PRG_to_G1", "PRG", "G1"))


def mean(values: list[float]) -> float:
    valid = [value for value in values if math.isfinite(value)]
    return statistics.fmean(valid) if valid else math.nan


def bootstrap(values: list[float], seed: int = 0, n_boot: int = 10000) -> tuple[float, float]:
    data = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if not len(data):
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    estimates = np.mean(rng.choice(data, size=(n_boot, len(data)), replace=True), axis=1)
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def mcnemar_exact(pass_to_fail: int, fail_to_pass: int) -> float:
    discordant = pass_to_fail + fail_to_pass
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(pass_to_fail, fail_to_pass) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def no_prg_authority() -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    target = {row["case_id"]: row for row in C.read_tsv(C.SOURCE_AUTHORITY)}
    rows: list[dict[str, str]] = []
    audit: list[dict[str, Any]] = []
    for object_key, source_exp, manifest in SOURCE_SPECS:
        for raw in C.read_tsv(manifest):
            case_id = raw.get("case_id", "")
            if raw.get("object_key") != object_key or case_id not in target:
                continue
            row = dict(raw)
            row.update({"source_exp": source_exp, "execution_source": source_exp,
                        "reused_full": "true", "worker": f"historical-{source_exp}",
                        "execution_profile": f"historical-{source_exp}"})
            rows.append(row)
            required = ("result_npz", "outdir_npz", "config_act", "scene_act", "trajectory", "contact_mask", "video")
            files_ok = all(C.repo_path(row[field]).is_file() and C.repo_path(row[field]).stat().st_size > 0 for field in required)
            variant_ok = row.get("retarget_variant_id") == target[case_id].get("retarget_variant_id")
            audit.append({"case_id": case_id, "object_key": object_key, "authority_exp": source_exp,
                          "manifest_status": row.get("status", ""), "retarget_variant_id": row.get("retarget_variant_id", ""),
                          "target_retarget_variant_id": target[case_id].get("retarget_variant_id", ""),
                          "variant_match": variant_ok, "required_artifacts_ok": files_ok,
                          "status": "pass" if variant_ok and files_ok else "fail",
                          "result_npz": row.get("result_npz", ""), "video": row.get("video", "")})
    rank = {key: index for index, key in enumerate(C.OBJECT_ORDER)}
    rows.sort(key=lambda row: (rank[row["object_key"]], row["case_id"]))
    if len(rows) != C.N_CASES or len({row["case_id"] for row in rows}) != C.N_CASES:
        raise ValueError(f"noPRG authority is not 72 unique rows: rows={len(rows)} unique={len({row['case_id'] for row in rows})}")
    if Counter(row["object_key"] for row in rows) != Counter(C.OBJECT_COUNTS):
        raise ValueError("noPRG authority object counts differ")
    if set(target) != {row["case_id"] for row in rows} or not all(row["status"] == "pass" for row in audit):
        raise ValueError("noPRG authority case/variant/artifact parity failed")
    return rows, audit


def load_current_arms() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = C.read_tsv(EXISTING_METRICS)
    prg = [dict(row, arm="PRG") for row in rows if row.get("arm") == "A0"]
    g1 = [dict(row, arm="G1") for row in rows if row.get("arm") == "G1"]
    if len(prg) != C.N_CASES or len(g1) != C.N_CASES:
        raise ValueError(f"existing arms incomplete: PRG={len(prg)} G1={len(g1)}")
    for arm_rows in (prg, g1):
        if len({row["case_id"] for row in arm_rows}) != C.N_CASES:
            raise ValueError("existing arm contains duplicate cases")
        if any(row.get("metric_standard_id") != EVAL_METRIC_STANDARD_ID for row in arm_rows):
            raise ValueError("existing arm metric standard mismatch")
    return prg, g1


def score_noprg(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    cached = {row["case_id"]: row for row in C.read_tsv(NOPRG_CACHE)} if NOPRG_CACHE.is_file() else {}
    scored: dict[str, dict[str, Any]] = dict(cached)
    errors: list[dict[str, str]] = []
    cfg = EvalConfig()
    for index, row in enumerate(rows, 1):
        case_id = row["case_id"]
        if case_id in scored and scored[case_id].get("metric_standard_id") == EVAL_METRIC_STANDARD_ID:
            print(f"[cached {index:02d}/72] noPRG {case_id}")
            continue
        try:
            item = base.score(row, "noPRG", cfg)
            item["arm"] = "noPRG"
            z_error = base.finite(item.get("track_obj_z_abs_err_cm_mean"))
            pos_error = base.finite(item.get("track_obj_pos_err_cm_mean"))
            if not math.isfinite(z_error) or not math.isfinite(pos_error) or z_error > pos_error + 1e-9:
                raise ValueError(f"metric_contract:z_cm={z_error}:pos_cm={pos_error}")
            scored[case_id] = item
            ordered = [scored[r["case_id"]] for r in rows if r["case_id"] in scored]
            C.write_tsv(NOPRG_CACHE, ordered)
            print(f"[scored {index:02d}/72] noPRG {case_id}")
        except Exception as exc:  # noqa: BLE001
            errors.append({"arm": "noPRG", "case_id": case_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[error] noPRG {case_id}: {errors[-1]['error']}", file=sys.stderr)
    return [scored[row["case_id"]] for row in rows if row["case_id"] in scored], errors


def paired_rows(arms: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    rank = {key: index for index, key in enumerate(C.OBJECT_ORDER)}
    case_ids = sorted(arms["PRG"], key=lambda case: (rank[arms["PRG"][case]["object_key"]], case))
    for comparison, before_arm, after_arm in COMPARISONS:
        for case_id in case_ids:
            before, after = arms[before_arm][case_id], arms[after_arm][case_id]
            row: dict[str, Any] = {"comparison": comparison, "before_arm": before_arm, "after_arm": after_arm,
                                    "case_id": case_id, "object_key": after["object_key"],
                                    "retarget_variant_id": after.get("retarget_variant_id", "")}
            for metric in base.KEY_METRICS:
                before_value, after_value = base.finite(before.get(metric)), base.finite(after.get(metric))
                row[f"before_{metric}"] = before_value
                row[f"after_{metric}"] = after_value
                row[f"delta_{metric}"] = after_value - before_value
            for gate_field in base.GATE_FIELDS:
                row[f"before_{gate_field}"] = before.get(gate_field, "")
                row[f"after_{gate_field}"] = after.get(gate_field, "")
            row.update({"before_result_sha256": before.get("result_sha256", ""),
                        "after_result_sha256": after.get("result_sha256", ""),
                        "before_video": before.get("video", ""), "after_video": after.get("video", "")})
            output.append(row)
    return output


def summarize(paired: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_object: list[dict[str, Any]] = []
    migrations: list[dict[str, Any]] = []
    for comparison, before_arm, after_arm in COMPARISONS:
        for object_key in C.OBJECT_ORDER:
            group = [row for row in paired if row["comparison"] == comparison and row["object_key"] == object_key]
            summary: dict[str, Any] = {"comparison": comparison, "before_arm": before_arm, "after_arm": after_arm,
                                        "object_key": object_key, "n_cases": len(group)}
            for metric in base.KEY_METRICS:
                before = [base.finite(row[f"before_{metric}"]) for row in group]
                after = [base.finite(row[f"after_{metric}"]) for row in group]
                delta = [base.finite(row[f"delta_{metric}"]) for row in group]
                lo, hi = bootstrap(delta)
                summary.update({f"before_{metric}_mean": mean(before), f"after_{metric}_mean": mean(after),
                                f"delta_{metric}_mean": mean(delta), f"delta_{metric}_ci95_lo": lo,
                                f"delta_{metric}_ci95_hi": hi})
            for gate in base.ALL_GATES:
                field = f"{gate}_gate_pass"
                before_passes = sum(C.truth(row[f"before_{field}"]) for row in group)
                after_passes = sum(C.truth(row[f"after_{field}"]) for row in group)
                summary[f"before_{gate}_pass_rate"] = before_passes / len(group)
                summary[f"after_{gate}_pass_rate"] = after_passes / len(group)
                summary[f"delta_{gate}_pass_rate"] = (after_passes - before_passes) / len(group)
                for row in group:
                    b, a = C.truth(row[f"before_{field}"]), C.truth(row[f"after_{field}"])
                    transition = "PASS_TO_PASS" if b and a else "PASS_TO_FAIL" if b else "FAIL_TO_PASS" if a else "FAIL_TO_FAIL"
                    migrations.append({"comparison": comparison, "case_id": row["case_id"], "object_key": object_key,
                                       "gate": gate, "before_pass": b, "after_pass": a, "migration": transition})
            summary["before_12gate_pass_rate"] = mean([mean([float(C.truth(row[f"before_{gate}_gate_pass"])) for gate in base.ALL_GATES]) for row in group])
            summary["after_12gate_pass_rate"] = mean([mean([float(C.truth(row[f"after_{gate}_gate_pass"])) for gate in base.ALL_GATES]) for row in group])
            summary["delta_12gate_pass_rate"] = summary["after_12gate_pass_rate"] - summary["before_12gate_pass_rate"]
            by_object.append(summary)
    return by_object, migrations


def gate_tables(arms: dict[str, dict[str, dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail: list[dict[str, Any]] = []
    overall: list[dict[str, Any]] = []
    for object_key in (*C.OBJECT_ORDER, "ALL"):
        case_ids = [case_id for case_id, row in arms["PRG"].items()
                    if object_key == "ALL" or row["object_key"] == object_key]
        for gate in base.ALL_GATES:
            field = f"{gate}_gate_pass"
            row: dict[str, Any] = {"object_key": object_key, "gate": gate, "n_cases": len(case_ids)}
            for arm in ("noPRG", "PRG", "G1"):
                count = sum(C.truth(arms[arm][case_id][field]) for case_id in case_ids)
                row[f"{arm}_pass_count"] = count
                row[f"{arm}_pass_rate"] = count / len(case_ids)
            row["delta_PRG_vs_noPRG_pp"] = 100.0 * (row["PRG_pass_rate"] - row["noPRG_pass_rate"])
            row["delta_G1_vs_PRG_pp"] = 100.0 * (row["G1_pass_rate"] - row["PRG_pass_rate"])
            row["delta_G1_vs_noPRG_pp"] = 100.0 * (row["G1_pass_rate"] - row["noPRG_pass_rate"])
            for comparison, before_arm, after_arm in COMPARISONS:
                transitions = Counter()
                for case_id in case_ids:
                    before = C.truth(arms[before_arm][case_id][field])
                    after = C.truth(arms[after_arm][case_id][field])
                    transition = "PASS_TO_PASS" if before and after else "PASS_TO_FAIL" if before else "FAIL_TO_PASS" if after else "FAIL_TO_FAIL"
                    transitions[transition] += 1
                row[f"{comparison}_pass_to_fail"] = transitions["PASS_TO_FAIL"]
                row[f"{comparison}_fail_to_pass"] = transitions["FAIL_TO_PASS"]
                row[f"{comparison}_mcnemar_exact_p"] = mcnemar_exact(
                    transitions["PASS_TO_FAIL"], transitions["FAIL_TO_PASS"])
            detail.append(row)
        aggregate: dict[str, Any] = {"object_key": object_key, "n_cases": len(case_ids),
                                     "n_gate_decisions": len(case_ids) * len(base.ALL_GATES)}
        for arm in ("noPRG", "PRG", "G1"):
            gate_passes = sum(C.truth(arms[arm][case_id][f"{gate}_gate_pass"])
                              for case_id in case_ids for gate in base.ALL_GATES)
            strict_passes = sum(C.truth(arms[arm][case_id]["numeric_release_pass_12gate"])
                                for case_id in case_ids)
            aggregate.update({f"{arm}_gate_pass_count": gate_passes,
                              f"{arm}_gate_pass_rate": gate_passes / aggregate["n_gate_decisions"],
                              f"{arm}_strict12_pass_count": strict_passes,
                              f"{arm}_strict12_pass_rate": strict_passes / len(case_ids)})
        for metric in ("gate_pass_rate", "strict12_pass_rate"):
            aggregate[f"delta_PRG_vs_noPRG_{metric}_pp"] = 100.0 * (aggregate[f"PRG_{metric}"] - aggregate[f"noPRG_{metric}"])
            aggregate[f"delta_G1_vs_PRG_{metric}_pp"] = 100.0 * (aggregate[f"G1_{metric}"] - aggregate[f"PRG_{metric}"])
            aggregate[f"delta_G1_vs_noPRG_{metric}_pp"] = 100.0 * (aggregate[f"G1_{metric}"] - aggregate[f"noPRG_{metric}"])
        overall.append(aggregate)
    return detail, overall


def main() -> int:
    authority, audit = no_prg_authority()
    C.write_tsv(OUT / "e194_noprg_authority_audit.tsv", audit)
    prg, g1 = load_current_arms()
    noprg, errors = score_noprg(authority)
    arms = {name: {row["case_id"]: row for row in rows} for name, rows in (("noPRG", noprg), ("PRG", prg), ("G1", g1))}
    complete = all(len(rows) == C.N_CASES for rows in arms.values()) and len(set.intersection(*(set(rows) for rows in arms.values()))) == C.N_CASES
    paired = paired_rows(arms) if complete else []
    by_object, migrations = summarize(paired) if paired else ([], [])
    gate_detail, gate_overall = gate_tables(arms) if complete else ([], [])
    combined = noprg + prg + g1
    C.write_tsv(OUT / "e194_three_arm_case_metrics.tsv", combined)
    C.write_tsv(OUT / "e194_three_arm_paired_deltas.tsv", paired)
    C.write_tsv(OUT / "e194_three_arm_by_object.tsv", by_object)
    C.write_tsv(OUT / "e194_three_arm_gate_migrations.tsv", migrations)
    C.write_tsv(OUT / "e194_three_arm_12gate_by_object.tsv", gate_detail)
    C.write_tsv(OUT / "e194_three_arm_12gate_overall.tsv", gate_overall)
    C.write_tsv(OUT / "e194_three_arm_eval_errors.tsv", errors)
    status = "pass" if complete and len(combined) == 216 and len(paired) == 144 and len(migrations) == 1728 and len(gate_detail) == 48 and len(gate_overall) == 4 and not errors else "incomplete"
    payload = {"created_at": C.now(), "metric_standard_id": EVAL_METRIC_STANDARD_ID, "status": status,
               "arm_rows": {name: len(rows) for name, rows in arms.items()}, "case_metric_rows": len(combined),
               "paired_rows": len(paired), "gate_migration_rows": len(migrations), "errors": len(errors),
               "gate_metric_rows": len(gate_detail), "gate_overall_rows": len(gate_overall),
               "noPRG_authority": {"box001": "E189 Full", "box023": "E179 Full", "box021": "E168 production"},
               "by_object": by_object,
               "sha256": {name: C.sha256(OUT / filename) for name, filename in {
                   "case_metrics": "e194_three_arm_case_metrics.tsv", "paired_deltas": "e194_three_arm_paired_deltas.tsv",
                   "by_object": "e194_three_arm_by_object.tsv", "gate_migrations": "e194_three_arm_gate_migrations.tsv",
                   "gate_by_object": "e194_three_arm_12gate_by_object.tsv", "gate_overall": "e194_three_arm_12gate_overall.tsv",
                   "authority_audit": "e194_noprg_authority_audit.tsv"}.items()}}
    C.write_json(OUT / "e194_three_arm_summary.json", payload)
    print(json.dumps({key: payload[key] for key in ("status", "arm_rows", "paired_rows", "gate_migration_rows", "errors")}, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
