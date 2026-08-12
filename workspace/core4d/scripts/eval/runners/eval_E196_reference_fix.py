#!/usr/bin/env python3
"""Three-arm public-core evaluation for E196 corrected-reference reruns."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/E196"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reports"))

from eval.core.core_metrics import EVAL_METRIC_STANDARD_ID, EvalConfig  # noqa: E402
import eval_E194_G1_expansion as E194  # noqa: E402
import e196_reference_fix_common as C  # noqa: E402
from e194_orientation_reference_conversion import audit_case  # noqa: E402


E194_METRICS = C.E194_RESULTS / (
    "s6_downstream/eval/full_g1_expansion/e194_g1_expansion_case_metrics.tsv"
)
EVAL_DIR = C.RESULTS / "s6_downstream/eval/full_reference_fix"
ARMS = ("PRG", "G1_contaminated", "G1_corrected")
HIGHER_BETTER = {
    "hand_object_physics_contact_3mm_in_mask_frac",
    "hand_object_physics_contact_in_mask_frac",
}


def finite(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    lowered = str(value).strip().lower()
    if lowered in {"true", "false"}:
        return float(lowered == "true")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def historical_rows(case_ids: set[str]) -> dict[str, dict[str, dict[str, str]]]:
    selected = {case_id: {} for case_id in case_ids}
    for row in C.read_tsv(E194_METRICS):
        case_id = row.get("case_id", "")
        if case_id not in selected:
            continue
        if row.get("metric_standard_id") != EVAL_METRIC_STANDARD_ID:
            raise ValueError(
                f"historical metric standard differs for {case_id}: "
                f"{row.get('metric_standard_id')!r} != {EVAL_METRIC_STANDARD_ID!r}"
            )
        arm = "PRG" if row.get("arm") == "A0" else "G1_contaminated" if row.get("arm") == "G1" else ""
        if arm:
            selected[case_id][arm] = dict(row, arm=arm)
    missing = {
        case_id: sorted(set(ARMS[:2]) - set(arms))
        for case_id, arms in selected.items()
        if set(arms) != set(ARMS[:2])
    }
    if missing:
        raise ValueError(f"historical three-arm authority incomplete: {missing}")
    return selected


def score_corrected(row: dict[str, str], cfg: EvalConfig) -> dict[str, Any]:
    scoring = dict(row)
    scoring.update(
        {
            "source_exp": "E196",
            "execution_source": "E196",
            "reused_full": "false",
            "spider_method_id": "E167A_zOnlyBody_PRG_gravcomp_G1_reference_corrected",
        }
    )
    item = E194.score(scoring, "G1", cfg)
    item.update(
        {
            "arm": "G1_corrected",
            "reference_contract_version": row["reference_contract_version"],
            "resolved_euler_convention": row["resolved_euler_convention"],
            "scene_act_meta_sha256": row["scene_act_meta_sha256"],
            "wave": row["wave"],
        }
    )
    return item


def paired_row(
    row: dict[str, str],
    prg: dict[str, Any],
    contaminated: dict[str, Any],
    corrected: dict[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "case_id": row["case_id"],
        "object_key": row["object_key"],
        "retarget_variant_id": row["retarget_variant_id"],
        "worker": row["worker"],
        "wave": row["wave"],
        "box001_primary": not (
            row["case_id"] == "box001_20231023_110_p1"
        ),
    }
    for metric in E194.KEY_METRICS:
        values = {
            "prg": finite(prg.get(metric)),
            "contaminated_g1": finite(contaminated.get(metric)),
            "corrected_g1": finite(corrected.get(metric)),
        }
        output.update({f"{arm}_{metric}": value for arm, value in values.items()})
        for baseline in ("prg", "contaminated_g1"):
            raw = values["corrected_g1"] - values[baseline]
            output[f"raw_delta_corrected_vs_{baseline}_{metric}"] = raw
            output[f"improvement_corrected_vs_{baseline}_{metric}"] = (
                raw if metric in HIGHER_BETTER else -raw
            )
    for gate in E194.GATE_FIELDS:
        output[f"prg_{gate}"] = prg.get(gate, "")
        output[f"contaminated_g1_{gate}"] = contaminated.get(gate, "")
        output[f"corrected_g1_{gate}"] = corrected.get(gate, "")
    for gate in E194.ALL_GATES:
        for baseline, source in (("prg", prg), ("contaminated_g1", contaminated)):
            before = truth(source.get(f"{gate}_gate_pass"))
            after = truth(corrected.get(f"{gate}_gate_pass"))
            output[f"flip_corrected_vs_{baseline}_{gate}"] = (
                "PASS_TO_PASS" if before and after else
                "PASS_TO_FAIL" if before else
                "FAIL_TO_PASS" if after else
                "FAIL_TO_FAIL"
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scope", nargs="?", choices=("wave0", "full"), default="full")
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    manifest = C.WAVE0_MANIFEST if args.scope == "wave0" else C.FULL_MANIFEST
    rows = C.read_tsv(manifest)
    expected = 3 if args.scope == "wave0" else C.N_CASES
    complete = [
        row for row in rows
        if C.repo_path(row["result_npz"]).is_file()
        and C.repo_path(row["outdir_npz"]).is_file()
        and C.repo_path(row["config_act"]).is_file()
        and C.repo_path(row["log"]).is_file()
    ]
    if args.require_all and len(complete) != expected:
        raise SystemExit(f"E196 complete rows={len(complete)} expected={expected}")
    authority = historical_rows({row["case_id"] for row in complete})
    cfg = EvalConfig()
    corrected_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for row in complete:
        case_id = row["case_id"]
        try:
            corrected = score_corrected(row, cfg)
            prg = authority[case_id]["PRG"]
            contaminated = authority[case_id]["G1_contaminated"]
            corrected_rows.append(corrected)
            pair = paired_row(row, prg, contaminated, corrected)
            paired_rows.append(pair)
            reference = audit_case(
                corrected,
                finite(pair["raw_delta_corrected_vs_prg_track_obj_ori_err_deg_mean"]),
                C.repo_path(row["log"]),
            )
            reference.update(
                {
                    "expected_meta_sha256": row["scene_act_meta_sha256"],
                    "expected_convention": row["resolved_euler_convention"],
                    "reference_contract_version": row["reference_contract_version"],
                }
            )
            reference["integrity_pass"] = all(
                (
                    reference["runtime_convention_matches_xml_axes"],
                    reference["current_meta_matches_runtime"],
                    reference["runtime_euler_convention"] == row["resolved_euler_convention"],
                    reference["runtime_target_vs_raw_ori_err_deg_max"] < 1e-4,
                    reference["axis_target_vs_raw_ori_err_deg_max"] < 1e-4,
                    reference["g1_vs_raw_ori_err_reproduction_abs_diff_deg"] < 1e-6,
                )
            )
            integrity_rows.append(reference)
            print(f"[scored] {case_id}", flush=True)
        except Exception as exc:  # noqa: BLE001
            errors.append({"case_id": case_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[error] {case_id}: {errors[-1]['error']}", file=sys.stderr)
    historical: list[dict[str, Any]] = []
    for row in complete:
        historical.extend(authority[row["case_id"]][arm] for arm in ARMS[:2])
    C.write_tsv(EVAL_DIR / "e196_reference_fix_case_metrics.tsv", historical + corrected_rows)
    C.write_tsv(EVAL_DIR / "e196_reference_fix_by_case.tsv", paired_rows)
    C.write_tsv(EVAL_DIR / "e196_reference_integrity_audit.tsv", integrity_rows)
    C.write_tsv(EVAL_DIR / "e196_reference_fix_eval_errors.tsv", errors)
    status = (
        "pass"
        if len(complete) == len(corrected_rows) == len(paired_rows) == len(integrity_rows) == expected
        and not errors
        and all(row["integrity_pass"] for row in integrity_rows)
        else "incomplete"
    )
    summary = {
        "created_at": C.now(),
        "scope": args.scope,
        "expected": expected,
        "complete": len(complete),
        "corrected_scored": len(corrected_rows),
        "paired": len(paired_rows),
        "integrity_pass": sum(bool(row["integrity_pass"]) for row in integrity_rows),
        "errors": len(errors),
        "metric_standard_id": EVAL_METRIC_STANDARD_ID,
        "status": status,
    }
    C.write_json(EVAL_DIR / "e196_reference_fix_eval_summary.json", summary)
    print(summary)
    return 1 if args.require_all and status != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
