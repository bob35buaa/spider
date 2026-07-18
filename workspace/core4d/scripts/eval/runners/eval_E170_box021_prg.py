#!/usr/bin/env python3
"""Unified 28-case E170 PRG evaluation with frozen E168 paired baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval.core.core_metrics import EVAL_METRIC_STANDARD_ID, METRIC_FIELDS, EvalConfig, evaluate_sequence, npz_qpos  # noqa: E402
from eval.core.motion_health import HEALTH_AGGS, METRIC_KEYS, run_health  # noqa: E402
from eval_E168_e167a_metrics import fixed_reference_z_metrics, release_window_info, source_person_idx  # noqa: E402


REPO = Path(__file__).resolve().parents[5]
DEFAULT_MANIFEST = REPO / "workspace/core4d/results/E170/s6_downstream/manifests/analysis_manifest.tsv"
DEFAULT_BASELINE = REPO / "workspace/core4d/results/E168/s6_downstream/cem/eval/box021_all28_reviewed/e168_case_metrics.tsv"
DEFAULT_OUT = REPO / "workspace/core4d/results/E170/s6_downstream/eval/full"

CONTACT_MIN = 0.50
BODY_Z_MAX = 0.20
HAND_PEN_MAX = 0.30
LEG_PEN_MAX = 0.10
RELEASE_MAX = 0.30
GATE_FALLBACK_MAX = 0.10
GATE_VALID_LAST_MIN = 0.05

PAIR_METRICS = {
    "body_z_err_p95_m": "lower",
    "track_pelvis_z_err_terminal_m": "lower",
    "hand_object_physics_contact_in_mask_frac": "higher",
    "hand_object_physics_contact_3mm_in_mask_frac": "higher",
    "hand_object_release_false_contact_3mm_frac": "lower",
    "hand_object_physics_penetration_3mm_frame_frac": "lower",
    "leg_penetration_frac": "lower",
    "leg_near_2cm_frac": "lower",
    "leg_object_physics_contact_frac": "lower",
    "track_root_pos_err_cm_mean": "lower",
    "track_root_ori_err_deg_mean": "lower",
    "track_eef_pos_err_cm_mean": "lower",
    "track_eef_ori_err_deg_mean": "lower",
    "track_obj_pos_err_cm_mean": "lower",
    "track_obj_ori_err_deg_mean": "lower",
    "qpos_accel_l2_p95": "lower",
    "qpos_jerk_l2_p95": "lower",
    "trackbody_jerk_p95": "lower",
    "ankle_jerk_p95": "lower",
    "obj_speed_max": "lower",
    "foot_slip_max_m": "lower",
}

LEG_GATE_KEYS = (
    "cem_leg_gate_valid_frac", "cem_leg_gate_selected_valid_frac",
    "cem_leg_gate_fallback_used", "cem_leg_gate_min_sdf_min_m",
    "cem_leg_gate_min_sdf_p05_m", "cem_leg_gate_violation_pct_mean",
    "cem_leg_gate_selected_all_valid",
)


def repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.exists():
        return path.resolve()
    text = str(raw)
    for marker in ("example_datasets/", "workspace/", "logs/"):
        if marker in text:
            return REPO / (marker + text.split(marker, 1)[1])
    return path if path.is_absolute() else REPO / path


def rel(raw: str | Path) -> str:
    path = Path(raw)
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path)


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def serial(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serial(row.get(key, "")) for key in fields})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any, default: float = math.nan) -> float:
    try:
        output = float(value)
    except Exception:
        return default
    return output if math.isfinite(output) else default


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def array_stat(data: np.lib.npyio.NpzFile, key: str, mode: str) -> float:
    if key not in data.files:
        return math.nan
    values = np.asarray(data[key], dtype=np.float64)
    if mode == "last":
        if values.ndim != 2 or not values.shape[1]:
            return math.nan
        values = values[:, -1]
    values = values[np.isfinite(values)]
    if not values.size:
        return math.nan
    return float(values.min() if mode == "min" else values.mean())


def gate_health(path: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    with np.load(path, allow_pickle=True) as data:
        for key in LEG_GATE_KEYS:
            output[f"{key}_mean"] = array_stat(data, key, "mean")
            output[f"{key}_last_iter_mean"] = array_stat(data, key, "last")
        output["cem_leg_gate_min_sdf_worst_m"] = array_stat(data, "cem_leg_gate_min_sdf_min_m", "min")
    fallback = finite(output["cem_leg_gate_fallback_used_mean"], math.inf)
    valid_last = finite(output["cem_leg_gate_valid_frac_last_iter_mean"], -math.inf)
    selected_all = finite(output["cem_leg_gate_selected_all_valid_mean"], -math.inf)
    output["leg_gate_fallback_pass"] = fallback <= GATE_FALLBACK_MAX
    output["leg_gate_valid_frac_pass"] = valid_last >= GATE_VALID_LAST_MIN
    output["leg_gate_selected_valid_pass"] = selected_all >= 1.0 - 1e-9
    output["leg_gate_health_pass"] = bool(output["leg_gate_fallback_pass"] and output["leg_gate_valid_frac_pass"] and output["leg_gate_selected_valid_pass"])
    return output


def review_fields(case_id: str, reviews: dict[str, dict[str, str]], codex: dict[str, dict[str, str]]) -> dict[str, Any]:
    manual = reviews.get(case_id, {})
    machine = codex.get(case_id, {})
    status = manual.get("user_manual_review_status", "pending")
    decision = manual.get("manual_use_decision", "PENDING")
    return {
        "codex_metric_verification_status": machine.get("codex_metric_verification_status", "PENDING_METRIC_VERIFICATION"),
        "codex_visual_spotcheck_status": machine.get("codex_visual_spotcheck_status", "NOT_SELECTED"),
        "codex_visual_findings": machine.get("codex_visual_findings", ""),
        "user_manual_review_status": status,
        "manual_use_decision": decision,
        "manual_quality_label": manual.get("manual_quality_label", ""),
        "manual_failure_taxonomy": manual.get("manual_failure_taxonomy", ""),
        "manual_review_note": manual.get("manual_review_note", ""),
        "manual_reviewer": manual.get("manual_reviewer", ""),
        "manual_reviewed_at": manual.get("manual_reviewed_at", ""),
        "manual_operational_use": status == "reviewed" and decision == "USE",
    }


def apply_gates(item: dict[str, Any]) -> None:
    release_applicable = bool(item.get("release_gate_applicable"))
    gates = {
        "fall": not bool(item.get("fall_flag")),
        "body_z": finite(item.get("body_z_err_p95_m"), math.inf) <= BODY_Z_MAX,
        "contact": finite(item.get("hand_object_physics_contact_in_mask_frac"), -math.inf) >= CONTACT_MIN,
        "release": (not release_applicable) or finite(item.get("hand_object_release_false_contact_3mm_frac"), math.inf) <= RELEASE_MAX,
        "hand_penetration": finite(item.get("hand_object_physics_penetration_3mm_frame_frac"), math.inf) <= HAND_PEN_MAX,
        "lower_body": finite(item.get("leg_penetration_frac"), math.inf) <= LEG_PEN_MAX,
    }
    for name, passed in gates.items():
        item[f"{name}_gate_pass"] = passed
    failures = [name for name, passed in gates.items() if not passed]
    item["numeric_release_pass"] = not failures
    item["numeric_failure_modes"] = ",".join(failures)
    item["strict_release_usable"] = bool(item["numeric_release_pass"] and item.get("manual_operational_use"))


def evaluate_row(row: dict[str, str], cfg: EvalConfig, reviews: dict[str, dict[str, str]], codex: dict[str, dict[str, str]]) -> dict[str, Any]:
    qpos_path, root_npz = repo_path(row["outdir_npz"]), repo_path(row["result_npz"])
    scene, trajectory, mask = repo_path(row["scene_act"]), repo_path(row["trajectory"]), repo_path(row["contact_mask"])
    item = evaluate_sequence(row=row, method=row["spider_method_id"], hand_collision_variant_id=row["hand_collision_variant_id"], qpos_path=qpos_path, scene_xml=scene, config=cfg, kin_ref_path=trajectory, contact_mask_path=mask, person_idx=source_person_idx(row))
    sim_qpos, _ = npz_qpos(qpos_path)
    item.update(run_health(qpos_path, scene, cfg))
    item.update(fixed_reference_z_metrics(qpos_path, scene, trajectory))
    item.update(release_window_info(mask, source_person_idx(row), len(sim_qpos)))
    item.update(gate_health(root_npz))
    item.update(review_fields(row["case_id"], reviews, codex))
    for key in ("ordinal", "variant", "case_id", "sequence_key", "source_person", "source_action", "source_obstacle_level", "execution_source", "reused_full", "retarget_variant_id", "target_variant_id", "hand_collision_variant_id", "spider_method_id", "assigned_gpu", "status"):
        item[key] = row.get(key, "")
    item.update({"metric_standard_id": EVAL_METRIC_STANDARD_ID, "result_npz": rel(root_npz), "outdir_npz": rel(qpos_path), "config_act": rel(row["config_act"]), "video": rel(row["video"]), "trajectory": rel(trajectory), "contact_mask": rel(mask), "scene_xml": rel(scene)})
    apply_gates(item)
    return item


def add_baseline(item: dict[str, Any], baseline: dict[str, str]) -> dict[str, Any]:
    row = {"case_id": item["case_id"], "e168_manual_use_decision": baseline.get("manual_use_decision", "")}
    for metric, direction in PAIR_METRICS.items():
        current, old = finite(item.get(metric)), finite(baseline.get(metric))
        delta = current - old if math.isfinite(current) and math.isfinite(old) else math.nan
        item[f"e168_{metric}"] = old
        item[f"delta_{metric}"] = delta
        item[f"improvement_{metric}"] = delta if direction == "higher" else -delta
        row["prg_" + metric] = current
        row["e168_" + metric] = old
        row["delta_" + metric] = delta
    row.update({"numeric_release_pass": item["numeric_release_pass"], "numeric_failure_modes": item["numeric_failure_modes"], "manual_use_decision": item["manual_use_decision"], "manual_operational_use": item["manual_operational_use"], "strict_release_usable": item["strict_release_usable"], "video": item["video"]})
    return row


def aggregate(values: list[Any]) -> tuple[int, float, float, float]:
    clean = [finite(value) for value in values]
    clean = [value for value in clean if math.isfinite(value)]
    if not clean:
        return 0, math.nan, math.nan, math.nan
    return len(clean), statistics.fmean(clean), statistics.median(clean), float(np.percentile(clean, 95))


def group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[tuple[str, str, list[dict[str, Any]]]] = [("overall", "all", rows)]
    for field in ("e168_manual_use_decision", "source_person", "retarget_variant_id", "execution_source", "source_action", "source_obstacle_level"):
        for value in sorted({str(row.get(field, "")) for row in rows}):
            groups.append((field, value, [row for row in rows if str(row.get(field, "")) == value]))
    output = []
    for group_type, value, subset in groups:
        base = {"group_type": group_type, "group_value": value, "rows": len(subset), "numeric_pass": sum(bool(row["numeric_release_pass"]) for row in subset), "manual_reviewed": sum(row["user_manual_review_status"] == "reviewed" for row in subset), "manual_operational_use": sum(bool(row["manual_operational_use"]) for row in subset), "strict_release_usable": sum(bool(row["strict_release_usable"]) for row in subset), "gate_health_pass": sum(bool(row["leg_gate_health_pass"]) for row in subset), "fall_count": sum(bool(row.get("fall_flag")) for row in subset)}
        for metric in PAIR_METRICS:
            n, mean, median, p95 = aggregate([row.get(metric) for row in subset])
            base.update({f"{metric}_n": n, f"{metric}_mean": mean, f"{metric}_median": median, f"{metric}_p95": p95})
        output.append(base)
    return output


def worst_cases(rows: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    output = []
    for metric, direction in PAIR_METRICS.items():
        valid = [row for row in rows if math.isfinite(finite(row.get(metric)))]
        valid.sort(key=lambda row: finite(row.get(metric)), reverse=direction == "lower")
        for rank, row in enumerate(valid[:top_k], 1):
            output.append({"metric": metric, "worse_direction": "high" if direction == "lower" else "low", "rank": rank, "case_id": row["case_id"], "value": row.get(metric), "numeric_failure_modes": row["numeric_failure_modes"], "video": row["video"]})
    return output


def manual_template(manifest_rows: list[dict[str, str]], existing: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for row in manifest_rows:
        old = existing.get(row["case_id"], {})
        output.append({"case_id": row["case_id"], "user_manual_review_status": old.get("user_manual_review_status", "pending"), "manual_use_decision": old.get("manual_use_decision", "PENDING"), "manual_quality_label": old.get("manual_quality_label", ""), "manual_failure_taxonomy": old.get("manual_failure_taxonomy", ""), "manual_review_note": old.get("manual_review_note", ""), "manual_reviewer": old.get("manual_reviewer", ""), "manual_reviewed_at": old.get("manual_reviewed_at", ""), "paired_video": f"workspace/core4d/results/E170/s6_downstream/render/full/paired/{row['case_id']}_E168_vs_E170_PRG.mp4"})
    return output


def recommendation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reviewed = sum(row["user_manual_review_status"] == "reviewed" for row in rows)
    overall = sum(bool(row["strict_release_usable"]) for row in rows)
    recovery = sum(bool(row["strict_release_usable"]) for row in rows if row["e168_manual_use_decision"] == "DO_NOT_USE")
    retention = sum(bool(row["strict_release_usable"]) for row in rows if row["e168_manual_use_decision"] == "USE")
    if len(rows) != 28 or reviewed != 28:
        value = "PENDING_USER_REVIEW"
    elif overall >= 22 and recovery >= 9 and retention >= 12:
        value = "STRONG"
    elif overall >= 18 and recovery >= 6 and retention >= 10:
        value = "PARTIAL"
    else:
        value = "FAIL"
    return {"machine_recommendation": value, "strict_overall": overall, "strict_recovery": recovery, "strict_retention": retention, "manual_reviewed": reviewed, "manual_operational_use": sum(bool(row["manual_operational_use"]) for row in rows), "gate_health_pass": sum(bool(row["leg_gate_health_pass"]) for row in rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    manifest, baseline_path, out_dir = repo_path(args.manifest), repo_path(args.baseline), repo_path(args.out_dir)
    manifest_rows = read_tsv(manifest)
    baselines = {row["case_id"]: row for row in read_tsv(baseline_path)}
    review_path, codex_path = out_dir / "user_manual_review_template.tsv", out_dir / "codex_verification.tsv"
    review_rows, codex_rows = read_tsv(review_path), read_tsv(codex_path)
    reviews, codex = {row["case_id"]: row for row in review_rows}, {row["case_id"]: row for row in codex_rows}
    required = ("result_npz", "outdir_npz", "config_act", "scene_act", "trajectory", "contact_mask")
    ready, not_ready = [], []
    for row in manifest_rows:
        missing = [key for key in required if not repo_path(row.get(key, "")).is_file()]
        if missing:
            not_ready.append({**row, "missing_artifacts": ",".join(missing)})
        else:
            ready.append(row)
    cfg, metrics, errors = EvalConfig(), [], []
    for index, row in enumerate(ready, 1):
        print(f"[{index}/{len(ready)}] {row['case_id']}", flush=True)
        try:
            metrics.append(evaluate_row(row, cfg, reviews, codex))
        except Exception as exc:
            errors.append({"case_id": row["case_id"], "variant": row["variant"], "error": f"{type(exc).__name__}: {exc}"})
    paired = []
    for item in metrics:
        baseline = baselines.get(item["case_id"])
        if baseline is None:
            errors.append({"case_id": item["case_id"], "variant": item["variant"], "error": "missing E168 baseline"})
        else:
            item["e168_manual_use_decision"] = baseline.get("manual_use_decision", "")
            paired.append(add_baseline(item, baseline))
    summaries, rankings = group_summary(metrics), worst_cases(metrics)
    rec = recommendation(metrics)
    failures = Counter(mode for row in metrics for mode in row["numeric_failure_modes"].split(",") if mode)
    payload = {"generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "metric_standard_id": EVAL_METRIC_STANDARD_ID, "manifest": rel(manifest), "manifest_sha256": sha256(manifest), "baseline": rel(baseline_path), "baseline_sha256": sha256(baseline_path), "counts": {"manifest_rows": len(manifest_rows), "evaluated": len(metrics), "not_ready": len(not_ready), "errors": len(errors), "numeric_pass": sum(bool(row["numeric_release_pass"]) for row in metrics)}, "numeric_failure_counts": dict(failures), "recommendation": rec, "thresholds": {"body_z_err_p95_m_max": BODY_Z_MAX, "raw_contact_min": CONTACT_MIN, "release_false_3mm_max": RELEASE_MAX, "hand_penetration_3mm_max": HAND_PEN_MAX, "leg_penetration_max": LEG_PEN_MAX, "gate_fallback_max": GATE_FALLBACK_MAX, "gate_valid_last_min": GATE_VALID_LAST_MIN}}
    priority = ["case_id", "variant", "execution_source", "reused_full", "e168_manual_use_decision", "numeric_release_pass", "numeric_failure_modes", "manual_operational_use", "strict_release_usable", "codex_metric_verification_status", "codex_visual_spotcheck_status", "codex_visual_findings", "user_manual_review_status", "manual_use_decision", "manual_quality_label", "manual_failure_taxonomy", "leg_gate_health_pass", *PAIR_METRICS]
    fields = []
    for key in [*priority, *METRIC_FIELDS, *METRIC_KEYS, *HEALTH_AGGS.keys()]:
        if key not in fields:
            fields.append(key)
    for row in metrics:
        for key in row:
            if key not in fields:
                fields.append(key)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(out_dir / "e170_case_metrics.tsv", metrics, fields)
    write_tsv(out_dir / "e170_paired_deltas.tsv", paired)
    write_tsv(out_dir / "e170_group_summary.tsv", summaries)
    write_tsv(out_dir / "e170_worst_cases.tsv", rankings)
    write_tsv(out_dir / "e170_not_ready.tsv", not_ready)
    write_tsv(out_dir / "e170_evaluation_errors.tsv", errors)
    write_tsv(out_dir / "evaluated_manifest_snapshot.tsv", ready)
    write_tsv(review_path, manual_template(manifest_rows, reviews))
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts"], sort_keys=True))
    if errors:
        return 1
    if args.require_all and (len(metrics) != 28 or not_ready):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
