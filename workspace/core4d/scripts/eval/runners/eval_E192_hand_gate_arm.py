#!/usr/bin/env python3
"""E192 paired A0/A2 evaluation using public core metrics and E189 12-gate scoring."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT / "experiments/E192"))
sys.path.insert(0, str(SCRIPT_ROOT / "experiments/E168"))
sys.path.insert(0, str(SCRIPT_ROOT / "experiments/E189"))

from eval.core.core_metrics import EvalConfig, evaluate_sequence, npz_qpos  # noqa: E402
from eval.core.motion_health import body_motion_health, qpos_kinematic_health  # noqa: E402
from eval_E168_e167a_metrics import fixed_reference_z_metrics, release_window_info, source_person_idx  # noqa: E402
import e192_common as C  # noqa: E402
import e189_common as E189  # noqa: E402


BASELINE_METRICS = {
    "box004": C.REPO / "workspace/core4d/results/E172/s6_downstream/eval/full/e171_case_metrics.tsv",
    "box024": C.REPO / "workspace/core4d/results/E173/s6_downstream/eval/full/e173_case_metrics.tsv",
}


def finite(value: Any, default: float = math.nan) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return default
    return output if math.isfinite(output) else default


def last_opt_values(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in data.files:
        return np.asarray([], dtype=np.float64)
    values = np.asarray(data[key], dtype=np.float64)
    if values.ndim == 1:
        return values[np.isfinite(values)]
    if values.ndim < 2:
        return np.asarray([], dtype=np.float64)
    if "opt_steps" in data.files:
        steps = np.asarray(data["opt_steps"]).reshape(-1)
        out = []
        for i, row in enumerate(values):
            if i >= len(steps):
                break
            if int(steps[i]) <= 0:
                continue
            index = min(int(steps[i]) - 1, row.shape[-1] - 1)
            out.append(row[index])
        values = np.asarray(out)
    else:
        values = values[:, -1]
    return values[np.isfinite(values)]


def cem_diagnostics(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not path.is_file():
        return out
    with np.load(path, allow_pickle=True) as data:
        for key in C.GATE_DIAGNOSTIC_KEYS:
            values = last_opt_values(data, key)
            out[f"{key}_mean"] = float(np.mean(values)) if values.size else math.nan
            out[f"{key}_min"] = float(np.min(values)) if values.size else math.nan
            out[f"{key}_p05"] = float(np.quantile(values, 0.05)) if values.size else math.nan
        for key in (
            "cem_hand_gate_valid_frac", "cem_gate_valid_frac", "cem_gate_fallback_used",
            "cem_hand_gate_selected_valid_frac",
        ):
            values = last_opt_values(data, key)
            out[f"{key}_mean"] = float(np.mean(values)) if values.size else math.nan
            out[f"{key}_min"] = float(np.min(values)) if values.size else math.nan
        selected_min = last_opt_values(data, "cem_hand_gate_selected_min_sdf_m")
        fallback = last_opt_values(data, "cem_gate_fallback_used")
        count = min(selected_min.size, fallback.size)
        nonfallback = selected_min[:count][fallback[:count] < 0.5]
        floor = C.A2_GATE["cem_hand_gate_hard_floor_m"]
        violations = nonfallback < floor
        out["cem_hand_gate_nonfallback_ticks"] = int(nonfallback.size)
        out["cem_hand_gate_nonfallback_selected_min_sdf_m"] = (
            float(np.min(nonfallback)) if nonfallback.size else math.nan
        )
        out["cem_hand_gate_nonfallback_hard_floor_violation_count"] = int(np.sum(violations))
        out["cem_hand_gate_nonfallback_hard_floor_violation_frac"] = (
            float(np.mean(violations)) if nonfallback.size else math.nan
        )
    return out


def evaluate_current(row: dict[str, str], cfg: EvalConfig) -> dict[str, Any]:
    qpos_path = C.repo_path(row["outdir_npz"])
    result_npz = C.repo_path(row["result_npz"])
    scene = C.repo_path(row["scene_act"])
    trajectory = C.repo_path(row["trajectory"])
    mask = C.repo_path(row["contact_mask"])
    for label, path in (("outdir_npz", qpos_path), ("result_npz", result_npz), ("scene", scene), ("trajectory", trajectory), ("contact_mask", mask)):
        if not path.is_file():
            raise FileNotFoundError(f"{row['case_id']}:{label}:{path}")
    person = source_person_idx(row)
    item = evaluate_sequence(
        row=row, method=C.METHOD_ID,
        hand_collision_variant_id=C.HAND_COLLISION_VARIANT_ID,
        qpos_path=qpos_path, scene_xml=scene, config=cfg,
        kin_ref_path=trajectory, contact_mask_path=mask, person_idx=person,
    )
    item.update(fixed_reference_z_metrics(qpos_path, scene, trajectory))
    item.update(release_window_info(mask, person, npz_qpos(qpos_path)[0].shape[0]))
    item.update(qpos_kinematic_health(qpos_path, cfg.fps))
    item.update(body_motion_health(qpos_path, scene, cfg))
    item.update(cem_diagnostics(result_npz))
    item.update({key: row.get(key, "") for key in ("case_id", "variant", "object_key", "arm", "stage", "worker_id", "gpu_id", "retarget_variant_id", "target_variant_id", "status")})
    item["result_npz"] = C.rel(result_npz)
    item["outdir_npz"] = C.rel(qpos_path)
    item["trajectory"] = C.rel(trajectory)
    item["contact_mask"] = C.rel(mask)
    item["scene_xml"] = C.rel(scene)
    return E189.apply_12gate_scoring(item)


def load_baseline() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for object_key, path in BASELINE_METRICS.items():
        for row in C.read_tsv(path):
            if row["case_id"] in C.CASES[object_key]:
                out[row["case_id"]] = {**row, "object_key": object_key, "arm": "A0-history"}
    if len(out) != 15:
        raise RuntimeError(f"baseline metrics rows={len(out)} != 15")
    return out


def baseline_fixed_depth(base: dict[str, str], source: dict[str, str], cfg: EvalConfig) -> dict[str, Any]:
    """Re-score only fixed-depth physics diagnostics for historical A0."""
    qpos_path = C.repo_path(source["outdir_npz"])
    scene = C.repo_path(source["scene_act"])
    mask = C.repo_path(source["contact_mask"])
    trajectory = C.repo_path(source["trajectory"])
    if not (qpos_path.is_file() and scene.is_file() and mask.is_file()):
        return {}
    item = evaluate_sequence(
        row=source, method="A0-history", hand_collision_variant_id=C.HAND_COLLISION_VARIANT_ID,
        qpos_path=qpos_path, scene_xml=scene, config=cfg,
        kin_ref_path=trajectory if trajectory.is_file() else None,
        contact_mask_path=mask, person_idx=source_person_idx(source),
    )
    return {key: item.get(key) for key in ("hand_gate_floor_saturation_frac", "hand_gate_fixed_depth_10mm_frame_frac", "hand_gate_fixed_depth_15mm_frame_frac", "hand_gate_fixed_depth_20mm_frame_frac")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("baseline_sentinel", "canary", "full"), nargs="?", default="full")
    parser.add_argument(
        "--skip-case",
        action="append",
        default=[],
        help="Exclude a case_id from this evaluation (repeatable; keeps the frozen manifest unchanged).",
    )
    args = parser.parse_args()
    manifest_name = "cem_baseline_sentinel_manifest.tsv" if args.stage == "baseline_sentinel" else f"cem_{args.stage}_manifest.tsv"
    manifest = C.RESULTS / "s6_downstream/manifests" / manifest_name
    skipped_cases = set(args.skip_case)
    rows = [row for row in C.read_tsv(manifest) if row["case_id"] not in skipped_cases]
    baseline = load_baseline()
    sources = C.load_source_rows()
    cfg = EvalConfig()
    current: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for row in rows:
        try:
            current.append(evaluate_current(row, cfg))
        except Exception as exc:  # keep complete error ledger; never hide missing rows
            errors.append({"variant": row["variant"], "error": f"{type(exc).__name__}:{exc}"})
    for case_id, base in baseline.items():
        base.update(cem_diagnostics(C.repo_path(sources[case_id]["result_npz"])))
        base.update(baseline_fixed_depth(base, sources[case_id], cfg))
        baseline[case_id] = E189.apply_12gate_scoring(base)

    out_dir = C.RESULTS / f"s6_downstream/eval/{args.stage}"
    fields: list[str] = []
    combined = list(baseline.values()) + current
    for row in combined:
        for key in row:
            if key not in fields:
                fields.append(key)
    C.write_tsv(out_dir / "e192_case_metrics.tsv", combined, fields)
    if args.stage == "baseline_sentinel":
        comparisons = []
        for item in current:
            old = baseline.get(item["case_id"], {})
            rec = {"case_id": item["case_id"], "object_key": item["object_key"]}
            for key in ("numeric_release_pass_12gate", "hand_object_physics_penetration_3mm_frame_frac", "hand_object_physics_contact_in_mask_frac", "leg_penetration_frac", "track_obj_pos_err_cm_mean", "track_obj_ori_err_deg_mean", "cem_hand_gate_valid_frac_mean", "cem_gate_fallback_used_mean"):
                rec[f"current_{key}"] = item.get(key, "")
                rec[f"history_{key}"] = old.get(key, "")
            comparisons.append(rec)
        C.write_tsv(out_dir / "e192_sentinel_comparison.tsv", comparisons, list(comparisons[0]) if comparisons else ["case_id", "object_key"])
    else:
        deltas = []
        for item in current:
            old = baseline.get(item["case_id"])
            if old is None: continue
            rec = {"case_id": item["case_id"], "object_key": item["object_key"], "arm": item.get("arm", "A2")}
            for key in ("hand_object_physics_penetration_3mm_frame_frac", "hand_object_physics_contact_3mm_in_mask_frac", "hand_object_physics_contact_in_mask_frac", "leg_penetration_frac", "track_obj_pos_err_cm_mean", "track_obj_ori_err_deg_mean", "hand_gate_fixed_depth_10mm_frame_frac", "hand_gate_fixed_depth_15mm_frame_frac", "hand_gate_fixed_depth_20mm_frame_frac"):
                a, b = finite(item.get(key)), finite(old.get(key))
                rec[f"a0_{key}"] = b; rec[f"a2_{key}"] = a; rec[f"delta_{key}"] = a - b if math.isfinite(a) and math.isfinite(b) else math.nan
            for gate in E189.ALL_GATES:
                rec[f"a0_{gate}_gate_pass"] = old.get(f"{gate}_gate_pass", "")
                rec[f"a2_{gate}_gate_pass"] = item.get(f"{gate}_gate_pass", "")
            for key in C.GATE_DIAGNOSTIC_KEYS:
                rec[f"a0_{key}"] = old.get(f"{key}_mean", old.get(key, ""))
                rec[f"a2_{key}"] = item.get(f"{key}_mean", "")
            for key in (
                "cem_hand_gate_nonfallback_ticks",
                "cem_hand_gate_nonfallback_selected_min_sdf_m",
                "cem_hand_gate_nonfallback_hard_floor_violation_count",
                "cem_hand_gate_nonfallback_hard_floor_violation_frac",
            ):
                rec[f"a2_{key}"] = item.get(key, "")
            deltas.append(rec)
        delta_fields = []
        for row in deltas:
            for key in row:
                if key not in delta_fields: delta_fields.append(key)
        C.write_tsv(out_dir / "e192_paired_deltas.tsv", deltas, delta_fields)
    C.write_json(out_dir / "e192_eval_summary.json", {"created_at": C.now(), "stage": args.stage, "manifest_rows": len(rows), "evaluated": len(current), "skipped_cases": sorted(skipped_cases), "errors": errors, "status": "pass" if not errors and len(current) == len(rows) else "incomplete"})
    print(f"E192 eval stage={args.stage}: evaluated={len(current)}/{len(rows)} errors={len(errors)}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
