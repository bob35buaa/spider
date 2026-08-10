#!/usr/bin/env python3
"""Evaluate E192 A2 and E195 A3 with shared metrics and produce paired deltas."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT / "experiments/E195"))
sys.path.insert(0, str(SCRIPT_ROOT / "experiments/E168"))
sys.path.insert(0, str(SCRIPT_ROOT / "experiments/E189"))

from eval.core.core_metrics import EvalConfig, evaluate_sequence, npz_qpos  # noqa: E402
from eval.core.motion_health import body_motion_health, qpos_kinematic_health  # noqa: E402
from eval_E168_e167a_metrics import (  # noqa: E402
    fixed_reference_z_metrics,
    release_window_info,
    source_person_idx,
)
import e189_common as E189  # noqa: E402
import e195_common as C  # noqa: E402


PAIRED_METRICS = (
    "hand_object_physics_penetration_3mm_frame_frac",
    "hand_object_physics_contact_3mm_in_mask_frac",
    "hand_object_physics_contact_in_mask_frac",
    "leg_penetration_frac",
    "track_obj_pos_err_cm_mean",
    "track_obj_ori_err_deg_mean",
    "fall_flag",
    "hand_object_release_false_contact_3mm_frac",
    "hand_gate_fixed_depth_8mm_frame_frac",
    "hand_gate_fixed_depth_10mm_frame_frac",
    "hand_gate_fixed_depth_12mm_frame_frac",
    "hand_gate_fixed_depth_15mm_frame_frac",
    "hand_gate_fixed_depth_20mm_frame_frac",
)


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
        output = []
        for index, row in enumerate(values):
            if index >= len(steps) or int(steps[index]) <= 0:
                continue
            last = min(int(steps[index]) - 1, row.shape[-1] - 1)
            output.append(row[last])
        values = np.asarray(output)
    else:
        values = values[:, -1]
    return values[np.isfinite(values)]


def cem_diagnostics(path: Path, floor: float) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if not path.is_file():
        return output
    with np.load(path, allow_pickle=True) as data:
        for key in C.GATE_DIAGNOSTIC_KEYS:
            values = last_opt_values(data, key)
            output[f"{key}_mean"] = float(np.mean(values)) if values.size else math.nan
            output[f"{key}_min"] = float(np.min(values)) if values.size else math.nan
            output[f"{key}_p05"] = float(np.quantile(values, 0.05)) if values.size else math.nan
        selected_min = last_opt_values(data, "cem_hand_gate_selected_min_sdf_m")
        fallback = last_opt_values(data, "cem_gate_fallback_used")
        count = min(selected_min.size, fallback.size)
        nonfallback = selected_min[:count][fallback[:count] < 0.5]
        violations = nonfallback < floor
        output["cem_hand_gate_nonfallback_ticks"] = int(nonfallback.size)
        output["cem_hand_gate_nonfallback_selected_min_sdf_m"] = (
            float(np.min(nonfallback)) if nonfallback.size else math.nan
        )
        output["cem_hand_gate_nonfallback_hard_floor_violation_count"] = int(np.sum(violations))
        output["cem_hand_gate_nonfallback_hard_floor_violation_frac"] = (
            float(np.mean(violations)) if nonfallback.size else math.nan
        )
    return output


def evaluate_arm(
    row: dict[str, str], *, arm: str, method: str, floor: float,
) -> dict[str, Any]:
    qpos_path = C.repo_path(row["outdir_npz"])
    result_npz = C.repo_path(row["result_npz"])
    scene = C.repo_path(row["scene_act"])
    trajectory = C.repo_path(row["trajectory"])
    mask = C.repo_path(row["contact_mask"])
    for label, path in (
        ("outdir_npz", qpos_path), ("result_npz", result_npz),
        ("scene", scene), ("trajectory", trajectory), ("contact_mask", mask),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{row['case_id']}:{label}:{path}")
    cfg = EvalConfig(hand_gate_hard_floor_m=floor)
    person = source_person_idx(row)
    item = evaluate_sequence(
        row=row, method=method, hand_collision_variant_id=C.HAND_COLLISION_VARIANT_ID,
        qpos_path=qpos_path, scene_xml=scene, config=cfg,
        kin_ref_path=trajectory, contact_mask_path=mask, person_idx=person,
    )
    item.update(fixed_reference_z_metrics(qpos_path, scene, trajectory))
    item.update(release_window_info(mask, person, npz_qpos(qpos_path)[0].shape[0]))
    item.update(qpos_kinematic_health(qpos_path, cfg.fps))
    item.update(body_motion_health(qpos_path, scene, cfg))
    item.update(cem_diagnostics(result_npz, floor))
    item.update({
        "case_id": row["case_id"], "variant": row["variant"],
        "object_key": row["object_key"], "arm": arm, "stage": "full",
        "worker_id": row["worker_id"], "gpu_id": row["gpu_id"],
        "status": row["status"], "result_npz": C.rel(result_npz),
        "outdir_npz": C.rel(qpos_path), "trajectory": C.rel(trajectory),
        "contact_mask": C.rel(mask), "scene_xml": C.rel(scene),
    })
    return E189.apply_12gate_scoring(item)


def main() -> int:
    current_rows = C.read_tsv(C.RESULTS / "s6_downstream/manifests/cem_full_manifest.tsv")
    baseline_rows = C.load_e192_rows()
    current: dict[str, dict[str, Any]] = {}
    baseline: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    for row in current_rows:
        case_id = row["case_id"]
        try:
            baseline[case_id] = evaluate_arm(
                baseline_rows[case_id], arm="E192_A2", method="E192_A2",
                floor=C.E192_GATE["cem_hand_gate_hard_floor_m"],
            )
            current[case_id] = evaluate_arm(
                row, arm="E195_A3", method=C.METHOD_ID,
                floor=C.E195_GATE["cem_hand_gate_hard_floor_m"],
            )
        except Exception as exc:
            errors.append({"case_id": case_id, "error": f"{type(exc).__name__}:{exc}"})

    out_dir = C.RESULTS / "s6_downstream/eval/full"
    combined = list(baseline.values()) + list(current.values())
    fields: list[str] = []
    for row in combined:
        for key in row:
            if key not in fields:
                fields.append(key)
    C.write_tsv(out_dir / "e195_case_metrics.tsv", combined, fields)

    deltas: list[dict[str, Any]] = []
    for case_id in C.all_case_ids():
        if case_id not in baseline or case_id not in current:
            continue
        old, new = baseline[case_id], current[case_id]
        record: dict[str, Any] = {
            "case_id": case_id,
            "object_key": C.object_key_of(case_id),
            "e192_worker_id": baseline_rows[case_id]["worker_id"],
            "e195_worker_id": next(row["worker_id"] for row in current_rows if row["case_id"] == case_id),
        }
        for key in PAIRED_METRICS:
            a, b = finite(old.get(key)), finite(new.get(key))
            record[f"e192_{key}"] = a
            record[f"e195_{key}"] = b
            record[f"delta_{key}"] = b - a if math.isfinite(a) and math.isfinite(b) else math.nan
        for gate in E189.ALL_GATES:
            record[f"e192_{gate}_gate_pass"] = old.get(f"{gate}_gate_pass", "")
            record[f"e195_{gate}_gate_pass"] = new.get(f"{gate}_gate_pass", "")
        for key in C.GATE_DIAGNOSTIC_KEYS:
            record[f"e192_{key}"] = old.get(f"{key}_mean", "")
            record[f"e195_{key}"] = new.get(f"{key}_mean", "")
        for key in (
            "cem_hand_gate_nonfallback_ticks",
            "cem_hand_gate_nonfallback_selected_min_sdf_m",
            "cem_hand_gate_nonfallback_hard_floor_violation_count",
            "cem_hand_gate_nonfallback_hard_floor_violation_frac",
        ):
            record[f"e192_{key}"] = old.get(key, "")
            record[f"e195_{key}"] = new.get(key, "")
        deltas.append(record)
    delta_fields: list[str] = []
    for row in deltas:
        for key in row:
            if key not in delta_fields:
                delta_fields.append(key)
    C.write_tsv(out_dir / "e195_paired_deltas.tsv", deltas, delta_fields)
    status = "pass" if not errors and len(current) == 15 and len(baseline) == 15 else "incomplete"
    C.write_json(out_dir / "e195_eval_summary.json", {
        "created_at": C.now(), "baseline_rows": len(baseline),
        "current_rows": len(current), "paired_rows": len(deltas),
        "errors": errors, "status": status,
    })
    print(f"E195 eval: baseline={len(baseline)}/15 current={len(current)}/15 errors={len(errors)}")
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

