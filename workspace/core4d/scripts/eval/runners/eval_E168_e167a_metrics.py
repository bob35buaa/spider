#!/usr/bin/env python3
"""Evaluate available E168 CEM rows with the E167A-aligned metric contract."""

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

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval.core.core_metrics import (  # noqa: E402
    EVAL_METRIC_STANDARD_ID,
    METRIC_FIELDS,
    EvalConfig,
    evaluate_sequence,
    npz_qpos,
)
from eval.core.motion_health import (  # noqa: E402
    HEALTH_AGGS,
    METRIC_KEYS,
    fps_from_npz,
    run_health,
)


REPO = Path(__file__).resolve().parents[5]
DEFAULT_MANIFEST = (
    REPO
    / "workspace/core4d/results/E168/s6_downstream/cem/manifests/cem_production_manifest.tsv"
)
DEFAULT_OUT_DIR = (
    REPO
    / "workspace/core4d/results/E168/s6_downstream/cem/eval/e167a_aligned_available"
)
DEFAULT_MANUAL_REVIEW = (
    REPO
    / "workspace/core4d/results/E168/s6_downstream/cem/eval/manual_review/e168_user_visual_review.tsv"
)

MONITORED_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]
BODY_Z_P95_THRESHOLD_M = 0.20
LEGACY_BODY_Z_PEAK_THRESHOLD_M = 0.25
SUGAR_3D_THRESHOLD_M = 0.30
CONTACT_MIN = 0.50
RELEASE_FALSE_MAX = 0.30
PENETRATION_3MM_MAX = 0.30
LOWER_BODY_INTERFERENCE_MAX = 0.10

SUMMARY_METRICS = [
    "track_pelvis_z_err_terminal_m",
    "body_z_err_p95_m",
    "body_z_err_peak_m",
    "sugar_3d_err_peak_m",
    "track_joint_err_deg_mean",
    "track_eef_pos_err_cm_mean",
    "track_eef_ori_err_deg_mean",
    "track_obj_pos_err_cm_mean",
    "track_obj_ori_err_deg_mean",
    "track_root_pos_err_cm_mean",
    "track_root_ori_err_deg_mean",
    "hand_object_physics_contact_in_mask_frac",
    "hand_object_physics_contact_3mm_in_mask_frac",
    "hand_object_release_false_contact_3mm_frac",
    "hand_object_physics_penetration_3mm_frame_frac",
    "hand_geom_penetration_2mm_frac",
    "hand_geom_penetration_5mm_frac",
    "leg_penetration_frac",
    "body_penetration_frac",
    "qpos_jerk_l2_p95",
    "trackbody_jerk_p95",
    "ankle_jerk_p95",
    "foot_slip_max_m",
]

RANKING_METRICS = {
    "track_pelvis_z_err_terminal_m": "high",
    "body_z_err_p95_m": "high",
    "body_z_err_peak_m": "high",
    "sugar_3d_err_peak_m": "high",
    "track_joint_err_deg_mean": "high",
    "track_eef_pos_err_cm_mean": "high",
    "track_obj_pos_err_cm_mean": "high",
    "hand_object_physics_contact_in_mask_frac": "low",
    "hand_object_release_false_contact_3mm_frac": "high",
    "hand_object_physics_penetration_3mm_frame_frac": "high",
    "leg_penetration_frac": "high",
    "qpos_jerk_l2_p95": "high",
}


def repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.exists():
        return path.resolve()
    text = str(raw)
    for marker in ("example_datasets/", "workspace/", "logs/"):
        if marker in text:
            return REPO / (marker + text.split(marker, 1)[1])
    return path if path.is_absolute() else REPO / path


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO.resolve()))
    except Exception:
        return str(path)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def format_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_tsv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def finite(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    return number if math.isfinite(number) else default


def mean(values: list[Any]) -> float:
    valid = [finite(value) for value in values]
    valid = [value for value in valid if math.isfinite(value)]
    return statistics.fmean(valid) if valid else math.nan


def percentile(values: list[Any], q: float) -> float:
    valid = np.asarray([finite(value) for value in values], dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    return float(np.percentile(valid, q)) if valid.size else math.nan


def p95(values: np.ndarray) -> float:
    valid = np.asarray(values, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    return float(np.percentile(valid, 95)) if valid.size else math.nan


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_person_idx(row: dict[str, str]) -> int:
    person = row.get("source_person", "").lower()
    if person in {"person1", "p1", "0"}:
        return 0
    if person in {"person2", "p2", "1"}:
        return 1
    return 0 if row.get("case_id", "").lower().endswith("_p1") else 1


def load_reference_qpos(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        qpos = np.asarray(data["qpos"], dtype=np.float64)
    if qpos.ndim == 3:
        qpos = qpos[:, 0, :]
    if qpos.ndim != 2:
        raise ValueError(f"unsupported reference qpos shape {qpos.shape}: {path}")
    return qpos


def scene_euler_convention(scene_xml: Path) -> str:
    meta_path = scene_xml.with_name("scene_act_meta.json")
    if not meta_path.is_file():
        return "XYZ"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    return str(payload.get("euler_convention", "XYZ"))


def convert_reference_to_scene_act(qpos: np.ndarray, scene_xml: Path) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    if qpos.shape[1] == model.nq:
        return qpos.astype(np.float64, copy=True)
    nq_robot = model.nq - 6
    if qpos.shape[1] < nq_robot + 7:
        raise ValueError(
            f"cannot convert reference shape {qpos.shape} to scene nq={model.nq}: {scene_xml}"
        )
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if object_id < 0:
        raise ValueError(f"scene has no object body: {scene_xml}")

    object_pos_world = qpos[:, nq_robot : nq_robot + 3]
    object_quat_wxyz = qpos[:, nq_robot + 3 : nq_robot + 7]
    body_pos = model.body_pos[object_id]
    body_quat_wxyz = model.body_quat[object_id]
    body_rot = R.from_quat(
        [
            body_quat_wxyz[1],
            body_quat_wxyz[2],
            body_quat_wxyz[3],
            body_quat_wxyz[0],
        ]
    )
    object_slide = body_rot.inv().apply(object_pos_world - body_pos[np.newaxis, :])
    object_quat_xyzw = np.column_stack(
        [
            object_quat_wxyz[:, 1],
            object_quat_wxyz[:, 2],
            object_quat_wxyz[:, 3],
            object_quat_wxyz[:, 0],
        ]
    )
    object_euler = (body_rot.inv() * R.from_quat(object_quat_xyzw)).as_euler(
        scene_euler_convention(scene_xml)
    )

    converted = np.zeros((qpos.shape[0], model.nq), dtype=np.float64)
    converted[:, :nq_robot] = qpos[:, :nq_robot]
    converted[:, nq_robot : nq_robot + 3] = object_slide
    converted[:, nq_robot + 3 : nq_robot + 6] = object_euler
    return converted


def body_positions(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    body_ids: list[int],
) -> np.ndarray:
    data = mujoco.MjData(model)
    out = np.zeros((qpos.shape[0], len(body_ids), 3), dtype=np.float64)
    for index, frame in enumerate(qpos):
        data.qpos[:] = frame
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        out[index] = data.xpos[body_ids]
    return out


def fixed_reference_z_metrics(
    qpos_path: Path,
    scene_xml: Path,
    trajectory_path: Path,
) -> dict[str, Any]:
    sim_qpos, intra_tick_qpos = npz_qpos(qpos_path)
    fixed_ref_qpos = convert_reference_to_scene_act(
        load_reference_qpos(trajectory_path), scene_xml
    )
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    body_ids: list[int] = []
    body_names: list[str] = []
    for name in MONITORED_BODY_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            body_ids.append(int(body_id))
            body_names.append(name)
    if len(body_ids) != len(MONITORED_BODY_NAMES):
        raise ValueError(
            f"missing monitored bodies in {scene_xml}: got={body_names}"
        )

    frame_count = min(len(sim_qpos), len(fixed_ref_qpos))
    sim_pos = body_positions(model, sim_qpos[:frame_count], body_ids)
    ref_pos = body_positions(model, fixed_ref_qpos[:frame_count], body_ids)
    diff = sim_pos - ref_pos
    z_err = np.abs(diff[..., 2])
    xy_err = np.linalg.norm(diff[..., :2], axis=-1)
    err_3d = np.linalg.norm(diff, axis=-1)
    fps = fps_from_npz(qpos_path, 50.0)
    z_accel = (
        np.diff(sim_pos[..., 2], n=2, axis=0) * (fps**2)
        if frame_count >= 3
        else np.empty((0,))
    )
    z_jerk = (
        np.diff(sim_pos[..., 2], n=3, axis=0) * (fps**3)
        if frame_count >= 4
        else np.empty((0,))
    )
    z_peak = float(np.max(z_err)) if z_err.size else math.nan
    z_p95 = p95(z_err)
    err_3d_peak = float(np.max(err_3d)) if err_3d.size else math.nan

    legacy_peak = math.nan
    if intra_tick_qpos is not None:
        legacy_frames = min(len(sim_qpos), len(intra_tick_qpos))
        legacy_ref_pos = body_positions(
            model, intra_tick_qpos[:legacy_frames], body_ids
        )
        legacy_err = np.abs(sim_pos[:legacy_frames, :, 2] - legacy_ref_pos[..., 2])
        legacy_peak = float(np.max(legacy_err)) if legacy_err.size else math.nan

    return {
        "z_reference_source": "fixed_kinematic_trajectory",
        "z_reference_path": rel(trajectory_path),
        "z_eval_frames": int(frame_count),
        "monitored_bodies": ",".join(body_names),
        "body_z_gate_metric": "body_z_err_p95_m",
        "body_z_gate_threshold_m": BODY_Z_P95_THRESHOLD_M,
        "body_z_err_peak_m": z_peak,
        "body_z_err_p95_m": z_p95,
        "body_z_over_frac": float(np.mean(z_err > BODY_Z_P95_THRESHOLD_M)),
        "holosoma_z_gate_pass": bool(z_p95 <= BODY_Z_P95_THRESHOLD_M),
        "sugar_3d_err_peak_m": err_3d_peak,
        "sugar_3d_err_p95_m": p95(err_3d),
        "sugar_3d_over_frac": float(np.mean(err_3d > SUGAR_3D_THRESHOLD_M)),
        "sugar_3d_gate_pass": bool(err_3d_peak <= SUGAR_3D_THRESHOLD_M),
        "xy_err_peak_m": float(np.max(xy_err)),
        "xy_err_p95_m": p95(xy_err),
        "xy_only_failure": bool(
            err_3d_peak > SUGAR_3D_THRESHOLD_M
            and z_p95 <= BODY_Z_P95_THRESHOLD_M
        ),
        "body_z_accel_p95": p95(np.abs(z_accel)),
        "body_z_jerk_p95": p95(np.abs(z_jerk)),
        "legacy_intra_tick_z_err_peak_m": legacy_peak,
        "legacy_intra_tick_z_gate_pass": bool(
            math.isfinite(legacy_peak)
            and legacy_peak <= LEGACY_BODY_Z_PEAK_THRESHOLD_M
        ),
    }


def manual_review_fields(
    case_id: str,
    reviews: dict[str, dict[str, str]],
) -> dict[str, Any]:
    review = reviews.get(case_id)
    if review is None:
        return {
            "manual_review_status": "pending",
            "manual_use_decision": "PENDING_MANUAL_REVIEW",
            "manual_quality_label": "",
            "visual_gate_pass": None,
            "manual_review_note": "",
            "manual_reviewer": "",
            "manual_reviewed_at": "",
            "manual_review_source_workbook": "",
        }
    visual_pass = str(review.get("visual_gate_pass", "")).strip().lower() == "true"
    return {
        "manual_review_status": review.get("manual_review_status", "reviewed"),
        "manual_use_decision": review["manual_use_decision"],
        "manual_quality_label": review["manual_quality_label"],
        "visual_gate_pass": visual_pass,
        "manual_review_note": review.get("manual_review_note", ""),
        "manual_reviewer": review.get("reviewer", ""),
        "manual_reviewed_at": review.get("reviewed_at", ""),
        "manual_review_source_workbook": review.get("source_workbook", ""),
    }


def artifact_status(row: dict[str, str]) -> dict[str, Any]:
    paths = {
        "root_npz": repo_path(row["result_npz"]),
        "outdir_npz": repo_path(row["outdir_npz"]),
        "config_act": repo_path(row["config_act"]),
        "video": repo_path(row["video"]),
        "scene_act": repo_path(row["scene_act"]),
        "trajectory": repo_path(row["trajectory"]),
        "contact_mask": repo_path(row["contact_mask"]),
    }
    out = {f"{name}_exists": path.is_file() for name, path in paths.items()}
    out["artifact_gate_pass"] = all(out.values())
    out["config_gate_pass"] = (
        row.get("config_audit_status") == "pass" and out["config_act_exists"]
    )
    return out


def release_window_info(
    contact_mask_path: Path,
    person_idx: int,
    frame_count: int,
) -> dict[str, Any]:
    """Describe whether the reference contains a trailing release window."""
    with np.load(contact_mask_path, allow_pickle=True) as data:
        if "spider_contact_mask_3cm" not in data.files:
            return {
                "release_window_frame_count": 0,
                "release_gate_applicable": False,
                "release_gate_status": "NOT_APPLICABLE_MASK_MISSING_KEY",
            }
        mask = np.asarray(data["spider_contact_mask_3cm"])
    if mask.ndim != 3 or mask.shape[2] != 2:
        raise ValueError(
            f"{contact_mask_path}: spider_contact_mask_3cm expected "
            f"(T, persons, 2), got {mask.shape}"
        )
    if person_idx < 0 or person_idx >= mask.shape[1]:
        raise ValueError(
            f"{contact_mask_path}: person_idx={person_idx} out of mask shape {mask.shape}"
        )
    frame_count = min(int(frame_count), int(mask.shape[0]))
    mask_any = np.any(mask[:frame_count, person_idx, :].astype(bool), axis=1)
    if not mask_any.any():
        return {
            "release_window_frame_count": 0,
            "release_gate_applicable": False,
            "release_gate_status": "NOT_APPLICABLE_NO_REFERENCE_CONTACT",
        }
    last_contact = int(np.flatnonzero(mask_any)[-1])
    release_frames = int(frame_count - last_contact - 1)
    return {
        "release_window_frame_count": release_frames,
        "release_gate_applicable": release_frames > 0,
        "release_gate_status": (
            "PENDING_EVALUATION"
            if release_frames > 0
            else "NOT_APPLICABLE_NO_RELEASE_WINDOW"
        ),
    }


def absolute_release_gates(item: dict[str, Any]) -> None:
    terminal_pelvis = finite(item.get("track_pelvis_z_err_terminal_m"), math.inf)
    raw_contact = finite(
        item.get("hand_object_physics_contact_in_mask_frac"), -math.inf
    )
    release_applicable = bool(item.get("release_gate_applicable"))
    false_release = finite(item.get("hand_object_release_false_contact_3mm_frac"))
    penetration = finite(
        item.get("hand_object_physics_penetration_3mm_frame_frac"), math.inf
    )
    lower_body = finite(item.get("leg_penetration_frac"), math.inf)

    item["tracking_gate_pass"] = bool(
        not item.get("fall_flag")
        and terminal_pelvis <= EvalConfig().track_pelvis_terminal_th_m
    )
    item["contact_gate_pass"] = raw_contact >= CONTACT_MIN
    if release_applicable:
        item["release_gate_pass"] = bool(
            math.isfinite(false_release) and false_release <= RELEASE_FALSE_MAX
        )
        item["release_gate_status"] = (
            "APPLICABLE_PASS" if item["release_gate_pass"] else "APPLICABLE_FAIL"
        )
    else:
        item["release_gate_pass"] = None
    item["penetration_gate_pass"] = penetration <= PENETRATION_3MM_MAX
    item["lower_body_gate_pass"] = lower_body <= LOWER_BODY_INTERFERENCE_MAX
    item["relative_gate_status"] = "not_available_no_comparable_baseline"
    item["visual_review_status"] = (
        "user_review_" + str(item.get("manual_quality_label", "")).lower()
        if item.get("manual_review_status") == "reviewed"
        else "pending_cem_visual_review"
    )

    failures = []
    for gate, failure in (
        ("artifact_gate_pass", "artifact"),
        ("config_gate_pass", "config"),
        ("tracking_gate_pass", "tracking"),
        ("holosoma_z_gate_pass", "holosoma_z"),
        ("contact_gate_pass", "contact"),
        ("penetration_gate_pass", "penetration"),
        ("lower_body_gate_pass", "lower_body"),
    ):
        if not bool(item.get(gate)):
            failures.append(failure)
    if release_applicable and not bool(item.get("release_gate_pass")):
        failures.append("release_false_contact")
    item["numeric_release_pass"] = not failures
    item["failure_modes"] = ",".join(failures)
    item["release_status"] = (
        "NUMERIC_PASS"
        if not failures
        else "NUMERIC_FAIL_" + "+".join(failures).upper()
    )


def evaluate_row(
    row: dict[str, str],
    cfg: EvalConfig,
    manual_reviews: dict[str, dict[str, str]],
) -> dict[str, Any]:
    qpos_path = repo_path(row["outdir_npz"])
    scene_xml = repo_path(row["scene_act"])
    trajectory_path = repo_path(row["trajectory"])
    contact_mask_path = repo_path(row["contact_mask"])
    item = evaluate_sequence(
        row=row,
        method=row["spider_method_id"],
        hand_collision_variant_id=row["hand_collision_variant_id"],
        qpos_path=qpos_path,
        scene_xml=scene_xml,
        config=cfg,
        kin_ref_path=trajectory_path,
        contact_mask_path=contact_mask_path,
        person_idx=source_person_idx(row),
    )
    terminal_pelvis = finite(item.get("track_pelvis_z_err_terminal_m"), math.inf)
    item["success_tracked"] = bool(
        not item.get("fall_flag")
        and terminal_pelvis <= cfg.track_pelvis_terminal_th_m
    )
    sim_qpos, _ = npz_qpos(qpos_path)
    item.update(run_health(qpos_path, scene_xml, cfg))
    item.update(fixed_reference_z_metrics(qpos_path, scene_xml, trajectory_path))
    item.update(artifact_status(row))
    item.update(
        release_window_info(
            contact_mask_path,
            source_person_idx(row),
            len(sim_qpos),
        )
    )
    item.update(
        {
            "metric_standard_id": EVAL_METRIC_STANDARD_ID,
            "case_id": row["case_id"],
            "sequence_key": row["sequence_key"],
            "source_person": row["source_person"],
            "source_action": row["source_action"],
            "source_obstacle_level": row["source_obstacle_level"],
            "object_key": row["object_key"],
            "retarget_variant_id": row["retarget_variant_id"],
            "target_variant_id": row["target_variant_id"],
            "hand_collision_variant_id": row["hand_collision_variant_id"],
            "spider_method_id": row["spider_method_id"],
            "preferred_pool": row["preferred_pool"],
            "gpu_id": row["gpu_id"],
            "manifest_status": row["status"],
            "result_npz": rel(row["result_npz"]),
            "outdir_npz": rel(row["outdir_npz"]),
            "config_act": rel(row["config_act"]),
            "video": rel(row["video"]) if repo_path(row["video"]).is_file() else "",
            "trajectory": rel(row["trajectory"]),
            "contact_mask": rel(row["contact_mask"]),
        }
    )
    item.update(manual_review_fields(row["case_id"], manual_reviews))
    absolute_release_gates(item)
    return item


def metric_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for metric in SUMMARY_METRICS:
        values = [finite(row.get(metric)) for row in rows]
        valid = [value for value in values if math.isfinite(value)]
        out.append(
            {
                "metric": metric,
                "n": len(valid),
                "mean": mean(valid),
                "median": percentile(valid, 50),
                "p95": percentile(valid, 95),
                "min": min(valid) if valid else math.nan,
                "max": max(valid) if valid else math.nan,
            }
        )
    return out


def group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[tuple[str, str, list[dict[str, Any]]]] = [("overall", "all", rows)]
    for field in ("object_key", "preferred_pool", "retarget_variant_id"):
        for value in sorted({str(row.get(field, "")) for row in rows}):
            groups.append((field, value, [row for row in rows if str(row.get(field, "")) == value]))

    out = []
    for group_type, group_value, group_rows in groups:
        item = {
            "group_type": group_type,
            "group_value": group_value,
            "n": len(group_rows),
            "numeric_pass": sum(bool(row.get("numeric_release_pass")) for row in group_rows),
            "tracked_pass": sum(bool(row.get("tracking_gate_pass")) for row in group_rows),
            "z_pass": sum(bool(row.get("holosoma_z_gate_pass")) for row in group_rows),
            "contact_pass": sum(bool(row.get("contact_gate_pass")) for row in group_rows),
            "release_pass": sum(bool(row.get("release_gate_pass")) for row in group_rows),
            "release_applicable": sum(
                bool(row.get("release_gate_applicable")) for row in group_rows
            ),
            "release_not_applicable": sum(
                not bool(row.get("release_gate_applicable")) for row in group_rows
            ),
            "penetration_pass": sum(bool(row.get("penetration_gate_pass")) for row in group_rows),
            "lower_body_pass": sum(bool(row.get("lower_body_gate_pass")) for row in group_rows),
            "manual_reviewed": sum(
                row.get("manual_review_status") == "reviewed" for row in group_rows
            ),
            "manual_use": sum(
                row.get("manual_use_decision") == "USE" for row in group_rows
            ),
            "manual_do_not_use": sum(
                row.get("manual_use_decision") == "DO_NOT_USE" for row in group_rows
            ),
            "fall_count": sum(bool(row.get("fall_flag")) for row in group_rows),
            "failed_cases": ",".join(
                row["case_id"] for row in group_rows if not row.get("numeric_release_pass")
            ),
        }
        for metric in SUMMARY_METRICS:
            item[f"{metric}_mean"] = mean([row.get(metric) for row in group_rows])
        out.append(item)
    return out


def metric_rankings(rows: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    out = []
    for metric, direction in RANKING_METRICS.items():
        valid = [row for row in rows if math.isfinite(finite(row.get(metric)))]
        valid.sort(
            key=lambda row: finite(row.get(metric)),
            reverse=direction == "high",
        )
        for rank, row in enumerate(valid[:top_k], start=1):
            out.append(
                {
                    "metric": metric,
                    "worse_direction": direction,
                    "rank": rank,
                    "case_id": row["case_id"],
                    "value": row.get(metric),
                    "release_status": row.get("release_status"),
                    "failure_modes": row.get("failure_modes"),
                    "video": row.get("video"),
                }
            )
    return out


def fmt(value: Any, digits: int = 3) -> str:
    number = finite(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "NA"


def write_summary_md(
    path: Path,
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    rankings: list[dict[str, Any]],
) -> None:
    counts = payload["counts"]
    lines = [
        "# E168 E167A-aligned available-case evaluation",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        f"Scope: `{counts['evaluated']}/{counts['manifest_rows']}` production rows evaluated; "
        f"`{counts['not_ready']}` not ready and `{counts['evaluation_errors']}` evaluation errors.",
        "",
        "## Method contract",
        "",
        f"- Core metric standard: `{EVAL_METRIC_STANDARD_ID}`.",
        "- E167A-aligned tracking gate: no fall and terminal pelvis-z error <= 0.08m.",
        "- Holosoma z-only gate: four monitored ankle/wrist bodies, fixed kinematic reference, p95 z error <= 0.20m; peak remains diagnostic.",
        "- New E168 cases have no same-case E163 baseline; relative regression gate is unavailable and is not fabricated.",
        "- E167 legacy zgate used the second intra-tick simulation step as reference. `legacy_intra_tick_*` is retained only as a diagnostic; release uses `fixed_kinematic_trajectory`.",
        "- A sequence with no trailing frames after its last reference-contact frame has no release window; its release gate is explicitly N/A rather than zero-filled or failed.",
        "- Numeric gates and user visual decisions are reported independently; unreviewed rows remain `PENDING_MANUAL_REVIEW`.",
        "",
        "## Gate summary",
        "",
        "| evaluated | numeric pass | tracking | z-only | contact | release (applicable) | release N/A | penetration | lower body | fall |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {counts['evaluated']} | {counts['numeric_pass']} | {counts['tracking_pass']} | "
        f"{counts['z_pass']} | {counts['contact_pass']} | "
        f"{counts['release_pass']}/{counts['release_applicable']} | "
        f"{counts['release_not_applicable']} | "
        f"{counts['penetration_pass']} | {counts['lower_body_pass']} | {counts['fall']} |",
        "",
        "## Per case",
        "",
        "| case | pool | status | manual use | pelvis term m | body-z p95 m | body-z peak m | joint deg | EEF cm | obj cm | raw contact | release false | pen3 | leg pen |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: item["case_id"]):
        lines.append(
            f"| `{row['case_id']}` | {row['preferred_pool']} | `{row['release_status']}` | "
            f"`{row['manual_use_decision']}` | {fmt(row.get('track_pelvis_z_err_terminal_m'))} | "
            f"{fmt(row.get('body_z_err_p95_m'))} | {fmt(row.get('body_z_err_peak_m'))} | "
            f"{fmt(row.get('track_joint_err_deg_mean'), 2)} | {fmt(row.get('track_eef_pos_err_cm_mean'), 2)} | "
            f"{fmt(row.get('track_obj_pos_err_cm_mean'), 2)} | {fmt(row.get('hand_object_physics_contact_in_mask_frac'))} | "
            f"{fmt(row.get('hand_object_release_false_contact_3mm_frac'))} | "
            f"{fmt(row.get('hand_object_physics_penetration_3mm_frame_frac'))} | "
            f"{fmt(row.get('leg_penetration_frac'))} |"
        )
    lines.extend(
        [
            "",
            "## Worst cases by metric",
            "",
            "| metric | direction | rank | case | value | release status |",
            "|---|---|---:|---|---:|---|",
        ]
    )
    for row in rankings:
        if int(row["rank"]) > 3:
            continue
        lines.append(
            f"| `{row['metric']}` | {row['worse_direction']} | {row['rank']} | "
            f"`{row['case_id']}` | {fmt(row['value'])} | `{row['release_status']}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--manual-review", type=Path, default=DEFAULT_MANUAL_REVIEW)
    parser.add_argument("--cases", nargs="*", default=[])
    parser.add_argument("--scope-label", default="")
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()

    manifest = repo_path(args.manifest)
    out_dir = repo_path(args.out_dir)
    manual_review_path = repo_path(args.manual_review)
    manual_review_rows = read_tsv(manual_review_path) if manual_review_path.is_file() else []
    manual_reviews = {row["case_id"]: row for row in manual_review_rows}
    all_rows = read_tsv(manifest)
    selectors = set(args.cases)
    scoped_rows = [
        row
        for row in all_rows
        if not selectors
        or row.get("case_id") in selectors
        or row.get("variant") in selectors
    ]
    scoped_ids = {row["case_id"] for row in scoped_rows}
    scoped_manual_review_rows = [
        row for row in manual_review_rows if row.get("case_id") in scoped_ids
    ]
    ready_rows = [
        row
        for row in scoped_rows
        if repo_path(row["outdir_npz"]).is_file()
        and repo_path(row["scene_act"]).is_file()
        and repo_path(row["trajectory"]).is_file()
        and repo_path(row["contact_mask"]).is_file()
    ]
    ready_ids = {row["case_id"] for row in ready_rows}
    not_ready_rows = [
        {
            **row,
            "missing_artifacts": ",".join(
                name
                for name, value in (
                    ("outdir_npz", row["outdir_npz"]),
                    ("scene_act", row["scene_act"]),
                    ("trajectory", row["trajectory"]),
                    ("contact_mask", row["contact_mask"]),
                )
                if not repo_path(value).is_file()
            ),
        }
        for row in scoped_rows
        if row["case_id"] not in ready_ids
    ]

    cfg = EvalConfig()
    metric_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    for index, row in enumerate(ready_rows, start=1):
        print(f"[{index}/{len(ready_rows)}] evaluate {row['case_id']}", flush=True)
        try:
            metric_rows.append(evaluate_row(row, cfg, manual_reviews))
        except Exception as exc:
            error_rows.append(
                {
                    "case_id": row["case_id"],
                    "variant": row["variant"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    metric_summary_rows = metric_summary(metric_rows)
    group_summary_rows = group_summary(metric_rows)
    ranking_rows = metric_rankings(metric_rows)
    failure_counter = Counter()
    for row in metric_rows:
        failure_counter.update(filter(None, str(row.get("failure_modes", "")).split(",")))
    counts = {
        "manifest_rows": len(scoped_rows),
        "evaluated": len(metric_rows),
        "not_ready": len(not_ready_rows),
        "evaluation_errors": len(error_rows),
        "numeric_pass": sum(bool(row.get("numeric_release_pass")) for row in metric_rows),
        "tracking_pass": sum(bool(row.get("tracking_gate_pass")) for row in metric_rows),
        "z_pass": sum(bool(row.get("holosoma_z_gate_pass")) for row in metric_rows),
        "contact_pass": sum(bool(row.get("contact_gate_pass")) for row in metric_rows),
        "release_pass": sum(bool(row.get("release_gate_pass")) for row in metric_rows),
        "release_applicable": sum(
            bool(row.get("release_gate_applicable")) for row in metric_rows
        ),
        "release_not_applicable": sum(
            not bool(row.get("release_gate_applicable")) for row in metric_rows
        ),
        "penetration_pass": sum(bool(row.get("penetration_gate_pass")) for row in metric_rows),
        "lower_body_pass": sum(bool(row.get("lower_body_gate_pass")) for row in metric_rows),
        "manual_reviewed": sum(
            row.get("manual_review_status") == "reviewed" for row in metric_rows
        ),
        "manual_use": sum(row.get("manual_use_decision") == "USE" for row in metric_rows),
        "manual_do_not_use": sum(
            row.get("manual_use_decision") == "DO_NOT_USE" for row in metric_rows
        ),
        "manual_pending": sum(
            row.get("manual_use_decision") == "PENDING_MANUAL_REVIEW"
            for row in metric_rows
        ),
        "fall": sum(bool(row.get("fall_flag")) for row in metric_rows),
    }
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "metric_standard_id": EVAL_METRIC_STANDARD_ID,
        "manifest": rel(manifest),
        "manifest_sha256": sha256(manifest),
        "scope_label": args.scope_label,
        "scope_case_ids": sorted(scoped_ids),
        "counts": counts,
        "failure_mode_counts": dict(failure_counter),
        "thresholds": {
            "fall_pelvis_z_m": cfg.fall_pelvis_z_m,
            "track_pelvis_terminal_th_m": cfg.track_pelvis_terminal_th_m,
            "body_z_gate_metric": "body_z_err_p95_m",
            "body_z_p95_m": BODY_Z_P95_THRESHOLD_M,
            "legacy_body_z_peak_m": LEGACY_BODY_Z_PEAK_THRESHOLD_M,
            "raw_contact_in_mask_min": CONTACT_MIN,
            "release_false_contact_3mm_max": RELEASE_FALSE_MAX,
            "physics_penetration_3mm_frame_frac_max": PENETRATION_3MM_MAX,
            "lower_body_interference_max": LOWER_BODY_INTERFERENCE_MAX,
        },
        "relative_gate_status": "not_available_no_comparable_baseline",
        "visual_review_status": (
            "pending_user_review"
            if counts["manual_reviewed"] == 0
            else (
                "partial_user_review"
                if counts["manual_reviewed"] < counts["evaluated"]
                else "complete_user_review"
            )
        ),
        "manual_review": rel(manual_review_path) if manual_review_path.is_file() else "",
    }

    metadata_fields = [
        "case_id",
        "sequence_key",
        "source_person",
        "object_key",
        "source_action",
        "source_obstacle_level",
        "preferred_pool",
        "gpu_id",
        "retarget_variant_id",
        "target_variant_id",
        "hand_collision_variant_id",
        "spider_method_id",
        "metric_standard_id",
        "manifest_status",
        "release_status",
        "failure_modes",
        "manual_review_status",
        "manual_use_decision",
        "manual_quality_label",
        "visual_gate_pass",
        "manual_review_note",
        "manual_reviewer",
        "manual_reviewed_at",
        "manual_review_source_workbook",
        "numeric_release_pass",
        "tracking_gate_pass",
        "holosoma_z_gate_pass",
        "contact_gate_pass",
        "release_gate_pass",
        "release_gate_applicable",
        "release_gate_status",
        "release_window_frame_count",
        "penetration_gate_pass",
        "lower_body_gate_pass",
        "artifact_gate_pass",
        "config_gate_pass",
        "relative_gate_status",
        "visual_review_status",
        "success_tracked",
        "fall_flag",
        "body_z_gate_metric",
        "body_z_gate_threshold_m",
        "body_z_err_peak_m",
        "body_z_err_p95_m",
        "body_z_over_frac",
        "sugar_3d_err_peak_m",
        "sugar_3d_err_p95_m",
        "sugar_3d_gate_pass",
        "xy_err_peak_m",
        "xy_err_p95_m",
        "xy_only_failure",
        "body_z_accel_p95",
        "body_z_jerk_p95",
        "legacy_intra_tick_z_err_peak_m",
        "legacy_intra_tick_z_gate_pass",
        "z_reference_source",
        "z_reference_path",
        "z_eval_frames",
        "monitored_bodies",
    ]
    health_fields = [
        *METRIC_KEYS,
        *HEALTH_AGGS.keys(),
        "has_smooth_health",
        "has_foot_health",
    ]
    path_fields = [
        "result_npz",
        "outdir_npz",
        "config_act",
        "video",
        "trajectory",
        "contact_mask",
        "scene_xml",
    ]
    metric_fields = []
    for field in [*metadata_fields, *METRIC_FIELDS, *health_fields, *path_fields]:
        if field not in metric_fields:
            metric_fields.append(field)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(out_dir / "e168_case_metrics.tsv", metric_rows, metric_fields)
    write_tsv(out_dir / "e168_metric_summary.tsv", metric_summary_rows)
    write_tsv(out_dir / "e168_group_summary.tsv", group_summary_rows)
    write_tsv(out_dir / "e168_worst_case_rankings.tsv", ranking_rows)
    write_tsv(out_dir / "e168_not_ready.tsv", not_ready_rows)
    write_tsv(out_dir / "e168_evaluation_errors.tsv", error_rows)
    write_tsv(
        out_dir / "e168_manual_review_snapshot.tsv",
        scoped_manual_review_rows,
        list(manual_review_rows[0].keys()) if manual_review_rows else None,
    )
    write_tsv(out_dir / "evaluated_manifest_snapshot.tsv", ready_rows)
    write_json(out_dir / "summary.json", payload)
    write_summary_md(out_dir / "summary.md", payload, metric_rows, ranking_rows)

    print(
        "E168 E167A-aligned eval: "
        f"evaluated={counts['evaluated']}/{counts['manifest_rows']} "
        f"numeric_pass={counts['numeric_pass']} not_ready={counts['not_ready']} "
        f"errors={counts['evaluation_errors']} out={rel(out_dir)}"
    )
    if error_rows:
        return 1
    if args.require_all and not_ready_rows:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
