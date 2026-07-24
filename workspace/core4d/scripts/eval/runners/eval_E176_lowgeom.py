#!/usr/bin/env python3
"""Evaluate low-geom non-box CEM and compare it with frozen E174.

E176 remains the default contract.  Later low-geom experiments can reuse the
same metric implementation through explicit experiment/output arguments.
"""

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
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
RESULT_ROOT = REPO / "workspace/core4d/results/E176/s6_downstream"
DEFAULT_BASELINE = (
    REPO
    / "workspace/core4d/results/E174/s6_downstream/eval/full/e174_case_metrics.tsv"
)

MONITORED_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
CONTACT_MIN = 0.50
BODY_Z_MAX = 0.20
HAND_PEN_MAX = 0.30
LEG_PEN_MAX = 0.10
RELEASE_MAX = 0.30
GATE_FALLBACK_MAX = 0.10
GATE_VALID_LAST_MIN = 0.05
TRACKING_GATE_KEYS = (
    "root_pos",
    "root_ori",
    "hand_pos",
    "hand_ori",
    "object_pos",
    "object_ori",
)
LEG_GATE_KEYS = (
    "cem_leg_gate_valid_frac",
    "cem_leg_gate_selected_valid_frac",
    "cem_leg_gate_fallback_used",
    "cem_leg_gate_min_sdf_min_m",
    "cem_leg_gate_min_sdf_p05_m",
    "cem_leg_gate_violation_pct_mean",
    "cem_leg_gate_selected_all_valid",
)
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


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path.resolve()
    text = str(value)
    for marker in ("example_datasets/", "workspace/", "logs/"):
        if marker in text:
            return REPO / (marker + text.split(marker, 1)[1])
    return path if path.is_absolute() else REPO / path


def rel(value: str | Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except (OSError, ValueError):
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
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: serial(row.get(key, "")) for key in fields}
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def p95(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, 95)) if array.size else math.nan


def person_idx(row: dict[str, str]) -> int:
    person = row.get("person", "").strip().lower()
    if person in {"person1", "p1", "0"}:
        return 0
    if person in {"person2", "p2", "1"}:
        return 1
    return 0 if row["case_id"].lower().endswith("_p1") else 1


def reference_qpos(trajectory: Path, scene_xml: Path) -> np.ndarray:
    with np.load(trajectory, allow_pickle=True) as payload:
        qpos = np.asarray(payload["qpos"], dtype=np.float64)
    if qpos.ndim == 3:
        qpos = qpos[:, 0, :]
    if qpos.ndim != 2:
        raise ValueError(f"invalid reference qpos shape {qpos.shape}")

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    if qpos.shape[1] == model.nq:
        return qpos.copy()
    nq_robot = model.nq - 6
    if qpos.shape[1] < nq_robot + 7:
        raise ValueError(
            f"cannot convert reference qpos={qpos.shape} to nq={model.nq}"
        )
    object_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "object"
    )
    if object_id < 0:
        raise ValueError("scene has no object body")

    convention = "XYZ"
    meta = scene_xml.with_name("scene_act_meta.json")
    if meta.is_file():
        convention = str(
            json.loads(meta.read_text(encoding="utf-8")).get(
                "euler_convention", "XYZ"
            )
        )
    body_quat = model.body_quat[object_id]
    body_rotation = Rotation.from_quat(
        [body_quat[1], body_quat[2], body_quat[3], body_quat[0]]
    )
    object_pos = qpos[:, nq_robot : nq_robot + 3]
    object_quat = qpos[:, nq_robot + 3 : nq_robot + 7]
    object_xyzw = np.column_stack(
        [
            object_quat[:, 1],
            object_quat[:, 2],
            object_quat[:, 3],
            object_quat[:, 0],
        ]
    )
    converted = np.zeros((len(qpos), model.nq), dtype=np.float64)
    converted[:, :nq_robot] = qpos[:, :nq_robot]
    converted[:, nq_robot : nq_robot + 3] = body_rotation.inv().apply(
        object_pos - model.body_pos[object_id][np.newaxis, :]
    )
    converted[:, nq_robot + 3 : nq_robot + 6] = (
        body_rotation.inv() * Rotation.from_quat(object_xyzw)
    ).as_euler(convention)
    return converted


def body_positions(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    body_ids: list[int],
) -> np.ndarray:
    data = mujoco.MjData(model)
    output = np.zeros((len(qpos), len(body_ids), 3), dtype=np.float64)
    for frame, values in enumerate(qpos):
        data.qpos[:] = values
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        output[frame] = data.xpos[body_ids]
    return output


def fixed_reference_z_metrics(
    qpos_path: Path,
    scene_xml: Path,
    trajectory: Path,
) -> dict[str, Any]:
    sim_qpos, intra_tick_qpos = npz_qpos(qpos_path)
    ref_qpos = reference_qpos(trajectory, scene_xml)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    body_ids = [
        int(
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
        )
        for body_name in MONITORED_BODY_NAMES
    ]
    if any(body_id < 0 for body_id in body_ids):
        raise ValueError("scene missing monitored body for body-z metrics")

    frame_count = min(len(sim_qpos), len(ref_qpos))
    sim_pos = body_positions(model, sim_qpos[:frame_count], body_ids)
    ref_pos = body_positions(model, ref_qpos[:frame_count], body_ids)
    difference = sim_pos - ref_pos
    z_error = np.abs(difference[..., 2])
    xy_error = np.linalg.norm(difference[..., :2], axis=-1)
    error_3d = np.linalg.norm(difference, axis=-1)
    fps = fps_from_npz(qpos_path, 50.0)
    z_accel = np.diff(sim_pos[..., 2], n=2, axis=0) * (fps**2)
    z_jerk = np.diff(sim_pos[..., 2], n=3, axis=0) * (fps**3)

    legacy_peak = math.nan
    if intra_tick_qpos is not None:
        legacy_frames = min(len(sim_qpos), len(intra_tick_qpos))
        legacy_pos = body_positions(
            model, intra_tick_qpos[:legacy_frames], body_ids
        )
        legacy_error = np.abs(
            sim_pos[:legacy_frames, :, 2] - legacy_pos[..., 2]
        )
        legacy_peak = (
            float(np.max(legacy_error))
            if legacy_error.size
            else math.nan
        )
    return {
        "z_reference_source": "fixed_kinematic_trajectory",
        "z_reference_path": rel(trajectory),
        "z_eval_frames": frame_count,
        "monitored_bodies": ",".join(MONITORED_BODY_NAMES),
        "body_z_gate_metric": "body_z_err_p95_m",
        "body_z_gate_threshold_m": BODY_Z_MAX,
        "body_z_err_peak_m": float(np.max(z_error)),
        "body_z_err_p95_m": p95(z_error),
        "body_z_over_frac": float(np.mean(z_error > BODY_Z_MAX)),
        "holosoma_z_gate_pass": bool(p95(z_error) <= BODY_Z_MAX),
        "sugar_3d_err_peak_m": float(np.max(error_3d)),
        "sugar_3d_err_p95_m": p95(error_3d),
        "xy_err_peak_m": float(np.max(xy_error)),
        "xy_err_p95_m": p95(xy_error),
        "body_z_accel_p95": p95(np.abs(z_accel)),
        "body_z_jerk_p95": p95(np.abs(z_jerk)),
        "legacy_intra_tick_z_err_peak_m": legacy_peak,
    }


def release_window_info(
    contact_mask: Path,
    selected_person_idx: int,
    frame_count: int,
) -> dict[str, Any]:
    with np.load(contact_mask, allow_pickle=True) as payload:
        if "spider_contact_mask_3cm" not in payload.files:
            return {
                "release_window_frame_count": 0,
                "release_gate_applicable": False,
                "release_gate_status": "NOT_APPLICABLE_MASK_MISSING_KEY",
            }
        mask = np.asarray(payload["spider_contact_mask_3cm"])
    if mask.ndim != 3 or mask.shape[2] != 2:
        raise ValueError(f"invalid 3cm contact mask shape {mask.shape}")
    frame_count = min(frame_count, len(mask))
    active = np.any(
        mask[:frame_count, selected_person_idx, :].astype(bool), axis=1
    )
    if not active.any():
        return {
            "release_window_frame_count": 0,
            "release_gate_applicable": False,
            "release_gate_status": "NOT_APPLICABLE_NO_REFERENCE_CONTACT",
        }
    release_frames = frame_count - int(np.flatnonzero(active)[-1]) - 1
    return {
        "release_window_frame_count": release_frames,
        "release_gate_applicable": release_frames > 0,
        "release_gate_status": (
            "EVALUATED"
            if release_frames > 0
            else "NOT_APPLICABLE_NO_RELEASE_WINDOW"
        ),
    }


def array_stat(
    payload: np.lib.npyio.NpzFile,
    key: str,
    mode: str,
) -> float:
    if key not in payload.files:
        return math.nan
    values = np.asarray(payload[key], dtype=np.float64)
    if mode == "last":
        if values.ndim != 2 or not values.shape[1]:
            return math.nan
        values = values[:, -1]
    values = values[np.isfinite(values)]
    if not values.size:
        return math.nan
    return float(values.min() if mode == "min" else values.mean())


def gate_health(result_npz: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    with np.load(result_npz, allow_pickle=True) as payload:
        for key in LEG_GATE_KEYS:
            output[f"{key}_mean"] = array_stat(payload, key, "mean")
            output[f"{key}_last_iter_mean"] = array_stat(
                payload, key, "last"
            )
        output["cem_leg_gate_min_sdf_worst_m"] = array_stat(
            payload, "cem_leg_gate_min_sdf_min_m", "min"
        )
    output["leg_gate_fallback_pass"] = (
        finite(output["cem_leg_gate_fallback_used_mean"], math.inf)
        <= GATE_FALLBACK_MAX
    )
    output["leg_gate_valid_frac_pass"] = (
        finite(
            output["cem_leg_gate_valid_frac_last_iter_mean"], -math.inf
        )
        >= GATE_VALID_LAST_MIN
    )
    output["leg_gate_selected_valid_pass"] = (
        finite(
            output["cem_leg_gate_selected_all_valid_mean"], -math.inf
        )
        >= 1.0 - 1e-9
    )
    output["leg_gate_health_pass"] = bool(
        output["leg_gate_fallback_pass"]
        and output["leg_gate_valid_frac_pass"]
        and output["leg_gate_selected_valid_pass"]
    )
    return output


def apply_gates(
    item: dict[str, Any],
    tracking_thresholds: dict[str, float] | None = None,
) -> None:
    release_applicable = bool(item.get("release_gate_applicable"))
    gates = {
        "fall": not bool(item.get("fall_flag")),
        "body_z": finite(item.get("body_z_err_p95_m"), math.inf)
        <= BODY_Z_MAX,
        "contact": finite(
            item.get("hand_object_physics_contact_in_mask_frac"),
            -math.inf,
        )
        >= CONTACT_MIN,
        "release": (not release_applicable)
        or finite(
            item.get("hand_object_release_false_contact_3mm_frac"),
            math.inf,
        )
        <= RELEASE_MAX,
        "hand_penetration": finite(
            item.get("hand_object_physics_penetration_3mm_frame_frac"),
            math.inf,
        )
        <= HAND_PEN_MAX,
        "lower_body": finite(
            item.get("leg_penetration_frac"), math.inf
        )
        <= LEG_PEN_MAX,
    }
    if tracking_thresholds is not None:
        gates.update(
            {
                "root_pos": finite(
                    item.get("track_root_pos_err_cm_mean"), math.inf
                )
                <= tracking_thresholds["root_pos"],
                "root_ori": finite(
                    item.get("track_root_ori_err_deg_mean"), math.inf
                )
                <= tracking_thresholds["root_ori"],
                "hand_pos": finite(
                    item.get("track_eef_pos_err_cm_mean"), math.inf
                )
                <= tracking_thresholds["hand_pos"],
                "hand_ori": finite(
                    item.get("track_eef_ori_err_deg_mean"), math.inf
                )
                <= tracking_thresholds["hand_ori"],
                "object_pos": finite(
                    item.get("track_obj_pos_err_cm_mean"), math.inf
                )
                <= tracking_thresholds["object_pos"],
                "object_ori": finite(
                    item.get("track_obj_ori_err_deg_mean"), math.inf
                )
                <= tracking_thresholds["object_ori"],
            }
        )
    for gate, passed in gates.items():
        item[f"{gate}_gate_pass"] = passed
    failures = [gate for gate, passed in gates.items() if not passed]
    item["numeric_release_pass"] = not failures
    item["numeric_failure_modes"] = ",".join(failures)


def evaluate_row(
    row: dict[str, str],
    config: EvalConfig,
    video_dir: Path | None = None,
    tracking_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    qpos_path = repo_path(row["outdir_npz"])
    result_npz = repo_path(row["result_npz"])
    scene_xml = repo_path(row["scene_act"])
    trajectory = repo_path(row["trajectory"])
    contact_mask = repo_path(row["contact_mask"])
    selected_person_idx = person_idx(row)
    item = evaluate_sequence(
        row=row,
        method=row["spider_method_id"],
        hand_collision_variant_id=row["hand_collision_variant_id"],
        qpos_path=qpos_path,
        scene_xml=scene_xml,
        config=config,
        kin_ref_path=trajectory,
        contact_mask_path=contact_mask,
        person_idx=selected_person_idx,
    )
    sim_qpos, _ = npz_qpos(qpos_path)
    item.update(run_health(qpos_path, scene_xml, config))
    item.update(
        fixed_reference_z_metrics(qpos_path, scene_xml, trajectory)
    )
    item.update(
        release_window_info(
            contact_mask, selected_person_idx, len(sim_qpos)
        )
    )
    item.update(gate_health(result_npz))
    for key in (
        "ordinal",
        "case_id",
        "variant",
        "object_key",
        "person",
        "retarget_variant_id",
        "target_variant_id",
        "hand_collision_variant_id",
        "spider_method_id",
        "status",
        "reference_first5_union_min_distance_m",
        "reference_first5_lowerbody_penetration_frac",
    ):
        item[key] = row.get(key, "")
    video = repo_path(row["video"]) if row.get("video") else Path()
    if video_dir is not None:
        rendered = video_dir / f"{row['variant']}.mp4"
        if rendered.is_file():
            video = rendered
    item.update(
        {
            "metric_standard_id": EVAL_METRIC_STANDARD_ID,
            "result_npz": rel(result_npz),
            "outdir_npz": rel(qpos_path),
            "config_act": rel(row["config_act"]),
            "video": rel(video) if video.is_file() else rel(row.get("video", "")),
            "trajectory": rel(trajectory),
            "contact_mask": rel(contact_mask),
            "scene_xml": rel(scene_xml),
        }
    )
    apply_gates(item, tracking_thresholds)
    return item


def add_baseline(
    item: dict[str, Any],
    baseline: dict[str, str],
    current_prefix: str,
) -> dict[str, Any]:
    current_pass = f"{current_prefix}_numeric_release_pass"
    current_failures = f"{current_prefix}_numeric_failure_modes"
    paired: dict[str, Any] = {
        "case_id": item["case_id"],
        "e174_numeric_release_pass": baseline.get(
            "numeric_release_pass", ""
        ),
        current_pass: item["numeric_release_pass"],
        "e174_numeric_failure_modes": baseline.get(
            "numeric_failure_modes", ""
        ),
        current_failures: item["numeric_failure_modes"],
    }
    for metric, direction in PAIR_METRICS.items():
        current = finite(item.get(metric))
        old = finite(baseline.get(metric))
        delta = (
            current - old
            if math.isfinite(current) and math.isfinite(old)
            else math.nan
        )
        item[f"e174_{metric}"] = old
        item[f"delta_{metric}"] = delta
        item[f"improvement_{metric}"] = (
            delta if direction == "higher" else -delta
        )
        paired[f"e174_{metric}"] = old
        paired[f"{current_prefix}_{metric}"] = current
        paired[f"delta_{metric}"] = delta
    return paired


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def group_summary(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = [("overall", "all", metrics)]
    for field in ("object_key", "person", "retarget_variant_id"):
        for value in sorted({str(row.get(field, "")) for row in metrics}):
            groups.append(
                (
                    field,
                    value,
                    [
                        row
                        for row in metrics
                        if str(row.get(field, "")) == value
                    ],
                )
            )
    output = []
    for group_type, group_value, rows in groups:
        item: dict[str, Any] = {
            "group_type": group_type,
            "group_value": group_value,
            "rows": len(rows),
            "numeric_pass": sum(
                bool(row["numeric_release_pass"]) for row in rows
            ),
            "gate_health_pass": sum(
                bool(row["leg_gate_health_pass"]) for row in rows
            ),
            "fall_count": sum(bool(row["fall_flag"]) for row in rows),
        }
        for metric in PAIR_METRICS:
            values = [finite(row.get(metric)) for row in rows]
            values = [value for value in values if math.isfinite(value)]
            item[f"{metric}_n"] = len(values)
            item[f"{metric}_mean"] = (
                statistics.fmean(values) if values else math.nan
            )
            item[f"{metric}_median"] = (
                statistics.median(values) if values else math.nan
            )
        output.append(item)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("canary", "full"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--experiment-id", default="E176")
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--output-prefix")
    parser.add_argument(
        "--video-dir",
        type=Path,
        help="optional offline-render directory; <variant>.mp4 overrides a missing run video",
    )
    parser.add_argument("--enable-tracking-gates", action="store_true")
    parser.add_argument("--root-pos-max-cm", type=float, default=20.0)
    parser.add_argument("--root-ori-max-deg", type=float, default=20.0)
    parser.add_argument("--hand-pos-max-cm", type=float, default=20.0)
    parser.add_argument("--hand-ori-max-deg", type=float, default=20.0)
    parser.add_argument("--object-pos-max-cm", type=float, default=20.0)
    parser.add_argument("--object-ori-max-deg", type=float, default=10.0)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args(argv)

    expected_rows = (
        args.expected_rows
        if args.expected_rows is not None
        else (6 if args.mode == "canary" else 39)
    )
    output_prefix = (args.output_prefix or args.experiment_id.lower()).strip()
    if (
        not output_prefix
        or not output_prefix[0].isalpha()
        or any(not (char.isalnum() or char == "_") for char in output_prefix)
    ):
        parser.error("--output-prefix must contain only letters, digits, and underscores")
    manifest = repo_path(
        args.manifest
        or RESULT_ROOT
        / "manifests"
        / (
            "lowgeom_canary_manifest.tsv"
            if args.mode == "canary"
            else "lowgeom_full_manifest.tsv"
        )
    )
    out_dir = repo_path(
        args.out_dir or RESULT_ROOT / "eval" / args.mode
    )
    video_dir = repo_path(args.video_dir) if args.video_dir else None
    tracking_thresholds = (
        {
            "root_pos": args.root_pos_max_cm,
            "root_ori": args.root_ori_max_deg,
            "hand_pos": args.hand_pos_max_cm,
            "hand_ori": args.hand_ori_max_deg,
            "object_pos": args.object_pos_max_cm,
            "object_ori": args.object_ori_max_deg,
        }
        if args.enable_tracking_gates
        else None
    )
    baseline_path = repo_path(args.baseline)
    manifest_rows = read_tsv(manifest)
    baseline_rows = {
        row["case_id"]: row for row in read_tsv(baseline_path)
    }
    required = (
        "result_npz",
        "outdir_npz",
        "config_act",
        "scene_act",
        "trajectory",
        "contact_mask",
    )
    ready: list[dict[str, str]] = []
    not_ready: list[dict[str, Any]] = []
    for row in manifest_rows:
        missing = [
            key
            for key in required
            if not repo_path(row.get(key, "")).is_file()
        ]
        if row.get("status") != "run_complete_pending_eval":
            missing.append(f"status:{row.get('status', '')}")
        if missing:
            not_ready.append(
                {**row, "not_ready_reasons": ",".join(missing)}
            )
        else:
            ready.append(row)

    metrics: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    config = EvalConfig()
    for index, row in enumerate(ready, 1):
        print(f"[{index}/{len(ready)}] {row['case_id']}", flush=True)
        try:
            metrics.append(
                evaluate_row(row, config, video_dir, tracking_thresholds)
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "case_id": row["case_id"],
                    "variant": row["variant"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    paired = []
    missing_baseline = []
    for item in metrics:
        baseline = baseline_rows.get(item["case_id"])
        if baseline is None:
            missing_baseline.append(item["case_id"])
        else:
            paired.append(add_baseline(item, baseline, output_prefix))

    transitions = Counter()
    for row in paired:
        old = boolish(row["e174_numeric_release_pass"])
        new = bool(row[f"{output_prefix}_numeric_release_pass"])
        transitions[
            f"e174_{'pass' if old else 'fail'}_to_"
            f"{output_prefix}_{'pass' if new else 'fail'}"
        ] += 1
    failures = Counter(
        failure
        for row in metrics
        for failure in row["numeric_failure_modes"].split(",")
        if failure
    )
    thresholds = {
        "body_z_err_p95_m_max": BODY_Z_MAX,
        "contact_in_mask_min": CONTACT_MIN,
        "raw_contact_min": CONTACT_MIN,
        "release_false_3mm_max": RELEASE_MAX,
        "hand_penetration_3mm_max": HAND_PEN_MAX,
        "leg_penetration_max": LEG_PEN_MAX,
        "gate_fallback_max": GATE_FALLBACK_MAX,
        "gate_valid_last_min": GATE_VALID_LAST_MIN,
    }
    if tracking_thresholds is not None:
        thresholds.update(
            {
                "track_root_pos_err_cm_mean_max": tracking_thresholds["root_pos"],
                "track_root_ori_err_deg_mean_max": tracking_thresholds["root_ori"],
                "track_eef_pos_err_cm_mean_max": tracking_thresholds["hand_pos"],
                "track_eef_ori_err_deg_mean_max": tracking_thresholds["hand_ori"],
                "track_obj_pos_err_cm_mean_max": tracking_thresholds["object_pos"],
                "track_obj_ori_err_deg_mean_max": tracking_thresholds["object_ori"],
            }
        )
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "experiment_id": args.experiment_id,
        "mode": args.mode,
        "metric_standard_id": EVAL_METRIC_STANDARD_ID,
        "tracking_gates_enabled": args.enable_tracking_gates,
        "manifest": rel(manifest),
        "manifest_sha256": sha256(manifest),
        "baseline": rel(baseline_path),
        "baseline_sha256": sha256(baseline_path),
        "counts": {
            "expected_rows": expected_rows,
            "manifest_rows": len(manifest_rows),
            "evaluated": len(metrics),
            "not_ready": len(not_ready),
            "errors": len(errors),
            "paired_rows": len(paired),
            "missing_baseline": len(missing_baseline),
            "numeric_pass": sum(
                bool(row["numeric_release_pass"]) for row in metrics
            ),
            "gate_health_pass": sum(
                bool(row["leg_gate_health_pass"]) for row in metrics
            ),
        },
        "numeric_failure_counts": dict(failures),
        "paired_numeric_transitions": dict(transitions),
        "tracking_gate_pass_counts": {
            key: sum(bool(row.get(f"{key}_gate_pass")) for row in metrics)
            for key in TRACKING_GATE_KEYS
        }
        if tracking_thresholds is not None
        else {},
        "thresholds": thresholds,
        "status": (
            "pass"
            if len(metrics) == expected_rows
            and not not_ready
            and not errors
            and not missing_baseline
            else "incomplete"
        ),
    }

    fields: list[str] = []
    priority = [
        "case_id",
        "variant",
        "object_key",
        "person",
        "numeric_release_pass",
        "numeric_failure_modes",
        "leg_gate_health_pass",
        *PAIR_METRICS,
    ]
    for key in [*priority, *METRIC_FIELDS, *METRIC_KEYS, *HEALTH_AGGS]:
        if key not in fields:
            fields.append(key)
    for row in metrics:
        for key in row:
            if key not in fields:
                fields.append(key)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(out_dir / f"{output_prefix}_case_metrics.tsv", metrics, fields)
    write_tsv(out_dir / f"{output_prefix}_vs_e174_paired_deltas.tsv", paired)
    write_tsv(out_dir / f"{output_prefix}_group_summary.tsv", group_summary(metrics))
    write_tsv(out_dir / f"{output_prefix}_not_ready.tsv", not_ready)
    write_tsv(out_dir / f"{output_prefix}_evaluation_errors.tsv", errors)
    write_tsv(out_dir / "evaluated_manifest_snapshot.tsv", ready)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["counts"], sort_keys=True))
    if errors:
        return 1
    if args.require_all and summary["status"] != "pass":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
