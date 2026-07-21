"""Shared CEM trajectory motion-health metrics used by E166+ evaluations."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from eval.core.core_metrics import EvalConfig, mj_id, npz_qpos


METRIC_KEYS = [
    "success_tracked",
    "fall_flag",
    "hand_object_physics_contact_in_mask_frac",
    "hand_object_physics_contact_in_rl_mask_frac",
    "hand_object_physics_contact_3mm_in_mask_frac",
    "hand_object_physics_contact_5mm_in_mask_frac",
    "hand_object_physics_penetration_3mm_frame_frac",
    "hand_object_physics_penetration_5mm_frame_frac",
    "hand_geom_penetration_2mm_frac",
    "hand_geom_penetration_5mm_frac",
    "hand_object_release_false_contact_3mm_frac",
    "track_joint_err_deg_mean",
    "track_eef_pos_err_cm_mean",
    "track_eef_ori_err_deg_mean",
    "track_obj_pos_err_cm_mean",
    "track_obj_ori_err_deg_mean",
    "track_root_pos_err_cm_mean",
    "track_root_ori_err_deg_mean",
    "qpos_speed_l2_p95",
    "qpos_accel_l2_p95",
    "qpos_jerk_l2_p95",
    "trackbody_speed_max",
    "ankle_speed_max",
    "wrist_speed_max",
    "trackbody_acc_max",
    "ankle_acc_max",
    "trackbody_jerk_p95",
    "ankle_jerk_p95",
    "obj_speed_max",
    "foot_slip_max_m",
    "foot_ground_dev_max_m",
    "foot_grounded_frame_frac",
]

HEALTH_AGGS = {
    "sample_smooth_accel_p95_mean": "mean",
    "sample_smooth_accel_p95_max": "max",
    "sample_smooth_jerk_p95_mean": "mean",
    "sample_smooth_jerk_p95_max": "max",
    "sample_smooth_penalty_mean": "mean",
    "sample_smooth_penalty_max": "max",
    "sample_foot_slip_speed_mean_mean": "mean",
    "sample_foot_slip_speed_peak_mean": "mean",
    "sample_foot_slip_speed_peak_max": "max",
    "sample_foot_ground_dev_mean_mean": "mean",
    "sample_foot_ground_dev_peak_mean": "mean",
    "sample_foot_ground_dev_peak_max": "max",
    "sample_foot_penalty_mean": "mean",
    "sample_foot_penalty_max": "max",
}

TRACK_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]
ANKLE_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
WRIST_BODY_NAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]


def reduce_array(array: np.ndarray, aggregation: str) -> float:
    values = np.asarray(array, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    if aggregation == "max":
        return float(values.max())
    return float(values.mean())


def qpos_kinematic_health(qpos_path: Path) -> dict[str, Any]:
    out = {
        "qpos_speed_l2_p95": math.nan,
        "qpos_accel_l2_p95": math.nan,
        "qpos_jerk_l2_p95": math.nan,
        "qpos_frames": 0,
    }
    if not qpos_path.is_file():
        return out
    with np.load(qpos_path, allow_pickle=True) as data:
        if "qpos" not in data.files:
            return out
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        time = np.asarray(data["time"], dtype=np.float64) if "time" in data.files else None
    if qpos.ndim < 2 or qpos.shape[0] < 4:
        return out
    flat = qpos.reshape(qpos.shape[0], -1)
    fps = 50.0
    if time is not None:
        time_1d = time.reshape(qpos.shape[0], -1)[:, 0]
        dt = np.diff(time_1d)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size:
            fps = float(1.0 / np.median(dt))
    speed = np.linalg.norm(np.diff(flat, axis=0), axis=1) * fps
    accel = np.linalg.norm(np.diff(flat, n=2, axis=0), axis=1) * (fps**2)
    jerk = np.linalg.norm(np.diff(flat, n=3, axis=0), axis=1) * (fps**3)
    out.update(
        {
            "qpos_speed_l2_p95": float(np.percentile(speed, 95)) if speed.size else math.nan,
            "qpos_accel_l2_p95": float(np.percentile(accel, 95)) if accel.size else math.nan,
            "qpos_jerk_l2_p95": float(np.percentile(jerk, 95)) if jerk.size else math.nan,
            "qpos_frames": int(qpos.shape[0]),
        }
    )
    return out


def fps_from_npz(qpos_path: Path, default: float) -> float:
    if not qpos_path.is_file():
        return default
    with np.load(qpos_path, allow_pickle=True) as data:
        if "time" not in data.files:
            return default
        time = np.asarray(data["time"], dtype=np.float64)
    time_1d = time.reshape(time.shape[0], -1)[:, 0]
    dt = np.diff(time_1d)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(1.0 / np.median(dt)) if dt.size else default


def contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    breaks = np.where(np.diff(indices) > 1)[0]
    starts = np.r_[indices[0], indices[breaks + 1]]
    ends = np.r_[indices[breaks], indices[-1]]
    return [(int(start), int(end) + 1) for start, end in zip(starts, ends)]


def foot_motion_metrics(ankles: np.ndarray) -> dict[str, float]:
    ground_z = np.percentile(ankles[:, :, 2], 5, axis=0)
    grounded = ankles[:, :, 2] <= (ground_z[np.newaxis, :] + 0.05)
    slip_values: list[float] = []
    ground_dev_values: list[float] = []
    for foot_index in range(ankles.shape[1]):
        for start, end in contiguous_segments(grounded[:, foot_index]):
            if end - start < 2:
                continue
            xy = ankles[start:end, foot_index, :2]
            z = ankles[start:end, foot_index, 2]
            slip_values.append(float(np.linalg.norm(xy - xy[0], axis=-1).max()))
            ground_dev_values.append(float(np.abs(z - ground_z[foot_index]).max()))
    return {
        "foot_slip_max_m": max(slip_values) if slip_values else math.nan,
        "foot_ground_dev_max_m": max(ground_dev_values) if ground_dev_values else math.nan,
        "foot_grounded_frame_frac": float(np.mean(np.any(grounded, axis=1))),
    }


def body_motion_health(
    qpos_path: Path,
    scene_xml: Path,
    config: EvalConfig,
) -> dict[str, Any]:
    out = {
        "trackbody_speed_max": math.nan,
        "ankle_speed_max": math.nan,
        "wrist_speed_max": math.nan,
        "trackbody_acc_max": math.nan,
        "ankle_acc_max": math.nan,
        "trackbody_jerk_p95": math.nan,
        "ankle_jerk_p95": math.nan,
        "obj_speed_max": math.nan,
        "foot_slip_max_m": math.nan,
        "foot_ground_dev_max_m": math.nan,
        "foot_grounded_frame_frac": math.nan,
    }
    if not qpos_path.is_file() or not scene_xml.is_file():
        return out
    qpos, _ = npz_qpos(qpos_path)
    if qpos.shape[0] < 4:
        return out
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    track_ids = [
        body_id
        for name in TRACK_BODY_NAMES
        if (body_id := mj_id(model, mujoco.mjtObj.mjOBJ_BODY, name)) >= 0
    ]
    ankle_ids = [
        body_id
        for name in ANKLE_BODY_NAMES
        if (body_id := mj_id(model, mujoco.mjtObj.mjOBJ_BODY, name)) >= 0
    ]
    wrist_ids = [
        body_id
        for name in WRIST_BODY_NAMES
        if (body_id := mj_id(model, mujoco.mjtObj.mjOBJ_BODY, name)) >= 0
    ]
    object_id = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if not track_ids or not ankle_ids or object_id < 0:
        return out

    body_positions = []
    ankle_positions = []
    wrist_positions = []
    object_positions = []
    for frame in qpos:
        data.qpos[:] = frame
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        body_positions.append(data.xpos[track_ids].copy())
        ankle_positions.append(data.xpos[ankle_ids].copy())
        if wrist_ids:
            wrist_positions.append(data.xpos[wrist_ids].copy())
        object_positions.append(data.xpos[object_id].copy())

    fps = fps_from_npz(qpos_path, config.fps)
    tracked = np.asarray(body_positions, dtype=np.float64)
    ankles = np.asarray(ankle_positions, dtype=np.float64)
    wrists = (
        np.asarray(wrist_positions, dtype=np.float64)
        if wrist_positions
        else np.empty((len(qpos), 0, 3))
    )
    obj = np.asarray(object_positions, dtype=np.float64)
    track_speed = np.linalg.norm(np.diff(tracked, axis=0), axis=-1) * fps
    ankle_speed = np.linalg.norm(np.diff(ankles, axis=0), axis=-1) * fps
    track_acc = np.linalg.norm(np.diff(tracked, n=2, axis=0), axis=-1) * (fps**2)
    ankle_acc = np.linalg.norm(np.diff(ankles, n=2, axis=0), axis=-1) * (fps**2)
    track_jerk = np.linalg.norm(np.diff(tracked, n=3, axis=0), axis=-1) * (fps**3)
    ankle_jerk = np.linalg.norm(np.diff(ankles, n=3, axis=0), axis=-1) * (fps**3)
    object_speed = np.linalg.norm(np.diff(obj, axis=0), axis=-1) * fps
    out.update(
        {
            "trackbody_speed_max": float(np.max(track_speed)) if track_speed.size else math.nan,
            "ankle_speed_max": float(np.max(ankle_speed)) if ankle_speed.size else math.nan,
            "wrist_speed_max": float(
                np.max(np.linalg.norm(np.diff(wrists, axis=0), axis=-1) * fps)
            )
            if wrists.size
            else math.nan,
            "trackbody_acc_max": float(np.max(track_acc)) if track_acc.size else math.nan,
            "ankle_acc_max": float(np.max(ankle_acc)) if ankle_acc.size else math.nan,
            "trackbody_jerk_p95": float(np.percentile(track_jerk, 95))
            if track_jerk.size
            else math.nan,
            "ankle_jerk_p95": float(np.percentile(ankle_jerk, 95))
            if ankle_jerk.size
            else math.nan,
            "obj_speed_max": float(np.max(object_speed)) if object_speed.size else math.nan,
        }
    )
    out.update(foot_motion_metrics(ankles))
    return out


def run_health(
    qpos_path: Path,
    scene_xml: Path,
    config: EvalConfig,
) -> dict[str, Any]:
    out: dict[str, Any] = {key: math.nan for key in HEALTH_AGGS}
    out.update(qpos_kinematic_health(qpos_path))
    out.update(body_motion_health(qpos_path, scene_xml, config))
    if not qpos_path.is_file():
        return out
    with np.load(qpos_path, allow_pickle=True) as data:
        for key, aggregation in HEALTH_AGGS.items():
            if key in data.files:
                out[key] = reduce_array(data[key], aggregation)
        out["has_smooth_health"] = all(
            key in data.files
            for key in ("sample_smooth_accel_p95_mean", "sample_smooth_jerk_p95_mean")
        )
        out["has_foot_health"] = all(
            key in data.files
            for key in (
                "sample_foot_slip_speed_peak_mean",
                "sample_foot_ground_dev_peak_mean",
            )
        )
    return out
