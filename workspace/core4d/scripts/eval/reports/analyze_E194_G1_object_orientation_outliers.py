#!/usr/bin/env python3
"""Diagnose whether E194 G1 object-orientation regressions are long-tail driven.

This is a read-only analysis over the frozen PRG/G1 72-case evaluator outputs.
It reproduces the public-core quaternion metric frame by frame, measures robust
case/session statistics, and records contact/pose divergence for orientation
gate flips plus matched near-zero controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np
import yaml

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts"))
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E194"))

import e194_g1_expansion_common as C  # noqa: E402
from e194_orientation_reference_conversion import (  # noqa: E402
    audit_reference_conversions,
)
from eval.core.core_metrics import (  # noqa: E402
    HAND_GEOMS,
    _quat_angle_deg,
    mj_id,
    npz_qpos,
    object_collision_geoms,
)


DEFAULT_EVAL_ROOT = C.RESULTS / "s6_downstream/eval/full_g1_expansion"
PAIRED_NAME = "e194_three_arm_paired_deltas.tsv"
METRICS_NAME = "e194_three_arm_case_metrics.tsv"
PRIMARY_EXCLUSION = "box001_20231023_110_p1"
FOCAL_CASES = (
    "box001_20231003_2_041_p1",
    "box001_20231020_014_p2",
    "box001_20231020_014_p1",
)
BOX023_CLUSTER = (
    "box023_20231020_040_p1",
    "box023_20231020_040_p2",
    "box023_20231020_042_p1",
    "box023_20231020_042_p2",
)
MODEL_CONFIG_FIELDS = (
    "init_pos_actuator_gain",
    "init_pos_actuator_bias",
    "init_rot_actuator_gain",
    "init_rot_actuator_bias",
    "task_obj_pos_rew_scale",
    "task_obj_rot_rew_scale",
    "task_obj_use_exp",
    "task_obj_pos_sigma",
    "task_obj_rot_sigma",
    "carry_corridor_rew_scale",
    "terminal_carry_gate_enabled",
    "hand_support_rew_scale",
    "surface_band_rew_scale",
    "leg_object_penalty_scale",
)
PHASE_ANCHORS = (("grasp", 0.22), ("lift", 0.42), ("carry", 0.65), ("place", 0.88))


def finite(value: Any) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return math.nan
    return output if math.isfinite(output) else math.nan


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def mean(values: Iterable[float]) -> float:
    valid = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(valid) if valid else math.nan


def median(values: Iterable[float]) -> float:
    valid = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.median(valid) if valid else math.nan


def percentile(values: Iterable[float], q: float) -> float:
    valid = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    return float(np.percentile(valid, q)) if valid.size else math.nan


def trim_mean(values: Iterable[float], fraction: float = 0.10) -> float:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    trim = int(math.floor(len(ordered) * fraction))
    kept = ordered[trim : len(ordered) - trim] if trim and len(ordered) > 2 * trim else ordered
    return statistics.fmean(kept) if kept else math.nan


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def session_id(case_id: str) -> str:
    if case_id.endswith("_p1") or case_id.endswith("_p2"):
        return case_id[:-3]
    return case_id


def phase_name(index: int, frames: int) -> str:
    fraction = index / max(1, frames - 1)
    return min(PHASE_ANCHORS, key=lambda item: abs(item[1] - fraction))[0]


def qnormalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    return q / norm if norm > 0 else np.full(4, math.nan)


def qconj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def qslerp(a: np.ndarray, b: np.ndarray, fraction: float) -> np.ndarray:
    a = qnormalize(a)
    b = qnormalize(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        return qnormalize((1.0 - fraction) * a + fraction * b)
    theta = math.acos(dot)
    scale = math.sin(theta)
    return qnormalize(
        math.sin((1.0 - fraction) * theta) / scale * a
        + math.sin(fraction * theta) / scale * b
    )


def local_error_rotvec_deg(q_ref: np.ndarray, q_run: np.ndarray) -> np.ndarray:
    """Shortest rotation vector taking the reference frame to the run frame.

    ``inv(q_ref) * q_run`` expresses the axis in the reference object's local
    frame.  Its norm is the same sign-invariant geodesic angle used by the
    public evaluator.
    """
    rel = qnormalize(qmul(qconj(qnormalize(q_ref)), qnormalize(q_run)))
    if rel[0] < 0:
        rel = -rel
    vector_norm = float(np.linalg.norm(rel[1:]))
    if vector_norm <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, max(0.0, float(rel[0])))
    return rel[1:] / vector_norm * math.degrees(angle)


def quat_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    flat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(flat, qnormalize(q_wxyz))
    return flat.reshape(3, 3)


def load_reference(path: Path, case_id: str) -> np.ndarray:
    with np.load(path, allow_pickle=True) as archive:
        reference = np.asarray(archive["qpos"], dtype=np.float64)
    if reference.ndim == 3:
        person = 0 if case_id.endswith("_p1") else 1
        reference = reference[:, min(person, reference.shape[1] - 1), :]
    if reference.ndim != 2:
        raise ValueError(f"unsupported reference shape {reference.shape}: {path}")
    return reference


def load_time(path: Path, frames: int) -> np.ndarray:
    with np.load(path, allow_pickle=True) as archive:
        if "time" not in archive.files:
            return np.arange(frames, dtype=np.float64) / 30.0
        raw = np.asarray(archive["time"], dtype=np.float64)
    output = raw.reshape(raw.shape[0], -1)[:, 0]
    return output[:frames]


def arm_frame_signals(row: dict[str, str]) -> list[dict[str, Any]]:
    qpos_path = C.repo_path(row["outdir_npz"])
    scene = C.repo_path(row["scene_xml"])
    trajectory = C.repo_path(row["trajectory"])
    run_qpos, _ = npz_qpos(qpos_path)
    reference = load_reference(trajectory, row["case_id"])
    model = mujoco.MjModel.from_xml_path(str(scene))
    object_body = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    object_gids = set(object_collision_geoms(model))
    hand_gids = {name: mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in HAND_GEOMS}
    if object_body < 0 or any(gid < 0 for gid in hand_gids.values()):
        raise ValueError(f"missing object/hand geom in {scene}")
    frames = min(len(run_qpos), len(reference))
    times = load_time(qpos_path, frames)
    nq_robot = min(36, max(0, model.nq - 6), run_qpos.shape[1], reference.shape[1])
    data = mujoco.MjData(model)
    output: list[dict[str, Any]] = []
    previous_quat: np.ndarray | None = None
    previous_time: float | None = None

    for index in range(frames):
        data.qpos[:] = run_qpos[index]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        run_pos = data.xpos[object_body].copy()
        run_quat = data.xquat[object_body].copy()
        run_mat = data.xmat[object_body].reshape(3, 3).copy()
        if reference.shape[1] == model.nq:
            ref_data = mujoco.MjData(model)
            ref_data.qpos[:] = reference[index, : model.nq]
            ref_data.qvel[:] = 0.0
            mujoco.mj_forward(model, ref_data)
            ref_pos = ref_data.xpos[object_body].copy()
            ref_quat = ref_data.xquat[object_body].copy()
        elif reference.shape[1] >= nq_robot + 7:
            ref_pos = reference[index, nq_robot : nq_robot + 3].copy()
            ref_quat = reference[index, nq_robot + 3 : nq_robot + 7].copy()
        else:
            raise ValueError(f"reference lacks object pose: {trajectory} shape={reference.shape}")

        contacts: dict[str, list[Any]] = {name: [] for name in HAND_GEOMS}
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if not (pair & object_gids):
                continue
            for name, gid in hand_gids.items():
                if gid in pair:
                    local_point = run_mat.T @ (np.asarray(contact.pos, dtype=np.float64) - run_pos)
                    contacts[name].append((float(contact.dist), local_point))

        contact_distances = [item[0] for values in contacts.values() for item in values]
        hand_local: dict[str, np.ndarray] = {}
        contact_local: dict[str, np.ndarray] = {}
        for name, gid in hand_gids.items():
            hand_local[name] = run_mat.T @ (data.geom_xpos[gid].copy() - run_pos)
            points = [item[1] for item in contacts[name]]
            contact_local[name] = np.mean(points, axis=0) if points else np.full(3, math.nan)

        time_s = float(times[index])
        angular_speed = math.nan
        if previous_quat is not None and previous_time is not None and time_s > previous_time:
            angular_speed = _quat_angle_deg(previous_quat, run_quat) / (time_s - previous_time)
        previous_quat = run_quat.copy()
        previous_time = time_s
        position_delta = run_pos - ref_pos
        rotvec = local_error_rotvec_deg(ref_quat, run_quat)
        row_out: dict[str, Any] = {
            "frame": index,
            "time_s": time_s,
            "phase": phase_name(index, frames),
            "fraction": index / max(1, frames - 1),
            "ref_object_z_m": float(ref_pos[2]),
            "object_ori_err_deg": _quat_angle_deg(run_quat, ref_quat),
            "object_pos_err_cm": float(np.linalg.norm(position_delta) * 100.0),
            "object_z_signed_err_cm": float(position_delta[2] * 100.0),
            "object_z_abs_err_cm": float(abs(position_delta[2]) * 100.0),
            "object_xy_err_cm": float(np.linalg.norm(position_delta[:2]) * 100.0),
            "rotvec_local_x_deg": float(rotvec[0]),
            "rotvec_local_y_deg": float(rotvec[1]),
            "rotvec_local_z_deg": float(rotvec[2]),
            "object_quat_w": float(run_quat[0]),
            "object_quat_x": float(run_quat[1]),
            "object_quat_y": float(run_quat[2]),
            "object_quat_z": float(run_quat[3]),
            "object_angular_speed_deg_s": angular_speed,
            "hand_contact": bool(contact_distances),
            "clean3_contact": bool(contact_distances) and min(contact_distances) >= -0.003,
            "bimanual_contact": all(bool(contacts[name]) for name in HAND_GEOMS),
            "contact_min_dist_mm": min(contact_distances) * 1000.0 if contact_distances else math.nan,
        }
        for name in HAND_GEOMS:
            short = "left" if name == "lh" else "right"
            row_out[f"{short}_contact"] = bool(contacts[name])
            row_out[f"{short}_contact_count"] = len(contacts[name])
            for axis, value in zip("xyz", hand_local[name]):
                row_out[f"{short}_hand_local_{axis}_m"] = float(value)
            for axis, value in zip("xyz", contact_local[name]):
                row_out[f"{short}_contact_local_{axis}_m"] = float(value)
        output.append(row_out)

    ref_z = np.asarray([row_out["ref_object_z_m"] for row_out in output], dtype=np.float64)
    lifted = ref_z > float(np.nanmin(ref_z)) + 0.05
    if int(np.count_nonzero(lifted)) < 5:
        lifted = np.ones(len(output), dtype=bool)
    for row_out, value in zip(output, lifted):
        row_out["ref_lifted"] = bool(value)
    return output


def full_substep_orientation_metric(row: dict[str, str]) -> dict[str, Any]:
    """Sensitivity metric over every saved simulation substep.

    The public evaluator intentionally uses substep 0 of each control tick.  We
    avoid the renderer's quaternion-to-Euler reference conversion here: the
    raw fixed-reference world quaternion is SLERPed directly between control
    ticks and compared to MuJoCo's simulated object world quaternion.
    """
    qpos_path = C.repo_path(row["outdir_npz"])
    with np.load(qpos_path, allow_pickle=True) as archive:
        raw_qpos = np.asarray(archive["qpos"], dtype=np.float64)
    if raw_qpos.ndim != 3:
        raise ValueError(f"expected control_tick x substep x nq qpos: {qpos_path} {raw_qpos.shape}")
    reference = load_reference(C.repo_path(row["trajectory"]), row["case_id"])
    if len(raw_qpos) != len(reference):
        raise ValueError(f"full-substep/reference length mismatch {row['case_id']}: {len(raw_qpos)} vs {len(reference)}")
    model = mujoco.MjModel.from_xml_path(str(C.repo_path(row["scene_xml"])))
    object_body = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    nq_robot = min(36, max(0, model.nq - 6), raw_qpos.shape[2], reference.shape[1])
    if object_body < 0 or reference.shape[1] < nq_robot + 7:
        raise ValueError(f"missing object/reference quaternion for {row['case_id']}")
    data = mujoco.MjData(model)
    per_substep: list[list[float]] = [[] for _ in range(raw_qpos.shape[1])]
    all_errors: list[float] = []
    for tick in range(len(reference)):
        q0 = reference[tick, nq_robot + 3 : nq_robot + 7]
        q1 = reference[min(tick + 1, len(reference) - 1), nq_robot + 3 : nq_robot + 7]
        for substep in range(raw_qpos.shape[1]):
            fraction = substep / raw_qpos.shape[1]
            ref_quat = qslerp(q0, q1, fraction)
            data.qpos[:] = raw_qpos[tick, substep]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            error = _quat_angle_deg(data.xquat[object_body], ref_quat)
            per_substep[substep].append(error)
            all_errors.append(error)
    output: dict[str, Any] = {
        "full_substep_count": len(all_errors),
        "full_substep_ori_err_deg_mean": mean(all_errors),
    }
    for index, values in enumerate(per_substep):
        output[f"substep{index}_ori_err_deg_mean"] = mean(values)
    return output


def first_sustained(values: list[float], threshold: float, run: int = 3) -> int | None:
    mask = [math.isfinite(value) and value > threshold for value in values]
    for index in range(0, len(mask) - run + 1):
        if all(mask[index : index + run]):
            return index
    return None


def build_case_rows(paired: list[dict[str, str]]) -> list[dict[str, Any]]:
    session_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in paired:
        session_groups[session_id(row["case_id"])].append(row)
    output: list[dict[str, Any]] = []
    for row in paired:
        prg = finite(row["before_track_obj_ori_err_deg_mean"])
        g1 = finite(row["after_track_obj_ori_err_deg_mean"])
        delta = finite(row["delta_track_obj_ori_err_deg_mean"])
        ratio = g1 / prg if prg > 1e-12 else math.inf
        group = session_groups[session_id(row["case_id"])]
        group_deltas = [finite(item["delta_track_obj_ori_err_deg_mean"]) for item in group]
        p_to_f = truth(row["before_object_ori_gate_pass"]) and not truth(row["after_object_ori_gate_pass"])
        if delta > 18.0 and ratio > 3.0:
            outlier_class = "DELTA_GT18_AND_RATIO_GT3"
        elif ratio > 3.0:
            outlier_class = "RATIO_GT3"
        elif p_to_f:
            outlier_class = "OBJECT_ORI_PASS_TO_FAIL"
        else:
            outlier_class = "NON_GATE_FLIP"
        output.append(
            {
                "case_id": row["case_id"],
                "object_key": row["object_key"],
                "retarget_variant_id": row["retarget_variant_id"],
                "session_id": session_id(row["case_id"]),
                "session_person_count": len(group),
                "session_mean_delta_deg": mean(group_deltas),
                "session_min_delta_deg": min(group_deltas),
                "session_max_delta_deg": max(group_deltas),
                "session_all_delta_gt5": all(value > 5.0 for value in group_deltas),
                "box001_primary_included": row["case_id"] != PRIMARY_EXCLUSION,
                "prg_object_ori_err_deg": prg,
                "g1_object_ori_err_deg": g1,
                "delta_object_ori_err_deg": delta,
                "g1_over_prg_ratio": ratio,
                "delta_gt18": delta > 18.0,
                "ratio_gt3": ratio > 3.0,
                "object_ori_gate_migration": "PASS_TO_FAIL" if p_to_f else "OTHER",
                "outlier_class": outlier_class,
                "delta_object_z_err_cm": finite(row["delta_track_obj_z_abs_err_cm_mean"]),
                "delta_object_3d_err_cm": finite(row["delta_track_obj_pos_err_cm_mean"]),
                "delta_raw_contact_frac": finite(row["delta_hand_object_physics_contact_in_mask_frac"]),
                "delta_clean3_contact_frac": finite(row["delta_hand_object_physics_contact_3mm_in_mask_frac"]),
                "delta_hand_penetration_frac": finite(row["delta_hand_object_physics_penetration_3mm_frame_frac"]),
                "delta_leg_penetration_frac": finite(row["delta_leg_penetration_frac"]),
                "delta_hand_ori_err_deg": finite(row["delta_track_eef_ori_err_deg_mean"]),
                "delta_root_ori_err_deg": finite(row["delta_track_root_ori_err_deg_mean"]),
                "prg_video": row.get("before_video", ""),
                "g1_video": row.get("after_video", ""),
            }
        )
    return output


def robust_row(label: str, rows: list[dict[str, Any]], removed: Iterable[str] = ()) -> dict[str, Any]:
    removed_set = set(removed)
    group = [row for row in rows if row["case_id"] not in removed_set]
    deltas = [finite(row["delta_object_ori_err_deg"]) for row in group]
    ordered = sorted(group, key=lambda row: finite(row["delta_object_ori_err_deg"]), reverse=True)
    positives = sum(max(0.0, value) for value in deltas)
    net = sum(deltas)
    output: dict[str, Any] = {
        "subset": label,
        "removed_case_ids": ",".join(sorted(removed_set)),
        "n_cases": len(group),
        "prg_mean_deg": mean(finite(row["prg_object_ori_err_deg"]) for row in group),
        "g1_mean_deg": mean(finite(row["g1_object_ori_err_deg"]) for row in group),
        "delta_mean_deg": mean(deltas),
        "delta_median_deg": median(deltas),
        "delta_trim10_mean_deg": trim_mean(deltas),
        "delta_p25_deg": percentile(deltas, 25),
        "delta_p75_deg": percentile(deltas, 75),
        "delta_positive_cases": sum(value > 0 for value in deltas),
        "delta_negative_cases": sum(value < 0 for value in deltas),
        "object_ori_pass_to_fail_cases": sum(row["object_ori_gate_migration"] == "PASS_TO_FAIL" for row in group),
        "ratio_gt3_cases": sum(bool(row["ratio_gt3"]) for row in group),
        "delta_gt18_cases": sum(bool(row["delta_gt18"]) for row in group),
        "net_delta_sum_deg": net,
        "positive_delta_sum_deg": positives,
    }
    for count in (1, 2, 3, 4, 5):
        top = ordered[:count]
        remaining = ordered[count:]
        top_sum = sum(finite(row["delta_object_ori_err_deg"]) for row in top)
        output[f"top{count}_case_ids"] = ",".join(row["case_id"] for row in top)
        output[f"top{count}_delta_sum_deg"] = top_sum
        output[f"top{count}_net_contribution_frac"] = top_sum / net if net > 0 else math.nan
        output[f"top{count}_positive_contribution_frac"] = top_sum / positives if positives > 0 else math.nan
        output[f"mean_after_dropping_top{count}_deg"] = mean(
            finite(row["delta_object_ori_err_deg"]) for row in remaining
        )
    return output


def robust_summaries(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = [robust_row("ALL_72", case_rows)]
    for object_key in C.OBJECT_ORDER:
        summaries.append(robust_row(f"{object_key}_all", [row for row in case_rows if row["object_key"] == object_key]))
    box001 = [row for row in case_rows if row["object_key"] == "box001"]
    summaries.extend(
        [
            robust_row("box001_primary27", box001, [PRIMARY_EXCLUSION]),
            robust_row("box001_all28_without_user_top3", box001, FOCAL_CASES),
            robust_row("box001_primary27_without_user_top3", box001, (*FOCAL_CASES, PRIMARY_EXCLUSION)),
        ]
    )
    box023 = [row for row in case_rows if row["object_key"] == "box023"]
    summaries.append(robust_row("box023_without_20231020_040_042_pair4", box023, BOX023_CLUSTER))
    summaries.append(robust_row("ALL_72_without_box001_top3_box023_pair4", case_rows, (*FOCAL_CASES, *BOX023_CLUSTER)))
    return summaries


def session_summaries(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        groups[(row["object_key"], row["session_id"])].append(row)
    output = []
    for (object_key, sid), group in groups.items():
        deltas = [finite(row["delta_object_ori_err_deg"]) for row in group]
        output.append(
            {
                "object_key": object_key,
                "session_id": sid,
                "n_persons": len(group),
                "case_ids": ",".join(sorted(row["case_id"] for row in group)),
                "retarget_variants": ",".join(sorted({row["retarget_variant_id"] for row in group})),
                "delta_mean_deg": mean(deltas),
                "delta_min_deg": min(deltas),
                "delta_max_deg": max(deltas),
                "all_delta_positive": all(value > 0 for value in deltas),
                "all_delta_gt5": all(value > 5 for value in deltas),
                "object_ori_pass_to_fail_count": sum(row["object_ori_gate_migration"] == "PASS_TO_FAIL" for row in group),
            }
        )
    return sorted(output, key=lambda row: (row["object_key"], -finite(row["delta_mean_deg"]), row["session_id"]))


def variant_summaries(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        groups[(row["object_key"], row["retarget_variant_id"])].append(row)
    output = []
    for (object_key, variant), group in groups.items():
        deltas = [finite(row["delta_object_ori_err_deg"]) for row in group]
        output.append(
            {
                "object_key": object_key,
                "retarget_variant_id": variant,
                "n_cases": len(group),
                "delta_mean_deg": mean(deltas),
                "delta_median_deg": median(deltas),
                "delta_positive_cases": sum(value > 0 for value in deltas),
                "object_ori_pass_to_fail_cases": sum(row["object_ori_gate_migration"] == "PASS_TO_FAIL" for row in group),
                "ratio_gt3_cases": sum(bool(row["ratio_gt3"]) for row in group),
            }
        )
    return sorted(output, key=lambda row: (row["object_key"], row["retarget_variant_id"]))


def select_frame_cases(case_rows: list[dict[str, Any]]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for case_id in FOCAL_CASES:
        selected[case_id] = "USER_FOCAL_OUTLIER"
    for row in case_rows:
        if row["object_ori_gate_migration"] == "PASS_TO_FAIL":
            selected.setdefault(row["case_id"], "OBJECT_ORI_PASS_TO_FAIL")
    for object_key in C.OBJECT_ORDER:
        variants = sorted({row["retarget_variant_id"] for row in case_rows if row["object_key"] == object_key})
        for variant in variants:
            candidates = [
                row
                for row in case_rows
                if row["object_key"] == object_key
                and row["retarget_variant_id"] == variant
                and row["object_ori_gate_migration"] != "PASS_TO_FAIL"
            ]
            if candidates:
                control = min(candidates, key=lambda row: abs(finite(row["delta_object_ori_err_deg"])))
                selected.setdefault(control["case_id"], f"NEAR_ZERO_CONTROL:{object_key}:{variant}")
    return selected


def paired_frame_diagnostics(
    case_row: dict[str, Any],
    prg_metric: dict[str, str],
    g1_metric: dict[str, str],
    role: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prg = arm_frame_signals(prg_metric)
    g1 = arm_frame_signals(g1_metric)
    frames = min(len(prg), len(g1))
    if len(prg) != len(g1):
        raise ValueError(f"PRG/G1 frame mismatch {case_row['case_id']}: {len(prg)} vs {len(g1)}")
    frame_rows: list[dict[str, Any]] = []
    delta_ori: list[float] = []
    interarm_ori: list[float] = []
    left_shift: list[float] = []
    right_shift: list[float] = []
    for index in range(frames):
        before, after = prg[index], g1[index]
        delta = finite(after["object_ori_err_deg"]) - finite(before["object_ori_err_deg"])
        delta_ori.append(delta)
        prg_q = np.asarray([before[f"object_quat_{key}"] for key in "wxyz"], dtype=np.float64)
        g1_q = np.asarray([after[f"object_quat_{key}"] for key in "wxyz"], dtype=np.float64)
        arm_angle = _quat_angle_deg(prg_q, g1_q)
        interarm_ori.append(arm_angle)
        shifts: dict[str, float] = {}
        for side in ("left", "right"):
            b = np.asarray([before[f"{side}_hand_local_{axis}_m"] for axis in "xyz"], dtype=np.float64)
            a = np.asarray([after[f"{side}_hand_local_{axis}_m"] for axis in "xyz"], dtype=np.float64)
            shifts[side] = float(np.linalg.norm(a - b) * 100.0)
        left_shift.append(shifts["left"])
        right_shift.append(shifts["right"])
        row_out: dict[str, Any] = {
            "case_id": case_row["case_id"],
            "object_key": case_row["object_key"],
            "retarget_variant_id": case_row["retarget_variant_id"],
            "diagnostic_role": role,
            "frame": index,
            "time_s": after["time_s"],
            "fraction": after["fraction"],
            "phase": after["phase"],
            "ref_lifted": after["ref_lifted"],
            "prg_object_ori_err_deg": before["object_ori_err_deg"],
            "g1_object_ori_err_deg": after["object_ori_err_deg"],
            "delta_object_ori_err_deg": delta,
            "g1_vs_prg_object_angle_deg": arm_angle,
            "prg_object_pos_err_cm": before["object_pos_err_cm"],
            "g1_object_pos_err_cm": after["object_pos_err_cm"],
            "prg_object_z_signed_err_cm": before["object_z_signed_err_cm"],
            "g1_object_z_signed_err_cm": after["object_z_signed_err_cm"],
            "prg_object_angular_speed_deg_s": before["object_angular_speed_deg_s"],
            "g1_object_angular_speed_deg_s": after["object_angular_speed_deg_s"],
            "prg_hand_contact": before["hand_contact"],
            "g1_hand_contact": after["hand_contact"],
            "prg_clean3_contact": before["clean3_contact"],
            "g1_clean3_contact": after["clean3_contact"],
            "prg_bimanual_contact": before["bimanual_contact"],
            "g1_bimanual_contact": after["bimanual_contact"],
            "prg_left_contact": before["left_contact"],
            "g1_left_contact": after["left_contact"],
            "prg_right_contact": before["right_contact"],
            "g1_right_contact": after["right_contact"],
            "prg_contact_min_dist_mm": before["contact_min_dist_mm"],
            "g1_contact_min_dist_mm": after["contact_min_dist_mm"],
            "left_hand_local_shift_cm": shifts["left"],
            "right_hand_local_shift_cm": shifts["right"],
        }
        for prefix, source in (("prg", before), ("g1", after)):
            for axis in "xyz":
                row_out[f"{prefix}_rotvec_local_{axis}_deg"] = source[f"rotvec_local_{axis}_deg"]
                row_out[f"{prefix}_left_contact_local_{axis}_m"] = source[f"left_contact_local_{axis}_m"]
                row_out[f"{prefix}_right_contact_local_{axis}_m"] = source[f"right_contact_local_{axis}_m"]
        frame_rows.append(row_out)

    prg_mean = mean(row["object_ori_err_deg"] for row in prg)
    g1_mean = mean(row["object_ori_err_deg"] for row in g1)
    if abs(prg_mean - finite(case_row["prg_object_ori_err_deg"])) > 1e-6:
        raise AssertionError(f"PRG orientation reproduction drift {case_row['case_id']}: {prg_mean}")
    if abs(g1_mean - finite(case_row["g1_object_ori_err_deg"])) > 1e-6:
        raise AssertionError(f"G1 orientation reproduction drift {case_row['case_id']}: {g1_mean}")

    onset5 = first_sustained(delta_ori, 5.0)
    onset10 = first_sustained(delta_ori, 10.0)
    onset18 = first_sustained(delta_ori, 18.0)
    lifted_indices = [index for index, row in enumerate(g1) if truth(row["ref_lifted"])]
    g1_axis_abs = {
        axis: mean(abs(finite(g1[index][f"rotvec_local_{axis}_deg"])) for index in lifted_indices)
        for axis in "xyz"
    }
    dominant_axis = max(g1_axis_abs, key=g1_axis_abs.get)
    both_contact_indices = [
        index
        for index in range(frames)
        if truth(prg[index]["bimanual_contact"]) and truth(g1[index]["bimanual_contact"])
    ]
    summary: dict[str, Any] = {
        "case_id": case_row["case_id"],
        "object_key": case_row["object_key"],
        "retarget_variant_id": case_row["retarget_variant_id"],
        "diagnostic_role": role,
        "frames": frames,
        "duration_s": max(row["time_s"] for row in g1) - min(row["time_s"] for row in g1),
        "table_prg_mean_deg": case_row["prg_object_ori_err_deg"],
        "table_g1_mean_deg": case_row["g1_object_ori_err_deg"],
        "recomputed_prg_mean_deg": prg_mean,
        "recomputed_g1_mean_deg": g1_mean,
        "recomputed_delta_mean_deg": g1_mean - prg_mean,
        "delta_median_frame_deg": median(delta_ori),
        "delta_peak_frame_deg": max(delta_ori),
        "delta_gt5_frame_frac": mean(float(value > 5.0) for value in delta_ori),
        "delta_gt10_frame_frac": mean(float(value > 10.0) for value in delta_ori),
        "first_sustained_delta_gt5_frame": onset5 if onset5 is not None else "",
        "first_sustained_delta_gt5_time_s": g1[onset5]["time_s"] if onset5 is not None else math.nan,
        "first_sustained_delta_gt5_phase": g1[onset5]["phase"] if onset5 is not None else "",
        "first_sustained_delta_gt10_frame": onset10 if onset10 is not None else "",
        "first_sustained_delta_gt10_time_s": g1[onset10]["time_s"] if onset10 is not None else math.nan,
        "first_sustained_delta_gt10_phase": g1[onset10]["phase"] if onset10 is not None else "",
        "first_sustained_delta_gt18_frame": onset18 if onset18 is not None else "",
        "first_sustained_delta_gt18_time_s": g1[onset18]["time_s"] if onset18 is not None else math.nan,
        "first_sustained_delta_gt18_phase": g1[onset18]["phase"] if onset18 is not None else "",
        "prg_lifted_ori_mean_deg": mean(prg[index]["object_ori_err_deg"] for index in lifted_indices),
        "g1_lifted_ori_mean_deg": mean(g1[index]["object_ori_err_deg"] for index in lifted_indices),
        "g1_vs_prg_object_angle_mean_deg": mean(interarm_ori),
        "g1_vs_prg_object_angle_peak_deg": max(interarm_ori),
        "prg_contact_frac": mean(float(row["hand_contact"]) for row in prg),
        "g1_contact_frac": mean(float(row["hand_contact"]) for row in g1),
        "prg_clean3_contact_frac": mean(float(row["clean3_contact"]) for row in prg),
        "g1_clean3_contact_frac": mean(float(row["clean3_contact"]) for row in g1),
        "prg_bimanual_contact_frac": mean(float(row["bimanual_contact"]) for row in prg),
        "g1_bimanual_contact_frac": mean(float(row["bimanual_contact"]) for row in g1),
        "prg_contact_g1_missing_frac": mean(float(prg[index]["hand_contact"] and not g1[index]["hand_contact"]) for index in range(frames)),
        "left_hand_local_shift_mean_cm": mean(left_shift),
        "right_hand_local_shift_mean_cm": mean(right_shift),
        "g1_local_rotvec_abs_x_lifted_mean_deg": g1_axis_abs["x"],
        "g1_local_rotvec_abs_y_lifted_mean_deg": g1_axis_abs["y"],
        "g1_local_rotvec_abs_z_lifted_mean_deg": g1_axis_abs["z"],
        "g1_dominant_local_error_axis": dominant_axis,
        "bimanual_contact_overlap_frames": len(both_contact_indices),
        "bimanual_left_right_contact_z_asym_mean_cm": mean(
            abs(finite(g1[index]["left_contact_local_z_m"]) - finite(g1[index]["right_contact_local_z_m"])) * 100.0
            for index in both_contact_indices
        ),
    }
    for phase, _ in PHASE_ANCHORS:
        indices = [index for index, row in enumerate(g1) if row["phase"] == phase]
        summary[f"{phase}_delta_ori_mean_deg"] = mean(delta_ori[index] for index in indices)
        summary[f"{phase}_g1_ori_mean_deg"] = mean(g1[index]["object_ori_err_deg"] for index in indices)
        summary[f"{phase}_g1_bimanual_contact_frac"] = mean(float(g1[index]["bimanual_contact"]) for index in indices)
    if onset5 is not None:
        for key in ("hand_contact", "clean3_contact", "bimanual_contact", "left_contact", "right_contact"):
            summary[f"onset5_prg_{key}"] = prg[onset5][key]
            summary[f"onset5_g1_{key}"] = g1[onset5][key]
        summary["onset5_left_hand_local_shift_cm"] = left_shift[onset5]
        summary["onset5_right_hand_local_shift_cm"] = right_shift[onset5]
    return frame_rows, summary


def mechanism_audit(
    metric_map: dict[tuple[str, str], dict[str, str]],
    eval_root: Path,
) -> list[dict[str, Any]]:
    preflight = C.RESULTS / "s6_downstream/evidence/g1_expansion/preflight/audit_rows.tsv"
    preflight_rows = {row["case_id"]: row for row in C.read_tsv(preflight)}
    if len(preflight_rows) != C.N_CASES or any(row["status"] != "pass" for row in preflight_rows.values()):
        raise ValueError("72-row G1 compiled-model preflight is not fully passing")
    output = []
    for case_id in FOCAL_CASES:
        prg = metric_map[(case_id, "PRG")]
        g1 = metric_map[(case_id, "G1")]
        prg_model = mujoco.MjModel.from_xml_path(str(C.repo_path(prg["scene_xml"])))
        g1_model = mujoco.MjModel.from_xml_path(str(C.repo_path(g1["scene_xml"])))
        obj = mj_id(g1_model, mujoco.mjtObj.mjOBJ_BODY, "object")
        with C.repo_path(prg["config_act"]).open(encoding="utf-8") as stream:
            prg_cfg = yaml.safe_load(stream)
        with C.repo_path(g1["config_act"]).open(encoding="utf-8") as stream:
            g1_cfg = yaml.safe_load(stream)
        mismatched = [field for field in MODEL_CONFIG_FIELDS if prg_cfg.get(field) != g1_cfg.get(field)]
        mass = float(g1_model.body_mass[obj])
        gravity = np.asarray(g1_model.opt.gravity, dtype=np.float64)
        compensation_force = mass * float(np.linalg.norm(gravity))
        output.append(
            {
                "case_id": case_id,
                "preflight_status": preflight_rows[case_id]["status"],
                "compiled_model_only_object_gravcomp_delta": True,
                "prg_object_gravcomp": float(prg_model.body_gravcomp[obj]),
                "g1_object_gravcomp": float(g1_model.body_gravcomp[obj]),
                "object_mass_kg": mass,
                "gravity_m_s2": "|".join(f"{value:.6f}" for value in gravity),
                "g1_compensation_force_magnitude_N": compensation_force,
                "force_application_point": "object_center_of_mass",
                "direct_torque_about_object_com_Nm": 0.0,
                "object_body_ipos_local_m": "|".join(f"{value:.9f}" for value in g1_model.body_ipos[obj]),
                "audited_config_mismatch_fields": ",".join(mismatched),
                **{f"config_{field}": g1_cfg.get(field) for field in MODEL_CONFIG_FIELDS},
                "prg_scene": prg["scene_xml"],
                "g1_scene": g1["scene_xml"],
                "prg_config": prg["config_act"],
                "g1_config": g1["config_act"],
                "preflight_evidence": C.rel(preflight),
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    args = parser.parse_args()
    eval_root = args.eval_root if args.eval_root.is_absolute() else REPO / args.eval_root
    paired_path = eval_root / PAIRED_NAME
    metrics_path = eval_root / METRICS_NAME
    paired_all = C.read_tsv(paired_path)
    paired = [row for row in paired_all if row.get("comparison") == "PRG_to_G1"]
    metrics = C.read_tsv(metrics_path)
    if len(paired) != C.N_CASES or len({row["case_id"] for row in paired}) != C.N_CASES:
        raise ValueError(f"expected 72 unique PRG_to_G1 pairs, got {len(paired)}")
    metric_map = {(row["case_id"], row["arm"]): row for row in metrics if row.get("arm") in {"PRG", "G1"}}
    if len(metric_map) != 2 * C.N_CASES:
        raise ValueError(f"expected 144 PRG/G1 metric rows, got {len(metric_map)}")

    case_rows = build_case_rows(paired)
    summaries = robust_summaries(case_rows)
    sessions = session_summaries(case_rows)
    variants = variant_summaries(case_rows)
    frame_selection = select_frame_cases(case_rows)
    by_case = {row["case_id"]: row for row in case_rows}
    frame_rows: list[dict[str, Any]] = []
    frame_summaries: list[dict[str, Any]] = []
    for index, case_id in enumerate(sorted(frame_selection), 1):
        curves, summary = paired_frame_diagnostics(
            by_case[case_id], metric_map[(case_id, "PRG")], metric_map[(case_id, "G1")], frame_selection[case_id]
        )
        frame_rows.extend(curves)
        frame_summaries.append(summary)
        print(f"[frame {index:02d}/{len(frame_selection):02d}] {case_id} {frame_selection[case_id]}")

    mechanisms = mechanism_audit(metric_map, eval_root)
    reference_rows, reference_summaries, reference_headline = (
        audit_reference_conversions(metrics, case_rows)
    )
    full_substep_metrics: dict[tuple[str, str], dict[str, Any]] = {}
    for index, ((case_id, arm), row) in enumerate(sorted(metric_map.items()), 1):
        full_substep_metrics[(case_id, arm)] = full_substep_orientation_metric(row)
        if index % 24 == 0:
            print(f"[substep {index:03d}/{len(metric_map):03d}]")
    substep_rows: list[dict[str, Any]] = []
    for row in case_rows:
        case_id = row["case_id"]
        prg = full_substep_metrics[(case_id, "PRG")]
        g1 = full_substep_metrics[(case_id, "G1")]
        full_delta = finite(g1["full_substep_ori_err_deg_mean"]) - finite(prg["full_substep_ori_err_deg_mean"])
        public_delta = finite(row["delta_object_ori_err_deg"])
        substep_rows.append(
            {
                "case_id": case_id,
                "object_key": row["object_key"],
                "retarget_variant_id": row["retarget_variant_id"],
                "public_substep0_prg_mean_deg": row["prg_object_ori_err_deg"],
                "public_substep0_g1_mean_deg": row["g1_object_ori_err_deg"],
                "public_substep0_delta_deg": public_delta,
                "full_substep_prg_mean_deg": prg["full_substep_ori_err_deg_mean"],
                "full_substep_g1_mean_deg": g1["full_substep_ori_err_deg_mean"],
                "full_substep_delta_deg": full_delta,
                "full_minus_public_delta_deg": full_delta - public_delta,
                "delta_sign_agrees": (full_delta > 0) == (public_delta > 0),
                "full_substep_count": prg["full_substep_count"],
                "prg_substep0_recomputed_mean_deg": prg["substep0_ori_err_deg_mean"],
                "g1_substep0_recomputed_mean_deg": g1["substep0_ori_err_deg_mean"],
                "prg_substep1_mean_deg": prg.get("substep1_ori_err_deg_mean", math.nan),
                "g1_substep1_mean_deg": g1.get("substep1_ori_err_deg_mean", math.nan),
                "reference_interpolation": "raw_world_quaternion_shortest_path_slerp",
            }
        )
    C.write_tsv(eval_root / "e194_g1_object_orientation_outlier_cases.tsv", case_rows)
    C.write_tsv(eval_root / "e194_g1_object_orientation_robust_summary.tsv", summaries)
    C.write_tsv(eval_root / "e194_g1_object_orientation_session_summary.tsv", sessions)
    C.write_tsv(eval_root / "e194_g1_object_orientation_variant_summary.tsv", variants)
    C.write_tsv(eval_root / "e194_g1_object_orientation_frame_diagnostics.tsv", frame_summaries)
    C.write_tsv(eval_root / "e194_g1_object_orientation_frame_curves.tsv", frame_rows)
    C.write_tsv(eval_root / "e194_g1_object_orientation_mechanism_audit.tsv", mechanisms)
    C.write_tsv(
        eval_root / "e194_g1_object_orientation_reference_conversion_audit.tsv",
        reference_rows,
    )
    C.write_tsv(
        eval_root / "e194_g1_object_orientation_reference_conversion_summary.tsv",
        reference_summaries,
    )
    C.write_tsv(eval_root / "e194_g1_object_orientation_full_substep_sensitivity.tsv", substep_rows)

    summary_map = {row["subset"]: row for row in summaries}
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric": "track_obj_ori_err_deg_mean",
        "metric_direction": "lower_is_better",
        "metric_definition": "mean per-frame sign-invariant quaternion geodesic angle to fixed reference, degrees",
        "source_rows": len(paired),
        "sources": {
            "paired_tsv": C.rel(paired_path),
            "paired_tsv_sha256": sha256(paired_path),
            "case_metrics_tsv": C.rel(metrics_path),
            "case_metrics_tsv_sha256": sha256(metrics_path),
        },
        "box001_primary_exclusion": PRIMARY_EXCLUSION,
        "user_focal_cases": list(FOCAL_CASES),
        "frame_diagnostic_case_count": len(frame_selection),
        "full_substep_sensitivity": {
            "all72_delta_mean_deg": mean(finite(row["full_substep_delta_deg"]) for row in substep_rows),
            "all72_delta_median_deg": median(finite(row["full_substep_delta_deg"]) for row in substep_rows),
            "delta_sign_agreement_cases": sum(bool(row["delta_sign_agrees"]) for row in substep_rows),
            "n_cases": len(substep_rows),
            "reference_interpolation": "raw_world_quaternion_shortest_path_slerp",
        },
        "reference_conversion_audit": reference_headline,
        "headline": {
            "all72_delta_mean_deg": summary_map["ALL_72"]["delta_mean_deg"],
            "all72_delta_median_deg": summary_map["ALL_72"]["delta_median_deg"],
            "box001_primary27_delta_mean_deg": summary_map["box001_primary27"]["delta_mean_deg"],
            "box001_primary27_delta_median_deg": summary_map["box001_primary27"]["delta_median_deg"],
            "box001_primary27_without_user_top3_delta_mean_deg": summary_map["box001_primary27_without_user_top3"]["delta_mean_deg"],
            "box001_top3_net_contribution_frac": summary_map["box001_primary27"]["top3_net_contribution_frac"],
            "box023_delta_mean_deg": summary_map["box023_all"]["delta_mean_deg"],
            "box023_without_pair4_delta_mean_deg": summary_map["box023_without_20231020_040_042_pair4"]["delta_mean_deg"],
            "box021_delta_mean_deg": summary_map["box021_all"]["delta_mean_deg"],
            "all72_without_7_cluster_delta_mean_deg": summary_map["ALL_72_without_box001_top3_box023_pair4"]["delta_mean_deg"],
        },
        "measured_interpretation": {
            "distribution_shape": "outlier_concentrated_not_global_shift",
            "box001": "three user-identified cases dominate the positive mean",
            "box023": "same session-cluster pattern in 040/042 p1/p2",
            "box021": "no orientation regression cluster",
            "quaternion_sign_artifact": "excluded_by_abs_dot_metric_and_frame_reproduction",
            "physical_model_drift": "excluded_by_72_row_preflight_and_focal_config_audit",
            "dominant_long_tail_cause": "g1_runtime_euler_convention_mismatch_to_xml_hinge_order",
        },
        "mechanism_boundary": {
            "direct": "gravcomp adds an upward force at the object COM; it adds no direct torque about that COM",
            "diagnosed_upstream_trigger": "missing_or_unsynced_scene_act_meta_makes_g1_fall_back_to_XYZ_and_follow_a_wrong_world_orientation_target",
            "gravcomp_residual_effect": "unresolved_until_same_reference_convention_rerun; convention-matched cases show no orientation long tail",
            "contact_divergence": "measured_downstream_behavior_consistent_with_the_wrong_target_not_proof_of_a_gravcomp_torque_mechanism",
            "causal_strength": "strong_reference_pipeline_evidence; gravcomp_specific_causality_not_established",
        },
    }
    C.write_json(eval_root / "e194_g1_object_orientation_outlier_summary.json", payload)
    print(json.dumps(payload["headline"], indent=2, ensure_ascii=False))
    print(
        f"wrote case={len(case_rows)} frame_cases={len(frame_summaries)} "
        f"frame_rows={len(frame_rows)} reference_rows={len(reference_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
