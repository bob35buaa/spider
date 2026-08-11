"""Audit E194 G1 runtime Euler references against raw quaternion authority."""

from __future__ import annotations

import json
import math
import re
import statistics
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[5]
RUNTIME_CONVENTION_RE = re.compile(r"quat→([A-Za-z]+) euler")
CLUSTER7 = frozenset(
    {
        "box001_20231003_2_041_p1",
        "box001_20231020_014_p1",
        "box001_20231020_014_p2",
        "box023_20231020_040_p1",
        "box023_20231020_040_p2",
        "box023_20231020_042_p1",
        "box023_20231020_042_p2",
    }
)


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO / path


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def parse_runtime_euler_convention(log_path: Path) -> str:
    matches = RUNTIME_CONVENTION_RE.findall(log_path.read_text(errors="replace"))
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise ValueError(f"expected one runtime Euler convention in {log_path}: {unique}")
    return unique[0]


def object_hinge_axis_sequence(model: mujoco.MjModel, body_id: int) -> str:
    start = int(model.body_jntadr[body_id])
    stop = start + int(model.body_jntnum[body_id])
    letters: list[str] = []
    for joint_id in range(start, stop):
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        axis = np.asarray(model.jnt_axis[joint_id], dtype=np.float64)
        index = int(np.argmax(np.abs(axis)))
        expected = np.zeros(3, dtype=np.float64)
        expected[index] = np.sign(axis[index])
        if not np.allclose(axis, expected, atol=1e-9) or axis[index] < 0.0:
            raise ValueError(f"unsupported object hinge axis: {axis.tolist()}")
        letters.append("XYZ"[index])
    sequence = "".join(letters)
    if len(sequence) != 3 or len(set(sequence)) != 3:
        raise ValueError(f"expected three unique object hinge axes, got {sequence!r}")
    return sequence


def load_reference(path: Path, case_id: str) -> np.ndarray:
    with np.load(path, allow_pickle=True) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
    if qpos.ndim == 3:
        person = 0 if case_id.endswith("_p1") else 1
        qpos = qpos[:, min(person, qpos.shape[1] - 1), :]
    if qpos.ndim != 2:
        raise ValueError(f"unsupported reference shape {qpos.shape}: {path}")
    return qpos


def load_public_qpos(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
    if qpos.ndim == 3:
        qpos = qpos[:, 0, :]
    if qpos.ndim != 2:
        raise ValueError(f"unsupported rollout shape {qpos.shape}: {path}")
    return qpos


def convert_reference_qpos(
    reference: np.ndarray,
    model: mujoco.MjModel,
    body_id: int,
    convention: str,
) -> np.ndarray:
    nq_robot = int(model.nq) - 6
    if reference.shape[1] < nq_robot + 7:
        raise ValueError(f"reference lacks freejoint object pose: {reference.shape}")
    body_quat = np.asarray(model.body_quat[body_id], dtype=np.float64)
    body_rot = Rotation.from_quat(np.r_[body_quat[1:], body_quat[0]])
    object_pos = reference[:, nq_robot : nq_robot + 3]
    object_quat = reference[:, nq_robot + 3 : nq_robot + 7]
    object_xyzw = np.c_[object_quat[:, 1:], object_quat[:, 0]]
    object_slide = body_rot.inv().apply(
        object_pos - np.asarray(model.body_pos[body_id])[None, :]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        object_euler = (body_rot.inv() * Rotation.from_quat(object_xyzw)).as_euler(
            convention
        )
    converted = np.zeros((len(reference), int(model.nq)), dtype=np.float64)
    converted[:, :nq_robot] = reference[:, :nq_robot]
    converted[:, nq_robot : nq_robot + 3] = object_slide
    converted[:, nq_robot + 3 : nq_robot + 6] = object_euler
    return converted


def world_object_poses(
    model: mujoco.MjModel,
    body_id: int,
    qpos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    positions = np.empty((len(qpos), 3), dtype=np.float64)
    quaternions = np.empty((len(qpos), 4), dtype=np.float64)
    for index, frame in enumerate(qpos):
        data.qpos[:] = frame
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        positions[index] = data.xpos[body_id]
        quaternions[index] = data.xquat[body_id]
    return positions, quaternions


def quaternion_angle_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = left / np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = right / np.linalg.norm(right, axis=1, keepdims=True)
    dots = np.abs(np.sum(left_norm * right_norm, axis=1))
    return np.degrees(2.0 * np.arccos(np.clip(dots, -1.0, 1.0)))


def current_meta(scene_path: Path) -> tuple[bool, str, Path]:
    meta_path = scene_path.with_name("scene_act_meta.json")
    if not meta_path.is_file():
        return False, "", meta_path
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    return True, str(payload.get("euler_convention", "")), meta_path


def error_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_max": float(np.max(values)),
    }


def audit_case(
    metric_row: dict[str, str],
    public_delta_deg: float,
    log_path: Path,
) -> dict[str, Any]:
    scene_path = resolve_path(metric_row["scene_xml"])
    trajectory_path = resolve_path(metric_row["trajectory"])
    rollout_path = resolve_path(metric_row["outdir_npz"])
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if body_id < 0:
        raise ValueError(f"scene has no object body: {scene_path}")
    runtime_convention = parse_runtime_euler_convention(log_path)
    axis_sequence = object_hinge_axis_sequence(model, body_id)
    reference = load_reference(trajectory_path, metric_row["case_id"])
    rollout = load_public_qpos(rollout_path)
    if len(reference) != len(rollout):
        raise ValueError(
            f"reference/rollout length mismatch {metric_row['case_id']}: "
            f"{len(reference)} vs {len(rollout)}"
        )
    nq_robot = int(model.nq) - 6
    raw_pos = reference[:, nq_robot : nq_robot + 3]
    raw_quat = reference[:, nq_robot + 3 : nq_robot + 7]
    runtime_qpos = convert_reference_qpos(reference, model, body_id, runtime_convention)
    axis_qpos = convert_reference_qpos(reference, model, body_id, axis_sequence)
    runtime_pos, runtime_quat = world_object_poses(model, body_id, runtime_qpos)
    axis_pos, axis_quat = world_object_poses(model, body_id, axis_qpos)
    rollout_pos, rollout_quat = world_object_poses(model, body_id, rollout)
    runtime_raw = quaternion_angle_deg(runtime_quat, raw_quat)
    axis_raw = quaternion_angle_deg(axis_quat, raw_quat)
    rollout_raw = quaternion_angle_deg(rollout_quat, raw_quat)
    rollout_runtime = quaternion_angle_deg(rollout_quat, runtime_quat)
    meta_exists, meta_convention, meta_path = current_meta(scene_path)
    public_g1 = float(metric_row["track_obj_ori_err_deg_mean"])
    recomputed_g1 = float(np.mean(rollout_raw))
    result: dict[str, Any] = {
        "case_id": metric_row["case_id"],
        "object_key": metric_row["object_key"],
        "retarget_variant_id": metric_row["retarget_variant_id"],
        "worker": metric_row["worker"],
        "runtime_euler_convention": runtime_convention,
        "xml_hinge_axis_sequence": axis_sequence,
        "runtime_convention_matches_xml_axes": runtime_convention == axis_sequence,
        "current_meta_exists": meta_exists,
        "current_meta_euler_convention": meta_convention,
        "current_meta_matches_runtime": meta_exists and meta_convention == runtime_convention,
        "cluster7": metric_row["case_id"] in CLUSTER7,
        "public_prg_to_g1_delta_deg": public_delta_deg,
        "public_g1_ori_err_deg_mean": public_g1,
        "g1_vs_raw_ori_err_recomputed_deg_mean": recomputed_g1,
        "g1_vs_raw_ori_err_reproduction_abs_diff_deg": abs(recomputed_g1 - public_g1),
        "runtime_target_vs_raw_pos_err_cm_mean": float(
            np.mean(np.linalg.norm(runtime_pos - raw_pos, axis=1)) * 100.0
        ),
        "runtime_target_vs_raw_pos_err_cm_max": float(
            np.max(np.linalg.norm(runtime_pos - raw_pos, axis=1)) * 100.0
        ),
        "axis_target_vs_raw_pos_err_cm_max": float(
            np.max(np.linalg.norm(axis_pos - raw_pos, axis=1)) * 100.0
        ),
        "g1_raw_minus_runtime_target_err_deg_mean": float(
            np.mean(rollout_raw) - np.mean(rollout_runtime)
        ),
        "frames": len(reference),
        "runtime_log": relative_path(log_path),
        "current_meta_path": relative_path(meta_path),
        "scene_xml": relative_path(scene_path),
        "trajectory": relative_path(trajectory_path),
        "g1_rollout": relative_path(rollout_path),
    }
    result.update(error_stats(runtime_raw, "runtime_target_vs_raw_ori_err_deg"))
    result.update(error_stats(axis_raw, "axis_target_vs_raw_ori_err_deg"))
    result.update(error_stats(rollout_runtime, "g1_vs_runtime_target_ori_err_deg"))
    return result


def finite_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else math.nan


def summary_row(subset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["public_prg_to_g1_delta_deg"]) for row in rows]
    return {
        "subset": subset,
        "n_cases": len(rows),
        "convention_match_cases": sum(
            bool(row["runtime_convention_matches_xml_axes"]) for row in rows
        ),
        "convention_mismatch_cases": sum(
            not bool(row["runtime_convention_matches_xml_axes"]) for row in rows
        ),
        "delta_mean_deg": finite_mean(deltas),
        "delta_median_deg": statistics.median(deltas) if deltas else math.nan,
        "delta_gt5_cases": sum(value > 5.0 for value in deltas),
        "runtime_target_vs_raw_ori_err_deg_mean": finite_mean(
            row["runtime_target_vs_raw_ori_err_deg_mean"] for row in rows
        ),
        "g1_vs_raw_ori_err_deg_mean": finite_mean(
            row["g1_vs_raw_ori_err_recomputed_deg_mean"] for row in rows
        ),
        "g1_vs_runtime_target_ori_err_deg_mean": finite_mean(
            row["g1_vs_runtime_target_ori_err_deg_mean"] for row in rows
        ),
        "cluster7_cases": sum(bool(row["cluster7"]) for row in rows),
    }


def audit_reference_conversions(
    metric_rows: list[dict[str, str]],
    case_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    delta_by_case = {
        str(row["case_id"]): float(row["delta_object_ori_err_deg"])
        for row in case_rows
    }
    g1_rows = [row for row in metric_rows if row.get("arm") == "G1"]
    if len(g1_rows) != len(delta_by_case):
        raise ValueError(
            f"expected one G1 metric per case, got {len(g1_rows)} for {len(delta_by_case)}"
        )
    audited: list[dict[str, Any]] = []
    for index, row in enumerate(sorted(g1_rows, key=lambda item: item["case_id"]), 1):
        log_path = REPO / f"logs/E194/cem/full_g1_expansion/{row['variant']}.log"
        audited.append(audit_case(row, delta_by_case[row["case_id"]], log_path))
        if index % 24 == 0:
            print(f"[reference {index:02d}/{len(g1_rows):02d}]")
    match = [row for row in audited if row["runtime_convention_matches_xml_axes"]]
    mismatch = [row for row in audited if not row["runtime_convention_matches_xml_axes"]]
    cluster = [row for row in audited if row["cluster7"]]
    mismatch_without_cluster = [row for row in mismatch if not row["cluster7"]]
    summaries = [
        summary_row("ALL_72", audited),
        summary_row("RUNTIME_CONVENTION_MATCH", match),
        summary_row("RUNTIME_CONVENTION_MISMATCH", mismatch),
        summary_row("MISMATCH_WITHOUT_CLUSTER7", mismatch_without_cluster),
        summary_row("CLUSTER7", cluster),
    ]
    for worker in sorted({str(row["worker"]) for row in audited}):
        worker_rows = [row for row in audited if row["worker"] == worker]
        summaries.extend(
            [
                summary_row(f"WORKER_{worker}_ALL", worker_rows),
                summary_row(
                    f"WORKER_{worker}_MATCH",
                    [row for row in worker_rows if row["runtime_convention_matches_xml_axes"]],
                ),
                summary_row(
                    f"WORKER_{worker}_MISMATCH",
                    [row for row in worker_rows if not row["runtime_convention_matches_xml_axes"]],
                ),
            ]
        )
    conversion_errors = np.asarray(
        [row["runtime_target_vs_raw_ori_err_deg_mean"] for row in audited],
        dtype=np.float64,
    )
    deltas = np.asarray(
        [row["public_prg_to_g1_delta_deg"] for row in audited], dtype=np.float64
    )
    mismatch_net = sum(float(row["public_prg_to_g1_delta_deg"]) for row in mismatch)
    cluster_net = sum(float(row["public_prg_to_g1_delta_deg"]) for row in cluster)
    headline = {
        "n_cases": len(audited),
        "convention_match_cases": len(match),
        "convention_mismatch_cases": len(mismatch),
        "runtime_convention_counts": dict(
            Counter(str(row["runtime_euler_convention"]) for row in audited)
        ),
        "mismatch_by_worker": dict(
            Counter(str(row["worker"]) for row in mismatch)
        ),
        "conversion_error_vs_delta_pearson": float(
            np.corrcoef(conversion_errors, deltas)[0, 1]
        ),
        "match_delta_mean_deg": finite_mean(
            row["public_prg_to_g1_delta_deg"] for row in match
        ),
        "mismatch_delta_mean_deg": finite_mean(
            row["public_prg_to_g1_delta_deg"] for row in mismatch
        ),
        "mismatch_without_cluster7_delta_mean_deg": finite_mean(
            row["public_prg_to_g1_delta_deg"] for row in mismatch_without_cluster
        ),
        "delta_gt5_all_in_mismatch": all(
            not row["runtime_convention_matches_xml_axes"]
            for row in audited
            if float(row["public_prg_to_g1_delta_deg"]) > 5.0
        ),
        "cluster7_share_of_mismatch_net_delta": (
            cluster_net / mismatch_net if abs(mismatch_net) > 1e-12 else math.nan
        ),
        "axis_conversion_max_error_deg": max(
            float(row["axis_target_vs_raw_ori_err_deg_max"]) for row in audited
        ),
    }
    return audited, summaries, headline
