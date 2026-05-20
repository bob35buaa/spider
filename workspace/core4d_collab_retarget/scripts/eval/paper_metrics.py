#!/usr/bin/env python3
"""Paper-aligned eval metrics for CORE4D collaborative retargeting.

The helper intentionally produces proxy metrics when the exact paper signal is
not available in SPIDER rollouts. Field names use `paper_*` and include the
source prefix in the name where useful.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


# CORE4D processed data is 30Hz. spider E018b NPZ verified: time delta
# 1/60s × 2 substeps = 1/30s base. holosoma v2 kinematic NPZ fps=30 explicit.
# If a future dataset has different fps, set summary["fps"] before calling
# add_paper_metrics — the entry warns when summary["fps"] != FPS.
FPS = 30.0
FOOT_STANCE_Z_M = 0.04
FOOT_REF_STICK_VEL_MPS = 0.05
FOOT_SKATE_VEL_MPS = 0.05
DEEP_PENETRATION_THRESHOLD_M = 0.02

# OmniRetarget Table II contact preservation thresholds
# (eval_paper_metrics.py:40)
CONTACT_PRESERVATION_LOCAL_RADIUS_M = 0.28  # 28cm in object local frame
# SMPL-X wrist indices for hand contact (eval_paper_metrics.py:59-60)
SMPLX_L_WRIST_IDX = 20
SMPLX_R_WRIST_IDX = 21
SMPLX_L_FOOT_IDX = 10
SMPLX_R_FOOT_IDX = 11

# OmniRetarget Table II penetration thresholds (aligned with
# holosoma/workspace/v1/scripts/eval_paper_metrics.py:37-40)
PEN_COLLISION_DETECTION_THRESHOLD = 0.1  # margin for broadphase prefilter (m)
PEN_TOLERANCE = 0.01  # 1cm tolerance per paper

# Pelvis / EEF body names in unitree_g1 MJCF.
# E018b scene_e018b_jointB_*.xml uses *_wrist_yaw_link as the last wrist body
# (rubber_hand is only a mesh, not a body).
PELVIS_BODY_NAME = "pelvis"
LEFT_EEF_BODY_NAME = "left_wrist_yaw_link"
RIGHT_EEF_BODY_NAME = "right_wrist_yaw_link"
# Non-robot bodies to exclude from MPKPE/orient set.
NON_ROBOT_BODY_NAMES = ("object", "support_weld_anchor", "support_dynamic_anchor")


def _as_float_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8, None)


def _quat_angle_rad(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    q0 = _quat_normalize(q0.astype(np.float64))
    q1 = _quat_normalize(q1.astype(np.float64))
    dot = np.abs(np.sum(q0 * q1, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def _window_bounds(summary: dict[str, Any], T: int) -> tuple[int, int]:
    start = min(max(int(summary["case_window_start_frame"]), 0), max(T - 1, 0))
    end = min(max(int(summary["case_window_end_frame"]), start), max(T - 1, 0))
    return start, end + 1


def _smoothness(q: np.ndarray) -> float:
    if len(q) < 3:
        return 0.0
    qdd = (q[2:] - 2.0 * q[1:-1] + q[:-2]) * (FPS * FPS)
    return float(np.abs(qdd).sum())


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if abs(den) > 1e-8 else 0.0


def _foot_xy(model: mujoco.MjModel, qpos: np.ndarray) -> dict[str, np.ndarray]:
    data = mujoco.MjData(model)
    out = {
        "left": np.zeros((len(qpos), 2), dtype=np.float64),
        "right": np.zeros((len(qpos), 2), dtype=np.float64),
    }
    for side in out:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_foot")
        if site_id < 0:
            raise ValueError(f"Missing site {side}_foot")
        for t, q in enumerate(qpos):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            out[side][t] = data.site_xpos[site_id, :2]
    return out


def _read_rows(path: Path, *, kind: str | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if kind is None or row.get("kind") == kind:
                rows.append(row)
    return rows


def _resize_bool_mask(mask: np.ndarray, target_len: int) -> np.ndarray:
    mask = mask.astype(bool)
    if len(mask) == target_len:
        return mask
    if len(mask) == 0:
        return np.zeros((target_len,) + mask.shape[1:], dtype=bool)
    idx = np.round(np.linspace(0, len(mask) - 1, target_len)).astype(int)
    return mask[idx]


def _load_contact_mask(
    summary: dict[str, Any],
    repo: Path,
    target_len: int,
    person_idx: int,
) -> np.ndarray | None:
    path_text = summary.get("case_window_mask_path")
    key = summary.get("case_window_mask_key")
    if not path_text or not key:
        return None
    path = repo / str(path_text)
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=True)
    if key not in data:
        return None
    raw = data[str(key)]
    if raw.ndim == 3:
        person_idx = min(max(person_idx, 0), raw.shape[1] - 1)
        raw = raw[:, person_idx, :]
    if raw.ndim == 1:
        raw = raw[:, None]
    if raw.shape[-1] == 1:
        raw = np.repeat(raw, 2, axis=-1)
    return _resize_bool_mask(raw[:, :2], target_len)


def _add_object_tracking_metrics(
    out: dict[str, Any],
    summary: dict[str, Any],
    qpos: np.ndarray,
    qpos_ref: np.ndarray,
) -> None:
    T = min(len(qpos), len(qpos_ref))
    start, end = _window_bounds(summary, T)
    obj = qpos[:T, -7:-4].astype(np.float64)
    ref = qpos_ref[:T, -7:-4].astype(np.float64)
    quat = qpos[:T, -4:].astype(np.float64)
    quat_ref = qpos_ref[:T, -4:].astype(np.float64)

    pos_err = np.linalg.norm(obj - ref, axis=1)
    xy_err = np.linalg.norm(obj[:, :2] - ref[:, :2], axis=1)
    z_err = np.abs(obj[:, 2] - ref[:, 2])
    rot_rad = _quat_angle_rad(quat, quat_ref)

    cw_pos = pos_err[start:end]
    cw_xy = xy_err[start:end]
    cw_z = z_err[start:end]
    cw_rot = rot_rad[start:end]

    out["paper_object_Epos_full_m"] = float(pos_err.mean())
    out["paper_object_Epos_case_m"] = float(cw_pos.mean())
    out["paper_object_Epos_case_max_m"] = float(cw_pos.max())
    out["paper_object_xy_error_case_m"] = float(cw_xy.mean())
    out["paper_object_z_error_case_m"] = float(cw_z.mean())
    out["paper_object_pos_error_final_m"] = float(pos_err[end - 1])
    out["paper_object_Erot_full_rad"] = float(rot_rad.mean())
    out["paper_object_Erot_case_rad"] = float(cw_rot.mean())
    out["paper_object_Erot_case_deg"] = float(np.degrees(cw_rot.mean()))
    out["paper_object_Erot_case_max_deg"] = float(np.degrees(cw_rot.max()))
    out["paper_spider_object_success"] = bool(
        out["paper_object_Epos_case_m"] < 0.10
        and out["paper_object_Erot_case_rad"] < 0.50
    )
    out["paper_dynaretarget_object_success"] = bool(
        out["paper_object_Epos_case_m"] < 0.10
        and out["paper_object_Erot_case_deg"] < 25.0
    )

    ref_disp = ref[end - 1, :2] - ref[start, :2]
    sim_disp = obj[end - 1, :2] - obj[start, :2]
    ref_dist = float(np.linalg.norm(ref_disp))
    sim_dist = float(np.linalg.norm(sim_disp))
    ref_unit = ref_disp / max(ref_dist, 1e-8)
    progress = float(np.dot(sim_disp, ref_unit))
    out["paper_carry_xy_displacement_case_m"] = sim_dist
    out["paper_carry_xy_progress_case_m"] = progress
    out["paper_carry_progress_ratio_case"] = _safe_ratio(progress, ref_dist)
    out["paper_object_z_height_case_mean_m"] = float(obj[start:end, 2].mean())
    out["paper_object_z_height_case_max_m"] = float(obj[start:end, 2].max())
    out["paper_transport_success"] = bool(
        out["paper_carry_progress_ratio_case"] >= 0.70
        and out["paper_object_z_height_case_mean_m"] >= 0.20
        and out["paper_object_Epos_case_m"] < 0.20
    )


def _add_smoothness_metrics(
    out: dict[str, Any],
    qpos: np.ndarray,
    qpos_ref: np.ndarray,
) -> None:
    T = min(len(qpos), len(qpos_ref))
    # G1 freejoint layout in this workspace: root(7) + 29 robot joints + object(7).
    robot_q = qpos[:T, 7:36].astype(np.float64)
    robot_ref = qpos_ref[:T, 7:36].astype(np.float64)
    s = _smoothness(robot_q)
    s_ref = _smoothness(robot_ref)
    out["paper_dynaretarget_smoothness"] = s
    out["paper_dynaretarget_ref_smoothness"] = s_ref
    out["paper_dynaretarget_relative_smoothness"] = _safe_ratio(s, s_ref)


def _add_keypoint_proxy_metrics(
    out: dict[str, Any],
    summary: dict[str, Any],
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qpos_ref: np.ndarray,
) -> None:
    import eval_E072 as e072  # Imported lazily to keep this helper standalone.

    T = min(len(qpos), len(qpos_ref))
    start, end = _window_bounds(summary, T)
    sim = e072.replay_metrics(model, qpos[:T])
    ref = e072.replay_metrics(model, qpos_ref[:T])
    sim_foot = _foot_xy(model, qpos[:T])
    ref_foot = _foot_xy(model, qpos_ref[:T])

    sim_kp = np.stack(
        [
            sim["pelvis_pos"],
            sim["left_palm_pos"],
            sim["right_palm_pos"],
            np.column_stack([sim_foot["left"], sim["left_foot_z"]]),
            np.column_stack([sim_foot["right"], sim["right_foot_z"]]),
        ],
        axis=1,
    )
    ref_kp = np.stack(
        [
            ref["pelvis_pos"],
            ref["left_palm_pos"],
            ref["right_palm_pos"],
            np.column_stack([ref_foot["left"], ref["left_foot_z"]]),
            np.column_stack([ref_foot["right"], ref["right_foot_z"]]),
        ],
        axis=1,
    )
    kp_err = np.linalg.norm(sim_kp - ref_kp, axis=-1)
    out["paper_mpkpe_proxy_case_cm"] = float(kp_err[start:end].mean() * 100.0)
    out["paper_pelvis_pos_error_case_cm"] = float(
        np.linalg.norm(
            sim["pelvis_pos"][start:end] - ref["pelvis_pos"][start:end], axis=1
        ).mean()
        * 100.0
    )

    left_ref_z = ref["left_foot_z"][:T]
    right_ref_z = ref["right_foot_z"][:T]
    sim_feet = {"left": sim_foot["left"][:T], "right": sim_foot["right"][:T]}
    ref_feet = {"left": ref_foot["left"][:T], "right": ref_foot["right"][:T]}
    ref_z = {"left": left_ref_z, "right": right_ref_z}

    desired = []
    skate = []
    velocities = []
    for side in ("left", "right"):
        sim_step = np.zeros(T)
        ref_step = np.zeros(T)
        sim_step[1:] = np.linalg.norm(np.diff(sim_feet[side], axis=0), axis=1)
        ref_step[1:] = np.linalg.norm(np.diff(ref_feet[side], axis=0), axis=1)
        sim_vel = sim_step * FPS
        ref_vel = ref_step * FPS
        stance = (
            (ref_z[side] < FOOT_STANCE_Z_M)
            & (ref_vel < FOOT_REF_STICK_VEL_MPS)
        )
        stance = stance[start:end]
        desired.append(stance)
        skate.append(stance & (sim_vel[start:end] > FOOT_SKATE_VEL_MPS))
        velocities.append(sim_vel[start:end][stance])

    desired_mask = np.concatenate(desired) if desired else np.zeros(0, dtype=bool)
    skate_mask = np.concatenate(skate) if skate else np.zeros(0, dtype=bool)
    nonempty_velocities = [v for v in velocities if len(v)]
    vel_vals = (
        np.concatenate(nonempty_velocities)
        if nonempty_velocities
        else np.zeros(0, dtype=np.float64)
    )
    denom = int(desired_mask.sum())
    out["paper_omniretarget_foot_stance_frames"] = denom
    out["paper_omniretarget_foot_skating_duration_pct"] = (
        float(skate_mask.sum() / denom * 100.0) if denom else 0.0
    )
    out["paper_omniretarget_foot_skating_max_vel_cm_s"] = (
        float(vel_vals.max() * 100.0) if len(vel_vals) else 0.0
    )


def _add_body_tracking_metrics(
    out: dict[str, Any],
    summary: dict[str, Any],
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qpos_ref: np.ndarray,
) -> None:
    """SPIDER Table 4 strict alignment via per-frame FK.

    Adds (case-window mean, full mean optional):
      paper_spider_joint_err_deg
      paper_spider_pos_err_cm        (MPKPE over robot bodies, body indices [1 .. nbody-2])
      paper_spider_ori_err_deg       (orientation Err over same body set)
      paper_spider_root_pos_err_cm   (pelvis)
      paper_spider_root_ori_err_deg  (pelvis)
      paper_spider_eef_pos_err_cm    (mean of L/R rubber hand)
      paper_spider_eef_ori_err_deg   (mean of L/R rubber hand)

    Object Pos/Ori already covered by _add_object_tracking_metrics — re-exposed
    here under spider_* names for table alignment.
    """
    T = min(len(qpos), len(qpos_ref))
    start, end = _window_bounds(summary, T)

    data_s = mujoco.MjData(model)
    data_r = mujoco.MjData(model)

    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PELVIS_BODY_NAME)
    left_eef_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LEFT_EEF_BODY_NAME)
    right_eef_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_EEF_BODY_NAME)
    if pelvis_id < 0 or left_eef_id < 0 or right_eef_id < 0:
        out["paper_spider_body_tracking_present"] = False
        return

    # Robot body set: all bodies except world (0), object body, and any mocap
    # support anchor (E014/E018b add support_weld_anchor at the end).
    non_robot = set()
    for nm in NON_ROBOT_BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        if bid >= 0:
            non_robot.add(bid)
    robot_body_ids = [b for b in range(1, model.nbody) if b not in non_robot]
    if not robot_body_ids:
        out["paper_spider_body_tracking_present"] = False
        return

    nq = model.nq
    q_sim = qpos[:T, :nq] if qpos.shape[1] >= nq else qpos[:T]
    q_ref = qpos_ref[:T, :nq] if qpos_ref.shape[1] >= nq else qpos_ref[:T]

    mpkpe = np.zeros(T, dtype=np.float64)
    body_ori = np.zeros(T, dtype=np.float64)
    root_pos = np.zeros(T, dtype=np.float64)
    root_ori = np.zeros(T, dtype=np.float64)
    eef_pos = np.zeros(T, dtype=np.float64)
    eef_ori = np.zeros(T, dtype=np.float64)
    joint_err = np.zeros(T, dtype=np.float64)

    for t in range(T):
        data_s.qpos[:] = q_sim[t]
        data_r.qpos[:] = q_ref[t]
        mujoco.mj_kinematics(model, data_s)
        mujoco.mj_kinematics(model, data_r)

        # Body Pos / Ori Err over robot bodies
        diffs = data_s.xpos[robot_body_ids] - data_r.xpos[robot_body_ids]
        mpkpe[t] = float(np.linalg.norm(diffs, axis=1).mean())
        # Orientation: 2*arccos(|q_sim · q_ref|) per body
        qs = data_s.xquat[robot_body_ids]
        qr = data_r.xquat[robot_body_ids]
        dot = np.abs(np.sum(_quat_normalize(qs) * _quat_normalize(qr), axis=1))
        dot = np.clip(dot, -1.0, 1.0)
        body_ori[t] = float((2.0 * np.arccos(dot)).mean())

        # Root (pelvis) Pos / Ori
        root_pos[t] = float(np.linalg.norm(data_s.xpos[pelvis_id] - data_r.xpos[pelvis_id]))
        qs_p = _quat_normalize(data_s.xquat[pelvis_id])
        qr_p = _quat_normalize(data_r.xquat[pelvis_id])
        root_ori[t] = float(2.0 * np.arccos(np.clip(abs(float(qs_p @ qr_p)), -1.0, 1.0)))

        # EEF (L/R rubber hand) Pos / Ori, mean of both
        eef_pos_pair = [
            float(np.linalg.norm(data_s.xpos[i] - data_r.xpos[i]))
            for i in (left_eef_id, right_eef_id)
        ]
        eef_pos[t] = float(np.mean(eef_pos_pair))
        eef_ori_pair = []
        for i in (left_eef_id, right_eef_id):
            qs_e = _quat_normalize(data_s.xquat[i])
            qr_e = _quat_normalize(data_r.xquat[i])
            eef_ori_pair.append(
                float(2.0 * np.arccos(np.clip(abs(float(qs_e @ qr_e)), -1.0, 1.0)))
            )
        eef_ori[t] = float(np.mean(eef_ori_pair))

        # Joint Err: robot joints only qpos[7:36] (29 dof)
        js = q_sim[t, 7:36]
        jr = q_ref[t, 7:36] if q_ref.shape[1] >= 36 else q_ref[t, 7:]
        n = min(len(js), len(jr))
        joint_err[t] = float(np.mean(np.abs(js[:n] - jr[:n])))

    out["paper_spider_body_tracking_present"] = True
    # Case-window means (primary reporting)
    out["paper_spider_joint_err_deg"] = float(np.degrees(joint_err[start:end].mean()))
    out["paper_spider_pos_err_cm"] = float(mpkpe[start:end].mean() * 100.0)
    out["paper_spider_ori_err_deg"] = float(np.degrees(body_ori[start:end].mean()))
    out["paper_spider_root_pos_err_cm"] = float(root_pos[start:end].mean() * 100.0)
    out["paper_spider_root_ori_err_deg"] = float(np.degrees(root_ori[start:end].mean()))
    out["paper_spider_eef_pos_err_cm"] = float(eef_pos[start:end].mean() * 100.0)
    out["paper_spider_eef_ori_err_deg"] = float(np.degrees(eef_ori[start:end].mean()))
    # Re-expose object Pos/Ori under spider naming for table alignment
    out["paper_spider_obj_pos_err_cm"] = float(out.get("paper_object_Epos_case_m", 0.0) * 100.0)
    out["paper_spider_obj_ori_err_deg"] = float(out.get("paper_object_Erot_case_deg", 0.0))
    # Std (case-window) for severity reporting
    out["paper_spider_joint_err_std_deg"] = float(np.degrees(joint_err[start:end].std()))
    out["paper_spider_pos_err_std_cm"] = float(mpkpe[start:end].std() * 100.0)


def _add_penetration_metrics_mj(
    out: dict[str, Any],
    summary: dict[str, Any],
    model: mujoco.MjModel,
    qpos: np.ndarray,
) -> None:
    """OmniRetarget Table II penetration via mj_geomDistance + prefilter.

    Ported from holosoma/workspace/v1/scripts/eval_paper_metrics.py:92-147
    with object-name detection from MJCF (looks up the freejoint body, then
    enumerates its geoms).
    """
    T = min(len(qpos), int(summary.get("T", len(qpos))))
    start, end = _window_bounds(summary, T)
    nq = model.nq
    if qpos.shape[1] < nq:
        out["paper_omniretarget_mj_penetration_present"] = False
        return

    # Identify object body = body that owns the trailing 7-qpos freejoint.
    # SPIDER convention places object body last; verify by checking last freejoint.
    obj_body_id = -1
    for b in range(model.nbody - 1, 0, -1):
        # find a freejoint whose qposadr is at nq-7
        for j in range(model.body_jntnum[b]):
            jid = model.body_jntadr[b] + j
            if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE and model.jnt_qposadr[jid] == nq - 7:
                obj_body_id = b
                break
        if obj_body_id >= 0:
            break
    if obj_body_id < 0:
        out["paper_omniretarget_mj_penetration_present"] = False
        return

    # Object geoms = all geoms belonging to that body
    obj_geom_ids = set()
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == obj_body_id:
            obj_geom_ids.add(int(g))

    geom_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(model.ngeom)
    ]

    def is_obj(g: int) -> bool:
        return g in obj_geom_ids

    def is_ground(g: int) -> bool:
        n = geom_names[g].lower()
        return "ground" in n or "floor" in n or n == "world"

    data = mujoco.MjData(model)
    fromto = np.zeros(6, dtype=float)
    pen_frames = 0
    max_depth_cm_list: list[float] = []

    # Save margins once
    saved_margins = model.geom_margin.copy()
    try:
        for t in range(T):
            data.qpos[:nq] = qpos[t, :nq]
            mujoco.mj_forward(model, data)

            # Broadphase prefilter via expanded margins
            model.geom_margin[:] = PEN_COLLISION_DETECTION_THRESHOLD
            mujoco.mj_collision(model, data)
            candidates: set[tuple[int, int]] = set()
            for k in range(data.ncon):
                c = data.contact[k]
                g1, g2 = int(c.geom1), int(c.geom2)
                if g1 < 0 or g2 < 0:
                    continue
                candidates.add((min(g1, g2), max(g1, g2)))
            model.geom_margin[:] = saved_margins

            # mj_geomDistance per candidate pair
            depths = []
            for g1, g2 in candidates:
                if model.geom_contype[g1] == 0 and model.geom_conaffinity[g1] == 0:
                    continue
                if model.geom_contype[g2] == 0 and model.geom_conaffinity[g2] == 0:
                    continue
                # Exclude object-ground pair (object resting on floor is fine)
                if (is_obj(g1) and is_ground(g2)) or (is_obj(g2) and is_ground(g1)):
                    continue
                # Keep only pairs involving object or ground (robot↔object, robot↔ground)
                if not (is_obj(g1) or is_obj(g2) or is_ground(g1) or is_ground(g2)):
                    continue
                fromto[:] = 0.0
                dist = mujoco.mj_geomDistance(
                    model, data, g1, g2, PEN_COLLISION_DETECTION_THRESHOLD, fromto
                )
                if dist < -PEN_TOLERANCE:
                    depths.append(-float(dist))
            if depths:
                pen_frames += 1
                max_depth_cm_list.append(float(max(depths)) * 100.0)
    finally:
        model.geom_margin[:] = saved_margins

    out["paper_omniretarget_mj_penetration_present"] = True
    out["paper_omniretarget_mj_penetration_duration_pct"] = float(pen_frames / max(T, 1) * 100.0)
    out["paper_omniretarget_mj_penetration_max_depth_cm"] = (
        float(max(max_depth_cm_list)) if max_depth_cm_list else 0.0
    )
    out["paper_omniretarget_mj_penetration_mean_depth_cm"] = (
        float(np.mean(max_depth_cm_list)) if max_depth_cm_list else 0.0
    )
    # Case-window restricted version (start..end frames)
    cw_max = [
        d for i, d in zip(range(T), [0.0] * T)  # placeholder count
    ]
    # Re-compute case-window summary from indices we kept (cheap: re-iterate stored)
    # We didn't store per-frame depths, so recompute case-window duration directly.
    # For simplicity, run an indices loop:
    # (We can't recover case-window stats without re-storing; recompute via single pass below.)
    # To keep memory low, just record full-trajectory stats; case-window stats follow below.

    # Recompute case-window stats with a short second pass on stored max_depth_list.
    # Since we tracked only penetrating-frame depths (not their frame index), we need
    # frame indices too. Add a quick re-run constrained to [start, end).
    cw_pen = 0
    cw_depths: list[float] = []
    model.geom_margin[:] = saved_margins
    for t in range(start, end):
        data.qpos[:nq] = qpos[t, :nq]
        mujoco.mj_forward(model, data)
        model.geom_margin[:] = PEN_COLLISION_DETECTION_THRESHOLD
        mujoco.mj_collision(model, data)
        candidates: set[tuple[int, int]] = set()
        for k in range(data.ncon):
            c = data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 < 0 or g2 < 0:
                continue
            candidates.add((min(g1, g2), max(g1, g2)))
        model.geom_margin[:] = saved_margins
        depths = []
        for g1, g2 in candidates:
            if model.geom_contype[g1] == 0 and model.geom_conaffinity[g1] == 0:
                continue
            if model.geom_contype[g2] == 0 and model.geom_conaffinity[g2] == 0:
                continue
            if (is_obj(g1) and is_ground(g2)) or (is_obj(g2) and is_ground(g1)):
                continue
            if not (is_obj(g1) or is_obj(g2) or is_ground(g1) or is_ground(g2)):
                continue
            fromto[:] = 0.0
            dist = mujoco.mj_geomDistance(
                model, data, g1, g2, PEN_COLLISION_DETECTION_THRESHOLD, fromto
            )
            if dist < -PEN_TOLERANCE:
                depths.append(-float(dist))
        if depths:
            cw_pen += 1
            cw_depths.append(float(max(depths)) * 100.0)
    out["paper_omniretarget_mj_penetration_case_duration_pct"] = float(
        cw_pen / max(end - start, 1) * 100.0
    )
    out["paper_omniretarget_mj_penetration_case_max_depth_cm"] = (
        float(max(cw_depths)) if cw_depths else 0.0
    )


def _add_contact_and_penetration_metrics(
    out: dict[str, Any],
    summary: dict[str, Any],
    repo: Path,
    results_dir: Path,
    variant: str,
    person_idx: int,
    T: int,
) -> None:
    start, end = _window_bounds(summary, T)
    ts_rows = _read_rows(results_dir / f"timeseries_{variant}.csv")
    leg_rows = _read_rows(results_dir / f"legobj_timeseries_{variant}.csv", kind="sim")
    if not ts_rows or not leg_rows:
        out["paper_metrics_contact_rows_present"] = False
        return
    out["paper_metrics_contact_rows_present"] = True
    ts_rows = ts_rows[:T]
    leg_rows = leg_rows[:T]

    leg_sdf = _as_float_array(leg_rows, "leg_box_sdf_min_m")[start:end]
    hand_sdf = _as_float_array(leg_rows, "hand_box_sdf_min_m")[start:end]
    min_robot_sdf = np.minimum(leg_sdf, hand_sdf)
    deep_threshold = -DEEP_PENETRATION_THRESHOLD_M
    out["paper_omniretarget_robot_object_penetration_duration_pct"] = float(
        (min_robot_sdf < 0.0).mean() * 100.0
    )
    out["paper_omniretarget_robot_object_deep_penetration_threshold_cm"] = float(
        DEEP_PENETRATION_THRESHOLD_M * 100.0
    )
    out["paper_omniretarget_robot_object_deep_penetration_duration_pct"] = float(
        (min_robot_sdf < deep_threshold).mean() * 100.0
    )
    out["paper_omniretarget_robot_object_max_penetration_cm"] = float(
        max(0.0, -float(min_robot_sdf.min()) * 100.0)
    )
    out["paper_omniretarget_hand_object_penetration_duration_pct"] = float(
        (hand_sdf < 0.0).mean() * 100.0
    )
    out["paper_omniretarget_hand_object_deep_penetration_duration_pct"] = float(
        (hand_sdf < deep_threshold).mean() * 100.0
    )
    out["paper_omniretarget_leg_object_penetration_duration_pct"] = float(
        (leg_sdf < 0.0).mean() * 100.0
    )
    out["paper_omniretarget_leg_object_deep_penetration_duration_pct"] = float(
        (leg_sdf < deep_threshold).mean() * 100.0
    )
    out["paper_omniretarget_robot_object_deep_penetration_ok"] = bool(
        out["paper_omniretarget_robot_object_deep_penetration_duration_pct"] <= 20.0
        and out["paper_omniretarget_robot_object_max_penetration_cm"] <= 5.0
    )

    mask = _load_contact_mask(summary, repo, T, person_idx)
    sim_left_contact = np.asarray(
        [int(row["sim_left_contact_count"]) > 0 for row in ts_rows], dtype=bool
    )
    sim_right_contact = np.asarray(
        [int(row["sim_right_contact_count"]) > 0 for row in ts_rows], dtype=bool
    )
    sim_left_near = np.asarray(
        [float(row["sim_left_sdf_m"]) < 0.05 for row in ts_rows], dtype=bool
    )
    sim_right_near = np.asarray(
        [float(row["sim_right_sdf_m"]) < 0.05 for row in ts_rows], dtype=bool
    )
    if mask is not None:
        desired = mask[start:end]
        actual_contact = np.column_stack(
            [sim_left_contact[start:end], sim_right_contact[start:end]]
        )
        actual_near = np.column_stack(
            [sim_left_near[start:end], sim_right_near[start:end]]
        )
        denom = int(desired.sum())
        out["paper_contact_desired_side_frames"] = denom
        out["paper_omniretarget_contact_preservation_pct"] = (
            float((actual_contact & desired).sum() / denom * 100.0)
            if denom
            else 0.0
        )
        out["paper_omniretarget_contact_preservation_5cm_pct"] = (
            float((actual_near & desired).sum() / denom * 100.0)
            if denom
            else 0.0
        )
    else:
        desired_any = np.asarray(
            [int(row["ref_total_contact_count"]) > 0 for row in ts_rows],
            dtype=bool,
        )[start:end]
        actual_any = (sim_left_contact | sim_right_contact)[start:end]
        denom = int(desired_any.sum())
        out["paper_contact_desired_side_frames"] = denom
        out["paper_omniretarget_contact_preservation_pct"] = (
            float((actual_any & desired_any).sum() / denom * 100.0)
            if denom
            else 0.0
        )
        out["paper_omniretarget_contact_preservation_5cm_pct"] = out[
            "paper_omniretarget_contact_preservation_pct"
        ]
    out["paper_omniretarget_contact_preservation_ok"] = bool(
        out["paper_omniretarget_contact_preservation_5cm_pct"] >= 70.0
    )


def _add_contact_preservation_omni_local(
    out: dict[str, Any],
    summary: dict[str, Any],
    model: mujoco.MjModel,
    qpos: np.ndarray,
    human_joints: np.ndarray | None,
    fps: float,
) -> None:
    """OmniRetarget Table II strict contact preservation (28cm obj-local).

    Ported from holosoma/workspace/v1/scripts/eval_paper_metrics.py:242-308.
    Detects demo contact when SMPL-X wrist is within ``CONTACT_PRESERVATION_LOCAL_RADIUS_M``
    of the demo object in object-local frame; preservation counts frames where
    the *sim* humanoid wrist (FK from qpos) is also within radius in sim object
    local frame.

    Requires ``human_joints`` ``(T, 22, 3)``; without it, sets
    ``paper_omniretarget_contact_preservation_local_present = False`` and falls
    back to the mask-gated 5cm proxy already in `_add_contact_and_penetration_metrics`.
    """
    if human_joints is None or human_joints.ndim != 3 or human_joints.shape[1] < 22:
        out["paper_omniretarget_contact_preservation_local_present"] = False
        return
    T = min(len(qpos), len(human_joints), int(summary.get("T", len(qpos))))
    start, end = _window_bounds(summary, T)

    # Identify object body (last freejoint, qposadr at nq-7)
    obj_body_id = -1
    for b in range(model.nbody - 1, 0, -1):
        for j in range(model.body_jntnum[b]):
            jid = int(model.body_jntadr[b] + j)
            if (
                int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE)
                and int(model.jnt_qposadr[jid]) == model.nq - 7
            ):
                obj_body_id = b
                break
        if obj_body_id >= 0:
            break
    if obj_body_id < 0:
        out["paper_omniretarget_contact_preservation_local_present"] = False
        return

    # Find robot wrist body ids (G1 unitree: left/right_wrist_yaw_link)
    l_wrist_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    r_wrist_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    if l_wrist_bid < 0 or r_wrist_bid < 0:
        out["paper_omniretarget_contact_preservation_local_present"] = False
        return

    # FK loop to get sim wrist + sim object pose
    data = mujoco.MjData(model)
    sim_obj_pos = np.zeros((T, 3))
    sim_obj_mat = np.zeros((T, 3, 3))
    sim_l_wrist = np.zeros((T, 3))
    sim_r_wrist = np.zeros((T, 3))
    for t in range(T):
        data.qpos[:] = qpos[t, : model.nq]
        mujoco.mj_kinematics(model, data)
        sim_obj_pos[t] = data.xpos[obj_body_id]
        sim_obj_mat[t] = data.xmat[obj_body_id].reshape(3, 3)
        sim_l_wrist[t] = data.xpos[l_wrist_bid]
        sim_r_wrist[t] = data.xpos[r_wrist_bid]

    # Demo wrist in world (Z-up SMPL-X)
    demo_l_wrist = human_joints[:T, SMPLX_L_WRIST_IDX, :].astype(np.float64)
    demo_r_wrist = human_joints[:T, SMPLX_R_WRIST_IDX, :].astype(np.float64)
    # Demo object pose: use the SIM object pose from qpos (kinematic retarget
    # produced both demo human + retarget object — we assume the qpos object is
    # already aligned with demo at this frame; this is true for holosoma kin).
    demo_obj_pos = qpos[:T, -7:-4].astype(np.float64)
    demo_obj_quat = qpos[:T, -4:].astype(np.float64)  # wxyz
    demo_obj_mat = _quat_to_matrix_batch(demo_obj_quat)

    # Transform demo wrist into demo object local frame
    demo_l_local = np.einsum("tij,tj->ti", demo_obj_mat.transpose(0, 2, 1), demo_l_wrist - demo_obj_pos)
    demo_r_local = np.einsum("tij,tj->ti", demo_obj_mat.transpose(0, 2, 1), demo_r_wrist - demo_obj_pos)
    sim_l_local = np.einsum("tij,tj->ti", sim_obj_mat.transpose(0, 2, 1), sim_l_wrist - sim_obj_pos)
    sim_r_local = np.einsum("tij,tj->ti", sim_obj_mat.transpose(0, 2, 1), sim_r_wrist - sim_obj_pos)

    radius = CONTACT_PRESERVATION_LOCAL_RADIUS_M
    demo_l_contact = np.linalg.norm(demo_l_local, axis=1) < radius
    demo_r_contact = np.linalg.norm(demo_r_local, axis=1) < radius
    sim_l_contact = np.linalg.norm(sim_l_local, axis=1) < radius
    sim_r_contact = np.linalg.norm(sim_r_local, axis=1) < radius

    out["paper_omniretarget_contact_preservation_local_present"] = True
    out["paper_omniretarget_contact_preservation_local_radius_m"] = float(radius)
    # OmniRetarget definition (eval_paper_metrics.py:298-303):
    #   miss_t = any(demo_contact & ~sim_contact)  per-frame
    #   preservation = 1 - miss_frames / T   (T = all frames, not just demo)
    # When demo has 0 contact frames, miss stays 0 → preservation = 1.0
    # (trivially perfect). This is the paper's convention.
    for tag, sl in (("full", slice(0, T)), ("case", slice(start, end))):
        seg_T = max(end - start, 1) if tag == "case" else T
        miss_l = demo_l_contact[sl] & (~sim_l_contact[sl])
        miss_r = demo_r_contact[sl] & (~sim_r_contact[sl])
        miss_frames = int((miss_l | miss_r).sum())
        demo_frames = int((demo_l_contact[sl] | demo_r_contact[sl]).sum())
        pct = float((1.0 - miss_frames / seg_T) * 100.0)
        out[f"paper_omniretarget_contact_preservation_local_{tag}_pct"] = pct
        out[f"paper_omniretarget_contact_preservation_local_{tag}_demo_frames"] = demo_frames
        out[f"paper_omniretarget_contact_preservation_local_{tag}_miss_frames"] = miss_frames
    out["paper_omniretarget_contact_preservation_local_ok"] = bool(
        out.get("paper_omniretarget_contact_preservation_local_case_pct", 0.0) >= 70.0
    )


def _quat_to_matrix_batch(quat_wxyz: np.ndarray) -> np.ndarray:
    """Batched wxyz -> rotation matrix (T, 3, 3)."""
    q = quat_wxyz / np.clip(np.linalg.norm(quat_wxyz, axis=1, keepdims=True), 1e-8, None)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    mat = np.zeros((len(q), 3, 3), dtype=np.float64)
    mat[:, 0, 0] = 1 - 2 * (y * y + z * z)
    mat[:, 0, 1] = 2 * (x * y - z * w)
    mat[:, 0, 2] = 2 * (x * z + y * w)
    mat[:, 1, 0] = 2 * (x * y + z * w)
    mat[:, 1, 1] = 1 - 2 * (x * x + z * z)
    mat[:, 1, 2] = 2 * (y * z - x * w)
    mat[:, 2, 0] = 2 * (x * z - y * w)
    mat[:, 2, 1] = 2 * (y * z + x * w)
    mat[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return mat


def add_paper_metrics_physics(
    *,
    model: mujoco.MjModel,
    qpos: np.ndarray,
    fps: float,
    human_joints: np.ndarray | None = None,
    case: str = "unknown",
    case_window: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Physics-only paper metrics for sources without a sim/ref split.

    Used by ``eval_holosoma_kinematic.py`` (E019 P1). Computes:
      * OmniRetarget mj_geomDistance penetration (no FPS dep)
      * OmniRetarget 28cm obj-local contact preservation (needs human_joints)
      * Smoothness (uses ``fps`` for q̈)
      * Object Pos / Ori relative to itself (degenerate but present for schema)

    For physical (sim/ref) data, use ``add_paper_metrics`` instead — it covers
    everything here plus SPIDER Table 4 alignment.
    """
    T = int(qpos.shape[0])
    if case_window is None:
        case_window = (0, T)
    start, end = case_window
    summary = {
        "variant": case,
        "case": case,
        "T": T,
        "case_window_start_frame": max(0, int(start)),
        "case_window_end_frame": max(0, min(int(end) - 1, T - 1)),
        "fps": fps,
    }
    out: dict[str, Any] = {
        "paper_metrics_version": "2026-05-20-P1",
        "paper_metrics_sources": "SPIDER,DynaRetarget,OmniRetarget,holosoma_v2",
        "paper_metrics_mode": "physics_only",
        "fps": fps,
        "case_window_start_frame": summary["case_window_start_frame"],
        "case_window_end_frame": summary["case_window_end_frame"],
        "T": T,
        "case": case,
    }

    # Smoothness — use per-case fps instead of module FPS
    if T > 2:
        robot_q = qpos[:T, 7:36].astype(np.float64)
        qdd = (robot_q[2:] - 2.0 * robot_q[1:-1] + robot_q[:-2]) * (fps * fps)
        out["paper_dynaretarget_smoothness"] = float(np.abs(qdd).sum())
    else:
        out["paper_dynaretarget_smoothness"] = 0.0
    out["paper_dynaretarget_ref_smoothness"] = out["paper_dynaretarget_smoothness"]
    out["paper_dynaretarget_relative_smoothness"] = 1.0

    # Object self-Pos/Ori (degenerate for kin self-eval)
    obj_pos = qpos[:T, -7:-4].astype(np.float64)
    obj_quat = qpos[:T, -4:].astype(np.float64)
    out["paper_object_Epos_full_m"] = 0.0
    out["paper_object_Epos_case_m"] = 0.0
    out["paper_object_Epos_case_max_m"] = 0.0
    out["paper_object_Erot_case_rad"] = 0.0
    out["paper_object_Erot_case_deg"] = 0.0
    out["paper_object_Erot_case_max_deg"] = 0.0
    out["paper_spider_obj_pos_err_cm"] = 0.0
    out["paper_spider_obj_ori_err_deg"] = 0.0

    # mj_geomDistance penetration (full + case-window)
    try:
        _add_penetration_metrics_mj(out, summary, model, qpos)
    except Exception as exc:  # pragma: no cover
        out["paper_omniretarget_mj_penetration_present"] = False
        out["paper_omniretarget_mj_penetration_error"] = str(exc)

    # 28cm local-frame contact preservation
    try:
        _add_contact_preservation_omni_local(
            out, summary, model, qpos, human_joints, fps
        )
    except Exception as exc:  # pragma: no cover
        out["paper_omniretarget_contact_preservation_local_present"] = False
        out["paper_omniretarget_contact_preservation_local_error"] = str(exc)

    return out


def add_paper_metrics(
    summary: dict[str, Any],
    *,
    repo: Path,
    results_dir: Path,
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qpos_ref: np.ndarray,
    person_idx: int,
) -> dict[str, Any]:
    """Return paper-aligned metrics for one evaluated rollout."""

    T = min(len(qpos), len(qpos_ref), int(summary["T"]))
    qpos = qpos[:T]
    qpos_ref = qpos_ref[:T]
    # Guardrail: if caller declared a per-case fps that disagrees with the
    # module FPS, smoothness (FPS²) and foot skating (FPS) numbers in this
    # invocation are wrong. Warn loudly instead of silently degrading.
    declared_fps = summary.get("fps")
    if declared_fps is not None and abs(float(declared_fps) - FPS) > 1e-3:
        warnings.warn(
            f"paper_metrics: summary['fps']={declared_fps} but module FPS={FPS}; "
            f"smoothness/foot_skating will be computed at {FPS}Hz. "
            f"Either set summary['fps'] to {FPS} or extend per-case plumbing.",
            stacklevel=2,
        )
    out: dict[str, Any] = {
        "paper_metrics_version": "2026-05-20-P2",
        "paper_metrics_sources": "SPIDER,DynaRetarget,OmniRetarget,holosoma_v2",
        "paper_metrics_fps": FPS,
    }
    _add_object_tracking_metrics(out, summary, qpos, qpos_ref)
    _add_smoothness_metrics(out, qpos, qpos_ref)
    _add_keypoint_proxy_metrics(out, summary, model, qpos, qpos_ref)
    _add_contact_and_penetration_metrics(
        out,
        summary,
        repo,
        results_dir,
        str(summary["variant"]),
        person_idx,
        T,
    )
    # E019 P0 additions: SPIDER Table 4 strict alignment + OmniRetarget Table II
    # penetration via mj_geomDistance. Both are independent FK passes; wrapped in
    # try/except so a failure on one variant does not break aggregate.
    try:
        _add_body_tracking_metrics(out, summary, model, qpos, qpos_ref)
    except Exception as exc:  # pragma: no cover - defensive
        out["paper_spider_body_tracking_present"] = False
        out["paper_spider_body_tracking_error"] = str(exc)
    try:
        _add_penetration_metrics_mj(out, summary, model, qpos)
    except Exception as exc:  # pragma: no cover - defensive
        out["paper_omniretarget_mj_penetration_present"] = False
        out["paper_omniretarget_mj_penetration_error"] = str(exc)
    # E019 P1: OmniRetarget 28cm obj-local contact preservation.
    # ``human_joints`` is opt-in via ``summary["human_joints"]`` (np.ndarray)
    # or ``summary["human_joints_npz"]`` (path). When absent the helper marks
    # the field absent and falls back silently.
    try:
        human_joints = summary.get("human_joints")
        if human_joints is None and summary.get("human_joints_npz"):
            arr = np.load(str(summary["human_joints_npz"]), allow_pickle=True)
            key = summary.get("human_joints_key", "human_joints")
            if key in arr.files:
                human_joints = np.asarray(arr[key], dtype=np.float64)
        fps = float(summary.get("fps", FPS))
        _add_contact_preservation_omni_local(
            out, summary, model, qpos, human_joints, fps
        )
    except Exception as exc:  # pragma: no cover - defensive
        out["paper_omniretarget_contact_preservation_local_present"] = False
        out["paper_omniretarget_contact_preservation_local_error"] = str(exc)
    return out
