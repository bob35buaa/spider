"""Shared evaluation metrics for CORE4D retargeting experiments.

Extracted from eval_E147_rubber_hand_collision.py to decouple downstream
evaluators (E148/E150/E151/E152) from that specific experiment script.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


# ---------------------------------------------------------------------------
# Configuration dataclass — defaults match E147 original global constants
# ---------------------------------------------------------------------------


@dataclass
class EvalConfig:
    """Evaluation parameters. Defaults reproduce E147 behaviour exactly."""

    fps: float = 30.0
    fall_pelvis_z_m: float = 0.45
    eef_offset: np.ndarray = field(default_factory=lambda: np.asarray([0.05, 0.0, 0.0], dtype=np.float64))
    near_thresholds_m: tuple[float, ...] = (0.03, 0.05, 0.08, 0.10)
    deep_penetration_m: float = -0.02
    clean_contact_penetration_m: float = -0.002
    deep_contact_dist_m: float = -0.005
    mesh_sample_count: int = 800
    # E154: body-tracking vs fixed kinematic truth + masked-contact (real 3cm).
    track_terminal_frac: float = 0.15  # terminal phase = last 15% of frames
    track_pelvis_terminal_th_m: float = 0.08  # success gate: terminal pelvis-z err
    release_false_contact_th: float = 0.30  # success gate: max false-contact frac
    # E191: object-support diagnostics (observation-only, no gate uses these).
    # The CORE4D scene_act object is driven by 6 P-only position actuators whose
    # gains are injected at runtime by examples/run_mjwp.py from the Hydra config;
    # they are NOT recoverable from the scene XML (which ships kp="0"). Defaults
    # below mirror the E163->E167A->E172/E173/E189 resolved chain.
    object_pos_actuator_gain: float = 500.0  # N/m, init_pos_actuator_gain
    object_rot_actuator_gain: float = 50.0  # Nm/rad, init_rot_actuator_gain
    object_lift_threshold_m: float = 0.05  # ref object z above its own min => "lifted"
    object_lift_min_frames: int = 5  # below this, fall back to all frames
    hand_gate_hard_floor_m: float = -0.020  # cem_hand_gate_hard_floor_m
    hand_gate_floor_tol_m: float = 0.0005  # counted as saturated within this band


# ---------------------------------------------------------------------------
# Body / geom name lists (shared across all evaluators)
# ---------------------------------------------------------------------------

HAND_GEOMS = ["lh", "rh"]

LOWERBODY_GEOMS = [
    "left_hip_collision",
    "right_hip_collision",
    "left_thigh_collision",
    "right_thigh_collision",
    "left_shin_collision",
    "right_shin_collision",
    "left_linkage_brace_collision",
    "right_linkage_brace_collision",
    "lf0",
    "lf1",
    "lf2",
    "lf3",
    "rf0",
    "rf1",
    "rf2",
    "rf3",
]

UPPER_BODY_NAMES = [
    "head_link",
    "torso_link",
    "pelvis_contour_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_elbow_link",
    "right_elbow_link",
]

# The canonical list of metric fields that evaluate_sequence() outputs.
METRIC_FIELDS = [
    "case_id",
    "variant",
    "method",
    "hand_collision_variant_id",
    "object_key",
    "object_category",
    "expected_quality",
    "qpos_path",
    "scene_xml",
    "qpos_frames",
    "duration_s",
    "pelvis_min_m",
    "pelvis_end_m",
    "fall_flag",
    "root_xy_displacement_m",
    "object_xy_displacement_m",
    "object_z_range_m",
    "object_floor_contact_frac",
    "eef_near_3cm_frac",
    "eef_near_5cm_frac",
    "eef_near_8cm_frac",
    "eef_near_10cm_frac",
    "hand_geom_near_3cm_frac",
    "hand_geom_near_5cm_frac",
    "hand_geom_near_8cm_frac",
    "hand_geom_near_10cm_frac",
    "hand_geom_penetration_frac",
    "hand_geom_penetration_2mm_frac",
    "hand_geom_penetration_5mm_frac",
    "hand_geom_deep_penetration_2cm_frac",
    "hand_object_physics_contact_frac",
    "hand_object_clean_physics_contact_frac",
    "hand_object_physics_contact_3mm_frac",
    "hand_object_physics_contact_5mm_frac",
    "hand_object_physics_penetration_3mm_frame_frac",
    "hand_object_physics_penetration_5mm_frame_frac",
    "hand_object_con_dist_mean_m",
    "hand_object_con_dist_min_m",
    "hand_object_con_dist_frac_lt_neg2mm",
    "hand_object_con_dist_frac_lt_neg3mm",
    "hand_object_con_dist_frac_lt_neg5mm",
    "hand_object_con_deep2mm_frame_frac",
    "hand_object_con_deep3mm_frame_frac",
    "hand_object_con_deep5mm_frame_frac",
    "hand_floor_min_z_m",
    "hand_floor_near_2cm_frac",
    "hand_floor_penetration_frac",
    "hand_floor_physics_contact_frac",
    "hand_floor_con_dist_mean_m",
    "hand_floor_con_dist_min_m",
    "hand_floor_con_dist_frac_lt_neg5mm",
    "hand_floor_con_deep5mm_frame_frac",
    "leg_near_2cm_frac",
    "leg_penetration_frac",
    "leg_object_physics_contact_frac",
    "body_penetration_frac",
    "head_penetration_frac",
    "upper_body_penetration_frac",
    "obj_err_mean_m",
    "obj_err_max_m",
    "notes",
]

# E154: body-tracking (vs fixed kinematic truth) + masked-contact (real 3cm).
# All NaN/empty unless evaluate_sequence is called with kin_ref_path / contact_mask_path.
TRACK_MASK_FIELDS = [
    "track_root_pos_err_mean_m",
    "track_root_pos_err_terminal_m",
    "track_root_quat_err_mean",
    "track_root_quat_err_terminal",
    "track_joint_err_mean_rad",
    "track_joint_err_terminal_rad",
    "track_pelvis_z_err_mean_m",
    "track_pelvis_z_err_terminal_m",
    "track_joint_err_deg_mean",
    "track_eef_pos_err_cm_mean",
    "track_eef_ori_err_deg_mean",
    "track_root_pos_err_cm_mean",
    "track_root_ori_err_deg_mean",
    "track_obj_pos_err_cm_mean",
    "track_obj_z_abs_err_cm_mean",
    "track_obj_ori_err_deg_mean",
    "ref_contact_frac",
    "hand_object_physics_contact_in_mask_frac",
    "hand_object_clean_physics_contact_in_mask_frac",
    "hand_object_physics_contact_3mm_in_mask_frac",
    "hand_object_physics_contact_5mm_in_mask_frac",
    "rl_object_contact_ref_frac",
    "rl_object_contact_gap_fill_frames",
    "rl_object_contact_filled_frame_count",
    "hand_object_physics_contact_in_rl_mask_frac",
    "hand_object_physics_contact_3mm_in_rl_mask_frac",
    "hand_object_physics_contact_5mm_in_rl_mask_frac",
    "hand_object_false_contact_frac",
    "hand_object_clean_false_contact_frac",
    "hand_object_false_contact_3mm_frac",
    "hand_object_false_contact_5mm_frac",
    "hand_object_approach_false_contact_frac",
    "hand_object_clean_approach_false_contact_frac",
    "hand_object_approach_false_contact_3mm_frac",
    "hand_object_approach_false_contact_5mm_frac",
    "hand_object_release_false_contact_frac",
    "hand_object_clean_release_false_contact_frac",
    "hand_object_release_false_contact_3mm_frac",
    "hand_object_release_false_contact_5mm_frac",
    "hand_geom_penetration_2mm_in_mask_frac",
    "hand_geom_penetration_5mm_in_mask_frac",
]
METRIC_FIELDS += TRACK_MASK_FIELDS

# E191: object-support diagnostics. Purely additive observation columns — no
# 12-gate rule reads them. They exist to separate three mechanisms that are
# perfectly collinear with object size in the existing box experiments:
#   (a) fixed-metre reward/gate geometry, (b) the soft object position servo
#   with no partner model, (c) grasp topology on large flat faces.
E191_SUPPORT_FIELDS = [
    # signed decomposition of track_obj_pos_err_cm_mean (which is an L2 norm)
    "track_obj_z_err_m_mean",
    "track_obj_z_err_m_p10",
    "track_obj_xy_err_cm_mean",
    # same decomposition restricted to the carry phase, where H1 is judged
    "track_obj_z_err_m_lifted_mean",
    "track_obj_xy_err_cm_lifted_mean",
    "track_obj_z_err_share_lifted",
    # side-resolved object height, probe aligned with SUGAR-side R010-6
    "obj_lifted_frame_frac",
    "obj_side_near_z_err_m",
    "obj_side_far_z_err_m",
    "obj_side_z_asym_cm",
    # implied restoring wrench of the object guidance servo
    "object_guidance_force_N_p95",
    "object_guidance_force_z_N_p95",
    "object_guidance_torque_Nm_p95",
    "object_weight_N",
    # how much hand penetration is inherited from the kinematic reference
    "ref_hand_geom_penetration_frac",
    "ref_hand_geom_penetration_3mm_frac",
    "ref_hand_geom_min_sdf_m",
    # is the CEM hand gate hard floor the binding constraint?
    "hand_gate_floor_saturation_frac",
    # fixed absolute-depth diagnostics shared by A0/A2 (E192); unlike the
    # arm-relative floor saturation above these never move with config.
    "hand_gate_fixed_depth_8mm_frame_frac",
    "hand_gate_fixed_depth_10mm_frame_frac",
    "hand_gate_fixed_depth_12mm_frame_frac",
    "hand_gate_fixed_depth_15mm_frame_frac",
    "hand_gate_fixed_depth_20mm_frame_frac",
    # object geometry / grasp lever arm (regression covariates)
    "object_mass_kg",
    "object_half_extents_m",
    "object_max_half_extent_m",
    "grip_near_arm_m",
    "grip_far_arm_m",
]
METRIC_FIELDS += E191_SUPPORT_FIELDS

# ---------------------------------------------------------------------------
# Canonical metric standard for E154+ experiments
# ---------------------------------------------------------------------------

EVAL_METRIC_STANDARD_ID = "core4d-e154-physics-contact-v1"
PHYSICS_CONTACT_THRESHOLDS_M = (0.003, 0.005)
DOWNSTREAM_RL_CONTACT_GAP_FILL_FRAMES = 5

# Minimal core metrics suitable as a shared baseline set. Kept for older scripts.
CORE_METRICS = [
    "pelvis_min_m",
    "fall_flag",
    "hand_geom_near_5cm_frac",
    "hand_geom_near_10cm_frac",
    "hand_geom_penetration_frac",
    "hand_geom_deep_penetration_2cm_frac",
    "hand_object_physics_contact_frac",
    "leg_penetration_frac",
    "body_penetration_frac",
    "object_floor_contact_frac",
    "obj_err_mean_m",
]

# Standard method summary fields for new CEM/retargeting experiments. Experiment
# scripts may append run-specific health fields, but should not redefine these.
STANDARD_SUMMARY_METRICS = [
    "hand_geom_near_5cm_frac",
    "hand_geom_near_10cm_frac",
    "hand_geom_penetration_frac",
    "hand_geom_penetration_2mm_frac",
    "hand_geom_penetration_5mm_frac",
    "hand_geom_deep_penetration_2cm_frac",
    "hand_object_physics_contact_frac",
    "hand_object_clean_physics_contact_frac",
    "hand_object_physics_contact_3mm_frac",
    "hand_object_physics_contact_5mm_frac",
    "hand_object_physics_penetration_3mm_frame_frac",
    "hand_object_physics_penetration_5mm_frame_frac",
    "hand_object_con_dist_mean_m",
    "hand_object_con_dist_min_m",
    "hand_object_con_dist_frac_lt_neg2mm",
    "hand_object_con_dist_frac_lt_neg3mm",
    "hand_object_con_dist_frac_lt_neg5mm",
    "hand_object_con_deep2mm_frame_frac",
    "hand_object_con_deep3mm_frame_frac",
    "hand_object_con_deep5mm_frame_frac",
    "hand_floor_near_2cm_frac",
    "hand_floor_penetration_frac",
    "hand_floor_min_z_m",
    "hand_floor_physics_contact_frac",
    "hand_floor_con_dist_min_m",
    "hand_floor_con_dist_frac_lt_neg5mm",
    "hand_floor_con_deep5mm_frame_frac",
    "leg_penetration_frac",
    "object_floor_contact_frac",
    "pelvis_min_m",
    "obj_err_mean_m",
]

STANDARD_DELTA_METRICS = [
    "hand_geom_near_5cm_frac",
    "hand_geom_near_10cm_frac",
    "hand_geom_penetration_frac",
    "hand_geom_penetration_2mm_frac",
    "hand_geom_penetration_5mm_frac",
    "hand_object_physics_contact_frac",
    "hand_object_clean_physics_contact_frac",
    "hand_object_physics_contact_3mm_frac",
    "hand_object_physics_contact_5mm_frac",
    "hand_object_physics_penetration_3mm_frame_frac",
    "hand_object_physics_penetration_5mm_frame_frac",
    "hand_object_con_dist_frac_lt_neg2mm",
    "hand_object_con_dist_frac_lt_neg3mm",
    "hand_object_con_dist_frac_lt_neg5mm",
    "hand_object_con_deep2mm_frame_frac",
    "hand_object_con_deep3mm_frame_frac",
    "hand_object_con_deep5mm_frame_frac",
    "hand_floor_near_2cm_frac",
    "hand_floor_penetration_frac",
    "hand_floor_physics_contact_frac",
    "hand_floor_con_deep5mm_frame_frac",
    "leg_penetration_frac",
    "obj_err_mean_m",
]

STANDARD_TRACK_DIAG = [
    "track_pelvis_z_err_terminal_m",
    "track_pelvis_z_err_mean_m",
    "track_root_pos_err_terminal_m",
    "track_joint_err_terminal_rad",
    "track_joint_err_deg_mean",
    "track_eef_pos_err_cm_mean",
    "track_eef_ori_err_deg_mean",
    "track_root_pos_err_cm_mean",
    "track_root_ori_err_deg_mean",
    "track_obj_pos_err_cm_mean",
    "track_obj_ori_err_deg_mean",
    "ref_contact_frac",
    "hand_object_physics_contact_in_mask_frac",
    "hand_object_clean_physics_contact_in_mask_frac",
    "hand_object_physics_contact_3mm_in_mask_frac",
    "hand_object_physics_contact_5mm_in_mask_frac",
    "rl_object_contact_ref_frac",
    "rl_object_contact_gap_fill_frames",
    "rl_object_contact_filled_frame_count",
    "hand_object_physics_contact_in_rl_mask_frac",
    "hand_object_physics_contact_3mm_in_rl_mask_frac",
    "hand_object_physics_contact_5mm_in_rl_mask_frac",
    "hand_object_false_contact_frac",
    "hand_object_clean_false_contact_frac",
    "hand_object_false_contact_3mm_frac",
    "hand_object_false_contact_5mm_frac",
    "hand_object_release_false_contact_frac",
    "hand_object_clean_release_false_contact_frac",
    "hand_object_release_false_contact_3mm_frac",
    "hand_object_release_false_contact_5mm_frac",
    "hand_geom_penetration_2mm_in_mask_frac",
    "hand_geom_penetration_5mm_in_mask_frac",
]

STANDARD_MASK_DELTA_METRICS = [
    "hand_object_physics_contact_3mm_in_mask_frac",
    "hand_object_physics_contact_5mm_in_mask_frac",
    "hand_geom_penetration_2mm_in_mask_frac",
]

STANDARD_TABLE_METRIC_FIELDS = {
    "5cm": "hand_geom_near_5cm_frac",
    "2mm_pen": "hand_geom_penetration_2mm_frac",
    "5mm_pen": "hand_geom_penetration_5mm_frac",
    "phys_contact3": "hand_object_physics_contact_3mm_frac",
    "phys_pen3": "hand_object_physics_penetration_3mm_frame_frac",
    "phys_contact5": "hand_object_physics_contact_5mm_frac",
    "phys_pen5": "hand_object_physics_penetration_5mm_frame_frac",
    "leg_pen": "leg_penetration_frac",
    "obj_err": "obj_err_mean_m",
}
STANDARD_TABLE_METRIC_ORDER = [
    "5cm",
    "2mm_pen",
    "5mm_pen",
    "phys_contact3",
    "phys_pen3",
    "phys_contact5",
    "phys_pen5",
    "leg_pen",
]
STANDARD_TABLE_METRIC_DIRECTIONS = [+1, -1, -1, +1, -1, +1, -1, -1]

STANDARD_LOWER_IS_WORST_METRICS = {
    "pelvis_min_m",
    "hand_floor_min_z_m",
    "hand_object_con_dist_mean_m",
    "hand_object_con_dist_min_m",
    "hand_floor_con_dist_mean_m",
    "hand_floor_con_dist_min_m",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def npz_qpos(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load qpos from NPZ. Returns (robot_qpos, optional_ref_qpos)."""
    data = np.load(path, allow_pickle=True)
    arr = np.asarray(data["qpos"], dtype=np.float64)
    if arr.ndim == 3 and arr.shape[1] >= 2:
        return arr[:, 0, :], arr[:, 1, :]
    if arr.ndim == 2:
        return arr, None
    raise ValueError(f"unsupported qpos shape {arr.shape}: {path}")


def mj_id(model: mujoco.MjModel, obj_type: int, name: str) -> int:
    return int(mujoco.mj_name2id(model, obj_type, name))


def mj_name(model: mujoco.MjModel, obj_type: int, idx: int) -> str:
    return mujoco.mj_id2name(model, obj_type, idx) or f"{obj_type}:{idx}"


def object_collision_geoms(model: mujoco.MjModel) -> list[int]:
    gids = []
    for gid in range(model.ngeom):
        name = mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name.startswith("object_collision"):
            gids.append(gid)
    if not gids:
        raise ValueError("no object_collision* geoms in scene")
    return gids


def signed_point_box(point: np.ndarray, box_pos: np.ndarray, box_mat: np.ndarray, half: np.ndarray) -> float:
    local = box_mat.T @ (point - box_pos)
    q = np.abs(local) - half
    outside = np.linalg.norm(np.maximum(q, 0.0))
    inside = min(max(q[0], q[1], q[2]), 0.0)
    return float(outside + inside)


def point_object_sdf(model: mujoco.MjModel, data: mujoco.MjData, point: np.ndarray, object_gids: list[int]) -> float:
    vals = []
    for gid in object_gids:
        vals.append(
            signed_point_box(
                point,
                data.geom_xpos[gid].copy(),
                data.geom_xmat[gid].reshape(3, 3).copy(),
                model.geom_size[gid, :3].copy(),
            )
        )
    return float(min(vals))


def geom_sample_points(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int, mesh_sample_count: int = 800) -> tuple[np.ndarray, float]:
    gtype = int(model.geom_type[geom_id])
    pos = data.geom_xpos[geom_id].copy()
    mat = data.geom_xmat[geom_id].reshape(3, 3).copy()
    if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return pos[None, :], float(model.geom_size[geom_id, 0])
    if gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        half_len = float(model.geom_size[geom_id, 1])
        axis = mat[:, 2]
        return np.asarray([pos + axis * s for s in np.linspace(-half_len, half_len, 13)]), float(model.geom_size[geom_id, 0])
    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        half = model.geom_size[geom_id, :3]
        corners = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    corners.append(pos + mat @ np.array([sx * half[0], sy * half[1], sz * half[2]]))
        return np.asarray(corners), 0.0
    if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        v0 = int(model.mesh_vertadr[mesh_id])
        nv = int(model.mesh_vertnum[mesh_id])
        verts = model.mesh_vert[v0 : v0 + nv].reshape(-1, 3)
        if nv > mesh_sample_count:
            idx = np.linspace(0, nv - 1, mesh_sample_count).astype(int)
            verts = verts[idx]
        return (mat @ verts.T).T + pos, 0.0
    return pos[None, :], 0.0


def geom_object_sdf(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int, object_gids: list[int], mesh_sample_count: int = 800) -> float:
    points, radius = geom_sample_points(model, data, geom_id, mesh_sample_count)
    best = math.inf
    for obj_gid in object_gids:
        obj_pos = data.geom_xpos[obj_gid].copy()
        obj_mat = data.geom_xmat[obj_gid].reshape(3, 3).copy()
        half = model.geom_size[obj_gid, :3].copy()
        vals = [signed_point_box(p, obj_pos, obj_mat, half) for p in points]
        best = min(best, min(vals) - radius)
    return float(best)


def quat_rotate(q_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    out = np.zeros(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(out, vec, q_wxyz)
    return out


def frac(mask: np.ndarray) -> float:
    if mask.size == 0:
        return math.nan
    return float(np.nanmean(mask))


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return math.nan
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else math.nan


def min_or_nan(values: list[float]) -> float:
    if not values:
        return math.nan
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.min(arr)) if arr.size else math.nan


def contact_frac_lt(values: list[float], threshold: float) -> float:
    if not values:
        return math.nan
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return frac(arr < threshold) if arr.size else math.nan


# ---------------------------------------------------------------------------
# Path utilities (used across evaluators)
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[5]


def rel(path: Path | str) -> str:
    """Return path relative to REPO root, for display/logging."""
    if not path:
        return ""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO.resolve()))
    except Exception:
        return str(path)


# ---------------------------------------------------------------------------
# E154 helpers: fixed kinematic-truth reference + real 3cm contact mask
# ---------------------------------------------------------------------------

CONTACT_MASK_ROOT = REPO / "workspace/core4d/results/E143/contact_masks"


def kin_ref_for_scene(scene_xml: Path | str) -> Path:
    """Fixed kinematic-truth trajectory for a case, from its scene path.

    Scenes live at .../humanoid_object/<case_dir>/scene_act_*.xml; the
    retargeted G1 kinematic reference is .../<case_dir>/0/trajectory_kinematic.npz.
    Pass the *original* scene path (not a /tmp snapshot copy).
    """
    return Path(scene_xml).resolve().parent / "0" / "trajectory_kinematic.npz"


def person_idx_from_case(case_id: str) -> int:
    """Map a case id to the retargeted person index (person1->0, person2->1)."""
    s = str(case_id).lower()
    if "person1" in s or "_p1" in s:
        return 0
    return 1  # person2 / _p2 (the default for these collab cases)


def contact_mask_for_case(case_key: str, root: Path | str | None = None) -> Path:
    """Real 3cm contact-mask npz for a case (key = short case id / mask dir name)."""
    base = Path(root) if root is not None else CONTACT_MASK_ROOT
    return base / str(case_key) / "raw_contact_mask_3cm.npz"


def _quat_geodesic_err(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Per-frame orientation error. Reuses spider.math.quat_sub when available
    (matches get_humanoid_tracking_err.py), else falls back to geodesic angle."""
    try:
        import torch

        from spider.math import quat_sub

        d = quat_sub(
            torch.from_numpy(np.ascontiguousarray(q1, dtype=np.float64)),
            torch.from_numpy(np.ascontiguousarray(q2, dtype=np.float64)),
        )
        return np.linalg.norm(d.numpy(), axis=1)
    except Exception:
        dot = np.clip(np.abs(np.sum(q1 * q2, axis=1)), -1.0, 1.0)
        return 2.0 * np.arccos(dot)


def _quat_angle_deg(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> float:
    q1 = np.asarray(q1_wxyz, dtype=np.float64)
    q2 = np.asarray(q2_wxyz, dtype=np.float64)
    n1 = np.linalg.norm(q1)
    n2 = np.linalg.norm(q2)
    if n1 <= 0.0 or n2 <= 0.0:
        return math.nan
    dot = float(np.clip(abs(np.dot(q1 / n1, q2 / n2)), -1.0, 1.0))
    return float(np.degrees(2.0 * math.acos(dot)))


def _table4_tracking_metrics(robot_qpos: np.ndarray, kin_qpos: np.ndarray, model: mujoco.MjModel) -> dict[str, float]:
    """SPIDER Table-4-style tracking metrics against fixed kin truth.

    The run scene uses a 6-DoF object representation (42 qpos in the current
    CORE4D scenes), while the OmniRetarget/SPIDER input reference commonly uses
    a 7-DoF freejoint object representation (43 qpos). For robot-body FK we
    only need the shared robot qpos prefix. Object tracking is reported when the
    reference object pose can be read either in scene qpos layout or as a world
    freejoint pose after the robot prefix.
    """
    keys = [
        "track_joint_err_deg_mean",
        "track_eef_pos_err_cm_mean",
        "track_eef_ori_err_deg_mean",
        "track_root_pos_err_cm_mean",
        "track_root_ori_err_deg_mean",
        "track_obj_pos_err_cm_mean",
        "track_obj_z_abs_err_cm_mean",
        "track_obj_ori_err_deg_mean",
    ]
    out: dict[str, float] = {k: math.nan for k in keys}
    H = min(robot_qpos.shape[0], kin_qpos.shape[0])
    if H == 0:
        return out

    # Current G1+object scenes use 36 robot qpos followed by 6 object qpos.
    # Keep this bounded by actual dimensions so older scenes fail to NaN rather
    # than indexing past layout.
    nq_robot = min(36, max(0, model.nq - 6), robot_qpos.shape[1], kin_qpos.shape[1])
    if nq_robot <= 7:
        return out

    joint = np.abs(robot_qpos[:H, 7:nq_robot] - kin_qpos[:H, 7:nq_robot])
    if joint.size:
        out["track_joint_err_deg_mean"] = float(np.degrees(np.mean(joint)))

    pelvis_id = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    left_wrist_id = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    right_wrist_id = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    object_id = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    eef_ids = [bid for bid in (left_wrist_id, right_wrist_id) if bid >= 0]

    data_run = mujoco.MjData(model)
    data_ref = mujoco.MjData(model)
    eef_pos: list[float] = []
    eef_ori: list[float] = []
    root_pos: list[float] = []
    root_ori: list[float] = []
    obj_pos: list[float] = []
    obj_z_abs: list[float] = []
    obj_ori: list[float] = []

    for i in range(H):
        q_run = robot_qpos[i]
        data_run.qpos[:] = q_run
        data_run.qvel[:] = 0.0
        mujoco.mj_forward(model, data_run)

        if kin_qpos.shape[1] == model.nq:
            q_ref = kin_qpos[i, : model.nq].copy()
        else:
            q_ref = q_run.copy()
            q_ref[:nq_robot] = kin_qpos[i, :nq_robot]
        data_ref.qpos[:] = q_ref
        data_ref.qvel[:] = 0.0
        mujoco.mj_forward(model, data_ref)

        if pelvis_id >= 0:
            root_pos.append(float(np.linalg.norm(data_run.xpos[pelvis_id] - data_ref.xpos[pelvis_id])))
            root_ori.append(_quat_angle_deg(data_run.xquat[pelvis_id], data_ref.xquat[pelvis_id]))

        frame_eef_pos = []
        frame_eef_ori = []
        for bid in eef_ids:
            frame_eef_pos.append(float(np.linalg.norm(data_run.xpos[bid] - data_ref.xpos[bid])))
            frame_eef_ori.append(_quat_angle_deg(data_run.xquat[bid], data_ref.xquat[bid]))
        if frame_eef_pos:
            eef_pos.append(float(np.nanmean(frame_eef_pos)))
            eef_ori.append(float(np.nanmean(frame_eef_ori)))

        if object_id >= 0:
            if kin_qpos.shape[1] == model.nq:
                ref_obj_pos = data_ref.xpos[object_id].copy()
                ref_obj_quat = data_ref.xquat[object_id].copy()
            elif kin_qpos.shape[1] >= nq_robot + 7:
                ref_obj_pos = kin_qpos[i, nq_robot : nq_robot + 3]
                ref_obj_quat = kin_qpos[i, nq_robot + 3 : nq_robot + 7]
            else:
                ref_obj_pos = None
                ref_obj_quat = None
            if ref_obj_pos is not None and ref_obj_quat is not None:
                obj_pos.append(float(np.linalg.norm(data_run.xpos[object_id] - ref_obj_pos)))
                obj_z_abs.append(float(abs(data_run.xpos[object_id, 2] - ref_obj_pos[2])))
                obj_ori.append(_quat_angle_deg(data_run.xquat[object_id], ref_obj_quat))

    if eef_pos:
        out["track_eef_pos_err_cm_mean"] = float(np.nanmean(eef_pos) * 100.0)
    if eef_ori:
        out["track_eef_ori_err_deg_mean"] = float(np.nanmean(eef_ori))
    if root_pos:
        out["track_root_pos_err_cm_mean"] = float(np.nanmean(root_pos) * 100.0)
    if root_ori:
        out["track_root_ori_err_deg_mean"] = float(np.nanmean(root_ori))
    if obj_pos:
        out["track_obj_pos_err_cm_mean"] = float(np.nanmean(obj_pos) * 100.0)
    if obj_z_abs:
        out["track_obj_z_abs_err_cm_mean"] = float(np.nanmean(obj_z_abs) * 100.0)
    if obj_ori:
        out["track_obj_ori_err_deg_mean"] = float(np.nanmean(obj_ori))
    return out


def _tracking_metrics(
    robot_qpos: np.ndarray,
    kin_ref_path: Path | None,
    config: EvalConfig,
    model: mujoco.MjModel | None = None,
) -> dict[str, float]:
    """Body tracking error of the run's robot channel vs the fixed kin truth.

    Robot dofs [0:36] (7 root + 29 joints) share layout between the 42-dim run
    channel and the 43-dim kin qpos; only object representation differs.
    """
    keys = [
        "track_root_pos_err_mean_m", "track_root_pos_err_terminal_m",
        "track_root_quat_err_mean", "track_root_quat_err_terminal",
        "track_joint_err_mean_rad", "track_joint_err_terminal_rad",
        "track_pelvis_z_err_mean_m", "track_pelvis_z_err_terminal_m",
        "track_joint_err_deg_mean",
        "track_eef_pos_err_cm_mean", "track_eef_ori_err_deg_mean",
        "track_root_pos_err_cm_mean", "track_root_ori_err_deg_mean",
        "track_obj_pos_err_cm_mean", "track_obj_z_abs_err_cm_mean",
        "track_obj_ori_err_deg_mean",
    ]
    out: dict[str, float] = {k: math.nan for k in keys}
    if kin_ref_path is None or not Path(kin_ref_path).is_file():
        return out
    kin = np.asarray(np.load(kin_ref_path, allow_pickle=True)["qpos"], dtype=np.float64)
    if kin.ndim == 3:
        kin = kin[:, 0, :]
    H = min(robot_qpos.shape[0], kin.shape[0])
    if H == 0:
        return out
    r = robot_qpos[:H]
    k = kin[:H]
    K = max(1, int(round(H * config.track_terminal_frac)))
    pos = np.linalg.norm(r[:, :3] - k[:, :3], axis=1)
    joint = np.linalg.norm(r[:, 7:36] - k[:, 7:36], axis=1)
    pelvis = np.abs(r[:, 2] - k[:, 2])
    quat = _quat_geodesic_err(r[:, 3:7], k[:, 3:7])

    def mt(arr: np.ndarray) -> tuple[float, float]:
        return float(np.mean(arr)), float(np.mean(arr[-K:]))

    out["track_root_pos_err_mean_m"], out["track_root_pos_err_terminal_m"] = mt(pos)
    out["track_root_quat_err_mean"], out["track_root_quat_err_terminal"] = mt(quat)
    out["track_joint_err_mean_rad"], out["track_joint_err_terminal_rad"] = mt(joint)
    out["track_pelvis_z_err_mean_m"], out["track_pelvis_z_err_terminal_m"] = mt(pelvis)
    if model is not None:
        out.update(_table4_tracking_metrics(r, k, model))
    return out


def _box_corners_world(center: np.ndarray, mat: np.ndarray, half: np.ndarray) -> np.ndarray:
    """8 world-space corners of an oriented box."""
    signs = np.asarray(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=np.float64,
    )
    return center + (signs * half) @ mat.T


def _object_support_metrics(
    run_qpos: np.ndarray,
    kin_ref_path: Path | None,
    model: mujoco.MjModel,
    config: EvalConfig,
    hand_frame_min_con_dist: list[float],
) -> dict[str, float]:
    """E191: object-support diagnostics (observation-only, additive).

    Splits the object tracking error into a signed vertical and a horizontal
    component, resolves object height per side (robot side vs the unsupported
    "partner" side), reports the restoring wrench implied by the object guidance
    servo, and measures how much hand-object penetration the kinematic reference
    already carries before any physics runs.

    Side assignment uses the horizontal pelvis -> object-centre direction of the
    *reference* pose, so it does not depend on how far the run drifted. Heights
    are the lowest corner of the object collision box on each side, which is the
    same probe the SUGAR-side R010-6 diagnosis used.
    """
    out: dict[str, float] = dict.fromkeys(E191_SUPPORT_FIELDS, math.nan)
    out["object_half_extents_m"] = ""

    object_body = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    pelvis_body = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if object_body < 0:
        return out

    # Geometry / provenance covariates are available even without a reference.
    try:
        obj_gid = object_collision_geoms(model)[0]
    except ValueError:
        return out
    half = model.geom_size[obj_gid, :3].astype(np.float64).copy()
    geom_pos_local = model.geom_pos[obj_gid].astype(np.float64).copy()
    geom_mat_local = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(geom_mat_local, model.geom_quat[obj_gid].astype(np.float64))
    geom_mat_local = geom_mat_local.reshape(3, 3)
    obj_mass = float(model.body_mass[object_body])
    out["object_mass_kg"] = obj_mass
    out["object_half_extents_m"] = "|".join(f"{v:.4f}" for v in half)
    out["object_max_half_extent_m"] = float(np.max(half))
    out["object_weight_N"] = obj_mass * float(-model.opt.gravity[2])

    # Hand gate saturation is contact-based and needs no reference.
    if hand_frame_min_con_dist:
        frame_min = np.asarray(hand_frame_min_con_dist, dtype=np.float64)
        floor_cut = config.hand_gate_hard_floor_m + config.hand_gate_floor_tol_m
        out["hand_gate_floor_saturation_frac"] = frac(
            frame_min <= floor_cut
        )
        out["hand_gate_fixed_depth_8mm_frame_frac"] = frac(frame_min <= -0.008)
        out["hand_gate_fixed_depth_10mm_frame_frac"] = frac(frame_min <= -0.010)
        out["hand_gate_fixed_depth_12mm_frame_frac"] = frac(frame_min <= -0.012)
        out["hand_gate_fixed_depth_15mm_frame_frac"] = frac(frame_min <= -0.015)
        out["hand_gate_fixed_depth_20mm_frame_frac"] = frac(frame_min <= -0.020)

    if kin_ref_path is None or not Path(kin_ref_path).is_file():
        return out
    kin = np.asarray(np.load(kin_ref_path, allow_pickle=True)["qpos"], dtype=np.float64)
    if kin.ndim == 3:
        kin = kin[:, 0, :]
    H = min(run_qpos.shape[0], kin.shape[0])
    nq_robot = min(36, max(0, model.nq - 6), run_qpos.shape[1], kin.shape[1])
    if H == 0 or nq_robot <= 7:
        return out
    ref_in_scene_layout = kin.shape[1] == model.nq
    if not ref_in_scene_layout and kin.shape[1] < nq_robot + 7:
        return out

    hand_gids = [gid for name in HAND_GEOMS if (gid := mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)) >= 0]
    long_axis = int(np.argmax(half))

    data_run = mujoco.MjData(model)
    data_ref = mujoco.MjData(model)
    dz: list[float] = []
    dxy: list[float] = []
    force: list[np.ndarray] = []
    torque: list[float] = []
    near_run: list[float] = []
    near_ref: list[float] = []
    far_run: list[float] = []
    far_ref: list[float] = []
    grip_t: list[float] = []
    ref_hand_sdf: list[float] = []
    ref_obj_z: list[float] = []

    for i in range(H):
        data_run.qpos[:] = run_qpos[i]
        data_run.qvel[:] = 0.0
        mujoco.mj_forward(model, data_run)
        run_center = data_run.geom_xpos[obj_gid].copy()
        run_mat = data_run.geom_xmat[obj_gid].reshape(3, 3).copy()

        # Reference object pose: either replay it in scene layout, or read the
        # 7-DoF freejoint pose that follows the shared robot prefix.
        if ref_in_scene_layout:
            data_ref.qpos[:] = kin[i, : model.nq]
            data_ref.qvel[:] = 0.0
            mujoco.mj_forward(model, data_ref)
            ref_center = data_ref.geom_xpos[obj_gid].copy()
            ref_mat = data_ref.geom_xmat[obj_gid].reshape(3, 3).copy()
            ref_body_pos = data_ref.xpos[object_body].copy()
            ref_body_quat = data_ref.xquat[object_body].copy()
        else:
            ref_body_pos = kin[i, nq_robot : nq_robot + 3].copy()
            ref_body_quat = kin[i, nq_robot + 3 : nq_robot + 7].copy()
            body_mat = np.zeros(9, dtype=np.float64)
            mujoco.mju_quat2Mat(body_mat, ref_body_quat)
            body_mat = body_mat.reshape(3, 3)
            ref_center = ref_body_pos + body_mat @ geom_pos_local
            ref_mat = body_mat @ geom_mat_local
            # robot-only FK for the reference hand pose
            data_ref.qpos[:] = 0.0
            data_ref.qpos[:nq_robot] = kin[i, :nq_robot]
            data_ref.qvel[:] = 0.0
            mujoco.mj_forward(model, data_ref)

        ref_obj_z.append(float(ref_center[2]))
        delta = ref_center - run_center
        dz.append(float(-delta[2]))  # run minus ref, negative = object sagged
        dxy.append(float(np.linalg.norm(delta[:2])))

        # P-only slide actuators along orthonormal body-local axes: the restoring
        # force expressed in world coordinates is simply kp * (target - actual),
        # because the body-fixed rotation cancels between the two representations.
        force.append(config.object_pos_actuator_gain * delta)
        ang = _quat_angle_deg(data_run.xquat[object_body], ref_body_quat)
        torque.append(config.object_rot_actuator_gain * math.radians(ang) if np.isfinite(ang) else math.nan)

        # Side split on the reference geometry, along pelvis -> object centre.
        if pelvis_body >= 0:
            lateral = ref_center[:2] - data_run.xpos[pelvis_body, :2]
            norm = float(np.linalg.norm(lateral))
            if norm > 1e-9:
                lateral = lateral / norm
                c_run = _box_corners_world(run_center, run_mat, half)
                c_ref = _box_corners_world(ref_center, ref_mat, half)
                proj = (c_ref[:, :2] - ref_center[:2]) @ lateral
                far = proj > 0.0
                if far.any() and (~far).any():
                    near_run.append(float(c_run[~far, 2].min()))
                    near_ref.append(float(c_ref[~far, 2].min()))
                    far_run.append(float(c_run[far, 2].min()))
                    far_ref.append(float(c_ref[far, 2].min()))

        # Grip position along the object's longest local axis, and the hand
        # penetration the reference already carries.
        if hand_gids:
            centers = [data_ref.geom_xpos[gid].copy() for gid in hand_gids]
            mid_local = ref_mat.T @ (np.mean(centers, axis=0) - ref_center)
            grip_t.append(float(mid_local[long_axis]))
            best = math.inf
            for gid in hand_gids:
                pts, radius = geom_sample_points(model, data_ref, gid, config.mesh_sample_count)
                vals = [signed_point_box(p, ref_center, ref_mat, half) for p in pts]
                best = min(best, min(vals) - radius)
            ref_hand_sdf.append(float(best))

    dz_arr = np.asarray(dz)
    dxy_arr = np.asarray(dxy)
    out["track_obj_z_err_m_mean"] = float(np.nanmean(dz_arr))
    out["track_obj_z_err_m_p10"] = float(np.nanpercentile(dz_arr, 10))
    out["track_obj_xy_err_cm_mean"] = float(np.nanmean(dxy_arr) * 100.0)

    # Lifted-frame selection: side heights, grip lever and the servo wrench are
    # only interpretable while the object is off the floor. The fraction column
    # makes the fallback-to-all-frames case visible.
    ref_z_arr = np.asarray(ref_obj_z)
    lifted = ref_z_arr > (float(np.nanmin(ref_z_arr)) + config.object_lift_threshold_m)
    out["obj_lifted_frame_frac"] = frac(lifted)
    if int(np.count_nonzero(lifted)) < config.object_lift_min_frames:
        lifted = np.ones_like(lifted, dtype=bool)

    z_lift = float(np.nanmean(np.abs(dz_arr[lifted])))
    xy_lift = float(np.nanmean(dxy_arr[lifted]))
    out["track_obj_z_err_m_lifted_mean"] = float(np.nanmean(dz_arr[lifted]))
    out["track_obj_xy_err_cm_lifted_mean"] = xy_lift * 100.0
    if z_lift + xy_lift > 1e-12:
        out["track_obj_z_err_share_lifted"] = z_lift / (z_lift + xy_lift)

    force_arr = np.asarray(force)[lifted]
    out["object_guidance_force_N_p95"] = float(np.nanpercentile(np.linalg.norm(force_arr, axis=1), 95))
    out["object_guidance_force_z_N_p95"] = float(np.nanpercentile(np.abs(force_arr[:, 2]), 95))
    torque_arr = np.asarray(torque, dtype=np.float64)[lifted]
    if np.isfinite(torque_arr).any():
        out["object_guidance_torque_Nm_p95"] = float(np.nanpercentile(torque_arr, 95))

    if ref_hand_sdf:
        ref_sdf_arr = np.asarray(ref_hand_sdf)
        out["ref_hand_geom_penetration_frac"] = frac(ref_sdf_arr < 0.0)
        out["ref_hand_geom_penetration_3mm_frac"] = frac(ref_sdf_arr < -0.003)
        out["ref_hand_geom_min_sdf_m"] = float(np.nanmin(ref_sdf_arr))

    if len(near_run) == len(lifted) and len(near_run) > 0:
        near_err = np.asarray(near_run)[lifted] - np.asarray(near_ref)[lifted]
        far_err = np.asarray(far_run)[lifted] - np.asarray(far_ref)[lifted]
        if near_err.size:
            out["obj_side_near_z_err_m"] = float(np.nanmean(near_err))
            out["obj_side_far_z_err_m"] = float(np.nanmean(far_err))
            out["obj_side_z_asym_cm"] = float((np.nanmean(near_err) - np.nanmean(far_err)) * 100.0)

    if len(grip_t) == len(lifted) and len(grip_t) > 0:
        t = float(np.nanmean(np.asarray(grip_t)[lifted]))
        h = float(half[long_axis])
        out["grip_near_arm_m"] = min(h - t, h + t)
        out["grip_far_arm_m"] = max(h - t, h + t)
    return out


def _fill_internal_false_gaps(mask: np.ndarray, max_gap_frames: int) -> tuple[np.ndarray, int]:
    filled = np.asarray(mask, dtype=np.bool_).copy()
    if filled.ndim == 1:
        filled = filled[:, None]
    changed_frames = np.zeros((filled.shape[0],), dtype=np.bool_)
    for col in range(filled.shape[1]):
        values = filled[:, col]
        i = 0
        while i < values.shape[0]:
            if values[i]:
                i += 1
                continue
            start = i
            while i < values.shape[0] and not values[i]:
                i += 1
            end = i
            if start == 0 or end == values.shape[0]:
                continue
            if end - start <= max_gap_frames and values[start - 1] and values[end]:
                values[start:end] = True
                changed_frames[start:end] = True
    return filled, int(np.count_nonzero(changed_frames))


def _masked_contact_metrics(
    hand_physics: list[bool],
    hand_clean_physics: list[bool],
    hand_clean3_physics: list[bool],
    hand_clean5_physics: list[bool],
    hand_arr: np.ndarray,
    contact_mask_path: Path | None,
    person_idx: int | None,
) -> dict[str, float]:
    """Contact/penetration restricted to the real 3cm reference contact window.

    `hand_physics` is the per-frame any-hand physics-contact flag; `hand_arr` is
    the per-frame min hand-object geom SDF. `hand_clean_physics` removes frames
    where the MuJoCo contact distance penetrates deeper than the clean-contact
    threshold. The mask
    `spider_contact_mask_3cm` is (N, 2 persons, 2 hands), frame-aligned.
    """
    keys = [
        "ref_contact_frac",
        "hand_object_physics_contact_in_mask_frac",
        "hand_object_clean_physics_contact_in_mask_frac",
        "hand_object_physics_contact_3mm_in_mask_frac",
        "hand_object_physics_contact_5mm_in_mask_frac",
        "rl_object_contact_ref_frac",
        "rl_object_contact_gap_fill_frames",
        "rl_object_contact_filled_frame_count",
        "hand_object_physics_contact_in_rl_mask_frac",
        "hand_object_physics_contact_3mm_in_rl_mask_frac",
        "hand_object_physics_contact_5mm_in_rl_mask_frac",
        "hand_object_false_contact_frac",
        "hand_object_clean_false_contact_frac",
        "hand_object_false_contact_3mm_frac",
        "hand_object_false_contact_5mm_frac",
        "hand_object_approach_false_contact_frac",
        "hand_object_clean_approach_false_contact_frac",
        "hand_object_approach_false_contact_3mm_frac",
        "hand_object_approach_false_contact_5mm_frac",
        "hand_object_release_false_contact_frac",
        "hand_object_clean_release_false_contact_frac",
        "hand_object_release_false_contact_3mm_frac",
        "hand_object_release_false_contact_5mm_frac",
        "hand_geom_penetration_2mm_in_mask_frac",
        "hand_geom_penetration_5mm_in_mask_frac",
    ]
    out: dict[str, float] = {k: math.nan for k in keys}
    if contact_mask_path is None or person_idx is None or not Path(contact_mask_path).is_file():
        return out
    data = np.load(contact_mask_path, allow_pickle=True)
    if "spider_contact_mask_3cm" not in data.files:
        return out
    sm = np.asarray(data["spider_contact_mask_3cm"])  # (N, persons, hands)
    pi = int(person_idx)
    if sm.ndim != 3 or sm.shape[2] != 2:
        raise ValueError(f"{contact_mask_path}: spider_contact_mask_3cm expected (T, persons, 2), got {sm.shape}")
    if pi < 0 or pi >= sm.shape[1]:
        raise ValueError(f"{contact_mask_path}: person_idx={pi} out of mask shape {sm.shape}")
    if sm.shape[0] != hand_arr.shape[0]:
        raise ValueError(
            f"{contact_mask_path}: contact mask length {sm.shape[0]} != qpos frames {hand_arr.shape[0]}"
        )
    mask_any = sm[:, pi, 0].astype(bool) | sm[:, pi, 1].astype(bool)
    H = min(
        len(hand_physics),
        len(hand_clean_physics),
        len(hand_clean3_physics),
        len(hand_clean5_physics),
        mask_any.shape[0],
        hand_arr.shape[0],
    )
    if H == 0:
        return out
    rp = np.asarray(hand_physics[:H], dtype=bool)
    cp = np.asarray(hand_clean_physics[:H], dtype=bool)
    c3 = np.asarray(hand_clean3_physics[:H], dtype=bool)
    c5 = np.asarray(hand_clean5_physics[:H], dtype=bool)
    m = mask_any[:H]
    ha = np.asarray(hand_arr[:H], dtype=np.float64)
    out["ref_contact_frac"] = float(np.mean(m))
    any_hand = np.max(sm[:H, pi, :].astype(bool), axis=1)
    rl_pair_mask = np.stack([any_hand, any_hand], axis=1)
    rl_pair_filled, rl_filled_count = _fill_internal_false_gaps(
        rl_pair_mask,
        DOWNSTREAM_RL_CONTACT_GAP_FILL_FRAMES,
    )
    rl_mask = np.max(rl_pair_filled, axis=1)
    out["rl_object_contact_gap_fill_frames"] = float(DOWNSTREAM_RL_CONTACT_GAP_FILL_FRAMES)
    out["rl_object_contact_filled_frame_count"] = float(rl_filled_count)
    out["rl_object_contact_ref_frac"] = float(np.mean(rl_mask))
    if m.any():
        out["hand_object_physics_contact_in_mask_frac"] = float(np.mean(rp[m]))
        out["hand_object_clean_physics_contact_in_mask_frac"] = float(np.mean(cp[m]))
        out["hand_object_physics_contact_3mm_in_mask_frac"] = float(np.mean(c3[m]))
        out["hand_object_physics_contact_5mm_in_mask_frac"] = float(np.mean(c5[m]))
        out["hand_geom_penetration_2mm_in_mask_frac"] = float(np.mean(ha[m] < -0.002))
        out["hand_geom_penetration_5mm_in_mask_frac"] = float(np.mean(ha[m] < -0.005))
    if rl_mask.any():
        out["hand_object_physics_contact_in_rl_mask_frac"] = float(np.mean(rp[rl_mask]))
        out["hand_object_physics_contact_3mm_in_rl_mask_frac"] = float(np.mean(c3[rl_mask]))
        out["hand_object_physics_contact_5mm_in_rl_mask_frac"] = float(np.mean(c5[rl_mask]))
    if (~m).any():
        out["hand_object_false_contact_frac"] = float(np.mean(rp[~m]))
        out["hand_object_clean_false_contact_frac"] = float(np.mean(cp[~m]))
        out["hand_object_false_contact_3mm_frac"] = float(np.mean(c3[~m]))
        out["hand_object_false_contact_5mm_frac"] = float(np.mean(c5[~m]))
    # Split the no-contact frames into the leading (approach) and trailing
    # (release) windows; the release window is where "won't let go" shows up.
    if m.any():
        first_c = int(np.argmax(m))
        last_c = int(H - 1 - np.argmax(m[::-1]))
        approach = np.zeros(H, dtype=bool)
        approach[:first_c] = True
        release = np.zeros(H, dtype=bool)
        release[last_c + 1:] = True
        if approach.any():
            out["hand_object_approach_false_contact_frac"] = float(np.mean(rp[approach]))
            out["hand_object_clean_approach_false_contact_frac"] = float(np.mean(cp[approach]))
            out["hand_object_approach_false_contact_3mm_frac"] = float(np.mean(c3[approach]))
            out["hand_object_approach_false_contact_5mm_frac"] = float(np.mean(c5[approach]))
        if release.any():
            out["hand_object_release_false_contact_frac"] = float(np.mean(rp[release]))
            out["hand_object_clean_release_false_contact_frac"] = float(np.mean(cp[release]))
            out["hand_object_release_false_contact_3mm_frac"] = float(np.mean(c3[release]))
            out["hand_object_release_false_contact_5mm_frac"] = float(np.mean(c5[release]))
    return out


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------


def evaluate_sequence(
    *,
    row: dict[str, str],
    method: str,
    hand_collision_variant_id: str,
    qpos_path: Path,
    scene_xml: Path,
    config: EvalConfig | None = None,
    kin_ref_path: Path | None = None,
    contact_mask_path: Path | None = None,
    person_idx: int | None = None,
) -> dict[str, Any]:
    """Evaluate a single retargeting trajectory against a MuJoCo scene.

    Parameters
    ----------
    row : dict
        Manifest row with at least case_id, variant, object_key, object_category,
        expected_quality fields.
    method : str
        Human-readable method label.
    hand_collision_variant_id : str
        Identifier for the hand collision variant (e.g. "sphere5cm", "rubber_hull").
    qpos_path : Path
        Path to NPZ file containing 'qpos' array.
    scene_xml : Path
        Path to MuJoCo scene XML.
    config : EvalConfig | None
        Evaluation parameters. Uses defaults (matching E147) if None.

    Returns
    -------
    dict with all METRIC_FIELDS populated.
    """
    if config is None:
        config = EvalConfig()

    qpos, ref_qpos = npz_qpos(qpos_path)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    if qpos.shape[1] != model.nq:
        raise ValueError(f"qpos nq={qpos.shape[1]} != scene nq={model.nq}: {qpos_path} vs {scene_xml}")
    data = mujoco.MjData(model)
    object_gids = object_collision_geoms(model)
    floor_gid = mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_z0 = 0.0
    object_body = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    pelvis_body = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    left_wrist = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    right_wrist = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    lower_gids = [gid for name in LOWERBODY_GEOMS if (gid := mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)) >= 0]
    hand_gids = [gid for name in HAND_GEOMS if (gid := mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)) >= 0]
    upper_bodies = [bid for name in UPPER_BODY_NAMES if (bid := mj_id(model, mujoco.mjtObj.mjOBJ_BODY, name)) >= 0]
    head_body = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "head_link")

    pelvis_z: list[float] = []
    root_xy: list[np.ndarray] = []
    obj_xyz: list[np.ndarray] = []
    eef_l: list[float] = []
    eef_r: list[float] = []
    hand_sdf: list[float] = []
    leg_sdf: list[float] = []
    body_sdf: list[float] = []
    head_sdf: list[float] = []
    upper_sdf: list[float] = []
    hand_physics: list[bool] = []
    hand_clean_physics: list[bool] = []
    hand_clean3_physics: list[bool] = []
    hand_clean5_physics: list[bool] = []
    hand_object_deep2mm_frame: list[bool] = []
    hand_object_deep3mm_frame: list[bool] = []
    hand_object_deep_frame: list[bool] = []
    hand_object_contact_dists: list[float] = []
    hand_frame_min_con_dist: list[float] = []  # E191: per-frame min, inf when no contact
    leg_physics: list[bool] = []
    object_floor: list[bool] = []
    hand_floor_sdf: list[float] = []
    hand_floor_physics: list[bool] = []
    hand_floor_deep_frame: list[bool] = []
    hand_floor_contact_dists: list[float] = []
    obj_err: list[float] = []

    object_set = set(object_gids)
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        pelvis_z.append(float(data.xpos[pelvis_body, 2]))
        root_xy.append(q[:2].copy())
        obj_xyz.append(data.xpos[object_body].copy())

        frame_eef = []
        for bid in (left_wrist, right_wrist):
            if bid < 0:
                frame_eef.append(math.nan)
            else:
                eef = data.xpos[bid].copy() + quat_rotate(data.xquat[bid].copy(), config.eef_offset)
                frame_eef.append(point_object_sdf(model, data, eef, object_gids))
        eef_l.append(float(frame_eef[0]))
        eef_r.append(float(frame_eef[1]))

        hand_vals = [geom_object_sdf(model, data, gid, object_gids, config.mesh_sample_count) for gid in hand_gids]
        hand_sdf.append(float(min(hand_vals)) if hand_vals else math.nan)
        # Hand-floor signed distance: min z of hand mesh sample points minus floor
        # plane (z=0), same geom-vertex sampling used for hand-object SDF.
        hf_min = math.inf
        for gid in hand_gids:
            pts, radius = geom_sample_points(model, data, gid, config.mesh_sample_count)
            hf_min = min(hf_min, float(pts[:, 2].min()) - radius - floor_z0)
        hand_floor_sdf.append(hf_min if hand_gids else math.nan)
        leg_vals = [geom_object_sdf(model, data, gid, object_gids, config.mesh_sample_count) for gid in lower_gids]
        leg_sdf.append(float(min(leg_vals)) if leg_vals else math.nan)
        body_vals = [point_object_sdf(model, data, data.xpos[bid].copy(), object_gids) for bid in upper_bodies]
        body_sdf.append(float(min(body_vals)) if body_vals else math.nan)
        head_sdf.append(point_object_sdf(model, data, data.xpos[head_body].copy(), object_gids) if head_body >= 0 else math.nan)
        upper_vals = [point_object_sdf(model, data, data.xpos[bid].copy(), object_gids) for bid in upper_bodies if bid != head_body]
        upper_sdf.append(float(min(upper_vals)) if upper_vals else math.nan)

        hand_contact = False
        hand_object_frame_dists: list[float] = []
        leg_contact = False
        floor_contact = False
        hand_floor_contact = False
        hand_floor_frame_dists: list[float] = []
        for ci in range(data.ncon):
            con = data.contact[ci]
            pair = {int(con.geom1), int(con.geom2)}
            if floor_gid >= 0 and floor_gid in pair and object_set & pair:
                floor_contact = True
            if floor_gid >= 0 and floor_gid in pair:
                other_floor = pair - {floor_gid}
                if other_floor and any(gid in hand_gids for gid in other_floor):
                    hand_floor_contact = True
                    hand_floor_frame_dists.append(float(con.dist))
            if not (object_set & pair):
                continue
            other = list(pair - object_set)
            if other:
                if other[0] in hand_gids:
                    hand_contact = True
                    hand_object_frame_dists.append(float(con.dist))
                leg_contact = leg_contact or other[0] in lower_gids
        hand_physics.append(hand_contact)
        hand_clean_physics.append(
            bool(hand_object_frame_dists)
            and min(hand_object_frame_dists) >= config.clean_contact_penetration_m
        )
        hand_clean3_physics.append(
            bool(hand_object_frame_dists)
            and min(hand_object_frame_dists) >= -0.003
        )
        hand_clean5_physics.append(
            bool(hand_object_frame_dists)
            and min(hand_object_frame_dists) >= -0.005
        )
        hand_object_deep2mm_frame.append(
            bool(hand_object_frame_dists)
            and min(hand_object_frame_dists) < config.clean_contact_penetration_m
        )
        hand_object_deep3mm_frame.append(
            bool(hand_object_frame_dists)
            and min(hand_object_frame_dists) < -0.003
        )
        hand_object_deep_frame.append(
            bool(hand_object_frame_dists)
            and min(hand_object_frame_dists) < config.deep_contact_dist_m
        )
        hand_object_contact_dists.extend(hand_object_frame_dists)
        hand_frame_min_con_dist.append(
            min(hand_object_frame_dists) if hand_object_frame_dists else math.inf
        )
        leg_physics.append(leg_contact)
        object_floor.append(floor_contact)
        hand_floor_physics.append(hand_floor_contact)
        hand_floor_deep_frame.append(
            bool(hand_floor_frame_dists)
            and min(hand_floor_frame_dists) < config.deep_contact_dist_m
        )
        hand_floor_contact_dists.extend(hand_floor_frame_dists)

        if ref_qpos is not None and i < len(ref_qpos):
            data.qpos[:] = ref_qpos[i]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            obj_err.append(float(np.linalg.norm(obj_xyz[-1] - data.xpos[object_body].copy())))

    pelvis = np.asarray(pelvis_z)
    root_arr = np.asarray(root_xy)
    obj_arr = np.asarray(obj_xyz)
    eef_l_arr = np.asarray(eef_l)
    eef_r_arr = np.asarray(eef_r)
    hand_arr = np.asarray(hand_sdf)
    hand_floor_arr = np.asarray(hand_floor_sdf)
    leg_arr = np.asarray(leg_sdf)
    body_arr = np.asarray(body_sdf)
    head_arr = np.asarray(head_sdf)
    upper_arr = np.asarray(upper_sdf)
    obj_err_arr = np.asarray(obj_err)

    out: dict[str, Any] = {
        "case_id": row["case_id"],
        "variant": row["variant"],
        "method": method,
        "hand_collision_variant_id": hand_collision_variant_id,
        "object_key": row.get("object_key", ""),
        "object_category": row.get("object_category", ""),
        "expected_quality": row.get("expected_quality", ""),
        "qpos_path": rel(qpos_path),
        "scene_xml": rel(scene_xml),
        "qpos_frames": int(qpos.shape[0]),
        "duration_s": float(qpos.shape[0] / config.fps),
        "pelvis_min_m": float(np.nanmin(pelvis)),
        "pelvis_end_m": float(pelvis[-1]),
        "fall_flag": bool(np.nanmin(pelvis) < config.fall_pelvis_z_m),
        "root_xy_displacement_m": float(np.linalg.norm(root_arr[-1] - root_arr[0])) if len(root_arr) else math.nan,
        "object_xy_displacement_m": float(np.linalg.norm(obj_arr[-1, :2] - obj_arr[0, :2])) if len(obj_arr) else math.nan,
        "object_z_range_m": float(np.nanmax(obj_arr[:, 2]) - np.nanmin(obj_arr[:, 2])) if len(obj_arr) else math.nan,
        "object_floor_contact_frac": frac(np.asarray(object_floor, dtype=bool)),
        "hand_geom_penetration_frac": frac(hand_arr < 0.0),
        "hand_geom_penetration_2mm_frac": frac(hand_arr < -0.002),
        "hand_geom_penetration_5mm_frac": frac(hand_arr < -0.005),
        "hand_geom_deep_penetration_2cm_frac": frac(hand_arr < config.deep_penetration_m),
        "hand_object_physics_contact_frac": frac(np.asarray(hand_physics, dtype=bool)),
        "hand_object_clean_physics_contact_frac": frac(np.asarray(hand_clean_physics, dtype=bool)),
        "hand_object_physics_contact_3mm_frac": frac(np.asarray(hand_clean3_physics, dtype=bool)),
        "hand_object_physics_contact_5mm_frac": frac(np.asarray(hand_clean5_physics, dtype=bool)),
        "hand_object_physics_penetration_3mm_frame_frac": frac(
            np.asarray(hand_object_deep3mm_frame, dtype=bool)
        ),
        "hand_object_physics_penetration_5mm_frame_frac": frac(
            np.asarray(hand_object_deep_frame, dtype=bool)
        ),
        "hand_object_con_dist_mean_m": mean_or_nan(hand_object_contact_dists),
        "hand_object_con_dist_min_m": min_or_nan(hand_object_contact_dists),
        "hand_object_con_dist_frac_lt_neg2mm": contact_frac_lt(
            hand_object_contact_dists, config.clean_contact_penetration_m
        ),
        "hand_object_con_dist_frac_lt_neg3mm": contact_frac_lt(
            hand_object_contact_dists, -0.003
        ),
        "hand_object_con_dist_frac_lt_neg5mm": contact_frac_lt(
            hand_object_contact_dists, config.deep_contact_dist_m
        ),
        "hand_object_con_deep2mm_frame_frac": frac(
            np.asarray(hand_object_deep2mm_frame, dtype=bool)
        ),
        "hand_object_con_deep3mm_frame_frac": frac(
            np.asarray(hand_object_deep3mm_frame, dtype=bool)
        ),
        "hand_object_con_deep5mm_frame_frac": frac(
            np.asarray(hand_object_deep_frame, dtype=bool)
        ),
        "hand_floor_min_z_m": float(np.nanmin(hand_floor_arr)) if hand_floor_arr.size else math.nan,
        "hand_floor_near_2cm_frac": frac(hand_floor_arr < 0.02),
        "hand_floor_penetration_frac": frac(hand_floor_arr < 0.0),
        "hand_floor_physics_contact_frac": frac(np.asarray(hand_floor_physics, dtype=bool)),
        "hand_floor_con_dist_mean_m": mean_or_nan(hand_floor_contact_dists),
        "hand_floor_con_dist_min_m": min_or_nan(hand_floor_contact_dists),
        "hand_floor_con_dist_frac_lt_neg5mm": contact_frac_lt(
            hand_floor_contact_dists, config.deep_contact_dist_m
        ),
        "hand_floor_con_deep5mm_frame_frac": frac(
            np.asarray(hand_floor_deep_frame, dtype=bool)
        ),
        "leg_near_2cm_frac": frac(leg_arr < 0.02),
        "leg_penetration_frac": frac(leg_arr < 0.0),
        "leg_object_physics_contact_frac": frac(np.asarray(leg_physics, dtype=bool)),
        "body_penetration_frac": frac(body_arr < 0.0),
        "head_penetration_frac": frac(head_arr < 0.0),
        "upper_body_penetration_frac": frac(upper_arr < 0.0),
        "obj_err_mean_m": float(np.nanmean(obj_err_arr)) if obj_err_arr.size else math.nan,
        "obj_err_max_m": float(np.nanmax(obj_err_arr)) if obj_err_arr.size else math.nan,
        "notes": "",
    }
    for threshold in config.near_thresholds_m:
        tag = int(round(threshold * 100))
        out[f"eef_near_{tag}cm_frac"] = max(frac(eef_l_arr < threshold), frac(eef_r_arr < threshold))
        out[f"hand_geom_near_{tag}cm_frac"] = frac(hand_arr < threshold)

    # E154: body tracking vs fixed kin truth + masked contact (real 3cm).
    # Always populated (NaN when refs not supplied) so METRIC_FIELDS stays complete.
    out.update(_tracking_metrics(qpos, kin_ref_path, config, model=model))
    out.update(_masked_contact_metrics(hand_physics, hand_clean_physics, hand_clean3_physics, hand_clean5_physics, hand_arr, contact_mask_path, person_idx))
    # E191: object-support diagnostics. Additive only — no existing column and no
    # 12-gate rule depends on these.
    out.update(_object_support_metrics(qpos, kin_ref_path, model, config, hand_frame_min_con_dist))
    return out
