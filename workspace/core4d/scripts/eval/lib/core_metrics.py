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
    deep_contact_dist_m: float = -0.005
    mesh_sample_count: int = 800


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
    "hand_object_con_dist_mean_m",
    "hand_object_con_dist_min_m",
    "hand_object_con_dist_frac_lt_neg5mm",
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

# Minimal core metrics suitable as a shared baseline set.
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
    hand_object_deep_frame: list[bool] = []
    hand_object_contact_dists: list[float] = []
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
        hand_object_deep_frame.append(
            bool(hand_object_frame_dists)
            and min(hand_object_frame_dists) < config.deep_contact_dist_m
        )
        hand_object_contact_dists.extend(hand_object_frame_dists)
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
        "hand_object_con_dist_mean_m": mean_or_nan(hand_object_contact_dists),
        "hand_object_con_dist_min_m": min_or_nan(hand_object_contact_dists),
        "hand_object_con_dist_frac_lt_neg5mm": contact_frac_lt(
            hand_object_contact_dists, config.deep_contact_dist_m
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
    return out
