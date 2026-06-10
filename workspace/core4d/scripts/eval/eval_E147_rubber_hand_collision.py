#!/usr/bin/env python3
"""Evaluate E147 sphere5cm vs rubber_hull CEM trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[4]
VARIANTS_TSV = REPO / "workspace/core4d/scripts/E147/variants.tsv"
RESULT_ROOT = REPO / "workspace/core4d/results/E147/rubber_hand_collision"

FPS = 30.0
FALL_PELVIS_Z_M = 0.45
EEF_OFFSET = np.asarray([0.05, 0.0, 0.0], dtype=np.float64)
NEAR_THRESHOLDS_M = (0.03, 0.05, 0.08, 0.10)
DEEP_PENETRATION_M = -0.02
DEEP_CONTACT_DIST_M = -0.005
MESH_SAMPLE_COUNT = 800

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

PAIR_FIELDS = [
    "case_id",
    "variant",
    "object_key",
    "object_category",
    "expected_quality",
    "sphere_historical_cem_status",
    "rubber_outputs_present",
    "sphere_hand_near_5cm",
    "rubber_hand_near_5cm",
    "delta_hand_near_5cm",
    "sphere_hand_deep_2cm",
    "rubber_hand_deep_2cm",
    "delta_hand_deep_2cm",
    "sphere_hand_penetration",
    "rubber_hand_penetration",
    "delta_hand_penetration",
    "sphere_physics_contact",
    "rubber_physics_contact",
    "delta_physics_contact",
    "sphere_pelvis_min_m",
    "rubber_pelvis_min_m",
    "rubber_fall_flag",
    "rubber_leg_penetration",
    "rubber_body_penetration",
    "rubber_object_floor_contact",
    "rubber_obj_err_mean_m",
    "ab_status",
]

EVIDENCE_FIELDS = [
    "case_id",
    "object_key",
    "object_name",
    "date",
    "seq",
    "person",
    "person_idx",
    "retarget_variant_id",
    "target_variant_id",
    "hand_collision_variant_id",
    "candidate_decision",
    "handoff_decision",
    "cem_status",
    "rl_status",
    "downstream_decision",
    "downstream_failure_mode",
    "downstream_notes",
    "cem_run_id",
    "cem_result_npz",
    "cem_video",
    "cem_metrics_ref",
    "rl_run_id",
    "rl_checkpoint",
    "rl_metrics_ref",
    "rl_video",
]


def rel(path: Path | str) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO.resolve()))
    except Exception:
        return str(path)


def repo_path(text: str) -> Path:
    p = Path(text)
    return p if p.is_absolute() else REPO / p


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.8g}"
    return str(value)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in fields})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def npz_qpos(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
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


def geom_sample_points(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> tuple[np.ndarray, float]:
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
        if nv > MESH_SAMPLE_COUNT:
            idx = np.linspace(0, nv - 1, MESH_SAMPLE_COUNT).astype(int)
            verts = verts[idx]
        return (mat @ verts.T).T + pos, 0.0
    return pos[None, :], 0.0


def geom_object_sdf(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int, object_gids: list[int]) -> float:
    points, radius = geom_sample_points(model, data, geom_id)
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


def evaluate_sequence(
    *,
    row: dict[str, str],
    method: str,
    hand_collision_variant_id: str,
    qpos_path: Path,
    scene_xml: Path,
) -> dict[str, Any]:
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
                eef = data.xpos[bid].copy() + quat_rotate(data.xquat[bid].copy(), EEF_OFFSET)
                frame_eef.append(point_object_sdf(model, data, eef, object_gids))
        eef_l.append(float(frame_eef[0]))
        eef_r.append(float(frame_eef[1]))

        hand_vals = [geom_object_sdf(model, data, gid, object_gids) for gid in hand_gids]
        hand_sdf.append(float(min(hand_vals)) if hand_vals else math.nan)
        # Hand-floor signed distance: min z of hand mesh sample points minus floor
        # plane (z=0), same geom-vertex sampling used for hand-object SDF.
        hf_min = math.inf
        for gid in hand_gids:
            pts, radius = geom_sample_points(model, data, gid)
            hf_min = min(hf_min, float(pts[:, 2].min()) - radius - floor_z0)
        hand_floor_sdf.append(hf_min if hand_gids else math.nan)
        leg_vals = [geom_object_sdf(model, data, gid, object_gids) for gid in lower_gids]
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
            and min(hand_object_frame_dists) < DEEP_CONTACT_DIST_M
        )
        hand_object_contact_dists.extend(hand_object_frame_dists)
        leg_physics.append(leg_contact)
        object_floor.append(floor_contact)
        hand_floor_physics.append(hand_floor_contact)
        hand_floor_deep_frame.append(
            bool(hand_floor_frame_dists)
            and min(hand_floor_frame_dists) < DEEP_CONTACT_DIST_M
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
        "duration_s": float(qpos.shape[0] / FPS),
        "pelvis_min_m": float(np.nanmin(pelvis)),
        "pelvis_end_m": float(pelvis[-1]),
        "fall_flag": bool(np.nanmin(pelvis) < FALL_PELVIS_Z_M),
        "root_xy_displacement_m": float(np.linalg.norm(root_arr[-1] - root_arr[0])) if len(root_arr) else math.nan,
        "object_xy_displacement_m": float(np.linalg.norm(obj_arr[-1, :2] - obj_arr[0, :2])) if len(obj_arr) else math.nan,
        "object_z_range_m": float(np.nanmax(obj_arr[:, 2]) - np.nanmin(obj_arr[:, 2])) if len(obj_arr) else math.nan,
        "object_floor_contact_frac": frac(np.asarray(object_floor, dtype=bool)),
        "hand_geom_penetration_frac": frac(hand_arr < 0.0),
        "hand_geom_penetration_2mm_frac": frac(hand_arr < -0.002),
        "hand_geom_penetration_5mm_frac": frac(hand_arr < -0.005),
        "hand_geom_deep_penetration_2cm_frac": frac(hand_arr < DEEP_PENETRATION_M),
        "hand_object_physics_contact_frac": frac(np.asarray(hand_physics, dtype=bool)),
        "hand_object_con_dist_mean_m": mean_or_nan(hand_object_contact_dists),
        "hand_object_con_dist_min_m": min_or_nan(hand_object_contact_dists),
        "hand_object_con_dist_frac_lt_neg5mm": contact_frac_lt(
            hand_object_contact_dists, DEEP_CONTACT_DIST_M
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
            hand_floor_contact_dists, DEEP_CONTACT_DIST_M
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
        "notes": "E147_eval_mesh_vertex_sdf_v1" if hand_collision_variant_id == "rubber_hull" else "E147_eval_sphere_sdf_v1",
    }
    for threshold in NEAR_THRESHOLDS_M:
        tag = int(round(threshold * 100))
        out[f"eef_near_{tag}cm_frac"] = max(frac(eef_l_arr < threshold), frac(eef_r_arr < threshold))
        out[f"hand_geom_near_{tag}cm_frac"] = frac(hand_arr < threshold)
    return out


def rubber_paths(row: dict[str, str], stage: str, results_dir: Path) -> tuple[Path, Path]:
    outdir_npz = results_dir / f"{row['variant']}_outdir_{stage}/trajectory_mjwp_act.npz"
    return outdir_npz, repo_path(row["rubber_scene_act"])


def build_pair_rows(rows: list[dict[str, str]], metrics: list[dict[str, Any]], stage: str, results_dir: Path) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for metric in metrics:
        by_key[(metric["case_id"], metric["hand_collision_variant_id"])] = metric
    out = []
    for row in rows:
        sphere = by_key.get((row["case_id"], "sphere5cm"), {})
        rubber = by_key.get((row["case_id"], "rubber_hull"), {})
        rubber_npz = results_dir / f"{row['variant']}_outdir_{stage}/trajectory_mjwp_act.npz"

        def delta(key: str) -> float:
            return float(rubber.get(key, math.nan)) - float(sphere.get(key, math.nan))

        contact_ok = delta("hand_geom_near_5cm_frac") >= -0.02
        deep_ok = delta("hand_geom_deep_penetration_2cm_frac") < 0.0
        stable_ok = not bool(rubber.get("fall_flag", True))
        if rubber_npz.is_file():
            if contact_ok and deep_ok and stable_ok:
                status = "rubber_primary_pass"
            elif contact_ok and stable_ok:
                status = "rubber_contact_stable_but_penetration_not_improved"
            else:
                status = "rubber_primary_fail"
        else:
            status = "missing_rubber"
        out.append(
            {
                "case_id": row["case_id"],
                "variant": row["variant"],
                "object_key": row["object_key"],
                "object_category": row["object_category"],
                "expected_quality": row["expected_quality"],
                "sphere_historical_cem_status": row["historical_cem_status"],
                "rubber_outputs_present": rubber_npz.is_file(),
                "sphere_hand_near_5cm": sphere.get("hand_geom_near_5cm_frac", ""),
                "rubber_hand_near_5cm": rubber.get("hand_geom_near_5cm_frac", ""),
                "delta_hand_near_5cm": delta("hand_geom_near_5cm_frac"),
                "sphere_hand_deep_2cm": sphere.get("hand_geom_deep_penetration_2cm_frac", ""),
                "rubber_hand_deep_2cm": rubber.get("hand_geom_deep_penetration_2cm_frac", ""),
                "delta_hand_deep_2cm": delta("hand_geom_deep_penetration_2cm_frac"),
                "sphere_hand_penetration": sphere.get("hand_geom_penetration_frac", ""),
                "rubber_hand_penetration": rubber.get("hand_geom_penetration_frac", ""),
                "delta_hand_penetration": delta("hand_geom_penetration_frac"),
                "sphere_physics_contact": sphere.get("hand_object_physics_contact_frac", ""),
                "rubber_physics_contact": rubber.get("hand_object_physics_contact_frac", ""),
                "delta_physics_contact": delta("hand_object_physics_contact_frac"),
                "sphere_pelvis_min_m": sphere.get("pelvis_min_m", ""),
                "rubber_pelvis_min_m": rubber.get("pelvis_min_m", ""),
                "rubber_fall_flag": rubber.get("fall_flag", ""),
                "rubber_leg_penetration": rubber.get("leg_penetration_frac", ""),
                "rubber_body_penetration": rubber.get("body_penetration_frac", ""),
                "rubber_object_floor_contact": rubber.get("object_floor_contact_frac", ""),
                "rubber_obj_err_mean_m": rubber.get("obj_err_mean_m", ""),
                "ab_status": status,
            }
        )
    return out


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else math.nan


def build_summary(metrics: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for metric in metrics:
        by_method[metric["hand_collision_variant_id"]].append(metric)
    method_summary = {}
    for method, rows in sorted(by_method.items()):
        method_summary[method] = {
            "rows": len(rows),
            "fall_count": sum(bool(r.get("fall_flag")) for r in rows),
            "mean_hand_near_5cm": mean([float(r["hand_geom_near_5cm_frac"]) for r in rows]),
            "mean_hand_near_8cm": mean([float(r["hand_geom_near_8cm_frac"]) for r in rows]),
            "mean_hand_penetration": mean([float(r["hand_geom_penetration_frac"]) for r in rows]),
            "mean_hand_deep_2cm": mean([float(r["hand_geom_deep_penetration_2cm_frac"]) for r in rows]),
            "mean_physics_contact": mean([float(r["hand_object_physics_contact_frac"]) for r in rows]),
            "mean_pelvis_min_m": mean([float(r["pelvis_min_m"]) for r in rows]),
            "mean_obj_err_mean_m": mean([float(r["obj_err_mean_m"]) for r in rows]),
            "mean_leg_penetration": mean([float(r["leg_penetration_frac"]) for r in rows]),
            "mean_body_penetration": mean([float(r["body_penetration_frac"]) for r in rows]),
        }
    return {
        "metric_rows": len(metrics),
        "pair_rows": len(pairs),
        "rubber_outputs_present": sum(str(p["rubber_outputs_present"]).lower() == "true" for p in pairs),
        "ab_status_counts": dict((status, sum(p["ab_status"] == status for p in pairs)) for status in sorted({p["ab_status"] for p in pairs})),
        "method_summary": method_summary,
        "near_thresholds_m": NEAR_THRESHOLDS_M,
        "deep_penetration_m": DEEP_PENETRATION_M,
        "mesh_sample_count": MESH_SAMPLE_COUNT,
    }


def parse_case_identity(case_id: str, person: str) -> dict[str, str]:
    parts = case_id.split("_")
    date = ""
    seq = ""
    if len(parts) >= 5 and parts[0] in {"e091", "d003"}:
        date = parts[2]
        seq = parts[3]
    elif len(parts) >= 4 and parts[0].startswith("bucket"):
        date = parts[1]
        seq = parts[2]
    person_idx = ""
    if person.startswith("person"):
        try:
            person_idx = str(int(person.replace("person", "")) - 1)
        except Exception:
            person_idx = ""
    return {"date": date, "seq": seq, "person_idx": person_idx}


def rubber_gate(row: dict[str, Any]) -> tuple[str, str]:
    if str(row.get("rubber_outputs_present", "")).lower() != "true":
        return "not_run", "missing_rubber_output"
    if str(row.get("rubber_fall_flag", "")).lower() == "true":
        return "fail", "pelvis_fall"
    try:
        if float(row.get("rubber_leg_penetration", "nan")) > 0.05:
            return "fail", "lowerbody_interference"
        if float(row.get("rubber_body_penetration", "nan")) > 0.0:
            return "fail", "body_object_interference"
        if float(row.get("rubber_object_floor_contact", "nan")) > 0.5:
            return "fail", "object_floor_contact"
        if float(row.get("rubber_obj_err_mean_m", "nan")) > 0.10:
            return "fail", "cem_work_status_fail"
    except Exception:
        return "fail", "rubber_metric_parse_error"
    return "pass", ""


def build_evidence_rows(rows: list[dict[str, str]], pairs: list[dict[str, Any]], stage: str, results_dir: Path, metrics_ref: Path) -> list[dict[str, Any]]:
    by_case = {row["case_id"]: row for row in rows}
    out = []
    for pair in pairs:
        src = by_case[pair["case_id"]]
        cem_status, failure = rubber_gate(pair)
        identity = parse_case_identity(pair["case_id"], src.get("person", ""))
        notes = (
            f"ab_status={pair['ab_status']}; "
            f"delta_hand_near_5cm={fmt(pair.get('delta_hand_near_5cm'))}; "
            f"delta_hand_deep_2cm={fmt(pair.get('delta_hand_deep_2cm'))}; "
            f"rubber_fall={pair.get('rubber_fall_flag', '')}; "
            f"mesh_sdf_sample_count={MESH_SAMPLE_COUNT}"
        )
        out.append(
            {
                "case_id": pair["case_id"],
                "object_key": src.get("object_key", ""),
                "object_name": src.get("object_key", ""),
                "date": identity["date"],
                "seq": identity["seq"],
                "person": src.get("person", ""),
                "person_idx": src.get("person_idx", identity["person_idx"]),
                "retarget_variant_id": src.get("retarget_variant_id", ""),
                "target_variant_id": src.get("target_variant_id", "ref_fk"),
                "hand_collision_variant_id": "rubber_hull",
                "candidate_decision": "PASS",
                "handoff_decision": "HANDOFF_READY",
                "cem_status": cem_status,
                "rl_status": "not_run",
                "downstream_decision": "DOWNSTREAM_CEM_PASS" if cem_status == "pass" else ("WAIT_CEM_NOT_RUN" if cem_status == "not_run" else "DOWNSTREAM_CEM_FAIL"),
                "downstream_failure_mode": failure,
                "downstream_notes": notes,
                "cem_run_id": pair["variant"],
                "cem_result_npz": rel(results_dir / f"{pair['variant']}.npz") if cem_status != "not_run" else "",
                "cem_video": rel(results_dir / f"{pair['variant']}_{stage}.mp4") if cem_status != "not_run" else "",
                "cem_metrics_ref": rel(metrics_ref),
                "rl_run_id": "",
                "rl_checkpoint": "",
                "rl_metrics_ref": "",
                "rl_video": "",
            }
        )
    return out


def write_markdown(path: Path, pairs: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# E147 rubber hand collision A/B eval",
        "",
        f"- rubber outputs present: {summary['rubber_outputs_present']}/{summary['pair_rows']}",
        f"- mesh SDF sample count: {MESH_SAMPLE_COUNT}",
        f"- status counts: `{summary['ab_status_counts']}`",
        "",
        "## Method Summary",
        "",
        "| method | rows | fall | hand5 | hand8 | hand pen | hand deep2 | physics contact | pelvis min | obj err | leg pen | body pen |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in summary["method_summary"].items():
        lines.append(
            f"| {method} | {row['rows']} | {row['fall_count']} | {row['mean_hand_near_5cm']:.4f} | "
            f"{row['mean_hand_near_8cm']:.4f} | {row['mean_hand_penetration']:.4f} | {row['mean_hand_deep_2cm']:.4f} | "
            f"{row['mean_physics_contact']:.4f} | {row['mean_pelvis_min_m']:.4f} | {row['mean_obj_err_mean_m']:.4f} | "
            f"{row['mean_leg_penetration']:.4f} | {row['mean_body_penetration']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Pair Delta",
            "",
            "| case | object | old | d hand5 | d deep2 | d pen | d phys | rubber pelvis | rubber fall | status |",
            "|---|---|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in pairs:
        lines.append(
            f"| `{row['case_id']}` | {row['object_key']} | {row['sphere_historical_cem_status']} | "
            f"{float(row['delta_hand_near_5cm']):.4f} | {float(row['delta_hand_deep_2cm']):.4f} | "
            f"{float(row['delta_hand_penetration']):.4f} | {float(row['delta_physics_contact']):.4f} | "
            f"{fmt(row['rubber_pelvis_min_m'])} | {row['rubber_fall_flag']} | {row['ab_status']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="full", choices=("smoke", "full"))
    parser.add_argument("--variants", type=Path, default=VARIANTS_TSV)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--allow-missing-rubber", action="store_true")
    args = parser.parse_args()

    results_dir = args.results_dir or (RESULT_ROOT / "cem" / args.stage)
    out_dir = args.out_dir or (RESULT_ROOT / "eval" / args.stage)
    rows = read_tsv(args.variants)
    metric_rows: list[dict[str, Any]] = []
    missing = []
    for row in rows:
        sphere_qpos = repo_path(row["sphere_outdir_npz"])
        metric_rows.append(
            evaluate_sequence(
                row=row,
                method="sphere historical CEM",
                hand_collision_variant_id="sphere5cm",
                qpos_path=sphere_qpos,
                scene_xml=repo_path(row["base_scene_act"]),
            )
        )
        rubber_qpos, rubber_scene = rubber_paths(row, args.stage, results_dir)
        if not rubber_qpos.is_file():
            missing.append(rel(rubber_qpos))
            continue
        metric_rows.append(
            evaluate_sequence(
                row=row,
                method="rubber_hull CEM",
                hand_collision_variant_id="rubber_hull",
                qpos_path=rubber_qpos,
                scene_xml=rubber_scene,
            )
        )
    if missing and not args.allow_missing_rubber:
        raise FileNotFoundError("missing rubber outputs:\n" + "\n".join(missing))

    pair_rows = build_pair_rows(rows, metric_rows, args.stage, results_dir)
    summary = build_summary(metric_rows, pair_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "e147_method_metrics.tsv"
    write_tsv(metrics_path, metric_rows, METRIC_FIELDS)
    write_tsv(out_dir / "e147_pair_delta.tsv", pair_rows, PAIR_FIELDS)
    write_tsv(out_dir / "e147_downstream_evidence_input.tsv", build_evidence_rows(rows, pair_rows, args.stage, results_dir, metrics_path), EVIDENCE_FIELDS)
    write_json(out_dir / "e147_eval_summary.json", summary)
    write_markdown(out_dir / "e147_eval_summary.md", pair_rows, summary)
    print(f"[E147 eval] wrote {len(metric_rows)} metric rows, missing rubber={len(missing)} to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
