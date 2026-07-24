#!/usr/bin/env python3
"""E175 offline audit of ref targets, proxy coverage, physics pairs and PRG SDF."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import trimesh
import yaml
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.core.core_metrics import (  # noqa: E402
    LOWERBODY_GEOMS,
    geom_object_sdf,
    npz_qpos,
    object_collision_geoms,
    signed_point_box,
)


REPO = Path(__file__).resolve().parents[5]
DEFAULT_MANIFEST = (
    REPO
    / "workspace/core4d/results/E174/s6_downstream/manifests/cem_full_manifest.tsv"
)
DEFAULT_METRICS = (
    REPO
    / "workspace/core4d/results/E174/s6_downstream/eval/full/e174_case_metrics.tsv"
)
DEFAULT_OUT = REPO / "workspace/core4d/results/E175/diagnostics"
HAND_GEOMS = ("lh", "rh")


@dataclass(frozen=True)
class SurfaceHit:
    geom_id: int
    geom_name: str
    signed_distance_m: float
    surface_distance_m: float


def repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.exists():
        return path.resolve()
    text = str(raw)
    for marker in ("example_datasets/", "workspace/", "logs/"):
        if marker in text:
            candidate = REPO / (marker + text.split(marker, 1)[1])
            if candidate.exists():
                return candidate.resolve()
    return path if path.is_absolute() else REPO / path


def rel(raw: str | Path) -> str:
    path = Path(raw)
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except (OSError, ValueError):
        return str(raw)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(
    path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None
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
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ""
                    if row.get(key) is None
                    or (
                        isinstance(row.get(key), float)
                        and not math.isfinite(row[key])
                    )
                    else row.get(key, "")
                    for key in fields
                }
            )


def geom_name(model: mujoco.MjModel, gid: int) -> str:
    return (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        or f"geom:{gid}"
    )


def nearest_object_surface(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    point: np.ndarray,
    object_gids: list[int],
) -> SurfaceHit:
    """Nearest individual box surface, preserving the signed distance."""
    if not object_gids:
        return SurfaceHit(-1, "", math.inf, math.inf)
    hits: list[SurfaceHit] = []
    for gid in object_gids:
        if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            raise ValueError(
                f"non-box object proxy unsupported: {geom_name(model, gid)}"
            )
        signed = signed_point_box(
            np.asarray(point, dtype=np.float64),
            data.geom_xpos[gid].copy(),
            data.geom_xmat[gid].reshape(3, 3).copy(),
            model.geom_size[gid, :3].copy(),
        )
        hits.append(
            SurfaceHit(gid, geom_name(model, gid), signed, abs(signed))
        )
    return min(hits, key=lambda item: item.surface_distance_m)


def paired_object_geom_ids(
    model: mujoco.MjModel,
    robot_gids: list[int],
    object_gids: list[int],
) -> list[int]:
    """Object geoms explicitly paired to any selected robot geom."""
    robot_set = set(robot_gids)
    object_set = set(object_gids)
    paired: set[int] = set()
    for pair_id in range(model.npair):
        gid1 = int(model.pair_geom1[pair_id])
        gid2 = int(model.pair_geom2[pair_id])
        if gid1 in robot_set and gid2 in object_set:
            paired.add(gid2)
        if gid2 in robot_set and gid1 in object_set:
            paired.add(gid1)
    return [gid for gid in object_gids if gid in paired]


def person_idx(case_id: str) -> int:
    return 0 if case_id.lower().endswith("_p1") else 1


def convert_reference_to_scene(
    qpos: np.ndarray, scene_xml: Path, model: mujoco.MjModel
) -> np.ndarray:
    """Convert freejoint-object reference qpos to scene_act 6-DoF layout."""
    if qpos.ndim == 3:
        qpos = qpos[:, 0, :]
    if qpos.shape[1] == model.nq:
        return qpos.astype(np.float64, copy=True)
    nq_robot = model.nq - 6
    if qpos.shape[1] < nq_robot + 7:
        raise ValueError(f"cannot convert qpos {qpos.shape} to nq={model.nq}")
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
    body_pos = model.body_pos[object_id]
    body_quat = model.body_quat[object_id]
    body_rot = Rotation.from_quat(
        [body_quat[1], body_quat[2], body_quat[3], body_quat[0]]
    )
    object_pos = qpos[:, nq_robot : nq_robot + 3]
    object_quat = qpos[:, nq_robot + 3 : nq_robot + 7]
    object_slide = body_rot.inv().apply(object_pos - body_pos[np.newaxis, :])
    object_xyzw = np.column_stack(
        [
            object_quat[:, 1],
            object_quat[:, 2],
            object_quat[:, 3],
            object_quat[:, 0],
        ]
    )
    object_euler = (
        body_rot.inv() * Rotation.from_quat(object_xyzw)
    ).as_euler(convention)
    converted = np.zeros((qpos.shape[0], model.nq), dtype=np.float64)
    converted[:, :nq_robot] = qpos[:, :nq_robot]
    converted[:, nq_robot : nq_robot + 3] = object_slide
    converted[:, nq_robot + 3 : nq_robot + 6] = object_euler
    return converted


def load_reference_qpos(
    trajectory: Path,
    rollout_path: Path,
    scene_xml: Path,
    model: mujoco.MjModel,
) -> tuple[np.ndarray, str]:
    """Load the reference in scene layout, with a CEM-output fallback.

    E174 manifests retain some stale absolute trajectory paths. Every completed
    CEM output stores the already-converted reference in qpos[:, 1, :].
    """
    if trajectory.is_file():
        with np.load(trajectory, allow_pickle=True) as payload:
            raw_ref = np.asarray(payload["qpos"], dtype=np.float64)
        return (
            convert_reference_to_scene(raw_ref, scene_xml, model),
            "manifest_trajectory",
        )

    with np.load(rollout_path, allow_pickle=True) as payload:
        qpos = np.asarray(payload["qpos"], dtype=np.float64)
    if qpos.ndim != 3 or qpos.shape[1] < 2:
        raise ValueError(
            f"{rollout_path} qpos must be (T,>=2,nq), got {qpos.shape}"
        )
    ref = qpos[:, 1, :]
    if ref.shape[1] != model.nq:
        raise ValueError(
            f"{rollout_path} reference nq={ref.shape[1]} != model.nq={model.nq}"
        )
    return ref.copy(), "cem_outdir_reference_channel"


def resize_time_nearest(values: np.ndarray, target_len: int) -> np.ndarray:
    """Match run_mjwp's nearest-neighbor time-axis resize."""
    if values.shape[0] == target_len:
        return values.copy()
    if values.shape[0] <= 0:
        raise ValueError("cannot resize a zero-frame array")
    indices = np.round(
        np.linspace(0, values.shape[0] - 1, target_len)
    ).astype(np.int64)
    return values[indices]


def ref_fk_contact_points(
    model: mujoco.MjModel,
    ref_qpos: np.ndarray,
    hand_body_names: list[str],
    eef_offset: np.ndarray,
    *,
    uses_eef_offset: bool,
) -> np.ndarray:
    """Reconstruct the world-space contact targets used by ref_fk.

    run_mjwp stores these points object-locally and reconstructs them under the
    simulated object pose. Evaluated at the reference object pose, the target
    is exactly the reference wrist/contact point returned here.
    """
    hand_body_ids: list[int] = []
    for name in hand_body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"missing ref_fk hand body: {name}")
        hand_body_ids.append(int(bid))
    if len(hand_body_ids) != 2:
        raise ValueError(
            f"expected two ref_fk hand bodies, got {hand_body_names}"
        )

    offset = np.asarray(eef_offset, dtype=np.float64)
    if offset.shape != (3,):
        raise ValueError(f"eef offset must be shape (3,), got {offset.shape}")
    data = mujoco.MjData(model)
    output = np.zeros((len(ref_qpos), 2, 3), dtype=np.float64)
    for frame, qpos in enumerate(ref_qpos):
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for hand_idx, bid in enumerate(hand_body_ids):
            point = data.xpos[bid].copy()
            if uses_eef_offset:
                quat = data.xquat[bid]
                rotation = Rotation.from_quat(
                    [quat[1], quat[2], quat[3], quat[0]]
                )
                point += rotation.apply(offset)
            output[frame, hand_idx] = point
    return output


def add_visual_mesh_distances(
    model: mujoco.MjModel,
    contact_rows: list[dict[str, Any]],
) -> None:
    """Annotate contact targets with exact distance to object_visual triangles."""
    if not contact_rows:
        return
    visual_gid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "object_visual"
    )
    if visual_gid < 0:
        raise ValueError("missing object_visual geom")
    if int(model.geom_type[visual_gid]) != int(mujoco.mjtGeom.mjGEOM_MESH):
        raise ValueError("object_visual must be a mesh")
    mesh_id = int(model.geom_dataid[visual_gid])
    vert_start = int(model.mesh_vertadr[mesh_id])
    vert_count = int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])
    vertices = np.asarray(
        model.mesh_vert[vert_start : vert_start + vert_count],
        dtype=np.float64,
    )
    faces = np.asarray(
        model.mesh_face[face_start : face_start + face_count],
        dtype=np.int64,
    )
    quat = model.geom_quat[visual_gid]
    rotation = Rotation.from_quat(
        [quat[1], quat[2], quat[3], quat[0]]
    )
    vertices = rotation.apply(vertices) + model.geom_pos[visual_gid]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    points = np.asarray(
        [
            [
                row["target_object_local_x"],
                row["target_object_local_y"],
                row["target_object_local_z"],
            ]
            for row in contact_rows
        ],
        dtype=np.float64,
    )
    closest, distances, _ = trimesh.proximity.closest_point(mesh, points)
    for row, point, distance in zip(contact_rows, closest, distances):
        mesh_near = bool(distance <= 0.03)
        proxy_far = bool(row["full_surface_m"] > 0.03)
        proxy_minus_mesh = float(row["full_surface_m"] - distance)
        row.update(
            {
                "mesh_surface_m": float(distance),
                "mesh_near_3cm": mesh_near,
                "mesh_miss_3cm": not mesh_near,
                "full_proxy_gap_when_mesh_near_3cm": (
                    mesh_near and proxy_far
                ),
                "full_proxy_minus_mesh_surface_m": proxy_minus_mesh,
                "full_proxy_undercoverage_gt_3cm": (
                    proxy_minus_mesh > 0.03
                ),
                "full_proxy_overcoverage_gt_3cm": (
                    proxy_minus_mesh < -0.03
                ),
                "full_proxy_mesh_abs_error_gt_3cm": (
                    abs(proxy_minus_mesh) > 0.03
                ),
                "mesh_closest_object_local_x": float(point[0]),
                "mesh_closest_object_local_y": float(point[1]),
                "mesh_closest_object_local_z": float(point[2]),
            }
        )


def percentile(values: list[float], q: float) -> float:
    return (
        float(np.percentile(np.asarray(values, dtype=np.float64), q))
        if values
        else math.nan
    )


def fraction(values: list[float], predicate) -> float:
    return (
        float(np.mean([bool(predicate(value)) for value in values]))
        if values
        else math.nan
    )


def contact_target_rows(
    *,
    row: dict[str, str],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ref_qpos: np.ndarray,
    contact_pos: np.ndarray,
    active_mask: np.ndarray,
    all_gids: list[int],
    physics_gids: list[int],
    prg_gids: list[int],
) -> list[dict[str, Any]]:
    object_bid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "object"
    )
    if object_bid < 0:
        raise ValueError("scene has no object body")
    count = min(
        len(ref_qpos), len(contact_pos), len(active_mask)
    )
    output: list[dict[str, Any]] = []
    for frame in range(count):
        data.qpos[:] = ref_qpos[frame]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for hand_idx, hand in enumerate(("left", "right")):
            if not bool(active_mask[frame, hand_idx]):
                continue
            point = np.asarray(contact_pos[frame, hand_idx], dtype=np.float64)
            object_pos = data.xpos[object_bid]
            object_mat = data.xmat[object_bid].reshape(3, 3)
            point_local = object_mat.T @ (point - object_pos)
            full = nearest_object_surface(model, data, point, all_gids)
            physics = nearest_object_surface(
                model, data, point, physics_gids
            )
            prg = nearest_object_surface(model, data, point, prg_gids)
            physics_minus_full = (
                physics.surface_distance_m - full.surface_distance_m
            )
            prg_minus_full = (
                prg.surface_distance_m - full.surface_distance_m
            )
            output.append(
                {
                    "case_id": row["case_id"],
                    "object_key": row["object_key"],
                    "frame": frame,
                    "hand": hand,
                    "target_x": point[0],
                    "target_y": point[1],
                    "target_z": point[2],
                    "target_object_local_x": point_local[0],
                    "target_object_local_y": point_local[1],
                    "target_object_local_z": point_local[2],
                    "full_geom": full.geom_name,
                    "full_signed_m": full.signed_distance_m,
                    "full_surface_m": full.surface_distance_m,
                    "full_miss_3cm": full.surface_distance_m > 0.03,
                    "full_miss_5cm": full.surface_distance_m > 0.05,
                    "physics_geom": physics.geom_name,
                    "physics_signed_m": physics.signed_distance_m,
                    "physics_surface_m": physics.surface_distance_m,
                    "physics_miss_3cm": physics.surface_distance_m > 0.03,
                    "physics_miss_5cm": physics.surface_distance_m > 0.05,
                    "prg_geom": prg.geom_name,
                    "prg_signed_m": prg.signed_distance_m,
                    "prg_surface_m": prg.surface_distance_m,
                    "prg_miss_3cm": prg.surface_distance_m > 0.03,
                    "prg_miss_5cm": prg.surface_distance_m > 0.05,
                    "physics_minus_full_surface_m": physics_minus_full,
                    "prg_minus_full_surface_m": prg_minus_full,
                    "physics_coverage_loss_gt_3cm": (
                        physics_minus_full > 0.03
                    ),
                    "prg_coverage_loss_gt_3cm": prg_minus_full > 0.03,
                    "physics_blind_vs_full": (
                        full.surface_distance_m <= 0.03
                        and physics.surface_distance_m > 0.03
                    ),
                    "prg_blind_vs_full": (
                        full.surface_distance_m <= 0.03
                        and prg.surface_distance_m > 0.03
                    ),
                }
            )
    return output


def lower_body_frame_rows(
    *,
    row: dict[str, str],
    source: str,
    model: mujoco.MjModel,
    qpos: np.ndarray,
    lower_gids: list[int],
    all_gids: list[int],
    physics_gids: list[int],
    prg_gids: list[int],
) -> list[dict[str, Any]]:
    data = mujoco.MjData(model)
    output: list[dict[str, Any]] = []
    sets = {
        "full": all_gids,
        "physics": physics_gids,
        "prg": prg_gids,
    }
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        record: dict[str, Any] = {
            "case_id": row["case_id"],
            "object_key": row["object_key"],
            "source": source,
            "frame": frame,
        }
        for label, object_gids in sets.items():
            candidates: list[tuple[float, int, int]] = []
            for lower_gid in lower_gids:
                for object_gid in object_gids:
                    candidates.append(
                        (
                            geom_object_sdf(
                                model, data, lower_gid, [object_gid]
                            ),
                            lower_gid,
                            object_gid,
                        )
                    )
            if candidates:
                value, lower_gid, object_gid = min(
                    candidates, key=lambda item: item[0]
                )
                record[f"{label}_min_sdf_m"] = value
                record[f"{label}_lower_geom"] = geom_name(
                    model, lower_gid
                )
                record[f"{label}_object_geom"] = geom_name(
                    model, object_gid
                )
            else:
                record[f"{label}_min_sdf_m"] = math.inf
                record[f"{label}_lower_geom"] = ""
                record[f"{label}_object_geom"] = ""
        record["physics_false_negative"] = (
            record["full_min_sdf_m"] < 0.0
            and record["physics_min_sdf_m"] >= 0.0
        )
        record["prg_false_negative"] = (
            record["full_min_sdf_m"] < 0.0
            and record["prg_min_sdf_m"] >= 0.0
        )
        output.append(record)
    return output


def summarize_case(
    row: dict[str, str],
    metric: dict[str, str],
    contact_rows: list[dict[str, Any]],
    lower_rows: list[dict[str, Any]],
    *,
    all_gids: list[int],
    hand_physics_gids: list[int],
    lower_physics_gids: list[int],
    prg_gids: list[int],
    model: mujoco.MjModel,
    ref_qpos_source: str,
    contact_target_source: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "case_id": row["case_id"],
        "object_key": row["object_key"],
        "numeric_release_pass": metric.get("numeric_release_pass", ""),
        "numeric_failure_modes": metric.get("numeric_failure_modes", ""),
        "ref_qpos_source": ref_qpos_source,
        "contact_target_source": contact_target_source,
        "full_proxy_geom_count": len(all_gids),
        "full_proxy_geoms": ",".join(geom_name(model, gid) for gid in all_gids),
        "hand_physics_geom_count": len(hand_physics_gids),
        "hand_physics_geoms": ",".join(
            geom_name(model, gid) for gid in hand_physics_gids
        ),
        "lower_physics_geom_count": len(lower_physics_gids),
        "lower_physics_geoms": ",".join(
            geom_name(model, gid) for gid in lower_physics_gids
        ),
        "prg_geom_count": len(prg_gids),
        "prg_geoms": ",".join(geom_name(model, gid) for gid in prg_gids),
        "active_contact_targets": len(contact_rows),
    }
    mesh_distances = [
        float(item["mesh_surface_m"]) for item in contact_rows
    ]
    output.update(
        {
            "contact_mesh_surface_p50_m": percentile(mesh_distances, 50),
            "contact_mesh_surface_p90_m": percentile(mesh_distances, 90),
            "contact_mesh_miss_3cm_frac": fraction(
                mesh_distances, lambda value: value > 0.03
            ),
            "contact_full_proxy_gap_when_mesh_near_3cm_frac": fraction(
                [
                    float(item["full_proxy_gap_when_mesh_near_3cm"])
                    for item in contact_rows
                ],
                lambda value: value > 0.5,
            ),
            "contact_full_proxy_undercoverage_gt_3cm_frac": fraction(
                [
                    float(item["full_proxy_undercoverage_gt_3cm"])
                    for item in contact_rows
                ],
                lambda value: value > 0.5,
            ),
            "contact_full_proxy_overcoverage_gt_3cm_frac": fraction(
                [
                    float(item["full_proxy_overcoverage_gt_3cm"])
                    for item in contact_rows
                ],
                lambda value: value > 0.5,
            ),
            "contact_full_proxy_mesh_abs_error_gt_3cm_frac": fraction(
                [
                    float(item["full_proxy_mesh_abs_error_gt_3cm"])
                    for item in contact_rows
                ],
                lambda value: value > 0.5,
            ),
        }
    )
    for label in ("full", "physics", "prg"):
        distances = [
            float(item[f"{label}_surface_m"]) for item in contact_rows
        ]
        output.update(
            {
                f"contact_{label}_surface_p50_m": percentile(distances, 50),
                f"contact_{label}_surface_p90_m": percentile(distances, 90),
                f"contact_{label}_surface_max_m": max(distances)
                if distances
                else math.nan,
                f"contact_{label}_miss_3cm_frac": fraction(
                    distances, lambda value: value > 0.03
                ),
                f"contact_{label}_miss_5cm_frac": fraction(
                    distances, lambda value: value > 0.05
                ),
            }
        )
    for label in ("physics", "prg"):
        output[f"contact_{label}_blind_vs_full_frac"] = fraction(
            [
                float(item[f"{label}_blind_vs_full"])
                for item in contact_rows
            ],
            lambda value: value > 0.5,
        )
        output[f"contact_{label}_coverage_loss_gt_3cm_frac"] = fraction(
            [
                float(item[f"{label}_coverage_loss_gt_3cm"])
                for item in contact_rows
            ],
            lambda value: value > 0.5,
        )
    for source in ("ref", "rollout"):
        subset = [item for item in lower_rows if item["source"] == source]
        for label in ("full", "physics", "prg"):
            values = [
                float(item[f"{label}_min_sdf_m"]) for item in subset
            ]
            output[f"{source}_{label}_min_sdf_m"] = (
                min(values) if values else math.nan
            )
            output[f"{source}_{label}_penetration_frac"] = fraction(
                values, lambda value: value < 0.0
            )
        output[f"{source}_physics_false_negative_frac"] = fraction(
            [float(item["physics_false_negative"]) for item in subset],
            lambda value: value > 0.5,
        )
        output[f"{source}_prg_false_negative_frac"] = fraction(
            [float(item["prg_false_negative"]) for item in subset],
            lambda value: value > 0.5,
        )
    return output


def summarize_objects(
    cases: list[dict[str, Any]],
    contact_rows: list[dict[str, Any]],
    lower_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    object_keys = sorted({str(row["object_key"]) for row in cases})
    for object_key in object_keys:
        case_subset = [
            row for row in cases if row["object_key"] == object_key
        ]
        contacts = [
            row for row in contact_rows if row["object_key"] == object_key
        ]
        lowers = [
            row for row in lower_rows if row["object_key"] == object_key
        ]
        record: dict[str, Any] = {
            "object_key": object_key,
            "cases": len(case_subset),
            "numeric_pass": sum(
                str(row["numeric_release_pass"]).lower() == "true"
                for row in case_subset
            ),
            "active_contact_targets": len(contacts),
        }
        mesh_distances = [
            float(row["mesh_surface_m"]) for row in contacts
        ]
        record["contact_mesh_miss_3cm_frac"] = fraction(
            mesh_distances, lambda value: value > 0.03
        )
        record["contact_full_proxy_gap_when_mesh_near_3cm_frac"] = fraction(
            [
                float(row["full_proxy_gap_when_mesh_near_3cm"])
                for row in contacts
            ],
            lambda value: value > 0.5,
        )
        for field in (
            "full_proxy_undercoverage_gt_3cm",
            "full_proxy_overcoverage_gt_3cm",
            "full_proxy_mesh_abs_error_gt_3cm",
        ):
            record[f"contact_{field}_frac"] = fraction(
                [float(row[field]) for row in contacts],
                lambda value: value > 0.5,
            )
        for label in ("full", "physics", "prg"):
            values = [
                float(row[f"{label}_surface_m"]) for row in contacts
            ]
            record[f"contact_{label}_miss_3cm_frac"] = fraction(
                values, lambda value: value > 0.03
            )
            record[f"contact_{label}_miss_5cm_frac"] = fraction(
                values, lambda value: value > 0.05
            )
        for label in ("physics", "prg"):
            record[f"contact_{label}_blind_vs_full_frac"] = fraction(
                [
                    float(row[f"{label}_blind_vs_full"])
                    for row in contacts
                ],
                lambda value: value > 0.5,
            )
            record[
                f"contact_{label}_coverage_loss_gt_3cm_frac"
            ] = fraction(
                [
                    float(row[f"{label}_coverage_loss_gt_3cm"])
                    for row in contacts
                ],
                lambda value: value > 0.5,
            )
        for source in ("ref", "rollout"):
            source_rows = [
                row for row in lowers if row["source"] == source
            ]
            for label in ("full", "physics", "prg"):
                values = [
                    float(row[f"{label}_min_sdf_m"])
                    for row in source_rows
                ]
                record[f"{source}_{label}_penetration_frac"] = fraction(
                    values, lambda value: value < 0.0
                )
            record[f"{source}_physics_false_negative_frac"] = fraction(
                [
                    float(row["physics_false_negative"])
                    for row in source_rows
                ],
                lambda value: value > 0.5,
            )
            record[f"{source}_prg_false_negative_frac"] = fraction(
                [
                    float(row["prg_false_negative"])
                    for row in source_rows
                ],
                lambda value: value > 0.5,
            )
        output.append(record)
    return output


def markdown_summary(
    case_rows: list[dict[str, Any]],
    object_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# E175 proxy fidelity diagnostic summary",
        "",
        "_E174 39-case offline audit · full proxy vs physics-paired proxy vs PRG proxy_",
        "",
        "---",
        "",
        "## 📊 Object summary",
        "",
        "| Object | Cases | Pass | Full proxy miss >3cm | Proxy undercoverage >3cm | Proxy overcoverage >3cm | Physics miss >3cm | Ref full pen | Ref physics FN | Rollout full pen |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in object_rows:
        lines.append(
            "| {object_key} | {cases} | {numeric_pass} | "
            "{contact_full_miss_3cm_frac:.3f} | "
            "{contact_full_proxy_undercoverage_gt_3cm_frac:.3f} | "
            "{contact_full_proxy_overcoverage_gt_3cm_frac:.3f} | "
            "{contact_physics_miss_3cm_frac:.3f} | "
            "{ref_full_penetration_frac:.3f} | "
            "{ref_physics_false_negative_frac:.3f} | "
            "{rollout_full_penetration_frac:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## 🔍 Contract finding",
            "",
            f"- Audited cases: `{len(case_rows)}`",
            "- `full` means every `object_collision*` box in XML",
            "- `physics` means object geoms explicitly paired to E174 hand/lower-body geoms",
            "- `prg` means the historical exact `object_collision` surrogate",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    manifest_path: Path,
    metrics_path: Path,
    out_dir: Path,
    *,
    allow_partial: bool,
) -> dict[str, Any]:
    manifest = read_tsv(manifest_path)
    metrics = {
        row["case_id"]: row for row in read_tsv(metrics_path)
    }
    if not allow_partial and len(manifest) != 39:
        raise ValueError(f"expected 39 E174 rows, got {len(manifest)}")
    case_rows: list[dict[str, Any]] = []
    all_contact_rows: list[dict[str, Any]] = []
    all_lower_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, row in enumerate(manifest, 1):
        case_id = row["case_id"]
        try:
            scene = repo_path(row["scene_act"])
            trajectory = repo_path(row["trajectory"])
            mask_path = repo_path(row["contact_mask"])
            rollout_path = repo_path(row["outdir_npz"])
            config_path = repo_path(row["config_act"])
            model = mujoco.MjModel.from_xml_path(str(scene))
            data = mujoco.MjData(model)
            all_gids = object_collision_geoms(model)
            primary_gid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision"
            )
            if primary_gid < 0:
                raise ValueError("missing exact object_collision geom")
            prg_gids = [int(primary_gid)]
            lower_gids = [
                gid
                for name in LOWERBODY_GEOMS
                if (
                    gid := mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_GEOM, name
                    )
                )
                >= 0
            ]
            hand_gids = [
                gid
                for name in HAND_GEOMS
                if (
                    gid := mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_GEOM, name
                    )
                )
                >= 0
            ]
            hand_physics_gids = paired_object_geom_ids(
                model, hand_gids, all_gids
            )
            lower_physics_gids = paired_object_geom_ids(
                model, lower_gids, all_gids
            )
            ref_qpos, ref_qpos_source = load_reference_qpos(
                trajectory, rollout_path, scene, model
            )
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if not mask_path.is_file():
                mask_path = repo_path(
                    str(config.get("contact_hdmi_mask_path", ""))
                )
            if not bool(config.get("contact_hdmi_dynamic_target", False)):
                raise ValueError("expected contact_hdmi_dynamic_target=true")
            target_source = str(
                config.get("contact_hdmi_target_source", "")
            )
            if target_source != "ref_fk":
                raise ValueError(
                    f"expected ref_fk contact target, got {target_source!r}"
                )
            hand_body_names = list(
                config.get(
                    "hand_approach_body_names",
                    ["left_wrist_yaw_link", "right_wrist_yaw_link"],
                )
            )
            eef_offset = np.asarray(
                config.get("contact_hdmi_eef_offset", [0.05, 0.0, 0.0]),
                dtype=np.float64,
            )
            uses_eef_offset = bool(
                config.get("contact_hdmi_target_uses_eef_offset", False)
            )
            contact_pos = ref_fk_contact_points(
                model,
                ref_qpos,
                hand_body_names,
                eef_offset,
                uses_eef_offset=uses_eef_offset,
            )
            with np.load(mask_path, allow_pickle=True) as payload:
                masks = np.asarray(
                    payload["spider_contact_mask_3cm"], dtype=np.bool_
                )
            active_mask = resize_time_nearest(
                masks[:, person_idx(case_id), :], len(ref_qpos)
            ).astype(np.bool_)
            contacts = contact_target_rows(
                row=row,
                model=model,
                data=data,
                ref_qpos=ref_qpos,
                contact_pos=contact_pos,
                active_mask=active_mask,
                all_gids=all_gids,
                physics_gids=hand_physics_gids,
                prg_gids=prg_gids,
            )
            add_visual_mesh_distances(model, contacts)
            rollout_qpos, _ = npz_qpos(rollout_path)
            lower_ref = lower_body_frame_rows(
                row=row,
                source="ref",
                model=model,
                qpos=ref_qpos,
                lower_gids=lower_gids,
                all_gids=all_gids,
                physics_gids=lower_physics_gids,
                prg_gids=prg_gids,
            )
            lower_rollout = lower_body_frame_rows(
                row=row,
                source="rollout",
                model=model,
                qpos=rollout_qpos,
                lower_gids=lower_gids,
                all_gids=all_gids,
                physics_gids=lower_physics_gids,
                prg_gids=prg_gids,
            )
            lower = lower_ref + lower_rollout
            case_rows.append(
                summarize_case(
                    row,
                    metrics.get(case_id, {}),
                    contacts,
                    lower,
                    all_gids=all_gids,
                    hand_physics_gids=hand_physics_gids,
                    lower_physics_gids=lower_physics_gids,
                    prg_gids=prg_gids,
                    model=model,
                    ref_qpos_source=ref_qpos_source,
                    contact_target_source=(
                        "ref_fk_eef_offset"
                        if uses_eef_offset
                        else "ref_fk_wrist_origin"
                    ),
                )
            )
            all_contact_rows.extend(contacts)
            all_lower_rows.extend(lower)
            print(f"[{index:02d}/{len(manifest):02d}] {case_id}: PASS")
        except Exception as exc:
            errors.append(
                {
                    "case_id": case_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"[{index:02d}/{len(manifest):02d}] {case_id}: {exc}")
    object_rows = summarize_objects(
        case_rows, all_contact_rows, all_lower_rows
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(out_dir / "case_proxy_fidelity.tsv", case_rows)
    write_tsv(out_dir / "contact_proxy_hits.tsv", all_contact_rows)
    write_tsv(out_dir / "lower_body_proxy_sdf.tsv", all_lower_rows)
    write_tsv(out_dir / "object_summary.tsv", object_rows)
    write_tsv(out_dir / "errors.tsv", errors)
    summary = {
        "source_experiment": "E174",
        "experiment": "E175",
        "manifest": rel(manifest_path),
        "metrics": rel(metrics_path),
        "expected_cases": len(manifest),
        "evaluated_cases": len(case_rows),
        "error_cases": len(errors),
        "contact_target_rows": len(all_contact_rows),
        "lower_body_frame_rows": len(all_lower_rows),
        "ref_qpos_source_counts": {
            source: sum(
                row["ref_qpos_source"] == source for row in case_rows
            )
            for source in sorted(
                {str(row["ref_qpos_source"]) for row in case_rows}
            )
        },
        "object_summary": object_rows,
    }
    (out_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "diagnostic_summary.md").write_text(
        markdown_summary(case_rows, object_rows),
        encoding="utf-8",
    )
    if errors and not allow_partial:
        raise RuntimeError(f"E175 diagnostics had {len(errors)} errors")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(
        args.manifest,
        args.metrics,
        args.out_dir,
        allow_partial=args.allow_partial,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
