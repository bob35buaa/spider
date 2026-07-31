#!/usr/bin/env python3
"""Build E182 object-local P/R/G query tensors from frozen dev3 trajectories."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from audit_query_tape_replay import chunk_manifest_path, resolve_chunk_path
from e182_common import atomic_json, relative_to_repo, sha256_file
from run_query_tape_replay import DEFAULT_RESULT_ROOT, load_dev_rows
from scipy.spatial.transform import Rotation

from spider.config import Config, filter_config_fields, load_config_yaml
from spider.io import load_data

DEFAULT_OUTPUT_ROOT = DEFAULT_RESULT_ROOT / "prg_query_tape"
MESH_SDF_SAMPLE_COUNT = 800


@dataclass
class CaseContext:
    """Resolved model, configuration, and consumer geometry for one dev case."""

    row: dict[str, str]
    config: Config
    model: mujoco.MjModel
    data: mujoco.MjData
    object_body_id: int
    raw_reference_qpos: np.ndarray
    consumer_names: list[str]
    consumer_geom_ids: dict[str, list[int]]
    point_geom_id: np.ndarray
    point_radius_m: np.ndarray
    point_offsets_geom_local: np.ndarray
    point_geom_column: np.ndarray
    point_consumer_mask: np.ndarray
    query_geom_ids: np.ndarray
    point_slices: dict[int, slice]


def _is_descendant(model: mujoco.MjModel, body_id: int, ancestor_id: int) -> bool:
    """Return whether one body belongs to an ancestor subtree."""
    current = body_id
    while current > 0:
        if current == ancestor_id:
            return True
        current = int(model.body_parentid[current])
    return current == ancestor_id


def _physics_robot_geom_ids(model: mujoco.MjModel, object_body_id: int) -> list[int]:
    """Resolve collision-enabled non-world, non-object geoms for P pose replay."""
    object_geoms = {
        geom_id
        for geom_id in range(model.ngeom)
        if _is_descendant(model, int(model.geom_bodyid[geom_id]), object_body_id)
    }
    explicit_pair_geoms: set[int] = set()
    for pair_index in range(model.npair):
        left = int(model.pair_geom1[pair_index])
        right = int(model.pair_geom2[pair_index])
        if left in object_geoms and right not in object_geoms:
            explicit_pair_geoms.add(right)
        if right in object_geoms and left not in object_geoms:
            explicit_pair_geoms.add(left)
    ids = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        if body_id == 0 or _is_descendant(model, body_id, object_body_id):
            continue
        collision_mask_enabled = not (
            int(model.geom_contype[geom_id]) == 0
            and int(model.geom_conaffinity[geom_id]) == 0
        )
        if geom_id not in explicit_pair_geoms and not collision_mask_enabled:
            continue
        ids.append(geom_id)
    return ids


def _consumer_geom_ids(
    config: Config, model: mujoco.MjModel, object_body_id: int
) -> dict[str, list[int]]:
    """Mirror the enabled MJWP P/R/G consumer groups in a stable order."""
    consumers: dict[str, list[int]] = {
        "P_collision": _physics_robot_geom_ids(model, object_body_id)
    }

    def add(name: str, enabled: bool, values: list[int]) -> None:
        if enabled and values:
            consumers[name] = [int(value) for value in values]

    add(
        "R_robot_penalty",
        config.robot_object_penalty_scale > 0.0,
        config.robot_object_penalty_geom_ids,
    )
    add(
        "R_leg_penalty",
        config.leg_object_penalty_scale > 0.0,
        config.leg_object_penalty_geom_ids,
    )
    add(
        "R_hand_deep",
        config.hand_object_deep_penalty_scale > 0.0,
        config.hand_object_deep_penalty_geom_ids,
    )
    add(
        "R_hand_support",
        config.hand_support_rew_scale > 0.0,
        config.hand_support_geom_ids,
    )
    surface_enabled = (
        config.surface_band_rew_scale > 0.0 or config.surface_band_penalty_scale > 0.0
    ) and bool(config.surface_band_geom_ids)
    if surface_enabled and config.surface_band_bimanual_required:
        add("R_surface_band_left", True, config.surface_band_left_geom_ids)
        add("R_surface_band_right", True, config.surface_band_right_geom_ids)
    else:
        add("R_surface_band", surface_enabled, config.surface_band_geom_ids)
    add(
        "R_nonhand_support",
        config.nonhand_support_penalty_scale > 0.0,
        config.nonhand_support_penalty_geom_ids,
    )
    add(
        "R_carry_corridor",
        config.carry_corridor_rew_scale > 0.0,
        config.carry_corridor_leg_geom_ids,
    )
    add("G_safety", config.cem_safety_gate_enabled, config.cem_safety_gate_geom_ids)
    add("G_hand", config.cem_hand_gate_enabled, config.cem_hand_gate_geom_ids)
    add("G_leg", config.cem_leg_gate_enabled, config.cem_leg_gate_geom_ids)
    return consumers


def _point_inventory(
    model: mujoco.MjModel,
    consumers: dict[str, list[int]],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[int, slice],
]:
    """Build the exact primitive/mesh point sampling used by MJWP union SDF."""
    consumer_names = list(consumers)
    geom_ids = list(
        dict.fromkeys(
            gid
            for name, ids in consumers.items()
            if name != "P_collision"
            for gid in ids
        )
    )
    point_geom_ids: list[int] = []
    point_geom_columns: list[int] = []
    point_radius: list[float] = []
    offsets: list[np.ndarray] = []
    slices: dict[int, slice] = {}
    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    capsule_type = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    for geom_column, geom_id in enumerate(geom_ids):
        start = len(point_geom_ids)
        if int(model.geom_type[geom_id]) == mesh_type:
            mesh_id = int(model.geom_dataid[geom_id])
            vert_start = int(model.mesh_vertadr[mesh_id])
            vert_count = int(model.mesh_vertnum[mesh_id])
            verts = np.asarray(
                model.mesh_vert[vert_start : vert_start + vert_count],
                dtype=np.float64,
            ).reshape(-1, 3)
            if vert_count > MESH_SDF_SAMPLE_COUNT:
                indices = np.linspace(0, vert_count - 1, MESH_SDF_SAMPLE_COUNT).astype(
                    int
                )
                verts = verts[indices]
            geom_offsets = verts
            radii = np.zeros(len(verts), dtype=np.float64)
        else:
            half_len = (
                float(model.geom_size[geom_id, 1])
                if int(model.geom_type[geom_id]) == capsule_type
                else 0.0
            )
            geom_offsets = np.asarray(
                [[0.0, 0.0, -half_len], [0.0, 0.0, 0.0], [0.0, 0.0, half_len]],
                dtype=np.float64,
            )
            radii = np.full(3, float(model.geom_size[geom_id, 0]), dtype=np.float64)
        point_geom_ids.extend([geom_id] * len(geom_offsets))
        point_geom_columns.extend([geom_column] * len(geom_offsets))
        point_radius.extend(radii.tolist())
        offsets.extend(geom_offsets)
        slices[geom_id] = slice(start, len(point_geom_ids))
    mask = np.zeros((len(point_geom_ids), len(consumer_names)), dtype=bool)
    for consumer_index, name in enumerate(consumer_names):
        if name == "P_collision":
            continue
        for geom_id in consumers[name]:
            mask[slices[geom_id], consumer_index] = True
    return (
        np.asarray(point_geom_ids, dtype=np.int32),
        np.asarray(point_radius, dtype=np.float32),
        np.asarray(offsets, dtype=np.float64),
        np.asarray(point_geom_columns, dtype=np.int32),
        mask,
        np.asarray(geom_ids, dtype=np.int32),
        slices,
    )


def load_case_context(row: dict[str, str]) -> CaseContext:
    """Load one frozen E178 config and its effective scene without absolute-path reuse."""
    config_values = filter_config_fields(load_config_yaml(row["e178_config_act"]))
    config_values["device"] = "cpu"
    config_values["model_path"] = row["source_e178_scene_act"]
    config_values["data_path"] = row["trajectory"]
    config = Config(**config_values)
    model = mujoco.MjModel.from_xml_path(config.model_path)
    if model.nq != config.nq:
        raise RuntimeError(f"{row['case_id']}: config/model nq mismatch")
    data = mujoco.MjData(model)
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if object_body_id < 0:
        raise RuntimeError(f"{row['case_id']}: object body is missing")
    raw_reference = load_data(config, config.data_path)[0].detach().cpu().numpy()
    consumers = _consumer_geom_ids(config, model, object_body_id)
    inventory = _point_inventory(model, consumers)
    return CaseContext(
        row=row,
        config=config,
        model=model,
        data=data,
        object_body_id=object_body_id,
        raw_reference_qpos=np.asarray(raw_reference, dtype=np.float64),
        consumer_names=list(consumers),
        consumer_geom_ids=consumers,
        point_geom_id=inventory[0],
        point_radius_m=inventory[1],
        point_offsets_geom_local=inventory[2],
        point_geom_column=inventory[3],
        point_consumer_mask=inventory[4],
        query_geom_ids=inventory[5],
        point_slices=inventory[6],
    )


def load_reference_qpos(row: dict[str, str], context: CaseContext) -> np.ndarray:
    """Apply the exact E027b freejoint-to-scene-act reference conversion."""
    qpos = context.raw_reference_qpos
    if qpos.shape[1] == context.model.nq:
        return qpos.copy()
    if qpos.shape[1] != context.model.nq + 1 or context.model.nq < 6:
        raise RuntimeError(
            f"{row['case_id']}: unsupported reference nq {qpos.shape[1]}"
        )
    nq_robot = context.model.nq - 6
    object_pos_world = qpos[:, nq_robot : nq_robot + 3]
    object_quat_wxyz = qpos[:, nq_robot + 3 : nq_robot + 7]
    body_pos = context.model.body_pos[context.object_body_id]
    body_quat_wxyz = context.model.body_quat[context.object_body_id]
    body_rotation = Rotation.from_quat(
        [
            body_quat_wxyz[1],
            body_quat_wxyz[2],
            body_quat_wxyz[3],
            body_quat_wxyz[0],
        ]
    )
    object_slide = body_rotation.inv().apply(object_pos_world - body_pos[None, :])
    object_rotation = Rotation.from_quat(
        np.column_stack(
            [
                object_quat_wxyz[:, 1],
                object_quat_wxyz[:, 2],
                object_quat_wxyz[:, 3],
                object_quat_wxyz[:, 0],
            ]
        )
    )
    meta_path = Path(context.config.model_path).parent / "scene_act_meta.json"
    convention = "XYZ"
    if meta_path.is_file():
        convention = json.loads(meta_path.read_text(encoding="utf-8"))[
            "euler_convention"
        ]
    object_euler = (body_rotation.inv() * object_rotation).as_euler(convention)
    converted = np.zeros((qpos.shape[0], context.model.nq), dtype=np.float64)
    converted[:, :nq_robot] = qpos[:, :nq_robot]
    converted[:, nq_robot : nq_robot + 3] = object_slide
    converted[:, nq_robot + 3 : nq_robot + 6] = object_euler
    return converted


def load_final_qpos(row: dict[str, str], context: CaseContext) -> np.ndarray:
    """Flatten the exact E178 physics trajectory record/ctrl-step axes."""
    with np.load(row["e178_result_npz"], allow_pickle=True) as values:
        qpos = np.asarray(values["qpos"], dtype=np.float64)
    if qpos.shape[-1] != context.model.nq:
        raise RuntimeError(f"{row['case_id']}: E178 final nq mismatch")
    return qpos.reshape(-1, context.model.nq)


def extract_query_chunk(
    context: CaseContext, qpos: np.ndarray
) -> dict[str, np.ndarray]:
    """Factor R/G query coordinates into geom poses and fixed local offsets."""
    original_shape = np.asarray(qpos).shape
    poses = np.asarray(qpos, dtype=np.float64).reshape(-1, context.model.nq)
    geom_positions = np.empty(
        (len(poses), len(context.query_geom_ids), 3), dtype=np.float32
    )
    geom_matrices = np.empty(
        (len(poses), len(context.query_geom_ids), 3, 3), dtype=np.float32
    )
    for pose_index, pose in enumerate(poses):
        context.data.qpos[:] = pose
        mujoco.mj_forward(context.model, context.data)
        object_pos = np.asarray(
            context.data.xpos[context.object_body_id], dtype=np.float64
        )
        object_mat = np.asarray(
            context.data.xmat[context.object_body_id], dtype=np.float64
        ).reshape(3, 3)
        for geom_column, geom_id in enumerate(context.query_geom_ids):
            geom_pos = np.asarray(context.data.geom_xpos[geom_id], dtype=np.float64)
            geom_mat = np.asarray(
                context.data.geom_xmat[geom_id], dtype=np.float64
            ).reshape(3, 3)
            geom_positions[pose_index, geom_column] = (
                (geom_pos - object_pos) @ object_mat
            ).astype(np.float32)
            geom_matrices[pose_index, geom_column] = (object_mat.T @ geom_mat).astype(
                np.float32
            )
    return {
        "qpos": poses,
        "qpos_original_shape": np.asarray(original_shape, dtype=np.int64),
        "geom_pos_object_local": geom_positions,
        "geom_mat_object_local": geom_matrices,
        "query_geom_ids": context.query_geom_ids,
        "point_offsets_geom_local": context.point_offsets_geom_local,
        "point_geom_column": context.point_geom_column,
        "point_radius_m": context.point_radius_m,
        "point_geom_id": context.point_geom_id,
        "point_consumer_mask": context.point_consumer_mask,
        "consumer_names": np.asarray(context.consumer_names, dtype="U64"),
    }


def materialize_query_points(chunk: dict[str, np.ndarray]) -> np.ndarray:
    """Materialize exact object-local query coordinates from a factored chunk."""
    columns = np.asarray(chunk["point_geom_column"], dtype=np.int64)
    positions = np.asarray(chunk["geom_pos_object_local"])[:, columns]
    matrices = np.asarray(chunk["geom_mat_object_local"])[:, columns]
    offsets = np.asarray(chunk["point_offsets_geom_local"])
    return positions + np.einsum("npij,pj->npi", matrices, offsets)


def verify_case_query_tape(
    output_root: Path, payload: dict[str, Any]
) -> dict[str, Any]:
    """Verify all chunk/source hashes and factored-array integrity for one case."""
    mismatches: list[dict[str, Any]] = []
    chunks = payload.get("chunks", [])
    if payload.get("status") != "COMPLETE":
        mismatches.append({"field": "status", "actual": payload.get("status")})
    if payload.get("chunk_count") != len(chunks):
        mismatches.append(
            {
                "field": "chunk_count",
                "actual": payload.get("chunk_count"),
                "expected": len(chunks),
            }
        )
    stored_size = 0
    for index, entry in enumerate(chunks):
        path = output_root / entry["relative_path"]
        if not path.is_file():
            mismatches.append({"chunk": index, "field": "file", "error": "missing"})
            continue
        stored_size += path.stat().st_size
        actual_sha = sha256_file(path)
        if actual_sha != entry["sha256"]:
            mismatches.append(
                {
                    "chunk": index,
                    "field": "sha256",
                    "actual": actual_sha,
                    "expected": entry["sha256"],
                }
            )
            continue
        source = Path(entry["source"])
        if not source.is_file() or sha256_file(source) != entry["source_sha256"]:
            mismatches.append({"chunk": index, "field": "source_file_or_sha"})
        with np.load(path, allow_pickle=False) as values:
            required = {
                "qpos",
                "geom_pos_object_local",
                "geom_mat_object_local",
                "query_geom_ids",
                "point_offsets_geom_local",
                "point_geom_column",
                "point_radius_m",
                "point_geom_id",
                "point_consumer_mask",
                "consumer_names",
            }
            missing = sorted(required - set(values.files))
            if missing:
                mismatches.append(
                    {"chunk": index, "field": "array_keys", "missing": missing}
                )
                continue
            for key in ("qpos", "geom_pos_object_local", "geom_mat_object_local"):
                if not np.isfinite(values[key]).all():
                    mismatches.append(
                        {"chunk": index, "field": key, "error": "nonfinite"}
                    )
            pose_count = values["qpos"].shape[0]
            if (
                values["geom_pos_object_local"].shape[0] != pose_count
                or values["geom_mat_object_local"].shape[0] != pose_count
            ):
                mismatches.append(
                    {"chunk": index, "field": "pose_axis", "error": "shape_mismatch"}
                )
    if stored_size != payload.get("stored_size_bytes"):
        mismatches.append(
            {
                "field": "stored_size_bytes",
                "actual": stored_size,
                "expected": payload.get("stored_size_bytes"),
            }
        )
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "case_id": payload.get("case_id"),
        "checked_chunks": len(chunks),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an NPZ atomically without changing numeric dtypes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def _write_source_chunk(
    *,
    context: CaseContext,
    qpos: np.ndarray,
    source_family: str,
    source_path: Path,
    output_path: Path,
    output_root: Path,
    source_chunk_index: int,
) -> dict[str, Any]:
    """Extract, write, and inventory one source chunk."""
    arrays = extract_query_chunk(context, qpos)
    _atomic_npz(output_path, arrays)
    return {
        "source_family": source_family,
        "source_chunk_index": source_chunk_index,
        "relative_path": output_path.relative_to(output_root).as_posix(),
        "sha256": sha256_file(output_path),
        "size_bytes": output_path.stat().st_size,
        "source": relative_to_repo(source_path),
        "source_sha256": sha256_file(source_path),
        "qpos_original_shape": list(np.asarray(qpos).shape),
        "pose_count": int(np.asarray(qpos).reshape(-1, context.model.nq).shape[0]),
        "points_per_pose": int(len(context.point_geom_id)),
    }


def build_case_query_tape(
    row: dict[str, str],
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    max_cem_chunks: int | None = None,
) -> dict[str, Any]:
    """Build ref, E178-final, and frozen on_a CEM query tensors for one dev case."""
    context = load_case_context(row)
    case_root = output_root / row["case_id"]
    chunks: list[dict[str, Any]] = []
    reference_path = Path(row["trajectory"])
    final_path = Path(row["e178_result_npz"])
    chunks.append(
        _write_source_chunk(
            context=context,
            qpos=load_reference_qpos(row, context),
            source_family="reference",
            source_path=reference_path,
            output_path=case_root / "reference/chunk_000000.npz",
            output_root=output_root,
            source_chunk_index=0,
        )
    )
    chunks.append(
        _write_source_chunk(
            context=context,
            qpos=load_final_qpos(row, context),
            source_family="e178_final",
            source_path=final_path,
            output_path=case_root / "e178_final/chunk_000000.npz",
            output_root=output_root,
            source_chunk_index=0,
        )
    )
    raw_manifest_path = chunk_manifest_path(DEFAULT_RESULT_ROOT, "on_a", row["case_id"])
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    raw_entries = raw_manifest["chunks"]
    if max_cem_chunks is not None:
        raw_entries = raw_entries[:max_cem_chunks]
    for index, entry in enumerate(raw_entries):
        raw_chunk = resolve_chunk_path(raw_manifest_path, entry)
        if not raw_chunk.is_file() or sha256_file(raw_chunk) != entry["sha256"]:
            raise RuntimeError(f"{row['case_id']}: invalid raw CEM chunk {index}")
        with np.load(raw_chunk, allow_pickle=False) as values:
            qpos = np.asarray(values["qpos"], dtype=np.float64)
        chunks.append(
            _write_source_chunk(
                context=context,
                qpos=qpos,
                source_family="cem_on_a",
                source_path=raw_chunk,
                output_path=case_root / f"cem_on_a/chunk_{index:06d}.npz",
                output_root=output_root,
                source_chunk_index=index,
            )
        )
    source_counts = {
        family: sum(entry["source_family"] == family for entry in chunks)
        for family in ("reference", "e178_final", "cem_on_a")
    }
    payload = {
        "experiment_id": "E182",
        "stage": "S1_object_local_prg_query_tape",
        "status": "COMPLETE",
        "case_id": row["case_id"],
        "object_key": row["object_key"],
        "selection_cem_mode": "on_a",
        "frame": "object_body_local",
        "point_contract": "MJWP_PRIMITIVE_AXIS3_RADIUS_OR_MESH_VERTICES_MAX800",
        "storage_contract": "FACTORED_GEOM_POSE_PLUS_FIXED_OFFSETS",
        "physics_contract": "P uses frozen qpos for full-scene replay; sampled points are R/G fidelity queries",
        "model": relative_to_repo(Path(context.config.model_path)),
        "model_sha256": sha256_file(Path(context.config.model_path)),
        "config": row["e178_config_act"],
        "config_sha256": sha256_file(Path(row["e178_config_act"])),
        "consumer_names": context.consumer_names,
        "consumer_geom_ids": context.consumer_geom_ids,
        "consumer_geom_names": {
            name: [
                mujoco.mj_id2name(context.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                or str(geom_id)
                for geom_id in ids
            ]
            for name, ids in context.consumer_geom_ids.items()
        },
        "points_per_pose": int(len(context.point_geom_id)),
        "source_chunk_counts": source_counts,
        "chunk_count": len(chunks),
        "chunks": chunks,
        "raw_cem_manifest": relative_to_repo(raw_manifest_path),
        "raw_cem_content_sha256": raw_manifest["content_sha256"],
    }
    payload["stored_size_bytes"] = sum(entry["size_bytes"] for entry in chunks)
    payload["estimated_expanded_point_bytes"] = sum(
        entry["pose_count"] * entry["points_per_pose"] * 3 * 4 for entry in chunks
    )
    atomic_json(case_root / "manifest.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    """Parse the builder CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-cem-chunks", type=int)
    return parser.parse_args()


def main() -> int:
    """Build selected dev3 tapes only; heldout rows are never loaded."""
    args = parse_args()
    rows = load_dev_rows()
    if args.case_id:
        requested = set(args.case_id)
        rows = [row for row in rows if row["case_id"] in requested]
        if {row["case_id"] for row in rows} != requested:
            raise RuntimeError("one or more requested cases are outside dev3")
    payloads = [
        build_case_query_tape(
            row,
            output_root=args.output_root,
            max_cem_chunks=args.max_cem_chunks,
        )
        for row in rows
    ]
    print(f"E182_PRG_QUERY_TAPE=PASS rows={len(payloads)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
