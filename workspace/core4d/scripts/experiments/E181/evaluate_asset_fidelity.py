#!/usr/bin/env python3
"""Evaluate all E181 CoACD candidates against the original-mesh oracle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import open3d as o3d
import trimesh
from build_authority import REPO_ROOT, relative_to_repo, sha256_file
from scipy.spatial import ConvexHull, cKDTree

ORACLE_ROOT = REPO_ROOT / "workspace/core4d/results/E181/s1_oracle"
COACD_ROOT = REPO_ROOT / "workspace/core4d/results/E181/s2_coacd"
OUTPUT_ROOT = REPO_ROOT / "workspace/core4d/results/E181/s2_asset_eval"
DEV_MANIFEST = (
    REPO_ROOT / "workspace/core4d/results/E181/s6_downstream/manifests/dev3.tsv"
)
OBJECT_KEYS = ("bucket003", "bucket004", "bucket007")
OBJECT_SEEDS = {
    "bucket003": 181_203,
    "bucket004": 181_204,
    "bucket007": 181_207,
}
M_SURFACE_POINTS = 20_000
C_RAW_SURFACE_POINTS = 120_000
C_ACCESSIBLE_SURFACE_POINTS = 50_000
CAVITY_UNIFORM_POINTS = 1_000_000
CAVITY_BROAD_POINTS = 20_000
CAVITY_CORE_POINTS = 10_000
CAVITY_BROAD_MIN_DISTANCE_M = 0.002
CAVITY_CORE_MIN_DISTANCE_M = 0.015
BOUNDARY_ACCESS_OFFSET_M = 0.002
CONVEX_CONTAINS_TOLERANCE_M = 1e-8
EVALUATOR_CONFIG = {
    "m_surface_points": M_SURFACE_POINTS,
    "c_raw_surface_points": C_RAW_SURFACE_POINTS,
    "c_accessible_surface_points": C_ACCESSIBLE_SURFACE_POINTS,
    "cavity_uniform_points": CAVITY_UNIFORM_POINTS,
    "cavity_broad_points": CAVITY_BROAD_POINTS,
    "cavity_core_points": CAVITY_CORE_POINTS,
    "cavity_broad_min_distance_m": CAVITY_BROAD_MIN_DISTANCE_M,
    "cavity_core_min_distance_m": CAVITY_CORE_MIN_DISTANCE_M,
    "boundary_access_offset_m": BOUNDARY_ACCESS_OFFSET_M,
    "convex_contains_tolerance_m": CONVEX_CONTAINS_TOLERANCE_M,
    "surface_distance_backend": "dense_accessible_boundary_kdtree",
    "c_to_m_backend": "trimesh_exact_nearest_triangle",
    "m_sign_distance_backend": "open3d_raycasting_scene_nsamples5",
    "contact_target_source": "trajectory_fk_palm_site_world_to_visual_mesh_local",
    "heldout_access": "sealed_until_c_star_is_frozen",
}
EVALUATOR_CONFIG_SHA256 = hashlib.sha256(
    json.dumps(EVALUATOR_CONFIG, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a TSV manifest."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def open3d_scene(mesh: trimesh.Trimesh) -> o3d.t.geometry.RaycastingScene:
    """Build an Open3D raycasting scene from a trimesh mesh."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)),
    )
    return scene


def scene_distance(
    scene: o3d.t.geometry.RaycastingScene,
    points: np.ndarray,
) -> np.ndarray:
    """Return unsigned float64 distance from an Open3D scene."""
    if len(points) == 0:
        return np.empty(0, dtype=np.float64)
    return (
        scene.compute_distance(o3d.core.Tensor(points.astype(np.float32)))
        .numpy()
        .astype(np.float64)
    )


def scene_signed_distance(
    scene: o3d.t.geometry.RaycastingScene,
    points: np.ndarray,
) -> np.ndarray:
    """Return negative-inside signed distance from an Open3D scene."""
    return (
        scene.compute_signed_distance(
            o3d.core.Tensor(points.astype(np.float32)),
            nsamples=5,
        )
        .numpy()
        .astype(np.float64)
    )


def convex_equations(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return normalized halfspaces for a convex mesh."""
    return np.asarray(ConvexHull(mesh.vertices).equations, dtype=np.float64)


def contains_convex_union(
    equations: list[np.ndarray],
    points: np.ndarray,
    *,
    chunk_size: int = 5_000,
) -> np.ndarray:
    """Evaluate occupancy of a union of convex halfspace sets."""
    occupied = np.zeros(len(points), dtype=bool)
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        query = points[start:stop]
        chunk_occupied = np.zeros(len(query), dtype=bool)
        for halfspaces in equations:
            if chunk_occupied.all():
                break
            active = ~chunk_occupied
            values = query[active] @ halfspaces[:, :3].T + halfspaces[:, 3]
            chunk_occupied[active] = np.all(
                values <= CONVEX_CONTAINS_TOLERANCE_M,
                axis=1,
            )
        occupied[start:stop] = chunk_occupied
    return occupied


def trajectory_targets_mesh_local(row: dict[str, str]) -> np.ndarray:
    """Transform FK palm targets from world to raw visual-mesh coordinates."""
    trajectory = np.load(REPO_ROOT / row["trajectory"], allow_pickle=False)
    qpos = np.asarray(trajectory["qpos"], dtype=np.float64)
    contact_pos = np.asarray(trajectory["contact_pos"], dtype=np.float64)
    contact = np.asarray(trajectory["contact"], dtype=bool)
    model = mujoco.MjModel.from_xml_path(
        str((REPO_ROOT / row["target_scene"]).resolve())
    )
    data = mujoco.MjData(model)
    visual_geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "object_visual",
    )
    if visual_geom_id < 0:
        raise RuntimeError(f"{row['case_id']}: object_visual geom missing")
    local_frames = []
    for frame_index in range(len(qpos)):
        data.qpos[:] = qpos[frame_index]
        mujoco.mj_forward(model, data)
        rotation = data.geom_xmat[visual_geom_id].reshape(3, 3)
        position = data.geom_xpos[visual_geom_id]
        local_frames.append((contact_pos[frame_index] - position) @ rotation)
    local = np.asarray(local_frames, dtype=np.float64)
    if contact.shape != local.shape[:2]:
        raise RuntimeError(f"{row['case_id']}: contact/position shape mismatch")
    return local[contact]


def choose_points(
    points: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose up to count deterministic points without replacement."""
    if len(points) <= count:
        return points
    return points[rng.choice(len(points), size=count, replace=False)]


def build_fixture(object_key: str, output_root: Path) -> dict[str, Any]:
    """Build deterministic M-surface, cavity, and target fixtures."""
    oracle_manifest_path = ORACLE_ROOT / object_key / "oracle_manifest.json"
    oracle = json.loads(oracle_manifest_path.read_text(encoding="utf-8"))
    if oracle.get("status") != "PASS":
        raise RuntimeError(f"{object_key}: Gate A oracle is not PASS")
    mesh_path = REPO_ROOT / oracle["cleaned_mesh"]["path"]
    mesh = trimesh.load(
        mesh_path,
        force="mesh",
        process=False,
        maintain_order=True,
    )
    rng = np.random.default_rng(OBJECT_SEEDS[object_key])
    surface_points, surface_face_ids = trimesh.sample.sample_surface(
        mesh,
        M_SURFACE_POINTS,
        seed=rng,
    )
    surface_normals = mesh.face_normals[surface_face_ids]

    uniform = rng.uniform(
        mesh.bounds[0],
        mesh.bounds[1],
        size=(CAVITY_UNIFORM_POINTS, 3),
    )
    hull_halfspaces = convex_equations(mesh.convex_hull)
    in_convex_hull = np.all(
        uniform @ hull_halfspaces[:, :3].T + hull_halfspaces[:, 3]
        <= CONVEX_CONTAINS_TOLERANCE_M,
        axis=1,
    )
    convex_queries = uniform[in_convex_hull]
    scene = open3d_scene(mesh)
    signed = scene_signed_distance(scene, convex_queries)
    broad = choose_points(
        convex_queries[signed >= CAVITY_BROAD_MIN_DISTANCE_M],
        CAVITY_BROAD_POINTS,
        rng,
    )
    core = choose_points(
        convex_queries[signed >= CAVITY_CORE_MIN_DISTANCE_M],
        CAVITY_CORE_POINTS,
        rng,
    )
    if len(broad) < 1_000 or len(core) < 300:
        raise RuntimeError(
            f"{object_key}: insufficient cavity fixture broad={len(broad)} "
            f"core={len(core)}"
        )

    object_rows = [
        row for row in read_tsv(DEV_MANIFEST) if row["object_key"] == object_key
    ]
    dev_targets = []
    dev_case_ids = []
    for row in object_rows:
        targets = trajectory_targets_mesh_local(row)
        dev_targets.append(targets)
        dev_case_ids.append(row["case_id"])
    dev_targets_array = np.vstack(dev_targets)
    fixture_path = output_root / "fixtures" / f"{object_key}.npz"
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        fixture_path,
        m_surface_points_m=surface_points.astype(np.float64),
        m_surface_normals=surface_normals.astype(np.float64),
        cavity_broad_points_m=broad.astype(np.float64),
        cavity_core_points_m=core.astype(np.float64),
        dev_target_points_m=dev_targets_array.astype(np.float64),
        dev_target_d_m_m=scene_distance(scene, dev_targets_array),
    )
    manifest = {
        "experiment_id": "E181",
        "stage": "S2_asset_fidelity_fixture",
        "status": "PASS",
        "object_key": object_key,
        "evaluator_config": EVALUATOR_CONFIG,
        "evaluator_config_sha256": EVALUATOR_CONFIG_SHA256,
        "oracle_manifest": {
            "path": relative_to_repo(oracle_manifest_path),
            "sha256": sha256_file(oracle_manifest_path),
        },
        "oracle_cleaned_mesh_sha256": oracle["cleaned_mesh"]["sha256"],
        "fixture": {
            "path": relative_to_repo(fixture_path),
            "sha256": sha256_file(fixture_path),
        },
        "counts": {
            "m_surface": len(surface_points),
            "cavity_uniform": len(uniform),
            "cavity_inside_convex_hull": len(convex_queries),
            "cavity_broad": len(broad),
            "cavity_core": len(core),
            "dev_targets": len(dev_targets_array),
        },
        "dev_case_ids": dev_case_ids,
        "heldout_status": "SEALED_UNTIL_C_STAR",
    }
    atomic_json(output_root / "fixtures" / f"{object_key}.json", manifest)
    return manifest


def load_parts(candidate: dict[str, Any]) -> list[trimesh.Trimesh]:
    """Load and SHA-check all ordered convex parts."""
    parts = []
    for part in candidate["parts"]:
        path = REPO_ROOT / part["path"]
        if sha256_file(path) != part["sha256"]:
            raise RuntimeError(f"part SHA changed: {path}")
        parts.append(
            trimesh.load(
                path,
                force="mesh",
                process=False,
                maintain_order=True,
            )
        )
    return parts


def accessible_candidate_surface(
    parts: list[trimesh.Trimesh],
    equations: list[np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Sample compound-union boundary accessible through a 2mm normal probe."""
    areas = np.asarray([part.area for part in parts], dtype=np.float64)
    counts = np.maximum(
        200,
        np.ceil(C_RAW_SURFACE_POINTS * areas / areas.sum()).astype(int),
    )
    points = []
    normals = []
    for part, count in zip(parts, counts, strict=True):
        sampled, face_ids = trimesh.sample.sample_surface(
            part,
            int(count),
            seed=rng,
        )
        points.append(sampled)
        normals.append(part.face_normals[face_ids])
    raw_points = np.vstack(points)
    raw_normals = np.vstack(normals)
    outside = ~contains_convex_union(
        equations,
        raw_points + BOUNDARY_ACCESS_OFFSET_M * raw_normals,
    )
    inside = contains_convex_union(
        equations,
        raw_points - BOUNDARY_ACCESS_OFFSET_M * raw_normals,
    )
    keep = outside & inside
    accessible_points = raw_points[keep]
    accessible_normals = raw_normals[keep]
    if len(accessible_points) < 5_000:
        raise RuntimeError(
            f"accessible candidate surface too small: {len(accessible_points)}"
        )
    if len(accessible_points) > C_ACCESSIBLE_SURFACE_POINTS:
        indices = rng.choice(
            len(accessible_points),
            size=C_ACCESSIBLE_SURFACE_POINTS,
            replace=False,
        )
        accessible_points = accessible_points[indices]
        accessible_normals = accessible_normals[indices]
    return accessible_points, accessible_normals, len(raw_points)


def percentile(values: np.ndarray, quantile: float) -> float:
    """Return a float percentile, failing on empty/non-finite input."""
    if len(values) == 0 or not np.isfinite(values).all():
        raise RuntimeError("metric input is empty or non-finite")
    return float(np.quantile(values, quantile))


def evaluate_candidate(
    object_key: str,
    candidate_path: Path,
    fixture_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one candidate and write its fidelity manifest."""
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if candidate.get("status") != "BUILD_PASS":
        raise RuntimeError(f"candidate is not BUILD_PASS: {candidate_path}")
    fidelity_path = candidate_path.parent / "fidelity.json"
    if fidelity_path.is_file():
        old = json.loads(fidelity_path.read_text(encoding="utf-8"))
        if (
            old.get("candidate_asset_sha256") == candidate["candidate_asset_sha256"]
            and old.get("evaluator_config_sha256") == EVALUATOR_CONFIG_SHA256
            and old.get("fixture_sha256") == fixture_manifest["fixture"]["sha256"]
        ):
            return old
    fixture = np.load(
        REPO_ROOT / fixture_manifest["fixture"]["path"],
        allow_pickle=False,
    )
    mesh = trimesh.load(
        ORACLE_ROOT / object_key / "cleaned.obj",
        force="mesh",
        process=False,
        maintain_order=True,
    )
    parts = load_parts(candidate)
    equations = [convex_equations(part) for part in parts]
    seed = int(
        hashlib.sha256(
            f"{object_key}/{candidate['candidate_id']}".encode()
        ).hexdigest()[:16],
        16,
    )
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    c_surface, c_normals, c_raw_count = accessible_candidate_surface(
        parts,
        equations,
        rng,
    )
    tree = cKDTree(c_surface)
    m_surface = fixture["m_surface_points_m"]
    m_normals = fixture["m_surface_normals"]
    m_to_c, nearest_indices = tree.query(m_surface, k=1, workers=-1)
    normal_dot = np.einsum(
        "ij,ij->i",
        m_normals,
        c_normals[nearest_indices],
    )
    normal_error_deg = np.degrees(np.arccos(np.clip(normal_dot, -1.0, 1.0)))
    _nearest_m, c_to_m, _triangle_ids = trimesh.proximity.closest_point(
        mesh,
        c_surface,
    )
    broad_occupied = contains_convex_union(
        equations,
        fixture["cavity_broad_points_m"],
    )
    core_occupied = contains_convex_union(
        equations,
        fixture["cavity_core_points_m"],
    )
    dev_distance, _ = tree.query(
        fixture["dev_target_points_m"],
        k=1,
        workers=-1,
    )
    dev_excess = np.maximum(
        0.0,
        dev_distance - fixture["dev_target_d_m_m"],
    )
    metrics = {
        "m_to_c_surface_p90_m": percentile(m_to_c, 0.90),
        "m_to_c_surface_p99_m": percentile(m_to_c, 0.99),
        "c_to_m_surface_p90_m": percentile(c_to_m, 0.90),
        "c_to_m_surface_p99_m": percentile(c_to_m, 0.99),
        "must_cover_recall_15mm": float(np.mean(m_to_c <= 0.015)),
        "must_stay_free_core_false_occupied_count": int(core_occupied.sum()),
        "must_stay_free_core_count": int(len(core_occupied)),
        "broader_cavity_false_occupied_fraction": float(broad_occupied.mean()),
        "broader_cavity_false_occupied_count": int(broad_occupied.sum()),
        "broader_cavity_count": int(len(broad_occupied)),
        "surface_normal_error_p90_deg": percentile(normal_error_deg, 0.90),
        "dev_contact_excess_p90_m": percentile(dev_excess, 0.90),
        "accessible_surface_raw_count": c_raw_count,
        "accessible_surface_retained_count": int(len(c_surface)),
        "evaluation_wall_seconds": time.perf_counter() - started,
    }
    dev_hard_gates = {
        "m_to_c_p90": metrics["m_to_c_surface_p90_m"] <= 0.015,
        "m_to_c_p99": metrics["m_to_c_surface_p99_m"] <= 0.030,
        "c_to_m_p90": metrics["c_to_m_surface_p90_m"] <= 0.015,
        "c_to_m_p99": metrics["c_to_m_surface_p99_m"] <= 0.030,
        "must_cover": metrics["must_cover_recall_15mm"] >= 0.98,
        "must_stay_free_core": (
            metrics["must_stay_free_core_false_occupied_count"] == 0
        ),
        "broader_cavity": (metrics["broader_cavity_false_occupied_fraction"] <= 0.001),
        "normal_p90": metrics["surface_normal_error_p90_deg"] <= 20.0,
        "dev_contact_excess": metrics["dev_contact_excess_p90_m"] <= 0.010,
        "hull_budget": candidate["hull_count"] <= 32,
        "vertex_budget": candidate["max_part_vertex_count"] <= 64,
    }
    result = {
        "experiment_id": "E181",
        "stage": "S2_asset_fidelity",
        "status": "PASS" if all(dev_hard_gates.values()) else "REJECT",
        "heldout_status": "SEALED_UNTIL_C_STAR",
        "object_key": object_key,
        "candidate_id": candidate["candidate_id"],
        "candidate_asset_sha256": candidate["candidate_asset_sha256"],
        "candidate_manifest": {
            "path": relative_to_repo(candidate_path),
            "sha256": sha256_file(candidate_path),
        },
        "fixture_sha256": fixture_manifest["fixture"]["sha256"],
        "evaluator_config": EVALUATOR_CONFIG,
        "evaluator_config_sha256": EVALUATOR_CONFIG_SHA256,
        "metrics": metrics,
        "dev_hard_gates": dev_hard_gates,
    }
    atomic_json(fidelity_path, result)
    print(
        f"E181_FIDELITY object={object_key} candidate={candidate['candidate_id']} "
        f"status={result['status']} cavity="
        f"{metrics['broader_cavity_false_occupied_fraction']:.6f}",
        flush=True,
    )
    return result


def metric_row(result: dict[str, Any]) -> dict[str, Any]:
    """Flatten one result for the aggregate TSV."""
    return {
        "object_key": result["object_key"],
        "candidate_id": result["candidate_id"],
        "candidate_asset_sha256": result["candidate_asset_sha256"],
        "status": result["status"],
        "heldout_status": result["heldout_status"],
        **result["metrics"],
        "failed_dev_gates": ",".join(
            key for key, value in result["dev_hard_gates"].items() if not value
        ),
    }


def summarize(results: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    """Write aggregate TSV and Gate B summary."""
    rows = [metric_row(result) for result in results]
    tsv_path = output_root / "candidate_metrics.tsv"
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    pass_counts = {
        object_key: sum(
            row["object_key"] == object_key and row["status"] == "PASS" for row in rows
        )
        for object_key in OBJECT_KEYS
    }
    summary = {
        "experiment_id": "E181",
        "stage": "S2_asset_fidelity",
        "status": (
            "PASS"
            if all(count > 0 for count in pass_counts.values())
            else "ASSET_REJECTED"
        ),
        "candidate_count": len(rows),
        "expected_candidate_count": 54,
        "pass_counts": pass_counts,
        "candidate_metrics": {
            "path": relative_to_repo(tsv_path),
            "sha256": sha256_file(tsv_path),
        },
        "evaluator_config": EVALUATOR_CONFIG,
        "evaluator_config_sha256": EVALUATOR_CONFIG_SHA256,
    }
    atomic_json(output_root / "pareto_summary.json", summary)
    print(
        f"E181_GATE_B={summary['status']} candidates={len(rows)} "
        f"pass_counts={pass_counts}"
    )
    return summary


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", choices=OBJECT_KEYS, action="append")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    selected = args.object or list(OBJECT_KEYS)
    results = []
    for object_key in selected:
        fixture_manifest = build_fixture(object_key, args.output_root)
        candidate_paths = sorted((COACD_ROOT / object_key).glob("t*/manifest.json"))
        if len(candidate_paths) != 18:
            raise RuntimeError(f"{object_key}: expected 18 candidates")
        results.extend(
            evaluate_candidate(object_key, path, fixture_manifest)
            for path in candidate_paths
        )
    if set(selected) == set(OBJECT_KEYS):
        summarize(results, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
