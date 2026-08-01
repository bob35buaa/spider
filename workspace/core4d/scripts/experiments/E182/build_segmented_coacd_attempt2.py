#!/usr/bin/env python3
"""Build the pre-registered bucket003 x2-by-y4 segmented CoACD attempt2 family."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TBB_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import trimesh
from build_task_query_fixture import DEFAULT_OUTPUT_ROOT
from e182_common import atomic_json, relative_to_repo, repo_path, sha256_file
from evaluate_task_queries import DEFAULT_FIXTURE

ATTEMPT_ROOT = DEFAULT_OUTPUT_ROOT / "attempt2_segmented_x2y4"
PROTOCOL_PATH = ATTEMPT_ROOT / "attempt2_protocol_manifest.json"
SEGMENT_ROOT = ATTEMPT_ROOT / "segments"
CANDIDATE_ROOT = ATTEMPT_ROOT / "candidates"
FIXTURE_PATH = ATTEMPT_ROOT / "query_fixture_manifest.json"
REPLAY_PATH = (
    DEFAULT_OUTPUT_ROOT / "p_contact_mujoco_replay/mujoco_replay_manifest.json"
)
ORACLE_MANIFEST_PATH = repo_path(
    "workspace/core4d/results/E181/s1_oracle/bucket003/oracle_manifest.json"
)

OBJECT_KEY = "bucket003"
CASE_ID = "bucket003_20231018_001_p1"
X_SEGMENTS = 2
Y_SEGMENTS = 4
SEGMENT_COUNT = X_SEGMENTS * Y_SEGMENTS
OUTER_PADDING_M = 0.01
THRESHOLDS_M = (0.005, 0.010, 0.020)
MAX_HULLS = (8, 16, 32)
MAX_VERTICES = 32
COACD_COMMON = {
    "preprocess_mode": "auto",
    "preprocess_resolution": 50,
    "resolution": 2000,
    "mcts_nodes": 20,
    "mcts_iterations": 150,
    "mcts_max_depth": 3,
    "pca": False,
    "merge": True,
    "decimate": True,
    "extrude": False,
    "extrude_margin_m": 0.01,
    "apx_mode": "ch",
    "seed": 1,
    "real_metric": True,
}


def _candidate_id(threshold_m: float, max_hulls: int) -> str:
    """Return a lexical attempt2 candidate identifier."""
    return (
        f"segx{X_SEGMENTS}y{Y_SEGMENTS}_t{round(threshold_m * 1000):03d}_"
        f"k{max_hulls:02d}_v{MAX_VERTICES:03d}"
    )


def _candidate_asset_sha256(candidate: dict[str, Any]) -> str:
    """Hash the same ordered candidate identity consumed by the S2 evaluator."""
    payload = {
        "object_key": candidate["object_key"],
        "parameters": candidate["parameters"],
        "parts": [
            {
                "part_index": part["part_index"],
                "sha256": part["sha256"],
                "vertex_count": part["vertex_count"],
                "face_count": part["face_count"],
            }
            for part in candidate["parts"]
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _candidate_identity_sha256(rows: list[dict[str, Any]]) -> str:
    """Hash the ordered augmented-fixture candidate identities."""
    payload = [
        {
            "object_key": row["object_key"],
            "candidate_id": row["candidate_id"],
            "candidate_asset_sha256": row["candidate_asset_sha256"],
            "manifest": row["manifest"],
            "max_hulls": row["max_hulls"],
            "actual_hulls": row["actual_hulls"],
            "threshold_m": row["threshold_m"],
            "max_vertices": row["max_vertices"],
        }
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_oracle() -> tuple[dict[str, Any], Path, trimesh.Trimesh]:
    """Load the unchanged E181 Gate-A bucket003 oracle."""
    manifest = json.loads(ORACLE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        raise RuntimeError("bucket003 oracle is not PASS")
    entry = manifest["cleaned_mesh"]
    path = repo_path(entry["path"])
    if sha256_file(path) != entry["sha256"]:
        raise RuntimeError("bucket003 oracle SHA changed")
    mesh = trimesh.load(path, force="mesh", process=False, maintain_order=True)
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise RuntimeError("bucket003 oracle topology changed")
    return manifest, path, mesh


def _load_protocol() -> dict[str, Any]:
    """Require the exact pre-score attempt2 protocol and source hashes."""
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_BUILD_OR_SCORE":
        raise RuntimeError("attempt2 protocol is not frozen")
    if protocol["builder_source"]["sha256"] != sha256_file(Path(__file__)):
        raise RuntimeError("attempt2 builder source changed after protocol freeze")
    if protocol["original_fixture"]["sha256"] != sha256_file(DEFAULT_FIXTURE):
        raise RuntimeError("original query fixture changed")
    if protocol["mujoco_replay"]["sha256"] != sha256_file(REPLAY_PATH):
        raise RuntimeError("MuJoCo replay evidence changed")
    return protocol


def freeze_protocol() -> dict[str, Any]:
    """Freeze attempt2 topology and parameter family before build or score."""
    existing_manifests = list(CANDIDATE_ROOT.glob("*/manifest.json"))
    existing_results = list((ATTEMPT_ROOT / "screen/results").glob("*.json"))
    if existing_manifests or existing_results:
        raise RuntimeError("cannot freeze attempt2 protocol after build or score")
    replay = json.loads(REPLAY_PATH.read_text(encoding="utf-8"))
    if (
        replay.get("status") != "COMPLETE"
        or replay.get("selection_eligible") is not False
        or replay.get("heldout_access") != "NOT_ACCESSED_DEV3_ONLY"
        or replay.get("candidate_count") != 3
        or any(
            row["point_phantom_confirmed_by_mujoco_fraction"] != 1.0
            for row in replay["candidates"]
        )
    ):
        raise RuntimeError("MuJoCo replay does not authorize attempt2")
    oracle, oracle_path, oracle_mesh = _load_oracle()
    candidates = [
        {
            "candidate_id": _candidate_id(threshold_m, max_hulls),
            "threshold_m": threshold_m,
            "max_hulls": max_hulls,
            "per_segment_max_hulls": max_hulls // SEGMENT_COUNT,
            "max_vertices": MAX_VERTICES,
        }
        for threshold_m in THRESHOLDS_M
        for max_hulls in MAX_HULLS
    ]
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_protocol",
        "status": "FROZEN_BEFORE_BUILD_OR_SCORE",
        "selection_eligible": False,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "reason": "BUCKET003_POINT_PHANTOMS_100PCT_CONFIRMED_BY_TRUE_MUJOCO_C",
        "object_key": OBJECT_KEY,
        "case_id": CASE_ID,
        "topology": {
            "name": "EXACT_MANIFOLD_X2_Y4_SOLID_INTERSECTION_THEN_COACD",
            "x_segments": X_SEGMENTS,
            "y_segments": Y_SEGMENTS,
            "z_segments": 1,
            "segment_count": SEGMENT_COUNT,
            "split_policy": "EQUAL_OBJECT_LOCAL_BOUNDS",
            "outer_padding_m": OUTER_PADDING_M,
            "overlap_policy": "NO_INTERNAL_OVERLAP_SHARED_PLANES_ONLY",
            "ordered_parts": "x_index_y_index_local_part_index",
        },
        "candidate_family": candidates,
        "candidate_count": len(candidates),
        "thresholds_m": list(THRESHOLDS_M),
        "max_hulls": list(MAX_HULLS),
        "max_vertices": MAX_VERTICES,
        "coacd_common": COACD_COMMON,
        "selection_contract": {
            "launch_floor": "UNCHANGED_E182_P_R_G_V1",
            "score": "UNCHANGED_P_R_G_WORST_NORMALIZED_V1",
            "per_K": "launch_floor_then_score_then_actual_hulls_runtime_threshold_id",
            "full_role": "FORBIDDEN_UNTIL_PRODUCTION_FREEZE",
        },
        "oracle": {
            "path": relative_to_repo(oracle_path),
            "sha256": oracle["cleaned_mesh"]["sha256"],
            "bounds_m": oracle_mesh.bounds.tolist(),
            "volume_m3": float(oracle_mesh.volume),
        },
        "mujoco_replay": {
            "path": relative_to_repo(REPLAY_PATH),
            "sha256": sha256_file(REPLAY_PATH),
        },
        "original_fixture": {
            "path": relative_to_repo(DEFAULT_FIXTURE),
            "sha256": sha256_file(DEFAULT_FIXTURE),
        },
        "builder_source": {
            "path": relative_to_repo(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
    }
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("existing attempt2 protocol differs")
        return existing
    atomic_json(PROTOCOL_PATH, payload)
    return payload


def _export_mesh(mesh: trimesh.Trimesh, path: Path) -> dict[str, Any]:
    """Export one deterministic closed mesh and verify its OBJ round trip."""
    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    if (
        mesh.is_empty
        or not mesh.is_watertight
        or not mesh.is_winding_consistent
        or not np.isfinite(mesh.vertices).all()
        or float(mesh.volume) <= 0.0
    ):
        raise RuntimeError(f"invalid mesh for export: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        trimesh.exchange.obj.export_obj(
            mesh,
            include_normals=False,
            include_color=False,
            include_texture=False,
            return_texture=False,
            write_texture=False,
            digits=17,
        ),
        encoding="utf-8",
    )
    loaded = trimesh.load(path, force="mesh", process=False, maintain_order=True)
    if not np.array_equal(loaded.faces, mesh.faces):
        raise RuntimeError(f"OBJ face round trip changed: {path}")
    displacement = float(np.max(np.abs(loaded.vertices - mesh.vertices), initial=0.0))
    if displacement > 1e-15:
        raise RuntimeError(f"OBJ vertex round trip changed: {path}")
    return {
        "path": relative_to_repo(path),
        "sha256": sha256_file(path),
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "volume_m3": float(mesh.volume),
        "bounds_m": mesh.bounds.tolist(),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "roundtrip_vertex_displacement_max_m": displacement,
    }


def partition_mesh(
    mesh: trimesh.Trimesh,
) -> list[tuple[int, int, np.ndarray, np.ndarray, trimesh.Trimesh]]:
    """Return the frozen exact x2-by-y4 object-local solid intersections."""
    low, high = mesh.bounds
    x_edges = np.linspace(low[0], high[0], X_SEGMENTS + 1)
    y_edges = np.linspace(low[1], high[1], Y_SEGMENTS + 1)
    x_edges[0] -= OUTER_PADDING_M
    x_edges[-1] += OUTER_PADDING_M
    y_edges[0] -= OUTER_PADDING_M
    y_edges[-1] += OUTER_PADDING_M
    z_edges = (low[2] - OUTER_PADDING_M, high[2] + OUTER_PADDING_M)
    cells = []
    for x_index in range(X_SEGMENTS):
        for y_index in range(Y_SEGMENTS):
            cell_low = np.asarray(
                [x_edges[x_index], y_edges[y_index], z_edges[0]], dtype=np.float64
            )
            cell_high = np.asarray(
                [
                    x_edges[x_index + 1],
                    y_edges[y_index + 1],
                    z_edges[1],
                ],
                dtype=np.float64,
            )
            box = trimesh.creation.box(
                extents=cell_high - cell_low,
                transform=trimesh.transformations.translation_matrix(
                    (cell_low + cell_high) / 2.0
                ),
            )
            segment = trimesh.boolean.intersection([mesh, box], engine="manifold")
            if not isinstance(segment, trimesh.Trimesh):
                raise RuntimeError("segment boolean did not return one mesh")
            cells.append((x_index, y_index, cell_low, cell_high, segment))
    return cells


def build_segments() -> dict[str, Any]:
    """Create and verify the exact x2-by-y4 solid partition."""
    protocol = _load_protocol()
    _, oracle_path, mesh = _load_oracle()
    rows = []
    for x_index, y_index, cell_low, cell_high, segment in partition_mesh(mesh):
        row = _export_mesh(
            segment,
            SEGMENT_ROOT / f"cell_x{x_index}_y{y_index}.obj",
        )
        row.update(
            {
                "segment_index": len(rows),
                "x_index": x_index,
                "y_index": y_index,
                "cell_bounds_m": [cell_low.tolist(), cell_high.tolist()],
            }
        )
        rows.append(row)
    volume_sum = float(sum(row["volume_m3"] for row in rows))
    volume_delta = volume_sum - float(mesh.volume)
    if len(rows) != SEGMENT_COUNT or abs(volume_delta) > 1e-8:
        raise RuntimeError("segmented solid volume does not close to oracle")
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_segments",
        "status": "PASS",
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "protocol": {
            "path": relative_to_repo(PROTOCOL_PATH),
            "sha256": sha256_file(PROTOCOL_PATH),
        },
        "oracle": {
            "path": relative_to_repo(oracle_path),
            "sha256": protocol["oracle"]["sha256"],
            "volume_m3": float(mesh.volume),
        },
        "segment_count": len(rows),
        "segment_volume_sum_m3": volume_sum,
        "segment_volume_delta_m3": volume_delta,
        "segments": rows,
    }
    atomic_json(SEGMENT_ROOT / "segment_manifest.json", payload)
    return payload


def _load_segments() -> dict[str, Any]:
    """Load the exact segment manifest and verify every immutable mesh."""
    path = SEGMENT_ROOT / "segment_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or manifest.get("segment_count") != 8:
        raise RuntimeError("attempt2 segment manifest is incomplete")
    if manifest["protocol"]["sha256"] != sha256_file(PROTOCOL_PATH):
        raise RuntimeError("attempt2 segment protocol SHA changed")
    for row in manifest["segments"]:
        path = repo_path(row["path"])
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"segment SHA changed: {path}")
    return manifest


def _export_part(
    vertices: np.ndarray,
    faces: np.ndarray,
    path: Path,
    *,
    part_index: int,
    segment_index: int,
    local_part_index: int,
) -> dict[str, Any]:
    """Validate and export one convex CoACD part."""
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
        maintain_order=True,
    )
    coacd_vertex_count = int(len(mesh.vertices))
    mesh.remove_unreferenced_vertices()
    if (
        not mesh.is_convex
        or not mesh.is_watertight
        or not mesh.is_winding_consistent
        or len(mesh.vertices) > MAX_VERTICES
    ):
        raise RuntimeError(f"invalid CoACD part {part_index}")
    row = _export_mesh(mesh, path)
    row.update(
        {
            "part_index": part_index,
            "segment_index": segment_index,
            "local_part_index": local_part_index,
            "coacd_vertex_count": coacd_vertex_count,
            "removed_unreferenced_vertex_count": coacd_vertex_count
            - row["vertex_count"],
            "convex": True,
        }
    )
    return row


def build_candidate(threshold_m: float, max_hulls: int) -> dict[str, Any]:
    """Run CoACD independently in each exact segment and concatenate ordered parts."""
    coacd = importlib.import_module("coacd")
    protocol = _load_protocol()
    segments = _load_segments()
    identifier = _candidate_id(threshold_m, max_hulls)
    expected = [
        row for row in protocol["candidate_family"] if row["candidate_id"] == identifier
    ]
    if len(expected) != 1:
        raise RuntimeError(f"candidate outside frozen family: {identifier}")
    per_segment_cap = int(expected[0]["per_segment_max_hulls"])
    parameters = {
        "threshold_m": threshold_m,
        "max_convex_hull": max_hulls,
        "max_ch_vertex": MAX_VERTICES,
        "decomposition_topology": protocol["topology"],
        "per_segment_max_convex_hull": per_segment_cap,
        **COACD_COMMON,
    }
    root = CANDIDATE_ROOT / identifier
    parts_root = root / "parts"
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "BUILD_PASS"
            and existing.get("parameters") == parameters
            and all(
                sha256_file(repo_path(part["path"])) == part["sha256"]
                for part in existing["parts"]
            )
        ):
            return existing
        raise RuntimeError(f"non-resumable attempt2 candidate: {identifier}")
    coacd.set_log_level("warn")
    started = time.perf_counter()
    parts = []
    per_segment_rows = []
    for segment in segments["segments"]:
        segment_mesh = trimesh.load(
            repo_path(segment["path"]),
            force="mesh",
            process=False,
            maintain_order=True,
        )
        result = coacd.run_coacd(
            coacd.Mesh(segment_mesh.vertices, segment_mesh.faces),
            threshold=threshold_m,
            max_convex_hull=per_segment_cap,
            preprocess_mode=COACD_COMMON["preprocess_mode"],
            preprocess_resolution=COACD_COMMON["preprocess_resolution"],
            resolution=COACD_COMMON["resolution"],
            mcts_nodes=COACD_COMMON["mcts_nodes"],
            mcts_iterations=COACD_COMMON["mcts_iterations"],
            mcts_max_depth=COACD_COMMON["mcts_max_depth"],
            pca=COACD_COMMON["pca"],
            merge=COACD_COMMON["merge"],
            decimate=COACD_COMMON["decimate"],
            max_ch_vertex=MAX_VERTICES,
            extrude=COACD_COMMON["extrude"],
            extrude_margin=COACD_COMMON["extrude_margin_m"],
            apx_mode=COACD_COMMON["apx_mode"],
            seed=COACD_COMMON["seed"],
            real_metric=COACD_COMMON["real_metric"],
        )
        if not 1 <= len(result) <= per_segment_cap:
            raise RuntimeError(f"{identifier}: segment hull cap violated")
        first_part = len(parts)
        for local_part_index, (vertices, faces) in enumerate(result):
            part_index = len(parts)
            parts.append(
                _export_part(
                    vertices,
                    faces,
                    parts_root / f"part_{part_index:03d}.obj",
                    part_index=part_index,
                    segment_index=segment["segment_index"],
                    local_part_index=local_part_index,
                )
            )
        per_segment_rows.append(
            {
                "segment_index": segment["segment_index"],
                "segment_sha256": segment["sha256"],
                "first_part_index": first_part,
                "hull_count": len(result),
            }
        )
    if len(parts) > max_hulls:
        raise RuntimeError(f"{identifier}: total hull cap violated")
    oracle = protocol["oracle"]
    manifest = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_build",
        "status": "BUILD_PASS",
        "selection_eligible": False,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "object_key": OBJECT_KEY,
        "candidate_id": identifier,
        "protocol": {
            "path": relative_to_repo(PROTOCOL_PATH),
            "sha256": sha256_file(PROTOCOL_PATH),
        },
        "segment_manifest": {
            "path": relative_to_repo(SEGMENT_ROOT / "segment_manifest.json"),
            "sha256": sha256_file(SEGMENT_ROOT / "segment_manifest.json"),
        },
        "oracle_cleaned_mesh": {
            "path": oracle["path"],
            "sha256": oracle["sha256"],
        },
        "parameters": parameters,
        "max_hulls": max_hulls,
        "hull_count": len(parts),
        "max_part_vertex_count": max(part["vertex_count"] for part in parts),
        "total_vertex_count": sum(part["vertex_count"] for part in parts),
        "total_face_count": sum(part["face_count"] for part in parts),
        "build_wall_seconds": time.perf_counter() - started,
        "per_segment": per_segment_rows,
        "parts": parts,
    }
    manifest["candidate_asset_sha256"] = _candidate_asset_sha256(manifest)
    atomic_json(manifest_path, manifest)
    print(
        "E182_SEGMENTED_COACD=PASS "
        f"candidate={identifier} hulls={len(parts)} "
        f"wall={manifest['build_wall_seconds']:.3f}"
    )
    return manifest


def build_candidates() -> dict[str, Any]:
    """Build or resume the complete frozen nine-candidate attempt2 family."""
    protocol = _load_protocol()
    manifests = [
        build_candidate(float(row["threshold_m"]), int(row["max_hulls"]))
        for row in protocol["candidate_family"]
    ]
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_build",
        "status": "BUILD_PASS",
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "protocol": {
            "path": relative_to_repo(PROTOCOL_PATH),
            "sha256": sha256_file(PROTOCOL_PATH),
        },
        "candidate_count": len(manifests),
        "candidate_ids": [manifest["candidate_id"] for manifest in manifests],
        "actual_hulls": [manifest["hull_count"] for manifest in manifests],
        "total_wall_seconds": float(
            sum(manifest["build_wall_seconds"] for manifest in manifests)
        ),
    }
    if payload["candidate_count"] != 9:
        raise RuntimeError("attempt2 candidate family is incomplete")
    atomic_json(ATTEMPT_ROOT / "build_summary.json", payload)
    return payload


def build_fixture() -> dict[str, Any]:
    """Freeze a bucket003-only augmented fixture after all candidate assets exist."""
    protocol = _load_protocol()
    summary = json.loads(
        (ATTEMPT_ROOT / "build_summary.json").read_text(encoding="utf-8")
    )
    if summary.get("status") != "BUILD_PASS" or summary.get("candidate_count") != 9:
        raise RuntimeError("attempt2 build summary is incomplete")
    original = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    case_rows = [case for case in original["cases"] if case["case_id"] == CASE_ID]
    if len(case_rows) != 1:
        raise RuntimeError("bucket003 dev case changed")
    rows = []
    for protocol_row in protocol["candidate_family"]:
        manifest_path = CANDIDATE_ROOT / protocol_row["candidate_id"] / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "BUILD_PASS":
            raise RuntimeError("attempt2 candidate is not BUILD_PASS")
        rows.append(
            {
                "object_key": OBJECT_KEY,
                "candidate_id": manifest["candidate_id"],
                "candidate_asset_sha256": manifest["candidate_asset_sha256"],
                "manifest": {
                    "path": relative_to_repo(manifest_path),
                    "sha256": sha256_file(manifest_path),
                },
                "max_hulls": int(protocol_row["max_hulls"]),
                "actual_hulls": int(manifest["hull_count"]),
                "threshold_m": float(protocol_row["threshold_m"]),
                "max_vertices": int(protocol_row["max_vertices"]),
            }
        )
    payload = {
        **original,
        "stage": "S2_task_query_fixture_attempt2_segmented_x2y4",
        "status": "FROZEN",
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "freeze_timing": "AFTER_BUILD_BEFORE_ATTEMPT2_SCORE",
        "attempt2_protocol": {
            "path": relative_to_repo(PROTOCOL_PATH),
            "sha256": sha256_file(PROTOCOL_PATH),
        },
        "original_fixture": {
            "path": relative_to_repo(DEFAULT_FIXTURE),
            "sha256": sha256_file(DEFAULT_FIXTURE),
        },
        "cases": case_rows,
        "candidates": rows,
        "candidate_count": len(rows),
        "candidate_identity_sha256": _candidate_identity_sha256(rows),
    }
    if FIXTURE_PATH.exists():
        existing = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("existing attempt2 fixture differs")
        return existing
    atomic_json(FIXTURE_PATH, payload)
    return payload


def parse_args() -> argparse.Namespace:
    """Parse attempt2 build stages."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "freeze-protocol",
            "build-segments",
            "build-candidates",
            "build-fixture",
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Execute one explicit attempt2 stage."""
    args = parse_args()
    if args.command == "freeze-protocol":
        payload = freeze_protocol()
        print(f"E182_ATTEMPT2_PROTOCOL={payload['status']}")
        return 0
    if args.command == "build-segments":
        payload = build_segments()
        print(
            f"E182_ATTEMPT2_SEGMENTS={payload['status']} count={payload['segment_count']}"
        )
        return 0
    if args.command == "build-candidates":
        payload = build_candidates()
        print(
            f"E182_ATTEMPT2_BUILD={payload['status']} "
            f"candidates={payload['candidate_count']}"
        )
        return 0
    payload = build_fixture()
    print(
        f"E182_ATTEMPT2_FIXTURE={payload['status']} "
        f"candidates={payload['candidate_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
