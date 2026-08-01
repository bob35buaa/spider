#!/usr/bin/env python3
"""Build attempt2-v2: exact segment hull K8 plus segmented CoACD K16/K32."""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from build_segmented_coacd_attempt2 import (
    COACD_COMMON,
    DEFAULT_FIXTURE,
    SEGMENT_COUNT,
    _candidate_asset_sha256,
    _candidate_identity_sha256,
    _export_mesh,
    _load_oracle,
)
from build_segmented_coacd_attempt2 import (
    MAX_VERTICES as COACD_MAX_VERTICES,
)
from build_segmented_coacd_attempt2 import (
    PROTOCOL_PATH as V1_PROTOCOL_PATH,
)
from build_segmented_coacd_attempt2 import (
    SEGMENT_ROOT as V1_SEGMENT_ROOT,
)
from build_task_query_fixture import DEFAULT_OUTPUT_ROOT
from e182_common import atomic_json, relative_to_repo, repo_path, sha256_file

ATTEMPT_ROOT = DEFAULT_OUTPUT_ROOT / "attempt2_segmented_x2y4_v2"
PROTOCOL_PATH = ATTEMPT_ROOT / "attempt2_v2_protocol_manifest.json"
CANDIDATE_ROOT = ATTEMPT_ROOT / "candidates"
FIXTURE_PATH = ATTEMPT_ROOT / "query_fixture_manifest.json"
SCREEN_ROOT = ATTEMPT_ROOT / "screen"
V1_SEGMENT_MANIFEST_PATH = V1_SEGMENT_ROOT / "segment_manifest.json"

OBJECT_KEY = "bucket003"
CASE_ID = "bucket003_20231018_001_p1"
THRESHOLDS_M = (0.005, 0.010, 0.020)
COACD_K = (16, 32)
K8_MAX_VERTICES = 160


def candidate_family() -> list[dict[str, Any]]:
    """Return the frozen seven-row K8/K16/K32 attempt2-v2 family."""
    rows = [
        {
            "candidate_id": "segx2y4_hull_k08_v160",
            "method": "EXACT_SEGMENT_CONVEX_HULL",
            "threshold_m": 0.0,
            "max_hulls": 8,
            "per_segment_max_hulls": 1,
            "max_vertices": K8_MAX_VERTICES,
        }
    ]
    rows.extend(
        {
            "candidate_id": (
                f"segx2y4_coacd_t{round(threshold_m * 1000):03d}_"
                f"k{max_hulls:02d}_v{COACD_MAX_VERTICES:03d}"
            ),
            "method": "SEGMENTED_COACD",
            "threshold_m": threshold_m,
            "max_hulls": max_hulls,
            "per_segment_max_hulls": max_hulls // SEGMENT_COUNT,
            "max_vertices": COACD_MAX_VERTICES,
        }
        for threshold_m in THRESHOLDS_M
        for max_hulls in COACD_K
    )
    return rows


def _load_v1_segments() -> dict[str, Any]:
    """Load the unchanged exact x2-by-y4 solid partition from attempt2-v1."""
    manifest = json.loads(V1_SEGMENT_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "PASS"
        or manifest.get("segment_count") != SEGMENT_COUNT
        or manifest["protocol"]["sha256"] != sha256_file(V1_PROTOCOL_PATH)
    ):
        raise RuntimeError("attempt2-v1 segment authority changed")
    for row in manifest["segments"]:
        if sha256_file(repo_path(row["path"])) != row["sha256"]:
            raise RuntimeError("attempt2-v1 segment SHA changed")
    return manifest


def _v1_failure_evidence() -> dict[str, Any]:
    """Resolve the preserved pre-manifest K8 cap1 build failure evidence."""
    candidate_root = V1_PROTOCOL_PATH.parent / "candidates" / "segx2y4_t005_k08_v032"
    manifest_path = candidate_root / "manifest.json"
    parts = sorted((candidate_root / "parts").glob("part_*.obj"))
    if manifest_path.exists() or len(parts) != 3:
        raise RuntimeError("attempt2-v1 partial failure evidence changed")
    return {
        "status": "BUILD_FAILED_BEFORE_CANDIDATE_MANIFEST",
        "signature": "segment hull cap violated",
        "candidate_id": "segx2y4_t005_k08_v032",
        "successful_prefix_part_count": len(parts),
        "successful_prefix_parts": [
            {
                "path": relative_to_repo(path),
                "sha256": sha256_file(path),
            }
            for path in parts
        ],
        "failed_segment_index": 3,
        "failed_per_segment_cap": 1,
    }


def freeze_protocol() -> dict[str, Any]:
    """Freeze v2 before any v2 candidate build or score."""
    if list(CANDIDATE_ROOT.glob("*/manifest.json")) or list(
        (SCREEN_ROOT / "results").glob("*.json")
    ):
        raise RuntimeError("cannot freeze attempt2-v2 after build or score")
    _, oracle_path, oracle_mesh = _load_oracle()
    segments = _load_v1_segments()
    rows = candidate_family()
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v2_protocol",
        "status": "FROZEN_BEFORE_V2_BUILD_OR_SCORE",
        "selection_eligible": False,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "reason": "V1_X2Y4_K8_COACD_CAP1_BUILD_INFEASIBLE",
        "object_key": OBJECT_KEY,
        "case_id": CASE_ID,
        "topology": {
            "name": "REUSE_EXACT_MANIFOLD_X2_Y4_SOLID_PARTITION",
            "segment_count": SEGMENT_COUNT,
            "ordered_parts": "x_index_y_index_local_part_index",
            "v1_segment_manifest": {
                "path": relative_to_repo(V1_SEGMENT_MANIFEST_PATH),
                "sha256": sha256_file(V1_SEGMENT_MANIFEST_PATH),
            },
        },
        "candidate_family": rows,
        "candidate_count": len(rows),
        "K8_contract": {
            "method": "ONE_DETERMINISTIC_CONVEX_HULL_PER_EXACT_SEGMENT",
            "justification": (
                "mathematical cap1 fallback after CoACD native merge graph refused cap1"
            ),
            "threshold_sweep": "NOT_APPLICABLE_SINGLE_GEOMETRY",
            "max_vertices": K8_MAX_VERTICES,
        },
        "K16_K32_contract": {
            "method": "COACD_PER_EXACT_SEGMENT",
            "thresholds_m": list(THRESHOLDS_M),
            "per_segment_caps": {"16": 2, "32": 4},
            "max_vertices": COACD_MAX_VERTICES,
            "coacd_common": COACD_COMMON,
        },
        "selection_contract": {
            "launch_floor": "UNCHANGED_E182_P_R_G_V1",
            "score": "UNCHANGED_P_R_G_WORST_NORMALIZED_V1",
            "per_K": "launch_floor_then_score_then_actual_hulls_runtime_threshold_id",
            "full_role": "FORBIDDEN_UNTIL_PRODUCTION_FREEZE",
        },
        "v1_protocol": {
            "path": relative_to_repo(V1_PROTOCOL_PATH),
            "sha256": sha256_file(V1_PROTOCOL_PATH),
        },
        "v1_build_failure": _v1_failure_evidence(),
        "oracle": {
            "path": relative_to_repo(oracle_path),
            "sha256": segments["oracle"]["sha256"],
            "bounds_m": oracle_mesh.bounds.tolist(),
            "volume_m3": float(oracle_mesh.volume),
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
            raise RuntimeError("existing attempt2-v2 protocol differs")
        return existing
    atomic_json(PROTOCOL_PATH, payload)
    return payload


def _load_protocol() -> dict[str, Any]:
    """Require the exact frozen v2 protocol and all parent authority hashes."""
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_V2_BUILD_OR_SCORE":
        raise RuntimeError("attempt2-v2 protocol is not frozen")
    if protocol["builder_source"]["sha256"] != sha256_file(Path(__file__)):
        raise RuntimeError("attempt2-v2 builder changed after freeze")
    if protocol["v1_protocol"]["sha256"] != sha256_file(V1_PROTOCOL_PATH):
        raise RuntimeError("attempt2-v1 protocol SHA changed")
    if protocol["topology"]["v1_segment_manifest"]["sha256"] != sha256_file(
        V1_SEGMENT_MANIFEST_PATH
    ):
        raise RuntimeError("attempt2-v1 segment manifest SHA changed")
    if protocol["original_fixture"]["sha256"] != sha256_file(DEFAULT_FIXTURE):
        raise RuntimeError("original task-query fixture changed")
    return protocol


def _coacd_parts(
    mesh: trimesh.Trimesh,
    *,
    threshold_m: float,
    max_hulls: int,
) -> list[tuple[Any, Any]]:
    """Run one frozen CoACD call without loading manifold in this process."""
    coacd = importlib.import_module("coacd")
    coacd.set_log_level("warn")
    return coacd.run_coacd(
        coacd.Mesh(mesh.vertices, mesh.faces),
        threshold=threshold_m,
        max_convex_hull=max_hulls,
        preprocess_mode=COACD_COMMON["preprocess_mode"],
        preprocess_resolution=COACD_COMMON["preprocess_resolution"],
        resolution=COACD_COMMON["resolution"],
        mcts_nodes=COACD_COMMON["mcts_nodes"],
        mcts_iterations=COACD_COMMON["mcts_iterations"],
        mcts_max_depth=COACD_COMMON["mcts_max_depth"],
        pca=COACD_COMMON["pca"],
        merge=COACD_COMMON["merge"],
        decimate=COACD_COMMON["decimate"],
        max_ch_vertex=COACD_MAX_VERTICES,
        extrude=COACD_COMMON["extrude"],
        extrude_margin=COACD_COMMON["extrude_margin_m"],
        apx_mode=COACD_COMMON["apx_mode"],
        seed=COACD_COMMON["seed"],
        real_metric=COACD_COMMON["real_metric"],
    )


def _export_part_v2(
    vertices: np.ndarray,
    faces: np.ndarray,
    path: Path,
    *,
    part_index: int,
    segment_index: int,
    local_part_index: int,
    max_vertices: int,
) -> dict[str, Any]:
    """Export one convex v2 part under its method-specific vertex ceiling."""
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
        or len(mesh.vertices) > max_vertices
    ):
        raise RuntimeError(f"invalid v2 convex part {part_index}")
    result = _export_mesh(mesh, path)
    result.update(
        {
            "part_index": part_index,
            "segment_index": segment_index,
            "local_part_index": local_part_index,
            "coacd_vertex_count": coacd_vertex_count,
            "removed_unreferenced_vertex_count": coacd_vertex_count
            - result["vertex_count"],
            "convex": True,
        }
    )
    return result


def build_candidate(row: dict[str, Any]) -> dict[str, Any]:
    """Build one exact-segment-hull or segmented-CoACD candidate."""
    protocol = _load_protocol()
    frozen_rows = [
        value
        for value in protocol["candidate_family"]
        if value["candidate_id"] == row["candidate_id"]
    ]
    if frozen_rows != [row]:
        raise RuntimeError("candidate row differs from frozen v2 family")
    segments = _load_v1_segments()
    root = CANDIDATE_ROOT / row["candidate_id"]
    manifest_path = root / "manifest.json"
    parameters = {
        "method": row["method"],
        "threshold_m": row["threshold_m"],
        "max_convex_hull": row["max_hulls"],
        "per_segment_max_convex_hull": row["per_segment_max_hulls"],
        "max_ch_vertex": row["max_vertices"],
        "topology": protocol["topology"],
        "coacd_common": COACD_COMMON if row["method"] == "SEGMENTED_COACD" else None,
    }
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
        raise RuntimeError(
            f"non-resumable attempt2-v2 candidate: {row['candidate_id']}"
        )
    started = time.perf_counter()
    parts = []
    per_segment = []
    for segment in segments["segments"]:
        mesh = trimesh.load(
            repo_path(segment["path"]),
            force="mesh",
            process=False,
            maintain_order=True,
        )
        if row["method"] == "EXACT_SEGMENT_CONVEX_HULL":
            hull = mesh.convex_hull
            result = [(hull.vertices, hull.faces)]
        else:
            result = _coacd_parts(
                mesh,
                threshold_m=float(row["threshold_m"]),
                max_hulls=int(row["per_segment_max_hulls"]),
            )
        if not 1 <= len(result) <= int(row["per_segment_max_hulls"]):
            raise RuntimeError(
                f"{row['candidate_id']}: segment {segment['segment_index']} "
                "hull cap violated"
            )
        first_part_index = len(parts)
        for local_part_index, (vertices, faces) in enumerate(result):
            part_index = len(parts)
            parts.append(
                _export_part_v2(
                    vertices,
                    faces,
                    root / "parts" / f"part_{part_index:03d}.obj",
                    part_index=part_index,
                    segment_index=segment["segment_index"],
                    local_part_index=local_part_index,
                    max_vertices=int(row["max_vertices"]),
                )
            )
        per_segment.append(
            {
                "segment_index": segment["segment_index"],
                "segment_sha256": segment["sha256"],
                "first_part_index": first_part_index,
                "hull_count": len(result),
            }
        )
    if len(parts) > int(row["max_hulls"]):
        raise RuntimeError(f"{row['candidate_id']}: total hull cap violated")
    manifest = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v2_build",
        "status": "BUILD_PASS",
        "selection_eligible": False,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "object_key": OBJECT_KEY,
        "candidate_id": row["candidate_id"],
        "protocol": {
            "path": relative_to_repo(PROTOCOL_PATH),
            "sha256": sha256_file(PROTOCOL_PATH),
        },
        "segment_manifest": {
            "path": relative_to_repo(V1_SEGMENT_MANIFEST_PATH),
            "sha256": sha256_file(V1_SEGMENT_MANIFEST_PATH),
        },
        "oracle_cleaned_mesh": {
            "path": protocol["oracle"]["path"],
            "sha256": protocol["oracle"]["sha256"],
        },
        "parameters": parameters,
        "method": row["method"],
        "max_hulls": row["max_hulls"],
        "hull_count": len(parts),
        "max_part_vertex_count": max(part["vertex_count"] for part in parts),
        "total_vertex_count": sum(part["vertex_count"] for part in parts),
        "total_face_count": sum(part["face_count"] for part in parts),
        "build_wall_seconds": time.perf_counter() - started,
        "per_segment": per_segment,
        "parts": parts,
    }
    manifest["candidate_asset_sha256"] = _candidate_asset_sha256(manifest)
    atomic_json(manifest_path, manifest)
    print(
        "E182_SEGMENTED_COACD_V2=PASS "
        f"candidate={row['candidate_id']} hulls={len(parts)} "
        f"wall={manifest['build_wall_seconds']:.3f}"
    )
    return manifest


def build_candidates() -> dict[str, Any]:
    """Build or resume all seven frozen v2 candidates."""
    protocol = _load_protocol()
    manifests = [build_candidate(row) for row in protocol["candidate_family"]]
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v2_build",
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
    if payload["candidate_count"] != 7:
        raise RuntimeError("attempt2-v2 candidate family incomplete")
    atomic_json(ATTEMPT_ROOT / "build_summary.json", payload)
    return payload


def build_fixture() -> dict[str, Any]:
    """Freeze the bucket003-only seven-candidate v2 query fixture."""
    protocol = _load_protocol()
    summary = json.loads(
        (ATTEMPT_ROOT / "build_summary.json").read_text(encoding="utf-8")
    )
    if summary.get("status") != "BUILD_PASS" or summary.get("candidate_count") != 7:
        raise RuntimeError("attempt2-v2 build summary incomplete")
    original = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    cases = [case for case in original["cases"] if case["case_id"] == CASE_ID]
    if len(cases) != 1:
        raise RuntimeError("bucket003 dev case changed")
    rows = []
    for frozen in protocol["candidate_family"]:
        manifest_path = CANDIDATE_ROOT / frozen["candidate_id"] / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "object_key": OBJECT_KEY,
                "candidate_id": manifest["candidate_id"],
                "candidate_asset_sha256": manifest["candidate_asset_sha256"],
                "manifest": {
                    "path": relative_to_repo(manifest_path),
                    "sha256": sha256_file(manifest_path),
                },
                "max_hulls": int(frozen["max_hulls"]),
                "actual_hulls": int(manifest["hull_count"]),
                "threshold_m": float(frozen["threshold_m"]),
                "max_vertices": int(frozen["max_vertices"]),
            }
        )
    payload = {
        **original,
        "stage": "S2_task_query_fixture_attempt2_segmented_x2y4_v2",
        "status": "FROZEN",
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "freeze_timing": "AFTER_V2_BUILD_BEFORE_V2_SCORE",
        "attempt2_v2_protocol": {
            "path": relative_to_repo(PROTOCOL_PATH),
            "sha256": sha256_file(PROTOCOL_PATH),
        },
        "original_fixture": {
            "path": relative_to_repo(DEFAULT_FIXTURE),
            "sha256": sha256_file(DEFAULT_FIXTURE),
        },
        "cases": cases,
        "candidates": rows,
        "candidate_count": len(rows),
        "candidate_identity_sha256": _candidate_identity_sha256(rows),
    }
    if FIXTURE_PATH.exists():
        existing = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("existing attempt2-v2 fixture differs")
        return existing
    atomic_json(FIXTURE_PATH, payload)
    return payload


def parse_args() -> argparse.Namespace:
    """Parse explicit v2 stages."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("freeze-protocol", "build-candidates", "build-fixture")
    )
    return parser.parse_args()


def main() -> int:
    """Execute one attempt2-v2 stage."""
    args = parse_args()
    if args.command == "freeze-protocol":
        payload = freeze_protocol()
        print(f"E182_ATTEMPT2_V2_PROTOCOL={payload['status']}")
        return 0
    if args.command == "build-candidates":
        payload = build_candidates()
        print(
            f"E182_ATTEMPT2_V2_BUILD={payload['status']} "
            f"candidates={payload['candidate_count']}"
        )
        return 0
    payload = build_fixture()
    print(
        f"E182_ATTEMPT2_V2_FIXTURE={payload['status']} "
        f"candidates={payload['candidate_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
