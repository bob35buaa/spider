#!/usr/bin/env python3
"""Build approved simultaneous double-plane candidates for E182 attempt2 v9."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import build_segmented_coacd_attempt2_v5 as v5
import build_task_aware_preseg_v8 as v8
import evaluate_task_queries as core
import evaluate_task_queries_v5 as v5_eval
import numpy as np
import trimesh
from build_task_query_fixture import DEFAULT_OUTPUT_ROOT
from e182_common import atomic_json, relative_to_repo, repo_path, sha256_file

ATTEMPT_ROOT = DEFAULT_OUTPUT_ROOT / "attempt2_task_aware_preseg_v9_double_plane"
PROTOCOL_PATH = ATTEMPT_ROOT / "attempt2_v9_protocol_manifest.json"
SEGMENT_ROOT = ATTEMPT_ROOT / "segments"
BASE_ROOT = ATTEMPT_ROOT / "coacd_bases"
CANDIDATE_ROOT = ATTEMPT_ROOT / "candidates"
STATIC_P_ROOT = ATTEMPT_ROOT / "static_p_screen"

OBJECT_KEY = v8.OBJECT_KEY
CASE_ID = v8.CASE_ID
SOURCE_SEGMENT_INDEX = v8.SOURCE_SEGMENT_INDEX
UNCHANGED_SEGMENT_INDICES = v8.UNCHANGED_SEGMENT_INDICES
THRESHOLDS_M = v8.THRESHOLDS_M
TARGET_HULLS = v8.TARGET_HULLS
BASE_MAX_HULLS_PER_SEGMENT = v8.BASE_MAX_HULLS_PER_SEGMENT
BASE_SAFETY_MAX_HULLS_PER_SEGMENT = v8.BASE_SAFETY_MAX_HULLS_PER_SEGMENT
FINAL_MAX_VERTICES = v8.FINAL_MAX_VERTICES
P_FLOOR = v8.P_FLOOR
VOLUME_CLOSURE_TOLERANCE_M3 = 1e-8
GEOMETRY_ROUNDTRIP_TOLERANCE_M = 1e-15

V8_PROTOCOL_PATH = v8.PROTOCOL_PATH
V8_BUILD_PATH = v8.ATTEMPT_ROOT / "build_summary.json"
V8_STATIC_PATH = v8.STATIC_P_ROOT / "static_p_aggregate.json"
V8_VISUAL_PATH = v8.ATTEMPT_ROOT / "visual_diagnostic" / "visual_manifest.json"

_coacd_parts = v8._coacd_parts


def _threshold_tag(threshold_m: float) -> str:
    return v8._threshold_tag(threshold_m)


def _artifact_reference(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise RuntimeError(f"required v9 parent artifact is missing: {path}")
    return {
        "path": relative_to_repo(path),
        "sha256": sha256_file(path),
    }


def _sorted_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    order = np.lexsort((vertices[:, 2], vertices[:, 1], vertices[:, 0]))
    return vertices[order]


def _sorted_face_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    rows = []
    for face in np.asarray(mesh.faces, dtype=np.int64):
        triangle = vertices[face]
        vertex_order = np.lexsort((triangle[:, 2], triangle[:, 1], triangle[:, 0]))
        rows.append(triangle[vertex_order].reshape(-1))
    values = np.asarray(rows, dtype=np.float64)
    row_order = np.lexsort(
        tuple(values[:, index] for index in reversed(range(values.shape[1])))
    )
    return values[row_order]


def _source_segment() -> tuple[dict[str, Any], trimesh.Trimesh]:
    manifest = v8._segment_manifest()
    source = next(
        row
        for row in manifest["segments"]
        if int(row["segment_index"]) == SOURCE_SEGMENT_INDEX
    )
    mesh = trimesh.load(
        repo_path(source["path"]),
        force="mesh",
        process=False,
        maintain_order=True,
    )
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("v9 source segment3 is not a Trimesh")
    if source["sha256"] != sha256_file(repo_path(source["path"])):
        raise RuntimeError("v9 source segment3 SHA changed")
    return source, mesh


@lru_cache(maxsize=1)
def _double_plane_meshes() -> tuple[trimesh.Trimesh, ...]:
    source, mesh = _source_segment()
    atlas = v8.derive_transition_atlas()
    axis = int(atlas["dominant_axis"])
    planes = [float(value) for value in atlas["candidate_planes_m"]]
    if planes != sorted(planes) or len(planes) != 2:
        raise RuntimeError("v9 requires exactly two ordered frozen planes")
    bounds = np.asarray(source["cell_bounds_m"], dtype=np.float64)
    edges = [float(bounds[0, axis]), *planes, float(bounds[1, axis])]
    children: list[trimesh.Trimesh] = []
    for low_axis, high_axis in zip(edges[:-1], edges[1:], strict=True):
        low = bounds[0].copy()
        high = bounds[1].copy()
        low[axis] = low_axis
        high[axis] = high_axis
        box = trimesh.creation.box(
            extents=high - low,
            transform=trimesh.transformations.translation_matrix((low + high) / 2.0),
        )
        child = trimesh.boolean.intersection([mesh, box], engine="manifold")
        if not isinstance(child, trimesh.Trimesh):
            raise RuntimeError("v9 double-plane boolean did not return a mesh")
        children.append(child)
    return tuple(children)


@lru_cache(maxsize=1)
def double_plane_topology() -> dict[str, Any]:
    """Validate the approved simultaneous double-plane topology without writing."""
    source, mesh = _source_segment()
    atlas = v8.derive_transition_atlas()
    axis = int(atlas["dominant_axis"])
    planes = [float(value) for value in atlas["candidate_planes_m"]]
    bounds = np.asarray(source["cell_bounds_m"], dtype=np.float64)
    edges = [float(bounds[0, axis]), *planes, float(bounds[1, axis])]
    children = list(_double_plane_meshes())
    child_rows = []
    for region_index, (child, low_axis, high_axis) in enumerate(
        zip(children, edges[:-1], edges[1:], strict=True)
    ):
        child_rows.append(
            {
                "region_index": region_index,
                "segment_index": SOURCE_SEGMENT_INDEX + region_index,
                "axis_interval_m": [low_axis, high_axis],
                "watertight": bool(child.is_watertight),
                "winding_consistent": bool(child.is_winding_consistent),
                "volume_m3": float(child.volume),
                "vertex_count": int(len(child.vertices)),
                "face_count": int(len(child.faces)),
                "bounds_m": child.bounds.tolist(),
            }
        )
    if not all(
        row["watertight"] and row["winding_consistent"] and row["volume_m3"] > 0.0
        for row in child_rows
    ):
        raise RuntimeError("v9 child topology is not closed positive geometry")
    volume_sum = float(sum(row["volume_m3"] for row in child_rows))
    volume_delta = volume_sum - float(mesh.volume)
    if abs(volume_delta) > VOLUME_CLOSURE_TOLERANCE_M3:
        raise RuntimeError("v9 double-plane volume does not close")
    transition_assignments = []
    for row in atlas["transition_rows"]:
        coordinate = float(row["point_object_local_m"][axis])
        region_index = int(np.searchsorted(planes, coordinate, side="right"))
        transition_assignments.append(
            {
                "label": row["label"],
                "source_family": row["source_family"],
                "source_pose_index": int(row["source_pose_index"]),
                "global_pose_index": int(row["global_pose_index"]),
                "coordinate_m": coordinate,
                "region_index": region_index,
            }
        )
    if sorted(row["region_index"] for row in transition_assignments) != [0, 1, 2]:
        raise RuntimeError("v9 frozen transitions are not isolated one per region")
    return {
        "name": "SIMULTANEOUS_DOUBLE_FROZEN_PLANE_SPLIT_OF_X2Y4_SEGMENT3",
        "source_segment_index": SOURCE_SEGMENT_INDEX,
        "source_segment_sha256": source["sha256"],
        "source_segment_manifest": _artifact_reference(v8.V1_SEGMENT_MANIFEST_PATH),
        "v8_split_manifests": [
            {
                "plane_index": plane_index,
                **_artifact_reference(
                    v8._plane_segment_root(plane_index) / "manifest.json"
                ),
            }
            for plane_index in (0, 1)
        ],
        "source_segment_count": 8,
        "final_segment_count": 10,
        "minimum_hulls": 10,
        "dominant_axis": axis,
        "dominant_axis_name": atlas["dominant_axis_name"],
        "planes_m": planes,
        "source_volume_m3": float(mesh.volume),
        "child_volume_sum_m3": volume_sum,
        "volume_delta_m3": volume_delta,
        "volume_closure_tolerance_m3": VOLUME_CLOSURE_TOLERANCE_M3,
        "children": child_rows,
        "transition_assignments": transition_assignments,
        "unchanged_original_segment_indices": list(UNCHANGED_SEGMENT_INDICES),
        "unchanged_segment_remap": {
            str(index): remap_original_segment_index(index)
            for index in UNCHANGED_SEGMENT_INDICES
        },
        "cross_presegment_merge": False,
        "direct_source_intersections": True,
    }


def candidate_family() -> list[dict[str, Any]]:
    """Return the frozen three-threshold by K16/K32 v9 family."""
    planes = double_plane_topology()["planes_m"]
    return [
        {
            "candidate_id": (
                f"taskpreseg_v9_both_{_threshold_tag(threshold_m)}_"
                f"k{max_hulls:02d}_v{FINAL_MAX_VERTICES:03d}"
            ),
            "method": "SIMULTANEOUS_DOUBLE_FROZEN_PLANE_PRESEG_THEN_ISOLATED_COACD",
            "planes_m": planes,
            "simultaneous_plane_count": 2,
            "threshold_m": threshold_m,
            "max_hulls": max_hulls,
            "minimum_segment_hulls": 10,
            "base_max_hulls_per_segment": BASE_MAX_HULLS_PER_SEGMENT,
            "max_vertices": FINAL_MAX_VERTICES,
        }
        for threshold_m in THRESHOLDS_M
        for max_hulls in TARGET_HULLS
    ]


def remap_original_segment_index(segment_index: int) -> int:
    """Map seven unchanged x2x4 segments into the ten-segment topology."""
    if segment_index in (0, 1, 2):
        return segment_index
    if segment_index in (4, 5, 6, 7):
        return segment_index + 2
    raise ValueError("source segment is replaced or outside the x2y4 topology")


def _geometry_fingerprint(
    direct: trimesh.Trimesh, frozen: trimesh.Trimesh
) -> dict[str, Any]:
    direct_vertices = _sorted_vertices(direct)
    frozen_vertices = _sorted_vertices(frozen)
    same_vertex_count = len(direct_vertices) == len(frozen_vertices)
    direct_faces = _sorted_face_vertices(direct)
    frozen_faces = _sorted_face_vertices(frozen)
    same_face_count = len(direct_faces) == len(frozen_faces)
    vertex_delta = (
        float(np.max(np.abs(direct_vertices - frozen_vertices)))
        if same_vertex_count
        else float("inf")
    )
    face_delta = (
        float(np.max(np.abs(direct_faces - frozen_faces)))
        if same_face_count
        else float("inf")
    )
    return {
        "same_vertex_count": same_vertex_count,
        "same_face_count": same_face_count,
        "sorted_vertex_max_abs_m": vertex_delta,
        "sorted_face_vertex_max_abs_m": face_delta,
        "volume_abs_delta_m3": abs(float(direct.volume) - float(frozen.volume)),
        "bounds_max_abs_m": float(np.max(np.abs(direct.bounds - frozen.bounds))),
    }


@lru_cache(maxsize=1)
def reuse_evidence() -> dict[str, Any]:
    """Bind 21 unchanged v4 bases and six exact v8 outer-child bases."""
    unchanged = v8.unaffected_base_evidence()
    direct_children = list(_double_plane_meshes())
    outer_specs = (
        (0, 0, 3, direct_children[0], "LEFT_OUTER"),
        (1, 1, 4, direct_children[2], "RIGHT_OUTER"),
    )
    outer_rows = []
    for threshold_m in THRESHOLDS_M:
        for plane_index, side_index, segment_index, direct, role in outer_specs:
            split_manifest_path = v8._plane_segment_root(plane_index) / "manifest.json"
            split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
            split_row = next(
                row
                for row in split_manifest["children"]
                if int(row["side_index"]) == side_index
                and int(row["segment_index"]) == segment_index
            )
            frozen = trimesh.load(
                repo_path(split_row["path"]),
                force="mesh",
                process=False,
                maintain_order=True,
            )
            if not isinstance(frozen, trimesh.Trimesh):
                raise RuntimeError("v9 frozen outer segment is not a mesh")
            fingerprint = _geometry_fingerprint(direct, frozen)
            fingerprint_exact = (
                fingerprint["same_vertex_count"]
                and fingerprint["same_face_count"]
                and fingerprint["sorted_vertex_max_abs_m"]
                <= GEOMETRY_ROUNDTRIP_TOLERANCE_M
                and fingerprint["sorted_face_vertex_max_abs_m"]
                <= GEOMETRY_ROUNDTRIP_TOLERANCE_M
                and fingerprint["volume_abs_delta_m3"] <= GEOMETRY_ROUNDTRIP_TOLERANCE_M
                and fingerprint["bounds_max_abs_m"] <= GEOMETRY_ROUNDTRIP_TOLERANCE_M
            )
            if not fingerprint_exact:
                raise RuntimeError("v9 outer child does not match frozen v8 geometry")
            base_path = (
                v8._new_base_root(plane_index, threshold_m)
                / f"segment_{segment_index:03d}"
                / "manifest.json"
            )
            base_payload = json.loads(base_path.read_text(encoding="utf-8"))
            if (
                base_payload.get("status") != "BUILD_PASS"
                or base_payload.get("segment_sha256") != split_row["sha256"]
                or not np.isclose(
                    float(base_payload.get("threshold_m", -1.0)),
                    threshold_m,
                    atol=0.0,
                    rtol=0.0,
                )
            ):
                raise RuntimeError("v9 frozen outer CoACD base changed")
            for part in base_payload["parts"]:
                if part["sha256"] != sha256_file(repo_path(part["path"])):
                    raise RuntimeError("v9 frozen outer CoACD part SHA changed")
            outer_rows.append(
                {
                    "role": role,
                    "threshold_m": threshold_m,
                    "v9_segment_index": 3 if role == "LEFT_OUTER" else 5,
                    "v8_plane_index": plane_index,
                    "v8_segment_index": segment_index,
                    "actual_hulls": int(base_payload["actual_hulls"]),
                    "segment": _artifact_reference(repo_path(split_row["path"])),
                    "path": relative_to_repo(base_path),
                    "sha256": sha256_file(base_path),
                    "geometry_fingerprint": fingerprint,
                    "geometry_fingerprint_exact": True,
                }
            )
    return {
        "source_segment_indices": list(UNCHANGED_SEGMENT_INDICES),
        "thresholds_m": list(THRESHOLDS_M),
        "unchanged_v4_manifest_count": int(unchanged["manifest_count"]),
        "unchanged_v4_manifests": unchanged["manifests"],
        "outer_v8_manifest_count": len(outer_rows),
        "outer_v8_manifests": outer_rows,
        "fresh_middle_coacd_count": len(THRESHOLDS_M),
        "middle_segment_index": 4,
    }


def static_p_gate(*, tp: int, phantom: int, missed: int) -> dict[str, Any]:
    """Apply the unchanged v8 P precision and recall floor."""
    return v8.static_p_gate(tp=tp, phantom=phantom, missed=missed)


def select_static_p_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain only true floor passes and apply the frozen v9 ordering."""
    passing = [dict(row) for row in rows if row["static_p_gate"]["status"] == "PASS"]
    passing.sort(
        key=lambda row: (
            float(row["p_score"]),
            int(row["actual_hulls"]),
            float(row["threshold_m"]),
            int(row["max_hulls"]),
            row["candidate_id"],
        )
    )
    return passing


def _source_dependencies() -> list[dict[str, str]]:
    sources = (
        ("UNCHANGED_EXACT_C_AND_P_MATH", Path(core.__file__).resolve()),
        ("V5_DETERMINISTIC_REDUCER", Path(v5.__file__).resolve()),
        ("V5_QUERY_EVALUATOR_AUTHORITY", Path(v5_eval.__file__).resolve()),
        ("V8_PARENT_BUILD_AND_STATIC_P", Path(v8.__file__).resolve()),
        ("V9_BUILD_AND_STATIC_P", Path(__file__).resolve()),
    )
    paths = [path for _, path in sources]
    if len(set(paths)) != len(paths):
        raise RuntimeError("v9 source dependency roles are not unique")
    return [
        {
            "role": role,
            "path": relative_to_repo(path),
            "sha256": sha256_file(path),
        }
        for role, path in sources
    ]


def protocol_payload() -> dict[str, Any]:
    """Construct the complete approved v9 protocol without writing artifacts."""
    topology = double_plane_topology()
    family = candidate_family()
    reuse = reuse_evidence()
    parent_bindings = {
        "v8_protocol": _artifact_reference(V8_PROTOCOL_PATH),
        "v8_build": _artifact_reference(V8_BUILD_PATH),
        "v8_static": _artifact_reference(V8_STATIC_PATH),
        "v8_visual": _artifact_reference(V8_VISUAL_PATH),
    }
    return {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v9_double_plane_protocol",
        "status": "FROZEN_BEFORE_V9_SEGMENT_OR_COACD_OR_SCORE",
        "selection_eligible": False,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "user_authorization": "APPROVED_V9_SIMULTANEOUS_DOUBLE_FROZEN_PLANES",
        "reason": "V8_SINGLE_PLANE_ZERO_OF_TWELVE_REQUIRES_ISOLATING_ALL_THREE_TRANSITIONS",
        "object_key": OBJECT_KEY,
        "case_id": CASE_ID,
        "topology": topology,
        "candidate_family": family,
        "candidate_count": len(family),
        "target_hulls": list(TARGET_HULLS),
        "thresholds_m": list(THRESHOLDS_M),
        "k8_policy": "FORBIDDEN_MINIMUM_TEN_SEGMENTS_EXCEEDS_K8",
        "reuse": reuse,
        "base_decomposition": {
            "middle_only_fresh_coacd": True,
            "fresh_middle_coacd_count": len(THRESHOLDS_M),
            "requested_max_hulls_per_segment": BASE_MAX_HULLS_PER_SEGMENT,
            "safety_max_hulls_per_segment": BASE_SAFETY_MAX_HULLS_PER_SEGMENT,
            "coacd_config_source": parent_bindings["v8_protocol"],
        },
        "reducer": {
            "source": _artifact_reference(v8.V5_PROTOCOL_PATH),
            "cross_presegment_merge": False,
            "minimum_hulls_per_candidate": 10,
        },
        "physics_contract": json.loads(v8.V5_PROTOCOL_PATH.read_text(encoding="utf-8"))[
            "physics_contract"
        ],
        "static_p_gate": {
            "authority_pose_count": 882,
            "oracle_contact_count": 27,
            "precision_floor": P_FLOOR,
            "recall_floor": P_FLOOR,
            "third_work_point": "TP_GE_19_AND_PHANTOM_LE_8",
            "required_before_full_prg": True,
        },
        "stop_action": "STOP_V9_NO_MORE_PLANES_OR_FLOOR_CHANGES",
        "selection_contract": {
            "static_p": "FLOOR_THEN_P_SCORE_ACTUAL_HULLS_THRESHOLD_K_ID",
            "full_prg": "UNCHANGED_E182_P_R_G_V1",
            "full_role": "FORBIDDEN_UNTIL_COMPLETE_P_R_G_AND_PHYSICS_PASS",
        },
        "parent_bindings": parent_bindings,
        "source_dependencies": _source_dependencies(),
    }


def freeze_protocol() -> dict[str, Any]:
    """Freeze v9 before any segment, CoACD, candidate, or score artifact."""
    non_protocol_files = [
        path
        for path in ATTEMPT_ROOT.rglob("*")
        if path.is_file() and path != PROTOCOL_PATH
    ]
    if non_protocol_files:
        raise RuntimeError("cannot freeze v9 after segment/build/score artifacts")
    payload = protocol_payload()
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("existing v9 protocol differs")
        return existing
    atomic_json(PROTOCOL_PATH, payload)
    return payload


def _load_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol != protocol_payload():
        raise RuntimeError("v9 protocol or a bound dependency changed")
    return protocol


def _require_artifact_reference(value: Any, path: Path, *, label: str) -> None:
    expected = _artifact_reference(path)
    if (
        not isinstance(value, dict)
        or value.get("path") != expected["path"]
        or value.get("sha256") != expected["sha256"]
    ):
        raise RuntimeError(f"v9 {label} reference changed")


def _validate_segment_manifest(
    payload: dict[str, Any], *, protocol: dict[str, Any]
) -> dict[str, Any]:
    topology = protocol["topology"]
    source, _ = _source_segment()
    if (
        payload.get("experiment_id") != "E182"
        or payload.get("stage") != "S2_decomposition_attempt2_v9_double_plane_segments"
        or payload.get("status") != "PASS"
        or payload.get("selection_eligible") is not False
        or payload.get("heldout_access") != "NOT_ACCESSED_DEV3_ONLY"
        or payload.get("planes_m") != topology["planes_m"]
        or int(payload.get("dominant_axis", -1)) != int(topology["dominant_axis"])
        or int(payload.get("child_count", -1)) != 3
        or int(payload.get("final_segment_count", -1)) != 10
    ):
        raise RuntimeError("non-resumable v9 segment manifest identity")
    _require_artifact_reference(
        payload.get("protocol"), PROTOCOL_PATH, label="segment protocol"
    )
    source_payload = payload.get("source_segment")
    if (
        not isinstance(source_payload, dict)
        or source_payload.get("path") != source["path"]
        or source_payload.get("sha256") != source["sha256"]
        or not np.isclose(
            float(source_payload.get("volume_m3", np.nan)),
            float(topology["source_volume_m3"]),
        )
    ):
        raise RuntimeError("non-resumable v9 segment source")
    children = payload.get("children")
    if not isinstance(children, list) or len(children) != 3:
        raise RuntimeError("non-resumable v9 segment children")
    for region_index, (child, expected) in enumerate(
        zip(children, topology["children"], strict=True)
    ):
        segment_index = SOURCE_SEGMENT_INDEX + region_index
        path = SEGMENT_ROOT / f"segment_{segment_index:03d}.obj"
        if (
            int(child.get("region_index", -1)) != region_index
            or int(child.get("segment_index", -1)) != segment_index
            or int(child.get("source_segment_index", -1)) != SOURCE_SEGMENT_INDEX
            or child.get("axis_interval_m") != expected["axis_interval_m"]
            or child.get("path") != relative_to_repo(path)
            or child.get("sha256") != sha256_file(path)
            or not bool(child.get("watertight"))
            or not bool(child.get("winding_consistent"))
            or float(child.get("volume_m3", 0.0)) <= 0.0
            or not np.isclose(
                float(child.get("volume_m3", np.nan)),
                float(expected["volume_m3"]),
            )
        ):
            raise RuntimeError("non-resumable v9 segment child")
    volume_sum = float(sum(float(row["volume_m3"]) for row in children))
    volume_delta = volume_sum - float(source_payload["volume_m3"])
    if (
        not np.isclose(float(payload.get("child_volume_sum_m3", np.nan)), volume_sum)
        or not np.isclose(float(payload.get("volume_delta_m3", np.nan)), volume_delta)
        or abs(volume_delta) > VOLUME_CLOSURE_TOLERANCE_M3
    ):
        raise RuntimeError("non-resumable v9 segment volume closure")
    return payload


def build_segments() -> dict[str, Any]:
    """Export the three exact double-plane child solids once."""
    protocol = _load_protocol()
    manifest_path = SEGMENT_ROOT / "manifest.json"
    if manifest_path.exists():
        return _validate_segment_manifest(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            protocol=protocol,
        )
    source, source_mesh = _source_segment()
    child_rows = []
    for region_index, (child, expected) in enumerate(
        zip(_double_plane_meshes(), protocol["topology"]["children"], strict=True)
    ):
        segment_index = SOURCE_SEGMENT_INDEX + region_index
        row = v8._export_mesh(
            child,
            SEGMENT_ROOT / f"segment_{segment_index:03d}.obj",
        )
        row.update(
            {
                "region_index": region_index,
                "segment_index": segment_index,
                "source_segment_index": SOURCE_SEGMENT_INDEX,
                "axis_interval_m": expected["axis_interval_m"],
            }
        )
        child_rows.append(row)
    volume_sum = float(sum(float(row["volume_m3"]) for row in child_rows))
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v9_double_plane_segments",
        "status": "PASS",
        "selection_eligible": False,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "protocol": _artifact_reference(PROTOCOL_PATH),
        "planes_m": protocol["topology"]["planes_m"],
        "dominant_axis": protocol["topology"]["dominant_axis"],
        "source_segment": {
            "path": source["path"],
            "sha256": source["sha256"],
            "volume_m3": float(source_mesh.volume),
        },
        "child_count": len(child_rows),
        "final_segment_count": 10,
        "child_volume_sum_m3": volume_sum,
        "volume_delta_m3": volume_sum - float(source_mesh.volume),
        "children": child_rows,
    }
    _validate_segment_manifest(payload, protocol=protocol)
    atomic_json(manifest_path, payload)
    return payload


def _base_root(threshold_m: float) -> Path:
    return BASE_ROOT / _threshold_tag(threshold_m)


def _frozen_threshold(threshold_m: float) -> float:
    protocol = _load_protocol()
    values = [
        float(value)
        for value in protocol["thresholds_m"]
        if np.isclose(float(value), threshold_m, atol=0.0, rtol=0.0)
    ]
    if len(values) != 1:
        raise ValueError("v9 threshold outside frozen family")
    return values[0]


def _load_segments() -> dict[str, Any]:
    path = SEGMENT_ROOT / "manifest.json"
    return _validate_segment_manifest(
        json.loads(path.read_text(encoding="utf-8")), protocol=_load_protocol()
    )


def _validate_middle_base_manifest(
    payload: dict[str, Any],
    *,
    threshold_m: float,
    segments: dict[str, Any],
    require_native_log: bool,
) -> dict[str, Any]:
    segment = next(
        row for row in segments["children"] if int(row["segment_index"]) == 4
    )
    root = _base_root(threshold_m) / "segment_004"
    if (
        payload.get("experiment_id") != "E182"
        or payload.get("stage") != "S2_decomposition_attempt2_v9_middle_base_child"
        or payload.get("status") != "BUILD_PASS"
        or payload.get("selection_eligible") is not False
        or payload.get("heldout_access") != "NOT_ACCESSED_DEV3_ONLY"
        or not np.isclose(
            float(payload.get("threshold_m", np.nan)),
            threshold_m,
            atol=0.0,
            rtol=0.0,
        )
        or int(payload.get("segment_index", -1)) != 4
        or payload.get("segment_path") != segment["path"]
        or payload.get("segment_sha256") != segment["sha256"]
        or int(payload.get("requested_max_hulls", -1)) != BASE_MAX_HULLS_PER_SEGMENT
    ):
        raise RuntimeError("non-resumable v9 middle base identity or parameters")
    _require_artifact_reference(
        payload.get("protocol"), PROTOCOL_PATH, label="middle-base protocol"
    )
    parts = payload.get("parts")
    if (
        not isinstance(parts, list)
        or not 1 <= len(parts) <= BASE_SAFETY_MAX_HULLS_PER_SEGMENT
        or int(payload.get("actual_hulls", -1)) != len(parts)
        or bool(payload.get("request_satisfied"))
        != (len(parts) <= BASE_MAX_HULLS_PER_SEGMENT)
    ):
        raise RuntimeError("non-resumable v9 middle base hull inventory")
    for part_index, part in enumerate(parts):
        expected_path = root / "parts" / f"part_{part_index:03d}.obj"
        if (
            int(part.get("part_index", -1)) != part_index
            or int(part.get("local_part_index", -1)) != part_index
            or int(part.get("segment_index", -1)) != 4
            or part.get("path") != relative_to_repo(expected_path)
            or part.get("sha256") != sha256_file(expected_path)
            or not bool(part.get("convex"))
            or not bool(part.get("watertight"))
            or not bool(part.get("winding_consistent"))
            or not 1 <= int(part.get("vertex_count", 0)) <= 32
        ):
            raise RuntimeError("non-resumable v9 middle base part")
    native_log = payload.get("native_log")
    if require_native_log or native_log is not None:
        _require_artifact_reference(
            native_log, root / "native.log", label="middle-base native log"
        )
    return payload


def build_middle_base(threshold_m: float) -> dict[str, Any]:
    """Run CoACD only for v9 segment4 at one frozen threshold."""
    threshold_m = _frozen_threshold(threshold_m)
    segments = _load_segments()
    segment = next(
        row for row in segments["children"] if int(row["segment_index"]) == 4
    )
    root = _base_root(threshold_m) / "segment_004"
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        return _validate_middle_base_manifest(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            threshold_m=threshold_m,
            segments=segments,
            require_native_log=False,
        )
    mesh = trimesh.load(
        repo_path(segment["path"]),
        force="mesh",
        process=False,
        maintain_order=True,
    )
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("v9 middle segment is not a mesh")
    started = time.perf_counter()
    result = _coacd_parts(
        mesh,
        threshold_m=threshold_m,
        max_hulls=BASE_MAX_HULLS_PER_SEGMENT,
    )
    if not 1 <= len(result) <= BASE_SAFETY_MAX_HULLS_PER_SEGMENT:
        raise RuntimeError("v9 middle CoACD safety hull count violated")
    parts = [
        v8._export_part_v2(
            vertices,
            faces,
            root / "parts" / f"part_{index:03d}.obj",
            part_index=index,
            segment_index=4,
            local_part_index=index,
            max_vertices=32,
        )
        for index, (vertices, faces) in enumerate(result)
    ]
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v9_middle_base_child",
        "status": "BUILD_PASS",
        "selection_eligible": False,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "protocol": _artifact_reference(PROTOCOL_PATH),
        "threshold_m": threshold_m,
        "segment_index": 4,
        "segment_path": segment["path"],
        "segment_sha256": segment["sha256"],
        "requested_max_hulls": BASE_MAX_HULLS_PER_SEGMENT,
        "request_satisfied": len(parts) <= BASE_MAX_HULLS_PER_SEGMENT,
        "actual_hulls": len(parts),
        "wall_seconds": time.perf_counter() - started,
        "parts": parts,
    }
    _validate_middle_base_manifest(
        payload,
        threshold_m=threshold_m,
        segments=segments,
        require_native_log=False,
    )
    atomic_json(manifest_path, payload)
    print(
        "E182_V9_MIDDLE_BASE=PASS "
        f"threshold={threshold_m:.3f} hulls={len(parts)} "
        f"requested={BASE_MAX_HULLS_PER_SEGMENT} "
        f"wall={payload['wall_seconds']:.3f}"
    )
    return payload


def _run_middle_base_subprocess(threshold_m: float) -> dict[str, Any]:
    root = _base_root(threshold_m) / "segment_004"
    manifest_path = root / "manifest.json"
    log_path = root / "native.log"
    if manifest_path.exists():
        existing = build_middle_base(threshold_m)
        return _validate_middle_base_manifest(
            existing,
            threshold_m=threshold_m,
            segments=_load_segments(),
            require_native_log=existing.get("native_log") is not None,
        )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "build-middle-base",
        "--threshold-m",
        str(threshold_m),
    ]
    completed = subprocess.run(
        command,
        cwd=v8.REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    log_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_log = log_path.with_suffix(".log.tmp")
    temporary_log.write_text(output, encoding="utf-8")
    os.replace(temporary_log, log_path)
    if completed.returncode != 0:
        raise RuntimeError(
            f"v9 middle threshold={threshold_m} RC={completed.returncode}; "
            f"log={log_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["native_log"] = _artifact_reference(log_path)
    _validate_middle_base_manifest(
        manifest,
        threshold_m=threshold_m,
        segments=_load_segments(),
        require_native_log=True,
    )
    atomic_json(manifest_path, manifest)
    if completed.stdout.strip():
        print(completed.stdout.strip(), flush=True)
    return manifest


def _reused_v4_entry(threshold_m: float, original_index: int) -> dict[str, Any]:
    path = (
        v8.V4_BASE_ROOT
        / _threshold_tag(threshold_m)
        / f"segment_{original_index:03d}"
        / "manifest.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    evidence = reuse_evidence()
    reference = next(
        row
        for row in evidence["unchanged_v4_manifests"]
        if np.isclose(float(row["threshold_m"]), threshold_m, atol=0.0, rtol=0.0)
        and int(row["segment_index"]) == original_index
    )
    if reference["path"] != relative_to_repo(path) or reference[
        "sha256"
    ] != sha256_file(path):
        raise RuntimeError("v9 reused v4 base reference changed")
    return {
        "source": "REUSED_V4_UNCHANGED_SEGMENT",
        "original_segment_index": original_index,
        "segment_index": remap_original_segment_index(original_index),
        "actual_hulls": int(payload["actual_hulls"]),
        "path": relative_to_repo(path),
        "sha256": sha256_file(path),
    }


def _reused_outer_entry(threshold_m: float, segment_index: int) -> dict[str, Any]:
    evidence = reuse_evidence()
    row = next(
        value
        for value in evidence["outer_v8_manifests"]
        if np.isclose(float(value["threshold_m"]), threshold_m, atol=0.0, rtol=0.0)
        and int(value["v9_segment_index"]) == segment_index
    )
    path = repo_path(row["path"])
    if row["sha256"] != sha256_file(path):
        raise RuntimeError("v9 reused outer base reference changed")
    return {
        "source": "REUSED_V8_OUTER_SEGMENT",
        "segment_index": segment_index,
        "v8_plane_index": int(row["v8_plane_index"]),
        "v8_segment_index": int(row["v8_segment_index"]),
        "actual_hulls": int(row["actual_hulls"]),
        "path": row["path"],
        "sha256": row["sha256"],
    }


def _fresh_middle_entry(threshold_m: float) -> dict[str, Any]:
    path = _base_root(threshold_m) / "segment_004" / "manifest.json"
    payload = _validate_middle_base_manifest(
        json.loads(path.read_text(encoding="utf-8")),
        threshold_m=threshold_m,
        segments=_load_segments(),
        require_native_log=False,
    )
    return {
        "source": "V9_FRESH_MIDDLE_SEGMENT",
        "segment_index": 4,
        "actual_hulls": int(payload["actual_hulls"]),
        "path": relative_to_repo(path),
        "sha256": sha256_file(path),
    }


def _composite_base_payload(threshold_m: float) -> dict[str, Any]:
    entries = [
        _reused_v4_entry(threshold_m, index) for index in UNCHANGED_SEGMENT_INDICES
    ]
    entries.extend(
        [
            _reused_outer_entry(threshold_m, 3),
            _fresh_middle_entry(threshold_m),
            _reused_outer_entry(threshold_m, 5),
        ]
    )
    entries.sort(key=lambda row: int(row["segment_index"]))
    if [int(row["segment_index"]) for row in entries] != list(range(10)):
        raise RuntimeError("v9 composite base segment set is incomplete")
    return {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v9_composite_base",
        "status": "BUILD_PASS",
        "selection_eligible": False,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "protocol": _artifact_reference(PROTOCOL_PATH),
        "planes_m": double_plane_topology()["planes_m"],
        "threshold_m": threshold_m,
        "segment_count": len(entries),
        "actual_hulls_per_segment": [int(row["actual_hulls"]) for row in entries],
        "total_hulls": sum(int(row["actual_hulls"]) for row in entries),
        "child_manifests": entries,
    }


def _write_composite_base(threshold_m: float) -> dict[str, Any]:
    path = _base_root(threshold_m) / "manifest.json"
    payload = _composite_base_payload(threshold_m)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("existing v9 composite base differs")
        return existing
    atomic_json(path, payload)
    return payload


def _validate_base_summary(
    payload: dict[str, Any], *, protocol: dict[str, Any]
) -> dict[str, Any]:
    if (
        payload.get("experiment_id") != "E182"
        or payload.get("stage") != "S2_decomposition_attempt2_v9_bases"
        or payload.get("status") != "BUILD_PASS"
        or payload.get("heldout_access") != "NOT_ACCESSED_DEV3_ONLY"
        or int(payload.get("fresh_middle_count", -1)) != 3
        or int(payload.get("composite_base_count", -1)) != 3
        or not np.isfinite(float(payload.get("wall_seconds", np.nan)))
        or float(payload["wall_seconds"]) < 0.0
    ):
        raise RuntimeError("non-resumable v9 base summary identity")
    _require_artifact_reference(
        payload.get("protocol"), PROTOCOL_PATH, label="base-summary protocol"
    )
    _require_artifact_reference(
        payload.get("segment_manifest"),
        SEGMENT_ROOT / "manifest.json",
        label="base-summary segments",
    )
    _load_segments()
    middle_refs = payload.get("fresh_middle_manifests")
    base_refs = payload.get("base_manifests")
    totals = payload.get("total_hulls")
    if not all(
        isinstance(value, list) and len(value) == 3
        for value in (middle_refs, base_refs, totals)
    ):
        raise RuntimeError("non-resumable v9 base summary inventory")
    for index, threshold_m in enumerate(protocol["thresholds_m"]):
        threshold_m = float(threshold_m)
        middle_path = _base_root(threshold_m) / "segment_004" / "manifest.json"
        if not np.isclose(
            float(middle_refs[index].get("threshold_m", np.nan)),
            threshold_m,
            atol=0.0,
            rtol=0.0,
        ):
            raise RuntimeError("non-resumable v9 middle summary order")
        _require_artifact_reference(
            middle_refs[index], middle_path, label="base-summary middle"
        )
        _validate_middle_base_manifest(
            json.loads(middle_path.read_text(encoding="utf-8")),
            threshold_m=threshold_m,
            segments=_load_segments(),
            require_native_log=True,
        )
        base_path = _base_root(threshold_m) / "manifest.json"
        if not np.isclose(
            float(base_refs[index].get("threshold_m", np.nan)),
            threshold_m,
            atol=0.0,
            rtol=0.0,
        ):
            raise RuntimeError("non-resumable v9 base summary order")
        _require_artifact_reference(base_refs[index], base_path, label="base summary")
        base = _write_composite_base(threshold_m)
        if int(totals[index]) != int(base["total_hulls"]):
            raise RuntimeError("non-resumable v9 base summary hull total")
    return payload


def build_bases() -> dict[str, Any]:
    """Build three fresh middle bases and three ten-segment composites."""
    protocol = _load_protocol()
    summary_path = ATTEMPT_ROOT / "base_summary.json"
    if summary_path.exists():
        return _validate_base_summary(
            json.loads(summary_path.read_text(encoding="utf-8")), protocol=protocol
        )
    _load_segments()
    started = time.perf_counter()
    thresholds = [float(value) for value in protocol["thresholds_m"]]
    middle = [_run_middle_base_subprocess(value) for value in thresholds]
    bases = [_write_composite_base(value) for value in thresholds]
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v9_bases",
        "status": "BUILD_PASS",
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "protocol": _artifact_reference(PROTOCOL_PATH),
        "segment_manifest": _artifact_reference(SEGMENT_ROOT / "manifest.json"),
        "fresh_middle_count": len(middle),
        "composite_base_count": len(bases),
        "total_hulls": [int(row["total_hulls"]) for row in bases],
        "wall_seconds": time.perf_counter() - started,
        "fresh_middle_manifests": [
            {
                "threshold_m": threshold_m,
                **_artifact_reference(
                    _base_root(threshold_m) / "segment_004" / "manifest.json"
                ),
            }
            for threshold_m in thresholds
        ],
        "base_manifests": [
            {
                "threshold_m": threshold_m,
                **_artifact_reference(_base_root(threshold_m) / "manifest.json"),
            }
            for threshold_m in thresholds
        ],
    }
    _validate_base_summary(payload, protocol=protocol)
    atomic_json(summary_path, payload)
    return payload


def _load_composite_base(threshold_m: float) -> dict[int, list[Any]]:
    base_path = _base_root(threshold_m) / "manifest.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base != _composite_base_payload(threshold_m):
        raise RuntimeError("v9 composite base changed")
    groups: dict[int, list[Any]] = {}
    for child_entry in base["child_manifests"]:
        child_path = repo_path(child_entry["path"])
        if child_entry["sha256"] != sha256_file(child_path):
            raise RuntimeError("v9 composite child manifest SHA changed")
        child = json.loads(child_path.read_text(encoding="utf-8"))
        segment_index = int(child_entry["segment_index"])
        group = []
        for local_index, part in enumerate(child["parts"]):
            part_path = repo_path(part["path"])
            if part["sha256"] != sha256_file(part_path):
                raise RuntimeError("v9 composite base part SHA changed")
            reducer_part = v8.ReducerPart(
                mesh=trimesh.load(
                    part_path,
                    force="mesh",
                    process=False,
                    maintain_order=True,
                ),
                segment_index=segment_index,
                lineage=(local_index,),
            )
            v8._checked_reducer_part(reducer_part)
            group.append(reducer_part)
        groups[segment_index] = group
    if set(groups) != set(range(10)) or any(not group for group in groups.values()):
        raise RuntimeError("v9 composite base groups are incomplete")
    return groups


def _candidate_parameters(
    row: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    return {
        "method": row["method"],
        "planes_m": row["planes_m"],
        "threshold_m": row["threshold_m"],
        "max_convex_hull": row["max_hulls"],
        "max_ch_vertex": row["max_vertices"],
        "base_decomposition": protocol["base_decomposition"],
        "reducer": protocol["reducer"],
        "topology": protocol["topology"],
        "physics_contract": protocol["physics_contract"],
    }


def _validate_candidate_manifest(
    payload: dict[str, Any],
    *,
    row: dict[str, Any],
    parameters: dict[str, Any],
    root: Path,
    base_manifest_path: Path,
) -> dict[str, Any]:
    if (
        payload.get("experiment_id") != "E182"
        or payload.get("stage") != "S2_decomposition_attempt2_v9_candidate"
        or payload.get("status") != "BUILD_PASS"
        or payload.get("selection_eligible") is not False
        or payload.get("heldout_access") != "NOT_ACCESSED_DEV3_ONLY"
        or payload.get("object_key") != OBJECT_KEY
        or payload.get("candidate_id") != row["candidate_id"]
        or payload.get("parameters") != parameters
    ):
        raise RuntimeError("non-resumable v9 candidate identity or parameters")
    for key in ("method", "planes_m", "threshold_m", "max_hulls"):
        if payload.get(key) != row[key]:
            raise RuntimeError(f"non-resumable v9 candidate field: {key}")
    _require_artifact_reference(
        payload.get("protocol"), PROTOCOL_PATH, label="candidate protocol"
    )
    _require_artifact_reference(
        payload.get("base_manifest"), base_manifest_path, label="candidate base"
    )
    _require_artifact_reference(
        payload.get("segment_manifest"),
        SEGMENT_ROOT / "manifest.json",
        label="candidate segments",
    )
    oracle = json.loads(V8_PROTOCOL_PATH.read_text(encoding="utf-8"))["topology"][
        "oracle"
    ]
    if payload.get("oracle_cleaned_mesh") != {
        "path": oracle["path"],
        "sha256": oracle["sha256"],
    }:
        raise RuntimeError("non-resumable v9 candidate oracle")
    parts = payload.get("parts")
    if not isinstance(parts, list) or not 10 <= len(parts) <= int(row["max_hulls"]):
        raise RuntimeError("non-resumable v9 candidate part inventory")
    groups: dict[int, list[dict[str, Any]]] = {}
    for part_index, part in enumerate(parts):
        expected_path = root / "parts" / f"part_{part_index:03d}.obj"
        if (
            int(part.get("part_index", -1)) != part_index
            or part.get("path") != relative_to_repo(expected_path)
            or part.get("sha256") != sha256_file(expected_path)
            or not bool(part.get("convex"))
            or not bool(part.get("watertight"))
            or not bool(part.get("winding_consistent"))
            or not 1 <= int(part.get("vertex_count", 0)) <= FINAL_MAX_VERTICES
        ):
            raise RuntimeError("non-resumable v9 candidate part")
        groups.setdefault(int(part.get("segment_index", -1)), []).append(part)
    if set(groups) != set(range(10)) or any(not group for group in groups.values()):
        raise RuntimeError("non-resumable v9 candidate segment coverage")
    per_segment = payload.get("per_segment")
    if not isinstance(per_segment, list) or len(per_segment) != 10:
        raise RuntimeError("non-resumable v9 candidate per-segment inventory")
    cursor = 0
    for segment_index, entry in enumerate(per_segment):
        count = len(groups[segment_index])
        if (
            int(entry.get("segment_index", -1)) != segment_index
            or int(entry.get("first_part_index", -1)) != cursor
            or int(entry.get("hull_count", -1)) != count
            or int(entry.get("source_base_hull_count", 0)) < count
        ):
            raise RuntimeError("non-resumable v9 candidate segment metadata")
        cursor += count
    base_hulls = int(payload.get("base_hull_count", -1))
    merge_trace = payload.get("merge_trace")
    merge_count = int(payload.get("merge_count", -1))
    if (
        int(payload.get("hull_count", -1)) != len(parts)
        or not isinstance(merge_trace, list)
        or merge_count != len(merge_trace)
        or base_hulls - len(parts) != merge_count
        or int(payload.get("max_part_vertex_count", -1))
        != max(int(part["vertex_count"]) for part in parts)
        or int(payload.get("total_vertex_count", -1))
        != sum(int(part["vertex_count"]) for part in parts)
        or int(payload.get("total_face_count", -1))
        != sum(int(part["face_count"]) for part in parts)
    ):
        raise RuntimeError("non-resumable v9 candidate aggregate metadata")
    if payload.get("candidate_asset_sha256") != v8._candidate_asset_sha256(payload):
        raise RuntimeError("non-resumable v9 candidate asset digest")
    return payload


def build_candidate(row: dict[str, Any]) -> dict[str, Any]:
    """Reduce one immutable ten-segment v9 base to K16 or K32."""
    protocol = _load_protocol()
    if [value for value in protocol["candidate_family"] if value == row] != [row]:
        raise RuntimeError("v9 candidate row differs from frozen family")
    root = CANDIDATE_ROOT / row["candidate_id"]
    manifest_path = root / "manifest.json"
    base_manifest_path = _base_root(float(row["threshold_m"])) / "manifest.json"
    parameters = _candidate_parameters(row, protocol)
    if manifest_path.exists():
        return _validate_candidate_manifest(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            row=row,
            parameters=parameters,
            root=root,
            base_manifest_path=base_manifest_path,
        )
    started = time.perf_counter()
    base_parts = _load_composite_base(float(row["threshold_m"]))
    base_hulls = sum(len(group) for group in base_parts.values())
    reduced, trace = v8.reduce_segmented_parts(
        base_parts, target_hulls=int(row["max_hulls"])
    )
    parts = []
    per_segment = []
    for segment_index, group in sorted(reduced.items()):
        first_part_index = len(parts)
        for local_part_index, part in enumerate(group):
            part_index = len(parts)
            parts.append(
                v8._export_reduced_part(
                    part,
                    root / "parts" / f"part_{part_index:03d}.obj",
                    part_index=part_index,
                    local_part_index=local_part_index,
                )
            )
        per_segment.append(
            {
                "segment_index": segment_index,
                "first_part_index": first_part_index,
                "hull_count": len(group),
                "source_base_hull_count": len(base_parts[segment_index]),
            }
        )
    if not 10 <= len(parts) <= int(row["max_hulls"]):
        raise RuntimeError("v9 final global hull budget violated")
    oracle = json.loads(V8_PROTOCOL_PATH.read_text(encoding="utf-8"))["topology"][
        "oracle"
    ]
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v9_candidate",
        "status": "BUILD_PASS",
        "selection_eligible": False,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "object_key": OBJECT_KEY,
        "candidate_id": row["candidate_id"],
        "protocol": _artifact_reference(PROTOCOL_PATH),
        "base_manifest": _artifact_reference(base_manifest_path),
        "segment_manifest": _artifact_reference(SEGMENT_ROOT / "manifest.json"),
        "oracle_cleaned_mesh": {
            "path": oracle["path"],
            "sha256": oracle["sha256"],
        },
        "parameters": parameters,
        "method": row["method"],
        "planes_m": row["planes_m"],
        "threshold_m": row["threshold_m"],
        "max_hulls": row["max_hulls"],
        "base_hull_count": base_hulls,
        "hull_count": len(parts),
        "merge_count": len(trace),
        "negative_roundoff_clamp_count": sum(
            step["negative_roundoff_clamped"] for step in trace
        ),
        "merge_added_volume_m3": float(sum(step["added_volume_m3"] for step in trace)),
        "max_part_vertex_count": max(int(part["vertex_count"]) for part in parts),
        "total_vertex_count": sum(int(part["vertex_count"]) for part in parts),
        "total_face_count": sum(int(part["face_count"]) for part in parts),
        "build_wall_seconds": time.perf_counter() - started,
        "per_segment": per_segment,
        "merge_trace": trace,
        "parts": parts,
    }
    payload["candidate_asset_sha256"] = v8._candidate_asset_sha256(payload)
    _validate_candidate_manifest(
        payload,
        row=row,
        parameters=parameters,
        root=root,
        base_manifest_path=base_manifest_path,
    )
    atomic_json(manifest_path, payload)
    print(
        "E182_TASK_PRESEG_V9=PASS "
        f"candidate={row['candidate_id']} base={base_hulls} final={len(parts)} "
        f"merges={len(trace)} wall={payload['build_wall_seconds']:.3f}"
    )
    return payload


def _validate_build_summary(
    payload: dict[str, Any], *, protocol: dict[str, Any]
) -> dict[str, Any]:
    family = protocol["candidate_family"]
    if (
        payload.get("experiment_id") != "E182"
        or payload.get("stage") != "S2_decomposition_attempt2_v9_build"
        or payload.get("status") != "BUILD_PASS"
        or payload.get("heldout_access") != "NOT_ACCESSED_DEV3_ONLY"
        or len(family) != 6
        or int(payload.get("candidate_count", -1)) != 6
        or not np.isfinite(float(payload.get("total_wall_seconds", np.nan)))
        or float(payload["total_wall_seconds"]) < 0.0
    ):
        raise RuntimeError("non-resumable v9 build summary identity")
    _require_artifact_reference(
        payload.get("protocol"), PROTOCOL_PATH, label="build-summary protocol"
    )
    base_summary_path = ATTEMPT_ROOT / "base_summary.json"
    _require_artifact_reference(
        payload.get("base_summary"), base_summary_path, label="build-summary bases"
    )
    _validate_base_summary(
        json.loads(base_summary_path.read_text(encoding="utf-8")), protocol=protocol
    )
    references = payload.get("candidate_manifests")
    identifiers = payload.get("candidate_ids")
    actual_hulls = payload.get("actual_hulls")
    if not all(
        isinstance(value, list) and len(value) == 6
        for value in (references, identifiers, actual_hulls)
    ):
        raise RuntimeError("non-resumable v9 build summary inventory")
    for index, (row, reference) in enumerate(zip(family, references, strict=True)):
        path = CANDIDATE_ROOT / row["candidate_id"] / "manifest.json"
        if reference.get("candidate_id") != row["candidate_id"]:
            raise RuntimeError("non-resumable v9 build summary order")
        _require_artifact_reference(reference, path, label="build-summary candidate")
        candidate = _validate_candidate_manifest(
            json.loads(path.read_text(encoding="utf-8")),
            row=row,
            parameters=_candidate_parameters(row, protocol),
            root=path.parent,
            base_manifest_path=_base_root(float(row["threshold_m"])) / "manifest.json",
        )
        if identifiers[index] != row["candidate_id"] or int(actual_hulls[index]) != int(
            candidate["hull_count"]
        ):
            raise RuntimeError("non-resumable v9 build summary metadata")
    return payload


def build_candidates() -> dict[str, Any]:
    """Build all six frozen v9 candidates from three composite bases."""
    protocol = _load_protocol()
    summary_path = ATTEMPT_ROOT / "build_summary.json"
    if summary_path.exists():
        return _validate_build_summary(
            json.loads(summary_path.read_text(encoding="utf-8")), protocol=protocol
        )
    base_summary_path = ATTEMPT_ROOT / "base_summary.json"
    _validate_base_summary(
        json.loads(base_summary_path.read_text(encoding="utf-8")), protocol=protocol
    )
    manifests = [build_candidate(row) for row in protocol["candidate_family"]]
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v9_build",
        "status": "BUILD_PASS",
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "protocol": _artifact_reference(PROTOCOL_PATH),
        "base_summary": _artifact_reference(base_summary_path),
        "candidate_count": len(manifests),
        "candidate_ids": [row["candidate_id"] for row in manifests],
        "actual_hulls": [int(row["hull_count"]) for row in manifests],
        "candidate_manifests": [
            {
                "candidate_id": row["candidate_id"],
                **_artifact_reference(
                    CANDIDATE_ROOT / row["candidate_id"] / "manifest.json"
                ),
            }
            for row in manifests
        ],
        "total_wall_seconds": float(
            sum(float(row["build_wall_seconds"]) for row in manifests)
        ),
    }
    _validate_build_summary(payload, protocol=protocol)
    atomic_json(summary_path, payload)
    return payload


def _candidate_segment_scenes(candidate: dict[str, Any]) -> list[Any]:
    groups: dict[int, list[trimesh.Trimesh]] = {}
    for part in candidate["parts"]:
        path = repo_path(part["path"])
        if part["sha256"] != sha256_file(path):
            raise RuntimeError("v9 static-P candidate part SHA changed")
        groups.setdefault(int(part["segment_index"]), []).append(
            trimesh.load(path, force="mesh", process=False, maintain_order=True)
        )
    if set(groups) != set(range(10)):
        raise RuntimeError("v9 static-P candidate segment set changed")
    scenes = []
    for segment_index in range(10):
        meshes = groups[segment_index]
        union = meshes[0] if len(meshes) == 1 else core.build_exact_union_mesh(meshes)
        scenes.append(core._raycasting_scene(union))
    return scenes


def _static_p_queries() -> tuple[list[dict[str, Any]], np.ndarray]:
    return v8.v6._static_p_queries()


def _validate_static_p_result(
    payload: dict[str, Any], *, row: dict[str, Any], manifest_path: Path
) -> dict[str, Any]:
    if (
        payload.get("experiment_id") != "E182"
        or payload.get("stage") != "S2_decomposition_attempt2_v9_static_P"
        or payload.get("status") != "COMPLETE"
        or payload.get("selection_eligible") is not True
        or payload.get("heldout_access") != "NOT_ACCESSED_DEV3_ONLY"
        or payload.get("candidate_id") != row["candidate_id"]
    ):
        raise RuntimeError("non-resumable v9 static-P identity")
    _require_artifact_reference(
        payload.get("protocol"), PROTOCOL_PATH, label="static-P protocol"
    )
    _require_artifact_reference(
        payload.get("candidate_manifest"), manifest_path, label="static-P candidate"
    )
    for key in ("planes_m", "threshold_m", "max_hulls"):
        if payload.get(key) != row[key]:
            raise RuntimeError(f"non-resumable v9 static-P field: {key}")
    metrics = payload.get("p_pose_contact")
    if not isinstance(metrics, dict):
        raise RuntimeError("non-resumable v9 static-P metrics")
    tp = int(metrics.get("true_positive_count", -1))
    phantom = int(metrics.get("phantom_contact_count", -1))
    missed = int(metrics.get("missed_contact_count", -1))
    true_negative = int(metrics.get("true_negative_count", -1))
    if tp + phantom + missed + true_negative != 882 or tp + missed != 27:
        raise RuntimeError("non-resumable v9 static-P confusion totals")
    gate = static_p_gate(tp=tp, phantom=phantom, missed=missed)
    precision = float(gate["precision"])
    recall = float(gate["recall"])
    expected_score = max((1.0 - precision) / 0.30, (1.0 - recall) / 0.30)
    if (
        payload.get("static_p_gate") != gate
        or bool(payload.get("full_prg_eligible")) != (gate["status"] == "PASS")
        or not np.isclose(float(metrics.get("precision", np.nan)), precision)
        or not np.isclose(float(metrics.get("recall", np.nan)), recall)
        or not np.isclose(float(payload.get("p_score", np.nan)), expected_score)
    ):
        raise RuntimeError("non-resumable v9 static-P gate or score")
    return payload


def evaluate_static_p(row: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one built v9 candidate on the frozen 882-pose P gate."""
    protocol = _load_protocol()
    if [value for value in protocol["candidate_family"] if value == row] != [row]:
        raise RuntimeError("v9 static-P row differs from frozen family")
    result_path = STATIC_P_ROOT / "results" / f"{row['candidate_id']}.json"
    manifest_path = CANDIDATE_ROOT / row["candidate_id"] / "manifest.json"
    candidate = _validate_candidate_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        row=row,
        parameters=_candidate_parameters(row, protocol),
        root=manifest_path.parent,
        base_manifest_path=_base_root(float(row["threshold_m"])) / "manifest.json",
    )
    if result_path.exists():
        return _validate_static_p_result(
            json.loads(result_path.read_text(encoding="utf-8")),
            row=row,
            manifest_path=manifest_path,
        )
    queries, oracle_contact = _static_p_queries()
    scenes = _candidate_segment_scenes(candidate)
    candidate_chunks = []
    started = time.perf_counter()
    for query in queries:
        segment_contacts = []
        for scene in scenes:
            clearance = (
                core._scene_signed_distance(scene, query["points"])
                - query["radii"][None, :]
            )
            segment_contacts.append(clearance.min(axis=1) <= 0.0)
        candidate_chunks.append(np.any(np.asarray(segment_contacts), axis=0))
    candidate_contact = np.concatenate(candidate_chunks)
    metrics = v8.v6._contact_metrics(candidate_contact, oracle_contact)
    gate = static_p_gate(
        tp=int(metrics["true_positive_count"]),
        phantom=int(metrics["phantom_contact_count"]),
        missed=int(metrics["missed_contact_count"]),
    )
    precision = float(metrics["precision"])
    recall = float(metrics["recall"])
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v9_static_P",
        "status": "COMPLETE",
        "selection_eligible": True,
        "full_prg_eligible": gate["status"] == "PASS",
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "candidate_id": row["candidate_id"],
        "planes_m": row["planes_m"],
        "threshold_m": row["threshold_m"],
        "max_hulls": row["max_hulls"],
        "actual_hulls": int(candidate["hull_count"]),
        "protocol": _artifact_reference(PROTOCOL_PATH),
        "candidate_manifest": _artifact_reference(manifest_path),
        "p_pose_contact": metrics,
        "static_p_gate": gate,
        "p_score": max((1.0 - precision) / 0.30, (1.0 - recall) / 0.30),
        "wall_seconds": time.perf_counter() - started,
    }
    _validate_static_p_result(payload, row=row, manifest_path=manifest_path)
    atomic_json(result_path, payload)
    print(
        "E182_V9_STATIC_P=COMPLETE "
        f"candidate={row['candidate_id']} gate={gate['status']} "
        f"precision={precision:.6f} recall={recall:.6f}"
    )
    return payload


def _validate_static_p_aggregate(
    payload: dict[str, Any], *, protocol: dict[str, Any]
) -> dict[str, Any]:
    family = protocol["candidate_family"]
    if (
        payload.get("experiment_id") != "E182"
        or payload.get("stage") != "S2_decomposition_attempt2_v9_static_P_aggregate"
        or payload.get("status") != "COMPLETE"
        or payload.get("selection_eligible") is not True
        or payload.get("heldout_access") != "NOT_ACCESSED_DEV3_ONLY"
        or len(family) != 6
        or int(payload.get("candidate_count", -1)) != 6
    ):
        raise RuntimeError("non-resumable v9 static-P aggregate identity")
    _require_artifact_reference(
        payload.get("protocol"), PROTOCOL_PATH, label="static-P aggregate protocol"
    )
    build_summary_path = ATTEMPT_ROOT / "build_summary.json"
    _require_artifact_reference(
        payload.get("build_summary"),
        build_summary_path,
        label="static-P aggregate build",
    )
    _validate_build_summary(
        json.loads(build_summary_path.read_text(encoding="utf-8")), protocol=protocol
    )
    references = payload.get("results")
    if not isinstance(references, list) or len(references) != 6:
        raise RuntimeError("non-resumable v9 static-P aggregate inventory")
    rows = []
    for row, reference in zip(family, references, strict=True):
        result_path = STATIC_P_ROOT / "results" / f"{row['candidate_id']}.json"
        manifest_path = CANDIDATE_ROOT / row["candidate_id"] / "manifest.json"
        if reference.get("candidate_id") != row["candidate_id"]:
            raise RuntimeError("non-resumable v9 static-P aggregate order")
        _require_artifact_reference(
            reference, result_path, label="static-P aggregate result"
        )
        rows.append(
            _validate_static_p_result(
                json.loads(result_path.read_text(encoding="utf-8")),
                row=row,
                manifest_path=manifest_path,
            )
        )
    selected = select_static_p_candidates(rows)
    expected_ids = [row["candidate_id"] for row in selected]
    if (
        int(payload.get("pass_count", -1)) != len(selected)
        or bool(payload.get("full_prg_eligible")) != bool(selected)
        or payload.get("zero_pass_action")
        != (None if selected else "STOP_V9_NO_MORE_PLANES_OR_FLOOR_CHANGES")
        or payload.get("selected_candidate_ids") != expected_ids
    ):
        raise RuntimeError("non-resumable v9 static-P aggregate selection")
    return payload


def run_static_p() -> dict[str, Any]:
    """Evaluate all six candidates and freeze the full-P/R/G eligible set."""
    protocol = _load_protocol()
    aggregate_path = STATIC_P_ROOT / "static_p_aggregate.json"
    if aggregate_path.exists():
        return _validate_static_p_aggregate(
            json.loads(aggregate_path.read_text(encoding="utf-8")), protocol=protocol
        )
    build_summary_path = ATTEMPT_ROOT / "build_summary.json"
    _validate_build_summary(
        json.loads(build_summary_path.read_text(encoding="utf-8")), protocol=protocol
    )
    rows = [evaluate_static_p(row) for row in protocol["candidate_family"]]
    selected = select_static_p_candidates(rows)
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v9_static_P_aggregate",
        "status": "COMPLETE",
        "selection_eligible": True,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "protocol": _artifact_reference(PROTOCOL_PATH),
        "build_summary": _artifact_reference(build_summary_path),
        "candidate_count": len(rows),
        "pass_count": len(selected),
        "full_prg_eligible": bool(selected),
        "zero_pass_action": (
            None if selected else "STOP_V9_NO_MORE_PLANES_OR_FLOOR_CHANGES"
        ),
        "selected_candidate_ids": [row["candidate_id"] for row in selected],
        "results": [
            {
                "candidate_id": row["candidate_id"],
                **_artifact_reference(
                    STATIC_P_ROOT / "results" / f"{row['candidate_id']}.json"
                ),
            }
            for row in rows
        ],
    }
    _validate_static_p_aggregate(payload, protocol=protocol)
    atomic_json(aggregate_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    """Parse one explicit v9 construction or static-P stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "protocol",
            "build-segments",
            "build-middle-base",
            "build-bases",
            "build-candidates",
            "static-p",
        ),
    )
    parser.add_argument("--threshold-m", type=float)
    return parser.parse_args()


def main() -> int:
    """Run one approved v9 stage."""
    args = parse_args()
    if args.stage == "protocol":
        payload = freeze_protocol()
        print(
            "E182_V9_PROTOCOL=FROZEN "
            f"candidates={payload['candidate_count']} segments={payload['topology']['final_segment_count']}"
        )
        return 0
    if args.stage == "build-segments":
        payload = build_segments()
        print(f"E182_V9_SEGMENTS={payload['status']} children={payload['child_count']}")
        return 0
    if args.stage == "build-middle-base":
        if args.threshold_m is None:
            raise SystemExit("build-middle-base requires --threshold-m")
        build_middle_base(args.threshold_m)
        return 0
    if args.stage == "build-bases":
        payload = build_bases()
        print(
            f"E182_V9_BASES={payload['status']} bases={payload['composite_base_count']}"
        )
        return 0
    if args.stage == "build-candidates":
        payload = build_candidates()
        print(
            f"E182_V9_BUILD={payload['status']} candidates={payload['candidate_count']}"
        )
        return 0
    payload = run_static_p()
    print(
        f"E182_V9_STATIC_P_AGGREGATE={payload['status']} "
        f"pass={payload['pass_count']}/{payload['candidate_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
