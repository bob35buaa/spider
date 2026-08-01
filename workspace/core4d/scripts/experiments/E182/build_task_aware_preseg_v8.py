#!/usr/bin/env python3
"""Build task-derived local pre-segmentation candidates for E182 attempt2 v8."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import evaluate_task_queries as core
import numpy as np
import render_segment3_partition_v7 as v7_visual
import search_segment_hybrid_v6 as v6
import trimesh
from build_segmented_coacd_attempt2 import (
    DEFAULT_FIXTURE,
    _candidate_asset_sha256,
    _export_mesh,
)
from build_segmented_coacd_attempt2 import SEGMENT_ROOT as V1_SEGMENT_ROOT
from build_segmented_coacd_attempt2_v2 import _coacd_parts, _export_part_v2
from build_segmented_coacd_attempt2_v4 import BASE_ROOT as V4_BASE_ROOT
from build_segmented_coacd_attempt2_v4 import PROTOCOL_PATH as V4_PROTOCOL_PATH
from build_segmented_coacd_attempt2_v4 import (
    ReducerPart,
    _checked_reducer_part,
    _export_reduced_part,
)
from build_segmented_coacd_attempt2_v5 import PROTOCOL_PATH as V5_PROTOCOL_PATH
from build_segmented_coacd_attempt2_v5 import reduce_segmented_parts
from build_task_query_fixture import DEFAULT_OUTPUT_ROOT
from e182_common import (
    REPO_ROOT,
    atomic_json,
    relative_to_repo,
    repo_path,
    sha256_file,
)
from search_segment3_partition_v7 import PROTOCOL_PATH as V7_PROTOCOL_PATH
from search_segment3_partition_v7 import SEARCH_PATH as V7_SEARCH_PATH

ATTEMPT_ROOT = DEFAULT_OUTPUT_ROOT / "attempt2_task_aware_preseg_v8"
PROTOCOL_PATH = ATTEMPT_ROOT / "attempt2_v8_protocol_manifest.json"
SEGMENT_ROOT = ATTEMPT_ROOT / "segments"
BASE_ROOT = ATTEMPT_ROOT / "coacd_bases"
CANDIDATE_ROOT = ATTEMPT_ROOT / "candidates"
STATIC_P_ROOT = ATTEMPT_ROOT / "static_p_screen"
FIXTURE_PATH = ATTEMPT_ROOT / "query_fixture_manifest.json"

OBJECT_KEY = "bucket003"
CASE_ID = "bucket003_20231018_001_p1"
SOURCE_SEGMENT_INDEX = 3
UNCHANGED_SEGMENT_INDICES = (0, 1, 2, 4, 5, 6, 7)
THRESHOLDS_M = (0.005, 0.010, 0.020)
TARGET_HULLS = (16, 32)
BASE_MAX_HULLS_PER_SEGMENT = 4
BASE_SAFETY_MAX_HULLS_PER_SEGMENT = 16
SEGMENT_PARALLEL_WORKERS = 4
FINAL_MAX_VERTICES = 256
P_FLOOR = 0.70
V1_SEGMENT_MANIFEST_PATH = V1_SEGMENT_ROOT / "segment_manifest.json"

for variable in ("OMP_NUM_THREADS", "TBB_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")


def _threshold_tag(threshold_m: float) -> str:
    return f"t{round(threshold_m * 1000):03d}"


def _segment_manifest() -> dict[str, Any]:
    manifest = json.loads(V1_SEGMENT_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "PASS"
        or manifest.get("segment_count") != 8
        or manifest.get("heldout_access") != "NOT_ACCESSED_DEV3_ONLY"
    ):
        raise RuntimeError("v8 source segment manifest changed")
    for row in manifest["segments"]:
        path = repo_path(row["path"])
        if row["sha256"] != sha256_file(path):
            raise RuntimeError("v8 source segment SHA changed")
    return manifest


def _oracle_and_queries() -> tuple[list[dict[str, Any]], np.ndarray, trimesh.Trimesh]:
    queries, oracle_contact = v6._static_p_queries()
    row = v7_visual._source_row(
        v7_visual.DIAGNOSTIC_K,
        int(v6.THRESHOLDS_M.index(0.020)),
    )
    _, _, oracle = core._load_candidate_and_oracle(row)
    return queries, oracle_contact, oracle


@lru_cache(maxsize=1)
def derive_transition_atlas() -> dict[str, Any]:
    """Derive every v8 split plane from frozen v7 contact transitions."""
    search = v7_visual._load_search()
    rows = v7_visual.select_diagnostic_rows(search)
    if len(rows) != 2:
        raise RuntimeError("v8 requires exactly two frozen v7 signatures")
    queries, oracle_contact, oracle = _oracle_and_queries()
    oracle_scene = core._raycasting_scene(oracle)
    scene_sets = [
        [core._raycasting_scene(part) for part in v7_visual.hybrid_parts(row)]
        for row in rows
    ]
    contacts_by_row = [[], []]
    transition_rows = []
    global_offset = 0
    for query in queries:
        clearances = [
            v7_visual._clearance(scenes, query["points"], query["radii"])
            for scenes in scene_sets
        ]
        contacts = [clearance.min(axis=1) <= 0.0 for clearance in clearances]
        for index in range(2):
            contacts_by_row[index].append(contacts[index])
        oracle_clearance = (
            core._scene_signed_distance(oracle_scene, query["points"])
            - query["radii"][None, :]
        )
        added = contacts[1] & ~contacts[0]
        removed = contacts[0] & ~contacts[1]
        if removed.any():
            raise RuntimeError("v8 transition family has removed contacts")
        for pose_index in np.flatnonzero(added):
            point_index = int(np.argmin(clearances[1][pose_index]))
            transition_rows.append(
                {
                    "global_pose_index": global_offset + int(pose_index),
                    "source_family": query["source_family"],
                    "source_pose_index": int(pose_index),
                    "point_index": point_index,
                    "point_object_local_m": query["points"][
                        pose_index, point_index
                    ].tolist(),
                    "label": (
                        "TP"
                        if bool(oracle_clearance[pose_index].min() <= 0.0)
                        else "PHANTOM"
                    ),
                    "precision_clearance_m": float(
                        clearances[0][pose_index, point_index]
                    ),
                    "recall_clearance_m": float(clearances[1][pose_index, point_index]),
                    "oracle_point_clearance_m": float(
                        oracle_clearance[pose_index, point_index]
                    ),
                    "oracle_pose_contact": bool(
                        oracle_clearance[pose_index].min() <= 0.0
                    ),
                }
            )
        global_offset += len(contacts[0])
    flat_contacts = [np.concatenate(values) for values in contacts_by_row]
    added_all = flat_contacts[1] & ~flat_contacts[0]
    removed_all = flat_contacts[0] & ~flat_contacts[1]
    if len(added_all) != 882 or not np.array_equal(
        oracle_contact, np.asarray(oracle_contact, dtype=bool)
    ):
        raise RuntimeError("v8 static authority changed")
    if len(transition_rows) != int(added_all.sum()) or removed_all.any():
        raise RuntimeError("v8 transition rows do not match contact delta")
    points = np.asarray(
        [row["point_object_local_m"] for row in transition_rows], dtype=np.float64
    )
    ranges = np.ptp(points, axis=0)
    dominant_axis = int(np.argmax(ranges))
    if int(np.count_nonzero(ranges == ranges[dominant_axis])) != 1:
        raise RuntimeError("v8 transition atlas has no unique dominant axis")
    order = np.argsort(points[:, dominant_axis], kind="stable")
    sorted_coordinates = points[order, dominant_axis]
    if len(np.unique(sorted_coordinates)) != len(sorted_coordinates):
        raise RuntimeError("v8 dominant-axis transition coordinates are not distinct")
    candidate_planes = 0.5 * (sorted_coordinates[:-1] + sorted_coordinates[1:])
    segment3 = next(
        row
        for row in _segment_manifest()["segments"]
        if int(row["segment_index"]) == SOURCE_SEGMENT_INDEX
    )
    cell_bounds = np.asarray(segment3["cell_bounds_m"], dtype=np.float64)
    if np.any(candidate_planes <= cell_bounds[0, dominant_axis]) or np.any(
        candidate_planes >= cell_bounds[1, dominant_axis]
    ):
        raise RuntimeError("v8 task-derived plane lies outside segment3")
    return {
        "authority_pose_count": len(added_all),
        "oracle_contact_count": int(np.asarray(oracle_contact, dtype=bool).sum()),
        "pose_count": len(transition_rows),
        "labels": [row["label"] for row in transition_rows],
        "transition_rows": transition_rows,
        "added_contact_count": int(added_all.sum()),
        "removed_contact_count": int(removed_all.sum()),
        "axis_ranges_m": ranges.tolist(),
        "dominant_axis": dominant_axis,
        "dominant_axis_name": "xyz"[dominant_axis],
        "sorted_transition_row_indices": order.tolist(),
        "sorted_dominant_coordinates_m": sorted_coordinates.tolist(),
        "candidate_planes_m": candidate_planes.tolist(),
        "source_signatures": [
            {
                "diagnostic_label": row["diagnostic_label"],
                "partition": row["partition"],
                "true_positive_count": int(row["true_positive_count"]),
                "phantom_contact_count": int(row["phantom_contact_count"]),
                "missed_contact_count": int(row["missed_contact_count"]),
            }
            for row in rows
        ],
        "v7_protocol": {
            "path": relative_to_repo(V7_PROTOCOL_PATH),
            "sha256": sha256_file(V7_PROTOCOL_PATH),
        },
        "v7_search": {
            "path": relative_to_repo(V7_SEARCH_PATH),
            "sha256": sha256_file(V7_SEARCH_PATH),
        },
    }


def _split_segment3_meshes(
    plane_m: float,
) -> tuple[dict[str, Any], trimesh.Trimesh, list[trimesh.Trimesh]]:
    manifest = _segment_manifest()
    source = next(
        row
        for row in manifest["segments"]
        if int(row["segment_index"]) == SOURCE_SEGMENT_INDEX
    )
    path = repo_path(source["path"])
    mesh = trimesh.load(path, force="mesh", process=False, maintain_order=True)
    axis = int(derive_transition_atlas()["dominant_axis"])
    cell_bounds = np.asarray(source["cell_bounds_m"], dtype=np.float64)
    if not cell_bounds[0, axis] < plane_m < cell_bounds[1, axis]:
        raise ValueError("v8 split plane is outside source segment3")
    child_bounds = []
    for side in range(2):
        low = cell_bounds[0].copy()
        high = cell_bounds[1].copy()
        if side == 0:
            high[axis] = plane_m
        else:
            low[axis] = plane_m
        child_bounds.append((low, high))
    children = []
    for low, high in child_bounds:
        box = trimesh.creation.box(
            extents=high - low,
            transform=trimesh.transformations.translation_matrix((low + high) / 2.0),
        )
        child = trimesh.boolean.intersection([mesh, box], engine="manifold")
        if not isinstance(child, trimesh.Trimesh):
            raise RuntimeError("v8 segment3 boolean split did not return a mesh")
        children.append(child)
    return source, mesh, children


def split_segment3(plane_m: float) -> dict[str, Any]:
    """Validate one task-derived exact local split without exporting artifacts."""
    source, mesh, children = _split_segment3_meshes(plane_m)
    rows = []
    for side, child in enumerate(children):
        rows.append(
            {
                "side_index": side,
                "watertight": bool(child.is_watertight),
                "winding_consistent": bool(child.is_winding_consistent),
                "volume_m3": float(child.volume),
                "vertex_count": int(len(child.vertices)),
                "face_count": int(len(child.faces)),
                "bounds_m": child.bounds.tolist(),
            }
        )
    return {
        "source_segment_index": int(source["segment_index"]),
        "source_segment_sha256": source["sha256"],
        "plane_m": float(plane_m),
        "segment_count_before_split": 8,
        "segment_count_after_split": 9,
        "source_volume_m3": float(mesh.volume),
        "child_volume_sum_m3": float(sum(child.volume for child in children)),
        "volume_delta_m3": float(sum(child.volume for child in children) - mesh.volume),
        "children": rows,
    }


def candidate_family(atlas: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return the frozen 2-plane × 3-threshold × K16/K32 family."""
    if atlas is None:
        atlas = derive_transition_atlas()
    return [
        {
            "candidate_id": (
                f"taskpreseg_v8_p{plane_index:02d}_{_threshold_tag(threshold_m)}_"
                f"k{max_hulls:02d}_v{FINAL_MAX_VERTICES:03d}"
            ),
            "method": "TASK_AWARE_LOCAL_EXACT_PRESEG_THEN_ISOLATED_COACD",
            "plane_index": plane_index,
            "plane_m": float(plane_m),
            "threshold_m": threshold_m,
            "max_hulls": max_hulls,
            "minimum_segment_hulls": 9,
            "base_max_hulls_per_segment": BASE_MAX_HULLS_PER_SEGMENT,
            "max_vertices": FINAL_MAX_VERTICES,
        }
        for plane_index, plane_m in enumerate(atlas["candidate_planes_m"])
        for threshold_m in THRESHOLDS_M
        for max_hulls in TARGET_HULLS
    ]


def remap_original_segment_index(segment_index: int) -> int:
    """Map seven unchanged x2×y4 indices into the nine-segment v8 topology."""
    if segment_index in (0, 1, 2):
        return segment_index
    if segment_index in (4, 5, 6, 7):
        return segment_index + 1
    raise ValueError("source segment is replaced or outside the x2y4 topology")


def unaffected_base_evidence() -> dict[str, Any]:
    """Bind every reused v4 child manifest for the seven unchanged segments."""
    protocol = json.loads(V4_PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_V4_BUILD_OR_SCORE":
        raise RuntimeError("v8 v4 base protocol changed")
    manifests = []
    for threshold_m in THRESHOLDS_M:
        for segment_index in UNCHANGED_SEGMENT_INDICES:
            path = (
                V4_BASE_ROOT
                / _threshold_tag(threshold_m)
                / f"segment_{segment_index:03d}"
                / "manifest.json"
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("status") != "BUILD_PASS"
                or int(payload.get("segment_index", -1)) != segment_index
                or not np.isclose(
                    float(payload.get("threshold_m", -1.0)),
                    threshold_m,
                    atol=0.0,
                    rtol=0.0,
                )
            ):
                raise RuntimeError("v8 reused v4 base manifest changed")
            for part in payload["parts"]:
                if part["sha256"] != sha256_file(repo_path(part["path"])):
                    raise RuntimeError("v8 reused v4 base part SHA changed")
            manifests.append(
                {
                    "threshold_m": threshold_m,
                    "segment_index": segment_index,
                    "actual_hulls": int(payload["actual_hulls"]),
                    "path": relative_to_repo(path),
                    "sha256": sha256_file(path),
                }
            )
    return {
        "source_segment_indices": list(UNCHANGED_SEGMENT_INDICES),
        "thresholds_m": list(THRESHOLDS_M),
        "manifest_count": len(manifests),
        "manifests": manifests,
        "v4_protocol": {
            "path": relative_to_repo(V4_PROTOCOL_PATH),
            "sha256": sha256_file(V4_PROTOCOL_PATH),
        },
    }


def static_p_gate(*, tp: int, phantom: int, missed: int) -> dict[str, Any]:
    """Apply the unchanged P precision/recall floor to one confusion signature."""
    precision = tp / (tp + phantom) if tp + phantom else 0.0
    recall = tp / (tp + missed) if tp + missed else 0.0
    passed = precision >= P_FLOOR and recall >= P_FLOOR
    return {
        "status": "PASS" if passed else "FAIL",
        "precision": precision,
        "recall": recall,
        "p_precision_floor": P_FLOOR,
        "p_recall_floor": P_FLOOR,
    }


def select_static_p_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain and deterministically order only true static-P floor passes."""
    passing = [dict(row) for row in rows if row["static_p_gate"]["status"] == "PASS"]
    passing.sort(
        key=lambda row: (
            float(row["p_score"]),
            int(row["actual_hulls"]),
            int(row["plane_index"]),
            float(row["threshold_m"]),
            int(row["max_hulls"]),
            row["candidate_id"],
        )
    )
    return passing


def _source_dependencies() -> list[dict[str, str]]:
    sources = (
        (
            "UNCHANGED_P_CONTACT_MATH",
            Path(core.__file__).resolve(),
        ),
        (
            "V6_STATIC_QUERY_AND_CONTACT_MATH",
            Path(v6.__file__).resolve(),
        ),
        (
            "V7_SIGNATURE_RECONSTRUCTION",
            Path(v7_visual.__file__).resolve(),
        ),
        (
            "V1_EXACT_SEGMENT_EXPORT",
            Path(_export_mesh.__code__.co_filename).resolve(),
        ),
        (
            "V2_ISOLATED_COACD_EXPORT",
            Path(_coacd_parts.__code__.co_filename).resolve(),
        ),
        (
            "V4_REDUCER_PART_CHECK_AND_EXPORT",
            Path(_checked_reducer_part.__code__.co_filename).resolve(),
        ),
        (
            "V5_DETERMINISTIC_REDUCER",
            Path(reduce_segmented_parts.__code__.co_filename).resolve(),
        ),
        (
            "V8_BUILD_AND_STATIC_P",
            Path(__file__).resolve(),
        ),
    )
    paths = [path for _, path in sources]
    if len(set(paths)) != len(paths):
        raise RuntimeError("v8 runtime source dependency roles are not unique")
    return [
        {
            "role": role,
            "path": relative_to_repo(path),
            "sha256": sha256_file(path),
        }
        for role, path in sources
    ]


def protocol_payload() -> dict[str, Any]:
    """Construct the complete v8 protocol without writing or scoring artifacts."""
    atlas = derive_transition_atlas()
    splits = [split_segment3(float(plane)) for plane in atlas["candidate_planes_m"]]
    family = candidate_family(atlas)
    reused = unaffected_base_evidence()
    v4_protocol = json.loads(V4_PROTOCOL_PATH.read_text(encoding="utf-8"))
    v5_protocol = json.loads(V5_PROTOCOL_PATH.read_text(encoding="utf-8"))
    if v5_protocol.get("status") != (
        "FROZEN_AFTER_V4_BASE_BEFORE_V5_CANDIDATE_OR_SCORE"
    ):
        raise RuntimeError("v8 v5 reducer protocol changed")
    segment_manifest = _segment_manifest()
    return {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v8_task_aware_preseg_protocol",
        "status": "FROZEN_BEFORE_V8_SEGMENT_OR_COACD_OR_SCORE",
        "selection_eligible": False,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "reason": "V5_V6_V7_P_NEGATIVE_REQUIRES_TASK_AWARE_LOCAL_PRESEGMENTATION",
        "object_key": OBJECT_KEY,
        "case_id": CASE_ID,
        "transition_atlas": atlas,
        "topology": {
            "name": "TASK_DERIVED_LOCAL_EXACT_SPLIT_OF_X2Y4_SEGMENT3",
            "source_segment_count": 8,
            "source_segment_index": SOURCE_SEGMENT_INDEX,
            "final_segment_count": 9,
            "dominant_axis": atlas["dominant_axis"],
            "dominant_axis_name": atlas["dominant_axis_name"],
            "plane_generation": "ALL_ADJACENT_MIDPOINTS_OF_SORTED_DISTINCT_TRANSITION_POINTS",
            "task_derived_splits": splits,
            "unchanged_original_segment_indices": list(UNCHANGED_SEGMENT_INDICES),
            "unchanged_segment_remap": {
                str(index): remap_original_segment_index(index)
                for index in UNCHANGED_SEGMENT_INDICES
            },
            "minimum_hulls": 9,
            "cross_presegment_merge": False,
            "volume_closure_tolerance_m3": 1e-8,
            "source_segment_manifest": {
                "path": relative_to_repo(V1_SEGMENT_MANIFEST_PATH),
                "sha256": sha256_file(V1_SEGMENT_MANIFEST_PATH),
            },
            "oracle": segment_manifest["oracle"],
        },
        "candidate_family": family,
        "candidate_count": len(family),
        "target_hulls": list(TARGET_HULLS),
        "thresholds_m": list(THRESHOLDS_M),
        "k8_policy": "NOT_RERUN_MINIMUM_NINE_SEGMENTS_V5_V6_EVIDENCE_RETAINED",
        "reused_v4_bases": reused,
        "base_decomposition": {
            **v4_protocol["base_decomposition"],
            "reused_unchanged_segments": True,
            "new_coacd_children_per_plane_threshold": 2,
        },
        "reducer": {
            **v5_protocol["reducer"],
            "cross_presegment_merge": False,
            "minimum_hulls_per_candidate": 9,
        },
        "physics_contract": v5_protocol["physics_contract"],
        "static_p_gate": {
            "authority_pose_count": 882,
            "oracle_contact_count": 27,
            "precision_floor": P_FLOOR,
            "recall_floor": P_FLOOR,
            "third_work_point": "TP_GE_19_AND_PHANTOM_LE_8",
            "required_before_full_prg": True,
            "zero_pass_action": "STOP_V8_NO_PLANE_OR_FLOOR_CHANGES",
        },
        "selection_contract": {
            "static_p": "FLOOR_THEN_P_SCORE_ACTUAL_HULLS_PLANE_THRESHOLD_K_ID",
            "full_prg": "UNCHANGED_E182_P_R_G_V1",
            "full_role": "FORBIDDEN_UNTIL_COMPLETE_P_R_G_AND_PHYSICS_PASS",
        },
        "original_fixture": {
            "path": relative_to_repo(DEFAULT_FIXTURE),
            "sha256": sha256_file(DEFAULT_FIXTURE),
        },
        "parent_protocols": [
            {
                "role": "V4_REUSED_BASE_AUTHORITY",
                "path": relative_to_repo(V4_PROTOCOL_PATH),
                "sha256": sha256_file(V4_PROTOCOL_PATH),
            },
            {
                "role": "V5_REDUCER_AUTHORITY",
                "path": relative_to_repo(V5_PROTOCOL_PATH),
                "sha256": sha256_file(V5_PROTOCOL_PATH),
            },
        ],
        "source_dependencies": _source_dependencies(),
    }


def freeze_protocol() -> dict[str, Any]:
    """Freeze the complete v8 build and static-P contract before any output."""
    non_protocol_files = [
        path
        for path in ATTEMPT_ROOT.rglob("*")
        if path.is_file() and path != PROTOCOL_PATH
    ]
    if non_protocol_files:
        raise RuntimeError("cannot freeze v8 after segment/build/score artifacts")
    payload = protocol_payload()
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("existing v8 protocol differs")
        return existing
    atomic_json(PROTOCOL_PATH, payload)
    return payload


def _load_protocol() -> dict[str, Any]:
    """Require exact v8 payload, runtime sources, parents, and frozen evidence."""
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol != protocol_payload():
        raise RuntimeError("v8 protocol or a bound dependency changed")
    return protocol


def _artifact_reference(path: Path) -> dict[str, str]:
    return {"path": relative_to_repo(path), "sha256": sha256_file(path)}


def _require_artifact_reference(value: Any, path: Path, *, label: str) -> None:
    expected = _artifact_reference(path)
    if (
        not isinstance(value, dict)
        or value.get("path") != expected["path"]
        or value.get("sha256") != expected["sha256"]
    ):
        raise RuntimeError(f"v8 {label} reference changed")


def _plane_segment_root(plane_index: int) -> Path:
    return SEGMENT_ROOT / f"plane_{plane_index:02d}"


def _validate_split_manifest(
    payload: dict[str, Any],
    *,
    protocol: dict[str, Any],
    plane_index: int,
    plane_m: float,
) -> dict[str, Any]:
    root = _plane_segment_root(plane_index)
    source = next(
        row
        for row in _segment_manifest()["segments"]
        if int(row["segment_index"]) == SOURCE_SEGMENT_INDEX
    )
    if (
        payload.get("experiment_id") != "E182"
        or payload.get("stage") != "S2_decomposition_attempt2_v8_split_segments"
        or payload.get("status") != "PASS"
        or payload.get("selection_eligible") is not False
        or payload.get("heldout_access") != "NOT_ACCESSED_DEV3_ONLY"
        or int(payload.get("plane_index", -1)) != plane_index
        or not np.isclose(
            float(payload.get("plane_m", np.nan)), plane_m, atol=0.0, rtol=0.0
        )
        or int(payload.get("dominant_axis", -1))
        != int(protocol["topology"]["dominant_axis"])
        or int(payload.get("child_count", -1)) != 2
    ):
        raise RuntimeError("non-resumable v8 split manifest identity")
    _require_artifact_reference(
        payload.get("protocol"), PROTOCOL_PATH, label="split protocol"
    )
    source_payload = payload.get("source_segment")
    if (
        not isinstance(source_payload, dict)
        or source_payload.get("path") != source["path"]
        or source_payload.get("sha256") != source["sha256"]
        or not np.isclose(
            float(source_payload.get("volume_m3", np.nan)),
            float(
                protocol["topology"]["task_derived_splits"][plane_index][
                    "source_volume_m3"
                ]
            ),
        )
    ):
        raise RuntimeError("non-resumable v8 split source")
    children = payload.get("children")
    if not isinstance(children, list) or len(children) != 2:
        raise RuntimeError("non-resumable v8 split children")
    for side_index, child in enumerate(children):
        segment_index = SOURCE_SEGMENT_INDEX + side_index
        path = root / f"segment_{segment_index:03d}.obj"
        if (
            int(child.get("side_index", -1)) != side_index
            or int(child.get("segment_index", -1)) != segment_index
            or int(child.get("source_segment_index", -1)) != SOURCE_SEGMENT_INDEX
            or child.get("path") != relative_to_repo(path)
            or child.get("sha256") != sha256_file(path)
            or not bool(child.get("watertight"))
            or not bool(child.get("winding_consistent"))
            or float(child.get("volume_m3", 0.0)) <= 0.0
        ):
            raise RuntimeError("non-resumable v8 split child")
    volume_sum = sum(float(child["volume_m3"]) for child in children)
    volume_delta = volume_sum - float(source_payload["volume_m3"])
    if (
        not np.isclose(float(payload.get("child_volume_sum_m3", np.nan)), volume_sum)
        or not np.isclose(float(payload.get("volume_delta_m3", np.nan)), volume_delta)
        or abs(volume_delta)
        > float(protocol["topology"]["volume_closure_tolerance_m3"])
    ):
        raise RuntimeError("non-resumable v8 split volume closure")
    return payload


def _validate_segment_summary(
    payload: dict[str, Any], *, protocol: dict[str, Any]
) -> dict[str, Any]:
    if (
        payload.get("experiment_id") != "E182"
        or payload.get("stage") != "S2_decomposition_attempt2_v8_all_split_segments"
        or payload.get("status") != "PASS"
        or payload.get("heldout_access") != "NOT_ACCESSED_DEV3_ONLY"
        or int(payload.get("plane_count", -1)) != 2
        or int(payload.get("child_count", -1)) != 4
    ):
        raise RuntimeError("non-resumable v8 segment summary identity")
    _require_artifact_reference(
        payload.get("protocol"), PROTOCOL_PATH, label="segment-summary protocol"
    )
    references = payload.get("manifests")
    if not isinstance(references, list) or len(references) != 2:
        raise RuntimeError("non-resumable v8 segment summary references")
    for plane_index, plane_m in enumerate(
        protocol["transition_atlas"]["candidate_planes_m"]
    ):
        path = _plane_segment_root(plane_index) / "manifest.json"
        reference = references[plane_index]
        if int(reference.get("plane_index", -1)) != plane_index:
            raise RuntimeError("non-resumable v8 segment summary plane order")
        _require_artifact_reference(reference, path, label="segment-summary child")
        child = json.loads(path.read_text(encoding="utf-8"))
        _validate_split_manifest(
            child,
            protocol=protocol,
            plane_index=plane_index,
            plane_m=float(plane_m),
        )
    return payload


def build_segments() -> dict[str, Any]:
    """Export both exact child solids for each frozen task-derived plane."""
    protocol = _load_protocol()
    summary_path = SEGMENT_ROOT / "segment_summary.json"
    if summary_path.exists():
        return _validate_segment_summary(
            json.loads(summary_path.read_text(encoding="utf-8")),
            protocol=protocol,
        )
    manifests = []
    for plane_index, plane_m in enumerate(
        protocol["transition_atlas"]["candidate_planes_m"]
    ):
        root = _plane_segment_root(plane_index)
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifests.append(
                _validate_split_manifest(
                    existing,
                    protocol=protocol,
                    plane_index=plane_index,
                    plane_m=float(plane_m),
                )
            )
            continue
        source, source_mesh, children = _split_segment3_meshes(float(plane_m))
        child_rows = []
        for side_index, child in enumerate(children):
            new_segment_index = SOURCE_SEGMENT_INDEX + side_index
            entry = _export_mesh(
                child,
                root / f"segment_{new_segment_index:03d}.obj",
            )
            entry.update(
                {
                    "side_index": side_index,
                    "segment_index": new_segment_index,
                    "source_segment_index": SOURCE_SEGMENT_INDEX,
                }
            )
            child_rows.append(entry)
        volume_sum = float(sum(row["volume_m3"] for row in child_rows))
        volume_delta = volume_sum - float(source_mesh.volume)
        if abs(volume_delta) > float(
            protocol["topology"]["volume_closure_tolerance_m3"]
        ):
            raise RuntimeError("v8 exported split volume does not close")
        payload = {
            "experiment_id": "E182",
            "stage": "S2_decomposition_attempt2_v8_split_segments",
            "status": "PASS",
            "selection_eligible": False,
            "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
            "protocol": {
                "path": relative_to_repo(PROTOCOL_PATH),
                "sha256": sha256_file(PROTOCOL_PATH),
            },
            "plane_index": plane_index,
            "plane_m": float(plane_m),
            "dominant_axis": protocol["topology"]["dominant_axis"],
            "source_segment": {
                "path": source["path"],
                "sha256": source["sha256"],
                "volume_m3": float(source_mesh.volume),
            },
            "child_count": len(child_rows),
            "child_volume_sum_m3": volume_sum,
            "volume_delta_m3": volume_delta,
            "children": child_rows,
        }
        _validate_split_manifest(
            payload,
            protocol=protocol,
            plane_index=plane_index,
            plane_m=float(plane_m),
        )
        atomic_json(manifest_path, payload)
        manifests.append(payload)
    summary = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v8_all_split_segments",
        "status": "PASS",
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "protocol": _artifact_reference(PROTOCOL_PATH),
        "plane_count": len(manifests),
        "child_count": sum(row["child_count"] for row in manifests),
        "manifests": [
            {
                "plane_index": row["plane_index"],
                "path": relative_to_repo(
                    _plane_segment_root(row["plane_index"]) / "manifest.json"
                ),
                "sha256": sha256_file(
                    _plane_segment_root(row["plane_index"]) / "manifest.json"
                ),
            }
            for row in manifests
        ],
    }
    if summary["plane_count"] != 2 or summary["child_count"] != 4:
        raise RuntimeError("v8 split segment family incomplete")
    _validate_segment_summary(summary, protocol=protocol)
    atomic_json(summary_path, summary)
    return summary


def _load_plane_segments(plane_index: int) -> dict[str, Any]:
    path = _plane_segment_root(plane_index) / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = _load_protocol()
    planes = protocol["transition_atlas"]["candidate_planes_m"]
    return _validate_split_manifest(
        payload,
        protocol=protocol,
        plane_index=plane_index,
        plane_m=float(planes[plane_index]),
    )


def _new_base_root(plane_index: int, threshold_m: float) -> Path:
    return BASE_ROOT / f"plane_{plane_index:02d}" / _threshold_tag(threshold_m)


def _validate_base_child_manifest(
    payload: dict[str, Any],
    *,
    plane_index: int,
    threshold_m: float,
    segment_index: int,
    segments: dict[str, Any],
    root: Path,
    require_native_log: bool,
) -> dict[str, Any]:
    segment = next(
        row
        for row in segments["children"]
        if int(row["segment_index"]) == segment_index
    )
    if (
        payload.get("experiment_id") != "E182"
        or payload.get("stage") != "S2_decomposition_attempt2_v8_base_child"
        or payload.get("status") != "BUILD_PASS"
        or payload.get("selection_eligible") is not False
        or payload.get("heldout_access") != "NOT_ACCESSED_DEV3_ONLY"
        or int(payload.get("plane_index", -1)) != plane_index
        or not np.isclose(
            float(payload.get("plane_m", np.nan)),
            float(segments["plane_m"]),
            atol=0.0,
            rtol=0.0,
        )
        or not np.isclose(
            float(payload.get("threshold_m", np.nan)),
            threshold_m,
            atol=0.0,
            rtol=0.0,
        )
        or int(payload.get("segment_index", -1)) != segment_index
        or payload.get("segment_path") != segment["path"]
        or payload.get("segment_sha256") != segment["sha256"]
        or int(payload.get("requested_max_hulls", -1)) != BASE_MAX_HULLS_PER_SEGMENT
    ):
        raise RuntimeError("non-resumable v8 base child identity or parameters")
    _require_artifact_reference(
        payload.get("protocol"), PROTOCOL_PATH, label="base-child protocol"
    )
    parts = payload.get("parts")
    if (
        not isinstance(parts, list)
        or not 1 <= len(parts) <= BASE_SAFETY_MAX_HULLS_PER_SEGMENT
        or int(payload.get("actual_hulls", -1)) != len(parts)
        or bool(payload.get("request_satisfied"))
        != (len(parts) <= BASE_MAX_HULLS_PER_SEGMENT)
    ):
        raise RuntimeError("non-resumable v8 base child hull inventory")
    for part_index, part in enumerate(parts):
        expected_path = root / "parts" / f"part_{part_index:03d}.obj"
        if (
            int(part.get("part_index", -1)) != part_index
            or int(part.get("local_part_index", -1)) != part_index
            or int(part.get("segment_index", -1)) != segment_index
            or part.get("path") != relative_to_repo(expected_path)
            or part.get("sha256") != sha256_file(expected_path)
            or not bool(part.get("convex"))
            or not bool(part.get("watertight"))
            or not bool(part.get("winding_consistent"))
            or not 1 <= int(part.get("vertex_count", 0)) <= 32
        ):
            raise RuntimeError("non-resumable v8 base child part")
    native_log = payload.get("native_log")
    if require_native_log or native_log is not None:
        _require_artifact_reference(
            native_log, root / "native.log", label="base-child native log"
        )
    return payload


def _frozen_family_values(*, plane_index: int, threshold_m: float) -> tuple[int, float]:
    protocol = _load_protocol()
    planes = protocol["transition_atlas"]["candidate_planes_m"]
    if not 0 <= plane_index < len(planes):
        raise ValueError("v8 plane index outside frozen family")
    thresholds = [
        float(value)
        for value in protocol["thresholds_m"]
        if np.isclose(float(value), threshold_m, atol=0.0, rtol=0.0)
    ]
    if len(thresholds) != 1:
        raise ValueError("v8 threshold outside frozen family")
    return plane_index, thresholds[0]


def build_base_child(
    *, plane_index: int, threshold_m: float, segment_index: int
) -> dict[str, Any]:
    """Run one isolated CoACD call for one of the two new exact children."""
    plane_index, threshold_m = _frozen_family_values(
        plane_index=plane_index,
        threshold_m=threshold_m,
    )
    if segment_index not in (3, 4):
        raise ValueError("v8 only builds new segment3 children")
    segments = _load_plane_segments(plane_index)
    segment = next(
        row
        for row in segments["children"]
        if int(row["segment_index"]) == segment_index
    )
    root = _new_base_root(plane_index, threshold_m) / f"segment_{segment_index:03d}"
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        return _validate_base_child_manifest(
            existing,
            plane_index=plane_index,
            threshold_m=threshold_m,
            segment_index=segment_index,
            segments=segments,
            root=root,
            require_native_log=False,
        )
    mesh = trimesh.load(
        repo_path(segment["path"]),
        force="mesh",
        process=False,
        maintain_order=True,
    )
    started = time.perf_counter()
    result = _coacd_parts(
        mesh,
        threshold_m=threshold_m,
        max_hulls=BASE_MAX_HULLS_PER_SEGMENT,
    )
    if not 1 <= len(result) <= BASE_SAFETY_MAX_HULLS_PER_SEGMENT:
        raise RuntimeError("v8 CoACD child safety hull count violated")
    parts = [
        _export_part_v2(
            vertices,
            faces,
            root / "parts" / f"part_{local_index:03d}.obj",
            part_index=local_index,
            segment_index=segment_index,
            local_part_index=local_index,
            max_vertices=32,
        )
        for local_index, (vertices, faces) in enumerate(result)
    ]
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v8_base_child",
        "status": "BUILD_PASS",
        "selection_eligible": False,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "protocol": {
            "path": relative_to_repo(PROTOCOL_PATH),
            "sha256": sha256_file(PROTOCOL_PATH),
        },
        "plane_index": plane_index,
        "plane_m": segments["plane_m"],
        "threshold_m": threshold_m,
        "segment_index": segment_index,
        "segment_path": segment["path"],
        "segment_sha256": segment["sha256"],
        "requested_max_hulls": BASE_MAX_HULLS_PER_SEGMENT,
        "request_satisfied": len(parts) <= BASE_MAX_HULLS_PER_SEGMENT,
        "actual_hulls": len(parts),
        "wall_seconds": time.perf_counter() - started,
        "parts": parts,
    }
    _validate_base_child_manifest(
        payload,
        plane_index=plane_index,
        threshold_m=threshold_m,
        segment_index=segment_index,
        segments=segments,
        root=root,
        require_native_log=False,
    )
    atomic_json(manifest_path, payload)
    print(
        "E182_V8_BASE_CHILD=PASS "
        f"plane={plane_index} threshold={threshold_m:.3f} segment={segment_index} "
        f"hulls={len(parts)} requested={BASE_MAX_HULLS_PER_SEGMENT} "
        f"wall={payload['wall_seconds']:.3f}"
    )
    return payload


def _run_base_subprocess(
    plane_index: int, threshold_m: float, segment_index: int
) -> dict[str, Any]:
    root = _new_base_root(plane_index, threshold_m) / f"segment_{segment_index:03d}"
    manifest_path = root / "manifest.json"
    log_path = root / "native.log"
    if manifest_path.exists():
        existing = build_base_child(
            plane_index=plane_index,
            threshold_m=threshold_m,
            segment_index=segment_index,
        )
        segments = _load_plane_segments(plane_index)
        return _validate_base_child_manifest(
            existing,
            plane_index=plane_index,
            threshold_m=threshold_m,
            segment_index=segment_index,
            segments=segments,
            root=root,
            require_native_log=True,
        )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "build-base-child",
        "--plane-index",
        str(plane_index),
        "--threshold-m",
        str(threshold_m),
        "--segment-index",
        str(segment_index),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    log_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_log_path = log_path.with_suffix(log_path.suffix + ".tmp")
    temporary_log_path.write_text(output, encoding="utf-8")
    os.replace(temporary_log_path, log_path)
    if completed.returncode != 0:
        raise RuntimeError(
            f"v8 child plane={plane_index} threshold={threshold_m} "
            f"segment={segment_index} RC={completed.returncode}; log={log_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["native_log"] = {
        "path": relative_to_repo(log_path),
        "sha256": sha256_file(log_path),
    }
    segments = _load_plane_segments(plane_index)
    _validate_base_child_manifest(
        manifest,
        plane_index=plane_index,
        threshold_m=threshold_m,
        segment_index=segment_index,
        segments=segments,
        root=root,
        require_native_log=True,
    )
    atomic_json(manifest_path, manifest)
    print(completed.stdout.strip(), flush=True)
    return manifest


def _reused_child_entry(threshold_m: float, original_index: int) -> dict[str, Any]:
    path = (
        V4_BASE_ROOT
        / _threshold_tag(threshold_m)
        / f"segment_{original_index:03d}"
        / "manifest.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "source": "REUSED_V4_UNCHANGED_SEGMENT",
        "original_segment_index": original_index,
        "segment_index": remap_original_segment_index(original_index),
        "actual_hulls": int(payload["actual_hulls"]),
        "path": relative_to_repo(path),
        "sha256": sha256_file(path),
    }


def _new_child_entry(
    plane_index: int, threshold_m: float, segment_index: int
) -> dict[str, Any]:
    path = (
        _new_base_root(plane_index, threshold_m)
        / f"segment_{segment_index:03d}"
        / "manifest.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = _load_plane_segments(plane_index)
    _validate_base_child_manifest(
        payload,
        plane_index=plane_index,
        threshold_m=threshold_m,
        segment_index=segment_index,
        segments=segments,
        root=path.parent,
        require_native_log=True,
    )
    return {
        "source": "V8_NEW_SPLIT_SEGMENT",
        "segment_index": segment_index,
        "actual_hulls": int(payload["actual_hulls"]),
        "path": relative_to_repo(path),
        "sha256": sha256_file(path),
    }


def _write_composite_base(plane_index: int, threshold_m: float) -> dict[str, Any]:
    root = _new_base_root(plane_index, threshold_m)
    path = root / "manifest.json"
    entries = [
        _reused_child_entry(threshold_m, index) for index in UNCHANGED_SEGMENT_INDICES
    ] + [_new_child_entry(plane_index, threshold_m, index) for index in (3, 4)]
    entries.sort(key=lambda row: int(row["segment_index"]))
    if [row["segment_index"] for row in entries] != list(range(9)):
        raise RuntimeError("v8 composite base segment set is incomplete")
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v8_composite_base",
        "status": "BUILD_PASS",
        "selection_eligible": False,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "protocol": {
            "path": relative_to_repo(PROTOCOL_PATH),
            "sha256": sha256_file(PROTOCOL_PATH),
        },
        "plane_index": plane_index,
        "plane_m": derive_transition_atlas()["candidate_planes_m"][plane_index],
        "threshold_m": threshold_m,
        "segment_count": len(entries),
        "actual_hulls_per_segment": [row["actual_hulls"] for row in entries],
        "total_hulls": sum(row["actual_hulls"] for row in entries),
        "child_manifests": entries,
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("existing v8 composite base differs")
        return existing
    atomic_json(path, payload)
    return payload


def _validate_base_summary(
    payload: dict[str, Any], *, protocol: dict[str, Any]
) -> dict[str, Any]:
    if (
        payload.get("experiment_id") != "E182"
        or payload.get("stage") != "S2_decomposition_attempt2_v8_bases"
        or payload.get("status") != "BUILD_PASS"
        or payload.get("heldout_access") != "NOT_ACCESSED_DEV3_ONLY"
        or int(payload.get("new_child_count", -1)) != 12
        or int(payload.get("composite_base_count", -1)) != 6
        or not np.isfinite(float(payload.get("wall_seconds", np.nan)))
        or float(payload["wall_seconds"]) < 0.0
    ):
        raise RuntimeError("non-resumable v8 base summary identity")
    _require_artifact_reference(
        payload.get("protocol"), PROTOCOL_PATH, label="base-summary protocol"
    )
    segment_summary_path = SEGMENT_ROOT / "segment_summary.json"
    _require_artifact_reference(
        payload.get("segment_summary"),
        segment_summary_path,
        label="base-summary segments",
    )
    _validate_segment_summary(
        json.loads(segment_summary_path.read_text(encoding="utf-8")),
        protocol=protocol,
    )
    expected_tasks = [
        (plane_index, float(threshold_m), segment_index)
        for plane_index in range(len(protocol["topology"]["task_derived_splits"]))
        for threshold_m in protocol["thresholds_m"]
        for segment_index in (3, 4)
    ]
    child_references = payload.get("new_child_manifests")
    if not isinstance(child_references, list) or len(child_references) != len(
        expected_tasks
    ):
        raise RuntimeError("non-resumable v8 base-summary child inventory")
    for reference, (plane_index, threshold_m, segment_index) in zip(
        child_references, expected_tasks, strict=True
    ):
        path = (
            _new_base_root(plane_index, threshold_m)
            / f"segment_{segment_index:03d}"
            / "manifest.json"
        )
        if (
            int(reference.get("plane_index", -1)) != plane_index
            or not np.isclose(
                float(reference.get("threshold_m", np.nan)),
                threshold_m,
                atol=0.0,
                rtol=0.0,
            )
            or int(reference.get("segment_index", -1)) != segment_index
        ):
            raise RuntimeError("non-resumable v8 base-summary child order")
        _require_artifact_reference(reference, path, label="base-summary child")
        segments = _load_plane_segments(plane_index)
        _validate_base_child_manifest(
            json.loads(path.read_text(encoding="utf-8")),
            plane_index=plane_index,
            threshold_m=threshold_m,
            segment_index=segment_index,
            segments=segments,
            root=path.parent,
            require_native_log=True,
        )
    expected_bases = [
        (plane_index, float(threshold_m))
        for plane_index in range(len(protocol["topology"]["task_derived_splits"]))
        for threshold_m in protocol["thresholds_m"]
    ]
    base_references = payload.get("base_manifests")
    total_hulls = payload.get("total_hulls")
    if (
        not isinstance(base_references, list)
        or len(base_references) != len(expected_bases)
        or not isinstance(total_hulls, list)
        or len(total_hulls) != len(expected_bases)
    ):
        raise RuntimeError("non-resumable v8 base-summary composite inventory")
    for index, (reference, (plane_index, threshold_m)) in enumerate(
        zip(base_references, expected_bases, strict=True)
    ):
        path = _new_base_root(plane_index, threshold_m) / "manifest.json"
        if int(reference.get("plane_index", -1)) != plane_index or not np.isclose(
            float(reference.get("threshold_m", np.nan)),
            threshold_m,
            atol=0.0,
            rtol=0.0,
        ):
            raise RuntimeError("non-resumable v8 base-summary composite order")
        _require_artifact_reference(reference, path, label="base-summary composite")
        base = _write_composite_base(plane_index, threshold_m)
        if int(total_hulls[index]) != int(base["total_hulls"]):
            raise RuntimeError("non-resumable v8 base-summary hull totals")
    return payload


def build_bases() -> dict[str, Any]:
    """Build 12 new isolated children and compose six nine-segment bases."""
    protocol = _load_protocol()
    summary_path = ATTEMPT_ROOT / "base_summary.json"
    if summary_path.exists():
        return _validate_base_summary(
            json.loads(summary_path.read_text(encoding="utf-8")),
            protocol=protocol,
        )
    segment_summary = json.loads(
        (SEGMENT_ROOT / "segment_summary.json").read_text(encoding="utf-8")
    )
    _validate_segment_summary(segment_summary, protocol=protocol)
    tasks = [
        (plane_index, float(threshold_m), segment_index)
        for plane_index in range(len(protocol["topology"]["task_derived_splits"]))
        for threshold_m in protocol["thresholds_m"]
        for segment_index in (3, 4)
    ]
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=SEGMENT_PARALLEL_WORKERS
    ) as executor:
        futures = [executor.submit(_run_base_subprocess, *task) for task in tasks]
        children = [future.result() for future in futures]
    bases = [
        _write_composite_base(plane_index, float(threshold_m))
        for plane_index in range(len(protocol["topology"]["task_derived_splits"]))
        for threshold_m in protocol["thresholds_m"]
    ]
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v8_bases",
        "status": "BUILD_PASS",
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "protocol": _artifact_reference(PROTOCOL_PATH),
        "segment_summary": _artifact_reference(SEGMENT_ROOT / "segment_summary.json"),
        "new_child_count": len(children),
        "composite_base_count": len(bases),
        "total_hulls": [row["total_hulls"] for row in bases],
        "wall_seconds": time.perf_counter() - started,
        "new_child_manifests": [
            {
                "plane_index": plane_index,
                "threshold_m": threshold_m,
                "segment_index": segment_index,
                **_artifact_reference(
                    _new_base_root(plane_index, threshold_m)
                    / f"segment_{segment_index:03d}"
                    / "manifest.json"
                ),
            }
            for plane_index, threshold_m, segment_index in tasks
        ],
        "base_manifests": [
            {
                "plane_index": row["plane_index"],
                "threshold_m": row["threshold_m"],
                "path": relative_to_repo(
                    _new_base_root(row["plane_index"], row["threshold_m"])
                    / "manifest.json"
                ),
                "sha256": sha256_file(
                    _new_base_root(row["plane_index"], row["threshold_m"])
                    / "manifest.json"
                ),
            }
            for row in bases
        ],
    }
    if payload["new_child_count"] != 12 or payload["composite_base_count"] != 6:
        raise RuntimeError("v8 base family incomplete")
    _validate_base_summary(payload, protocol=protocol)
    atomic_json(summary_path, payload)
    return payload


def _load_composite_base(
    plane_index: int, threshold_m: float
) -> dict[int, list[ReducerPart]]:
    base_path = _new_base_root(plane_index, threshold_m) / "manifest.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base.get("status") != "BUILD_PASS" or base.get("segment_count") != 9:
        raise RuntimeError("v8 composite base is incomplete")
    groups: dict[int, list[ReducerPart]] = {}
    for child_entry in base["child_manifests"]:
        child_path = repo_path(child_entry["path"])
        if child_entry["sha256"] != sha256_file(child_path):
            raise RuntimeError("v8 composite child manifest SHA changed")
        child = json.loads(child_path.read_text(encoding="utf-8"))
        segment_index = int(child_entry["segment_index"])
        group = []
        for local_index, part in enumerate(child["parts"]):
            path = repo_path(part["path"])
            if part["sha256"] != sha256_file(path):
                raise RuntimeError("v8 composite base part SHA changed")
            reducer_part = ReducerPart(
                mesh=trimesh.load(
                    path,
                    force="mesh",
                    process=False,
                    maintain_order=True,
                ),
                segment_index=segment_index,
                lineage=(local_index,),
            )
            _checked_reducer_part(reducer_part)
            group.append(reducer_part)
        groups[segment_index] = group
    if set(groups) != set(range(9)) or any(not group for group in groups.values()):
        raise RuntimeError("v8 composite base groups are incomplete")
    return groups


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
        or payload.get("stage") != "S2_decomposition_attempt2_v8_candidate"
        or payload.get("status") != "BUILD_PASS"
        or payload.get("selection_eligible") is not False
        or payload.get("heldout_access") != "NOT_ACCESSED_DEV3_ONLY"
        or payload.get("object_key") != OBJECT_KEY
        or payload.get("candidate_id") != row["candidate_id"]
        or payload.get("parameters") != parameters
    ):
        raise RuntimeError("non-resumable v8 candidate identity or parameters")
    for key in (
        "method",
        "plane_index",
        "plane_m",
        "threshold_m",
        "max_hulls",
    ):
        if payload.get(key) != row[key]:
            raise RuntimeError(f"non-resumable v8 candidate field: {key}")
    _require_artifact_reference(
        payload.get("protocol"), PROTOCOL_PATH, label="protocol"
    )
    _require_artifact_reference(
        payload.get("base_manifest"), base_manifest_path, label="candidate base"
    )
    segment_manifest_path = (
        _plane_segment_root(int(row["plane_index"])) / "manifest.json"
    )
    _require_artifact_reference(
        payload.get("segment_manifest"),
        segment_manifest_path,
        label="candidate segment",
    )
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if payload.get("oracle_cleaned_mesh") != {
        "path": protocol["topology"]["oracle"]["path"],
        "sha256": protocol["topology"]["oracle"]["sha256"],
    }:
        raise RuntimeError("non-resumable v8 candidate oracle")
    parts = payload.get("parts")
    if not isinstance(parts, list) or not 9 <= len(parts) <= int(row["max_hulls"]):
        raise RuntimeError("non-resumable v8 candidate part inventory")
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
            raise RuntimeError("non-resumable v8 candidate part")
        segment_index = int(part.get("segment_index", -1))
        groups.setdefault(segment_index, []).append(part)
    if set(groups) != set(range(9)) or any(not values for values in groups.values()):
        raise RuntimeError("non-resumable v8 candidate segment coverage")
    per_segment = payload.get("per_segment")
    if not isinstance(per_segment, list) or len(per_segment) != 9:
        raise RuntimeError("non-resumable v8 candidate per-segment inventory")
    cursor = 0
    for segment_index, entry in enumerate(per_segment):
        count = len(groups[segment_index])
        if (
            int(entry.get("segment_index", -1)) != segment_index
            or int(entry.get("first_part_index", -1)) != cursor
            or int(entry.get("hull_count", -1)) != count
            or int(entry.get("source_base_hull_count", 0)) < count
        ):
            raise RuntimeError("non-resumable v8 candidate segment metadata")
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
        raise RuntimeError("non-resumable v8 candidate aggregate metadata")
    if payload.get("candidate_asset_sha256") != _candidate_asset_sha256(payload):
        raise RuntimeError("non-resumable v8 candidate asset digest")
    return payload


def _candidate_parameters(
    row: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    return {
        "method": row["method"],
        "plane_index": row["plane_index"],
        "plane_m": row["plane_m"],
        "threshold_m": row["threshold_m"],
        "max_convex_hull": row["max_hulls"],
        "max_ch_vertex": row["max_vertices"],
        "base_decomposition": protocol["base_decomposition"],
        "reducer": protocol["reducer"],
        "topology": protocol["topology"],
        "physics_contract": protocol["physics_contract"],
    }


def build_candidate(row: dict[str, Any]) -> dict[str, Any]:
    """Reduce one immutable nine-segment v8 base to K16 or K32."""
    protocol = _load_protocol()
    if [value for value in protocol["candidate_family"] if value == row] != [row]:
        raise RuntimeError("v8 candidate row differs from frozen family")
    root = CANDIDATE_ROOT / row["candidate_id"]
    manifest_path = root / "manifest.json"
    base_manifest_path = (
        _new_base_root(int(row["plane_index"]), float(row["threshold_m"]))
        / "manifest.json"
    )
    parameters = _candidate_parameters(row, protocol)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        return _validate_candidate_manifest(
            existing,
            row=row,
            parameters=parameters,
            root=root,
            base_manifest_path=base_manifest_path,
        )
    started = time.perf_counter()
    base_parts = _load_composite_base(
        int(row["plane_index"]), float(row["threshold_m"])
    )
    base_hulls = sum(len(group) for group in base_parts.values())
    reduced, trace = reduce_segmented_parts(
        base_parts,
        target_hulls=int(row["max_hulls"]),
    )
    parts = []
    per_segment = []
    for segment_index, group in sorted(reduced.items()):
        first_part_index = len(parts)
        for local_part_index, part in enumerate(group):
            part_index = len(parts)
            parts.append(
                _export_reduced_part(
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
    if not 9 <= len(parts) <= int(row["max_hulls"]):
        raise RuntimeError("v8 final global hull budget violated")
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v8_candidate",
        "status": "BUILD_PASS",
        "selection_eligible": False,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "object_key": OBJECT_KEY,
        "candidate_id": row["candidate_id"],
        "protocol": {
            "path": relative_to_repo(PROTOCOL_PATH),
            "sha256": sha256_file(PROTOCOL_PATH),
        },
        "base_manifest": {
            "path": relative_to_repo(base_manifest_path),
            "sha256": sha256_file(base_manifest_path),
        },
        "segment_manifest": {
            "path": relative_to_repo(
                _plane_segment_root(int(row["plane_index"])) / "manifest.json"
            ),
            "sha256": sha256_file(
                _plane_segment_root(int(row["plane_index"])) / "manifest.json"
            ),
        },
        "oracle_cleaned_mesh": {
            "path": protocol["topology"]["oracle"]["path"],
            "sha256": protocol["topology"]["oracle"]["sha256"],
        },
        "parameters": parameters,
        "method": row["method"],
        "plane_index": row["plane_index"],
        "plane_m": row["plane_m"],
        "threshold_m": row["threshold_m"],
        "max_hulls": row["max_hulls"],
        "base_hull_count": base_hulls,
        "hull_count": len(parts),
        "merge_count": len(trace),
        "negative_roundoff_clamp_count": sum(
            step["negative_roundoff_clamped"] for step in trace
        ),
        "merge_added_volume_m3": float(sum(step["added_volume_m3"] for step in trace)),
        "max_part_vertex_count": max(part["vertex_count"] for part in parts),
        "total_vertex_count": sum(part["vertex_count"] for part in parts),
        "total_face_count": sum(part["face_count"] for part in parts),
        "build_wall_seconds": time.perf_counter() - started,
        "per_segment": per_segment,
        "merge_trace": trace,
        "parts": parts,
    }
    payload["candidate_asset_sha256"] = _candidate_asset_sha256(payload)
    _validate_candidate_manifest(
        payload,
        row=row,
        parameters=parameters,
        root=root,
        base_manifest_path=base_manifest_path,
    )
    atomic_json(manifest_path, payload)
    print(
        "E182_TASK_PRESEG_V8=PASS "
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
        or payload.get("stage") != "S2_decomposition_attempt2_v8_build"
        or payload.get("status") != "BUILD_PASS"
        or payload.get("heldout_access") != "NOT_ACCESSED_DEV3_ONLY"
        or len(family) != 12
        or int(payload.get("candidate_count", -1)) != len(family)
        or not np.isfinite(float(payload.get("total_wall_seconds", np.nan)))
        or float(payload["total_wall_seconds"]) < 0.0
    ):
        raise RuntimeError("non-resumable v8 build summary identity")
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
        isinstance(value, list) and len(value) == len(family)
        for value in (references, identifiers, actual_hulls)
    ):
        raise RuntimeError("non-resumable v8 build-summary candidate inventory")
    for index, (row, reference) in enumerate(zip(family, references, strict=True)):
        path = CANDIDATE_ROOT / row["candidate_id"] / "manifest.json"
        if reference.get("candidate_id") != row["candidate_id"]:
            raise RuntimeError("non-resumable v8 build-summary candidate order")
        _require_artifact_reference(reference, path, label="build-summary candidate")
        candidate = _validate_candidate_manifest(
            json.loads(path.read_text(encoding="utf-8")),
            row=row,
            parameters=_candidate_parameters(row, protocol),
            root=path.parent,
            base_manifest_path=(
                _new_base_root(int(row["plane_index"]), float(row["threshold_m"]))
                / "manifest.json"
            ),
        )
        if identifiers[index] != row["candidate_id"] or int(actual_hulls[index]) != int(
            candidate["hull_count"]
        ):
            raise RuntimeError("non-resumable v8 build-summary candidate metadata")
    return payload


def build_candidates() -> dict[str, Any]:
    """Build all 12 frozen v8 candidates from six composite bases."""
    protocol = _load_protocol()
    summary_path = ATTEMPT_ROOT / "build_summary.json"
    if summary_path.exists():
        return _validate_build_summary(
            json.loads(summary_path.read_text(encoding="utf-8")),
            protocol=protocol,
        )
    base_summary_path = ATTEMPT_ROOT / "base_summary.json"
    base_summary = json.loads(base_summary_path.read_text(encoding="utf-8"))
    _validate_base_summary(base_summary, protocol=protocol)
    manifests = [build_candidate(row) for row in protocol["candidate_family"]]
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v8_build",
        "status": "BUILD_PASS",
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "protocol": _artifact_reference(PROTOCOL_PATH),
        "base_summary": _artifact_reference(base_summary_path),
        "candidate_count": len(manifests),
        "candidate_ids": [row["candidate_id"] for row in manifests],
        "actual_hulls": [row["hull_count"] for row in manifests],
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
            sum(row["build_wall_seconds"] for row in manifests)
        ),
    }
    if payload["candidate_count"] != 12:
        raise RuntimeError("v8 candidate family incomplete")
    _validate_build_summary(payload, protocol=protocol)
    atomic_json(summary_path, payload)
    return payload


def _candidate_segment_scenes(candidate: dict[str, Any]) -> list[Any]:
    groups: dict[int, list[trimesh.Trimesh]] = {}
    for part in candidate["parts"]:
        path = repo_path(part["path"])
        if part["sha256"] != sha256_file(path):
            raise RuntimeError("v8 static-P candidate part SHA changed")
        groups.setdefault(int(part["segment_index"]), []).append(
            trimesh.load(path, force="mesh", process=False, maintain_order=True)
        )
    if set(groups) != set(range(9)):
        raise RuntimeError("v8 static-P candidate segment set changed")
    scenes = []
    for segment_index in range(9):
        meshes = groups[segment_index]
        union = meshes[0] if len(meshes) == 1 else core.build_exact_union_mesh(meshes)
        scenes.append(core._raycasting_scene(union))
    return scenes


def _validate_static_p_result(
    payload: dict[str, Any],
    *,
    row: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    if (
        payload.get("experiment_id") != "E182"
        or payload.get("stage") != "S2_decomposition_attempt2_v8_static_P"
        or payload.get("status") != "COMPLETE"
        or payload.get("selection_eligible") is not True
        or payload.get("heldout_access") != "NOT_ACCESSED_DEV3_ONLY"
        or payload.get("candidate_id") != row["candidate_id"]
    ):
        raise RuntimeError("non-resumable v8 static-P identity")
    _require_artifact_reference(
        payload.get("protocol"), PROTOCOL_PATH, label="static-P protocol"
    )
    _require_artifact_reference(
        payload.get("candidate_manifest"), manifest_path, label="static-P candidate"
    )
    for key in ("plane_index", "plane_m", "threshold_m", "max_hulls"):
        if payload.get(key) != row[key]:
            raise RuntimeError(f"non-resumable v8 static-P field: {key}")
    metrics = payload.get("p_pose_contact")
    if not isinstance(metrics, dict):
        raise RuntimeError("non-resumable v8 static-P metrics")
    tp = int(metrics.get("true_positive_count", -1))
    phantom = int(metrics.get("phantom_contact_count", -1))
    missed = int(metrics.get("missed_contact_count", -1))
    true_negative = int(metrics.get("true_negative_count", -1))
    if tp + phantom + missed + true_negative != 882 or tp + missed != 27:
        raise RuntimeError("non-resumable v8 static-P confusion totals")
    gate = static_p_gate(tp=tp, phantom=phantom, missed=missed)
    precision = gate["precision"]
    recall = gate["recall"]
    expected_score = max((1.0 - precision) / 0.30, (1.0 - recall) / 0.30)
    if (
        payload.get("static_p_gate") != gate
        or bool(payload.get("full_prg_eligible")) != (gate["status"] == "PASS")
        or not np.isclose(float(metrics.get("precision", np.nan)), precision)
        or not np.isclose(float(metrics.get("recall", np.nan)), recall)
        or not np.isclose(float(payload.get("p_score", np.nan)), expected_score)
    ):
        raise RuntimeError("non-resumable v8 static-P gate or score")
    return payload


def evaluate_static_p(row: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one built candidate on the frozen 882-pose P topology gate."""
    protocol = _load_protocol()
    if [value for value in protocol["candidate_family"] if value == row] != [row]:
        raise RuntimeError("v8 static-P row differs from frozen family")
    result_path = STATIC_P_ROOT / "results" / f"{row['candidate_id']}.json"
    manifest_path = CANDIDATE_ROOT / row["candidate_id"] / "manifest.json"
    candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
    if candidate.get("status") != "BUILD_PASS":
        raise RuntimeError("v8 static-P candidate is not BUILD_PASS")
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        return _validate_static_p_result(
            existing,
            row=row,
            manifest_path=manifest_path,
        )
    queries, oracle_contact = v6._static_p_queries()
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
    metrics = v6._contact_metrics(candidate_contact, oracle_contact)
    gate = static_p_gate(
        tp=int(metrics["true_positive_count"]),
        phantom=int(metrics["phantom_contact_count"]),
        missed=int(metrics["missed_contact_count"]),
    )
    precision = float(metrics["precision"])
    recall = float(metrics["recall"])
    p_score = max((1.0 - precision) / 0.30, (1.0 - recall) / 0.30)
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v8_static_P",
        "status": "COMPLETE",
        "selection_eligible": True,
        "full_prg_eligible": gate["status"] == "PASS",
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "candidate_id": row["candidate_id"],
        "plane_index": row["plane_index"],
        "plane_m": row["plane_m"],
        "threshold_m": row["threshold_m"],
        "max_hulls": row["max_hulls"],
        "actual_hulls": candidate["hull_count"],
        "protocol": _artifact_reference(PROTOCOL_PATH),
        "candidate_manifest": _artifact_reference(manifest_path),
        "p_pose_contact": metrics,
        "static_p_gate": gate,
        "p_score": p_score,
        "wall_seconds": time.perf_counter() - started,
    }
    _validate_static_p_result(payload, row=row, manifest_path=manifest_path)
    atomic_json(result_path, payload)
    print(
        "E182_V8_STATIC_P=COMPLETE "
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
        or payload.get("stage") != "S2_decomposition_attempt2_v8_static_P_aggregate"
        or payload.get("status") != "COMPLETE"
        or payload.get("selection_eligible") is not True
        or payload.get("heldout_access") != "NOT_ACCESSED_DEV3_ONLY"
        or len(family) != 12
        or int(payload.get("candidate_count", -1)) != len(family)
    ):
        raise RuntimeError("non-resumable v8 static-P aggregate identity")
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
    if not isinstance(references, list) or len(references) != len(family):
        raise RuntimeError("non-resumable v8 static-P aggregate result inventory")
    rows = []
    for row, reference in zip(family, references, strict=True):
        result_path = STATIC_P_ROOT / "results" / f"{row['candidate_id']}.json"
        manifest_path = CANDIDATE_ROOT / row["candidate_id"] / "manifest.json"
        if reference.get("candidate_id") != row["candidate_id"]:
            raise RuntimeError("non-resumable v8 static-P aggregate result order")
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
        != (None if selected else "STOP_V8_NO_PLANE_OR_FLOOR_CHANGES")
        or payload.get("selected_candidate_ids") != expected_ids
    ):
        raise RuntimeError("non-resumable v8 static-P aggregate selection")
    return payload


def run_static_p() -> dict[str, Any]:
    """Evaluate all 12 candidates and freeze the set eligible for full P/R/G."""
    protocol = _load_protocol()
    aggregate_path = STATIC_P_ROOT / "static_p_aggregate.json"
    if aggregate_path.exists():
        return _validate_static_p_aggregate(
            json.loads(aggregate_path.read_text(encoding="utf-8")),
            protocol=protocol,
        )
    build_summary_path = ATTEMPT_ROOT / "build_summary.json"
    build_summary = json.loads(build_summary_path.read_text(encoding="utf-8"))
    _validate_build_summary(build_summary, protocol=protocol)
    rows = [evaluate_static_p(row) for row in protocol["candidate_family"]]
    selected = select_static_p_candidates(rows)
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v8_static_P_aggregate",
        "status": "COMPLETE",
        "selection_eligible": True,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "protocol": _artifact_reference(PROTOCOL_PATH),
        "build_summary": _artifact_reference(build_summary_path),
        "candidate_count": len(rows),
        "pass_count": len(selected),
        "full_prg_eligible": bool(selected),
        "zero_pass_action": (None if selected else "STOP_V8_NO_PLANE_OR_FLOOR_CHANGES"),
        "selected_candidate_ids": [row["candidate_id"] for row in selected],
        "results": [
            {
                "candidate_id": row["candidate_id"],
                "path": relative_to_repo(
                    STATIC_P_ROOT / "results" / f"{row['candidate_id']}.json"
                ),
                "sha256": sha256_file(
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
    """Parse one explicit v8 stage command without implicit fall-through."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "freeze-protocol",
            "build-segments",
            "build-base-child",
            "build-bases",
            "build-candidates",
            "static-p",
        ),
    )
    parser.add_argument("--plane-index", type=int)
    parser.add_argument("--threshold-m", type=float)
    parser.add_argument("--segment-index", type=int)
    return parser.parse_args()


def main() -> int:
    """Dispatch one frozen v8 construction or static-P stage."""
    args = parse_args()
    if args.command == "freeze-protocol":
        payload = freeze_protocol()
        print(f"E182_V8_PROTOCOL={payload['status']}")
        return 0
    if args.command == "build-segments":
        payload = build_segments()
        print(f"E182_V8_SEGMENTS={payload['status']} children={payload['child_count']}")
        return 0
    if args.command == "build-base-child":
        if (
            args.plane_index is None
            or args.threshold_m is None
            or args.segment_index is None
        ):
            raise SystemExit("build-base-child requires plane/threshold/segment")
        build_base_child(
            plane_index=args.plane_index,
            threshold_m=args.threshold_m,
            segment_index=args.segment_index,
        )
        return 0
    if args.command == "build-bases":
        payload = build_bases()
        print(
            f"E182_V8_BASES={payload['status']} bases={payload['composite_base_count']}"
        )
        return 0
    if args.command == "build-candidates":
        payload = build_candidates()
        print(
            f"E182_V8_BUILD={payload['status']} candidates={payload['candidate_count']}"
        )
        return 0
    payload = run_static_p()
    print(
        f"E182_V8_STATIC_P_AGGREGATE={payload['status']} "
        f"pass={payload['pass_count']}/{payload['candidate_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
