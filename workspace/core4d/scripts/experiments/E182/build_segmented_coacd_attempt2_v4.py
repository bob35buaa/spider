#!/usr/bin/env python3
"""Build CoACD bases and deterministic global-K reductions for E182 attempt2-v4."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import combinations
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
)
from build_segmented_coacd_attempt2_v2 import (
    COACD_MAX_VERTICES,
    V1_SEGMENT_MANIFEST_PATH,
    _coacd_parts,
    _export_part_v2,
    _load_v1_segments,
)
from build_segmented_coacd_attempt2_v3 import (
    CANDIDATE_ROOT as V3_CANDIDATE_ROOT,
)
from build_segmented_coacd_attempt2_v3 import (
    CHILD_ROOT as V3_CHILD_ROOT,
)
from build_segmented_coacd_attempt2_v3 import (
    PROTOCOL_PATH as V3_PROTOCOL_PATH,
)
from build_task_query_fixture import DEFAULT_OUTPUT_ROOT
from e182_common import (
    REPO_ROOT,
    atomic_json,
    relative_to_repo,
    repo_path,
    sha256_file,
)
from evaluate_task_queries import build_exact_union_mesh

ATTEMPT_ROOT = DEFAULT_OUTPUT_ROOT / "attempt2_segmented_x2y4_v4_global_budget"
PROTOCOL_PATH = ATTEMPT_ROOT / "attempt2_v4_protocol_manifest.json"
BASE_ROOT = ATTEMPT_ROOT / "coacd_bases"
CANDIDATE_ROOT = ATTEMPT_ROOT / "candidates"
FIXTURE_PATH = ATTEMPT_ROOT / "query_fixture_manifest.json"
SCREEN_ROOT = ATTEMPT_ROOT / "screen"

OBJECT_KEY = "bucket003"
CASE_ID = "bucket003_20231018_001_p1"
THRESHOLDS_M = (0.005, 0.010, 0.020)
TARGET_HULLS = (8, 16, 32)
BASE_MAX_HULLS_PER_SEGMENT = 4
BASE_SAFETY_MAX_HULLS_PER_SEGMENT = 16
FINAL_MAX_VERTICES = 256
SEGMENT_PARALLEL_WORKERS = 4
V3_FAILED_CANDIDATE_ID = "segx2y4_iso_coacd_t005_k16_v032"
V1_BUILDER_PATH = Path(_export_mesh.__code__.co_filename).resolve()
V2_BUILDER_PATH = Path(_coacd_parts.__code__.co_filename).resolve()
EXACT_UNION_SOURCE_PATH = Path(build_exact_union_mesh.__code__.co_filename).resolve()

for variable in ("OMP_NUM_THREADS", "TBB_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")


@dataclass(frozen=True)
class ReducerPart:
    """One convex part with immutable segment membership and source lineage."""

    mesh: trimesh.Trimesh
    segment_index: int
    lineage: tuple[int, ...]


def _threshold_tag(threshold_m: float) -> str:
    return f"t{round(threshold_m * 1000):03d}"


def _candidate_id(threshold_m: float, max_hulls: int) -> str:
    return (
        f"segx2y4_gmerge_{_threshold_tag(threshold_m)}_"
        f"k{max_hulls:02d}_v{FINAL_MAX_VERTICES:03d}"
    )


def candidate_family() -> list[dict[str, Any]]:
    """Return three global hull budgets from each of three immutable CoACD bases."""
    return [
        {
            "candidate_id": _candidate_id(threshold_m, max_hulls),
            "method": "ISOLATED_COACD_GLOBAL_BUDGET_MERGE",
            "threshold_m": threshold_m,
            "max_hulls": max_hulls,
            "base_max_hulls_per_segment": BASE_MAX_HULLS_PER_SEGMENT,
            "max_vertices": FINAL_MAX_VERTICES,
        }
        for threshold_m in THRESHOLDS_M
        for max_hulls in TARGET_HULLS
    ]


def _v3_failure_evidence() -> dict[str, Any]:
    """Bind the isolated equal-cap failure which changes the v4 assumption."""
    protocol = json.loads(V3_PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_V3_BUILD_OR_SCORE":
        raise RuntimeError("attempt2-v3 protocol changed")
    failed_root = V3_CHILD_ROOT / V3_FAILED_CANDIDATE_ID
    failed_log = failed_root / "segment_003/native.log"
    failed_text = failed_log.read_text(encoding="utf-8")
    signature = "segment 3 hull cap violated actual=3 cap=2"
    if signature not in failed_text:
        raise RuntimeError("attempt2-v3 failure signature changed")
    successful = sorted(failed_root.glob("segment_*/manifest.json"))
    failed_candidate_manifest = (
        V3_CANDIDATE_ROOT / V3_FAILED_CANDIDATE_ID / "manifest.json"
    )
    if len(successful) != 7 or failed_candidate_manifest.exists():
        raise RuntimeError("attempt2-v3 partial evidence changed")
    for path in successful:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "BUILD_PASS" or manifest.get("actual_hulls") != 2:
            raise RuntimeError("attempt2-v3 successful child evidence changed")
    return {
        "status": "V3_ISOLATED_EQUAL_CAP_BUILD_FAILED",
        "candidate_id": V3_FAILED_CANDIDATE_ID,
        "failed_segment_index": 3,
        "actual_hulls": 3,
        "requested_cap": 2,
        "successful_segment_count": len(successful),
        "successful_child_manifests": [
            {"path": relative_to_repo(path), "sha256": sha256_file(path)}
            for path in successful
        ],
        "failed_native_log": {
            "path": relative_to_repo(failed_log),
            "sha256": sha256_file(failed_log),
        },
        "v3_protocol": {
            "path": relative_to_repo(V3_PROTOCOL_PATH),
            "sha256": sha256_file(V3_PROTOCOL_PATH),
        },
        "root_cause": "COACD_MAX_CONVEX_HULL_IS_NOT_A_RELIABLE_HARD_CAP",
        "changed_assumption": "GLOBAL_BUDGET_NOT_EQUAL_PER_SEGMENT_CAP",
    }


def freeze_protocol() -> dict[str, Any]:
    """Freeze v4 construction and selection semantics before any base build or score."""
    if list(BASE_ROOT.glob("*/*/manifest.json")):
        raise RuntimeError("cannot freeze v4 after base child build")
    if list(CANDIDATE_ROOT.glob("*/manifest.json")):
        raise RuntimeError("cannot freeze v4 after candidate build")
    if list((SCREEN_ROOT / "results").glob("*.json")):
        raise RuntimeError("cannot freeze v4 after score")
    segments = _load_v1_segments()
    rows = candidate_family()
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v4_protocol",
        "status": "FROZEN_BEFORE_V4_BUILD_OR_SCORE",
        "selection_eligible": False,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "reason": "V3_PROVED_COACD_LOCAL_MAX_HULLS_IS_NOT_A_HARD_GLOBAL_BUDGET",
        "object_key": OBJECT_KEY,
        "case_id": CASE_ID,
        "candidate_family": rows,
        "candidate_count": len(rows),
        "base_decomposition": {
            "one_base_per_threshold": True,
            "thresholds_m": list(THRESHOLDS_M),
            "max_convex_hull_request_per_segment": BASE_MAX_HULLS_PER_SEGMENT,
            "request_is_hard_gate": False,
            "safety_max_returned_hulls_per_segment": (
                BASE_SAFETY_MAX_HULLS_PER_SEGMENT
            ),
            "initial_max_ch_vertex": COACD_MAX_VERTICES,
            "coacd_common": COACD_COMMON,
            "fresh_process_per_segment": True,
            "coacd_calls_per_child": 1,
            "segment_parallel_workers": SEGMENT_PARALLEL_WORKERS,
            "candidates_share_threshold_base": True,
        },
        "reducer": {
            "name": "DETERMINISTIC_INTRA_SEGMENT_GREEDY_CONVEX_MERGE_V1",
            "targets": list(TARGET_HULLS),
            "hard_condition": "TOTAL_FINAL_HULLS_LE_TARGET",
            "minimum_hulls_per_nonempty_segment": 1,
            "cross_segment_merge": False,
            "pair_cost": "CONVEX_HULL_VOLUME_MINUS_EXACT_MANIFOLD_PAIR_UNION_VOLUME",
            "negative_roundoff_clamp": 0.0,
            "cost_round_digits": 15,
            "tie_break": [
                "rounded_added_volume_m3",
                "segment_index",
                "left_lineage",
                "right_lineage",
            ],
            "final_max_vertices": FINAL_MAX_VERTICES,
            "query_conditioned": False,
            "construction_uses_heldout": False,
        },
        "physics_contract": {
            "mesh_asset_maxhullvert_minimum": FINAL_MAX_VERTICES,
            "reason": "PREVENT_MUJOCO_FROM_SIMPLIFYING_FINAL_CONVEX_PARTS_BELOW_EXACT_C",
            "pre_full_compile_audit": True,
            "required_checks": [
                "ALL_PARTS_COMPILE",
                "ASSET_MAXHULLVERT_GE_PART_VERTEX_COUNT",
                "GEOM_AND_EXPLICIT_PAIR_COUNTS_EXACT",
                "REAL_MUJOCO_CONTACT_REPLAY",
            ],
            "full_role": "FORBIDDEN_UNTIL_COMPILE_AND_CONTACT_AUDIT_PASS",
        },
        "topology": {
            "name": "REUSE_EXACT_MANIFOLD_X2_Y4_SOLID_PARTITION",
            "segment_count": SEGMENT_COUNT,
            "segment_manifest": {
                "path": relative_to_repo(V1_SEGMENT_MANIFEST_PATH),
                "sha256": sha256_file(V1_SEGMENT_MANIFEST_PATH),
            },
            "ordered_parts": "segment_index_then_sorted_source_lineage",
        },
        "selection_contract": {
            "launch_floor": "UNCHANGED_E182_P_R_G_V1",
            "score": "UNCHANGED_P_R_G_WORST_NORMALIZED_V1",
            "per_K": "launch_floor_then_score_then_actual_hulls_runtime_threshold_id",
            "full_role": "FORBIDDEN_UNTIL_PRODUCTION_FREEZE",
        },
        "v3_failure_evidence": _v3_failure_evidence(),
        "oracle": segments["oracle"],
        "original_fixture": {
            "path": relative_to_repo(DEFAULT_FIXTURE),
            "sha256": sha256_file(DEFAULT_FIXTURE),
        },
        "thread_env": {
            name: os.environ[name]
            for name in ("OMP_NUM_THREADS", "TBB_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "source_dependencies": [
            {
                "role": "SEGMENT_EXPORT_AND_CANDIDATE_DIGEST",
                "path": relative_to_repo(V1_BUILDER_PATH),
                "sha256": sha256_file(V1_BUILDER_PATH),
            },
            {
                "role": "COACD_CALL_AND_BASE_PART_EXPORT",
                "path": relative_to_repo(V2_BUILDER_PATH),
                "sha256": sha256_file(V2_BUILDER_PATH),
            },
            {
                "role": "EXACT_MANIFOLD_PAIR_UNION",
                "path": relative_to_repo(EXACT_UNION_SOURCE_PATH),
                "sha256": sha256_file(EXACT_UNION_SOURCE_PATH),
            },
        ],
        "builder_source": {
            "path": relative_to_repo(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
    }
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("existing attempt2-v4 protocol differs")
        return existing
    atomic_json(PROTOCOL_PATH, payload)
    return payload


def _load_protocol() -> dict[str, Any]:
    """Require exact v4 source and all parent authority hashes."""
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_V4_BUILD_OR_SCORE":
        raise RuntimeError("attempt2-v4 protocol is not frozen")
    checks = (
        (protocol["builder_source"], Path(__file__)),
        (protocol["topology"]["segment_manifest"], V1_SEGMENT_MANIFEST_PATH),
        (protocol["original_fixture"], DEFAULT_FIXTURE),
        (protocol["v3_failure_evidence"]["v3_protocol"], V3_PROTOCOL_PATH),
        (
            protocol["v3_failure_evidence"]["failed_native_log"],
            repo_path(protocol["v3_failure_evidence"]["failed_native_log"]["path"]),
        ),
    )
    for entry, path in checks:
        if entry["sha256"] != sha256_file(path):
            raise RuntimeError(f"attempt2-v4 authority SHA changed: {path}")
    for dependency in protocol["source_dependencies"]:
        path = repo_path(dependency["path"])
        if dependency["sha256"] != sha256_file(path):
            raise RuntimeError(f"attempt2-v4 source dependency changed: {path}")
    for child in protocol["v3_failure_evidence"]["successful_child_manifests"]:
        path = repo_path(child["path"])
        if child["sha256"] != sha256_file(path):
            raise RuntimeError(f"attempt2-v3 failure child changed: {path}")
    return protocol


def _base_root(threshold_m: float) -> Path:
    return BASE_ROOT / _threshold_tag(threshold_m)


def _frozen_threshold(threshold_m: float) -> float:
    protocol = _load_protocol()
    matches = [
        value
        for value in protocol["base_decomposition"]["thresholds_m"]
        if np.isclose(float(value), threshold_m, atol=0.0, rtol=0.0)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"threshold outside frozen v4 family: {threshold_m}")
    return float(matches[0])


def build_base_child(threshold_m: float, segment_index: int) -> dict[str, Any]:
    """Run exactly one isolated CoACD base call and retain over-cap returns."""
    threshold_m = _frozen_threshold(threshold_m)
    segments = _load_v1_segments()["segments"]
    if not 0 <= segment_index < len(segments):
        raise RuntimeError("segment index outside frozen partition")
    segment = segments[segment_index]
    root = _base_root(threshold_m) / f"segment_{segment_index:03d}"
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "BUILD_PASS"
            and existing.get("segment_sha256") == segment["sha256"]
            and all(
                sha256_file(repo_path(part["path"])) == part["sha256"]
                for part in existing["parts"]
            )
        ):
            return existing
        raise RuntimeError("non-resumable attempt2-v4 base child manifest")
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
        raise RuntimeError(
            f"v4 base safety hull count violated actual={len(result)} "
            f"max={BASE_SAFETY_MAX_HULLS_PER_SEGMENT}"
        )
    parts = [
        _export_part_v2(
            vertices,
            faces,
            root / "parts" / f"part_{local_index:03d}.obj",
            part_index=local_index,
            segment_index=segment_index,
            local_part_index=local_index,
            max_vertices=COACD_MAX_VERTICES,
        )
        for local_index, (vertices, faces) in enumerate(result)
    ]
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v4_base_child",
        "status": "BUILD_PASS",
        "selection_eligible": False,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
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
    atomic_json(manifest_path, payload)
    print(
        "E182_V4_BASE_CHILD=PASS "
        f"threshold={threshold_m:.3f} segment={segment_index} "
        f"hulls={len(parts)} requested={BASE_MAX_HULLS_PER_SEGMENT} "
        f"wall={payload['wall_seconds']:.3f}"
    )
    return payload


def _run_base_subprocess(threshold_m: float, segment_index: int) -> dict[str, Any]:
    """Launch one base child and bind its complete native output."""
    root = _base_root(threshold_m) / f"segment_{segment_index:03d}"
    manifest_path = root / "manifest.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "build-base-child",
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
    log_path = root / "native.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"v4 base threshold={threshold_m} segment={segment_index} "
            f"child RC={completed.returncode}; log={log_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["native_log"] = {
        "path": relative_to_repo(log_path),
        "sha256": sha256_file(log_path),
    }
    atomic_json(manifest_path, manifest)
    print(completed.stdout.strip(), flush=True)
    return manifest


def build_base(threshold_m: float) -> dict[str, Any]:
    """Build or resume the eight isolated children for one threshold base."""
    threshold_m = _frozen_threshold(threshold_m)
    manifest_path = _base_root(threshold_m) / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "BUILD_PASS" and all(
            sha256_file(repo_path(entry["path"])) == entry["sha256"]
            for entry in existing["child_manifests"]
        ):
            return existing
        raise RuntimeError("non-resumable attempt2-v4 base manifest")
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=SEGMENT_PARALLEL_WORKERS
    ) as executor:
        futures = [
            executor.submit(_run_base_subprocess, threshold_m, segment_index)
            for segment_index in range(SEGMENT_COUNT)
        ]
        children = [future.result() for future in futures]
    children.sort(key=lambda value: value["segment_index"])
    child_manifests = []
    for child in children:
        path = (
            _base_root(threshold_m)
            / f"segment_{child['segment_index']:03d}/manifest.json"
        )
        child_manifests.append(
            {"path": relative_to_repo(path), "sha256": sha256_file(path)}
        )
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v4_base",
        "status": "BUILD_PASS",
        "selection_eligible": False,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "protocol": {
            "path": relative_to_repo(PROTOCOL_PATH),
            "sha256": sha256_file(PROTOCOL_PATH),
        },
        "threshold_m": threshold_m,
        "segment_count": len(children),
        "requested_max_hulls_per_segment": BASE_MAX_HULLS_PER_SEGMENT,
        "actual_hulls_per_segment": [child["actual_hulls"] for child in children],
        "total_hulls": sum(child["actual_hulls"] for child in children),
        "request_violation_segments": [
            child["segment_index"]
            for child in children
            if not child["request_satisfied"]
        ],
        "child_manifests": child_manifests,
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(manifest_path, payload)
    print(
        "E182_V4_BASE=PASS "
        f"threshold={threshold_m:.3f} hulls={payload['total_hulls']} "
        f"wall={payload['wall_seconds']:.3f}"
    )
    return payload


def build_bases() -> dict[str, Any]:
    """Build the three threshold bases serially while each uses four child workers."""
    protocol = _load_protocol()
    manifests = [
        build_base(float(threshold_m))
        for threshold_m in protocol["base_decomposition"]["thresholds_m"]
    ]
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v4_bases",
        "status": "BUILD_PASS",
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "base_count": len(manifests),
        "thresholds_m": [manifest["threshold_m"] for manifest in manifests],
        "total_hulls": [manifest["total_hulls"] for manifest in manifests],
        "total_wall_seconds": float(sum(m["wall_seconds"] for m in manifests)),
        "base_manifests": [
            {
                "path": relative_to_repo(
                    _base_root(m["threshold_m"]) / "manifest.json"
                ),
                "sha256": sha256_file(_base_root(m["threshold_m"]) / "manifest.json"),
            }
            for m in manifests
        ],
    }
    if payload["base_count"] != len(THRESHOLDS_M):
        raise RuntimeError("attempt2-v4 base family incomplete")
    atomic_json(ATTEMPT_ROOT / "base_summary.json", payload)
    return payload


def _checked_reducer_part(part: ReducerPart) -> None:
    mesh = part.mesh
    if (
        mesh.is_empty
        or not mesh.is_convex
        or not mesh.is_watertight
        or not mesh.is_winding_consistent
        or not np.isfinite(mesh.vertices).all()
        or float(mesh.volume) <= 0.0
        or not part.lineage
        or tuple(sorted(set(part.lineage))) != part.lineage
    ):
        raise ValueError("invalid reducer part")


def _merged_part(left: ReducerPart, right: ReducerPart) -> ReducerPart:
    if left.segment_index != right.segment_index:
        raise ValueError("cross-segment merge is forbidden")
    points = np.vstack((left.mesh.vertices, right.mesh.vertices))
    mesh = trimesh.convex.convex_hull(points)
    lineage = tuple(sorted((*left.lineage, *right.lineage)))
    result = ReducerPart(
        mesh=mesh,
        segment_index=left.segment_index,
        lineage=lineage,
    )
    _checked_reducer_part(result)
    return result


def _merge_cost(
    left: ReducerPart, right: ReducerPart
) -> tuple[float, ReducerPart, float]:
    merged = _merged_part(left, right)
    union = build_exact_union_mesh((left.mesh, right.mesh))
    union_volume = float(union.volume)
    raw_added = float(merged.mesh.volume) - union_volume
    tolerance = 1e-10 * max(1.0, union_volume)
    if raw_added < -tolerance:
        raise RuntimeError("convex merge volume is smaller than exact pair union")
    return max(0.0, raw_added), merged, union_volume


def reduce_segmented_parts(
    parts_by_segment: dict[int, list[ReducerPart]],
    *,
    target_hulls: int,
) -> tuple[dict[int, list[ReducerPart]], list[dict[str, Any]]]:
    """Greedily merge the least-inflating pair until the global K cap is met."""
    groups = {
        int(segment_index): sorted(parts, key=lambda part: part.lineage)
        for segment_index, parts in sorted(parts_by_segment.items())
    }
    if not groups or any(not parts for parts in groups.values()):
        raise ValueError("each reducer segment must be nonempty")
    if target_hulls < len(groups):
        raise ValueError("target hulls cannot be below nonempty segment count")
    for segment_index, parts in groups.items():
        for part in parts:
            _checked_reducer_part(part)
            if part.segment_index != segment_index:
                raise ValueError("reducer part has wrong segment membership")
    trace: list[dict[str, Any]] = []
    while sum(len(parts) for parts in groups.values()) > target_hulls:
        choices: list[
            tuple[
                tuple[Any, ...],
                int,
                ReducerPart,
                ReducerPart,
                ReducerPart,
                float,
                float,
            ]
        ] = []
        for segment_index, parts in groups.items():
            for left, right in combinations(parts, 2):
                added_volume, merged, union_volume = _merge_cost(left, right)
                key = (
                    round(added_volume, 15),
                    segment_index,
                    left.lineage,
                    right.lineage,
                )
                choices.append(
                    (
                        key,
                        segment_index,
                        left,
                        right,
                        merged,
                        added_volume,
                        union_volume,
                    )
                )
        if not choices:
            raise RuntimeError(
                "global hull target is infeasible without cross-segment merge"
            )
        (
            _,
            segment_index,
            left,
            right,
            merged,
            added_volume,
            union_volume,
        ) = min(choices, key=lambda choice: choice[0])
        groups[segment_index] = sorted(
            [part for part in groups[segment_index] if part not in (left, right)]
            + [merged],
            key=lambda part: part.lineage,
        )
        trace.append(
            {
                "step": len(trace),
                "segment_index": segment_index,
                "left_lineage": list(left.lineage),
                "right_lineage": list(right.lineage),
                "output_lineage": list(merged.lineage),
                "exact_pair_union_volume_m3": union_volume,
                "merged_convex_volume_m3": float(merged.mesh.volume),
                "added_volume_m3": added_volume,
            }
        )
    return groups, trace


def _load_base_parts(threshold_m: float) -> dict[int, list[ReducerPart]]:
    manifest_path = _base_root(threshold_m) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "BUILD_PASS":
        raise RuntimeError("attempt2-v4 base is not BUILD_PASS")
    groups: dict[int, list[ReducerPart]] = {}
    for child_entry in manifest["child_manifests"]:
        child_path = repo_path(child_entry["path"])
        if sha256_file(child_path) != child_entry["sha256"]:
            raise RuntimeError("attempt2-v4 base child manifest SHA changed")
        child = json.loads(child_path.read_text(encoding="utf-8"))
        segment_index = int(child["segment_index"])
        group = []
        for local_index, part in enumerate(child["parts"]):
            path = repo_path(part["path"])
            if sha256_file(path) != part["sha256"]:
                raise RuntimeError("attempt2-v4 base part SHA changed")
            mesh = trimesh.load(path, force="mesh", process=False, maintain_order=True)
            reducer_part = ReducerPart(
                mesh=mesh,
                segment_index=segment_index,
                lineage=(local_index,),
            )
            _checked_reducer_part(reducer_part)
            group.append(reducer_part)
        groups[segment_index] = group
    if set(groups) != set(range(SEGMENT_COUNT)):
        raise RuntimeError("attempt2-v4 base segment set changed")
    return groups


def _export_reduced_part(
    part: ReducerPart,
    path: Path,
    *,
    part_index: int,
    local_part_index: int,
) -> dict[str, Any]:
    _checked_reducer_part(part)
    if len(part.mesh.vertices) > FINAL_MAX_VERTICES:
        raise RuntimeError(
            f"v4 final vertex ceiling violated actual={len(part.mesh.vertices)} "
            f"max={FINAL_MAX_VERTICES}"
        )
    result = _export_mesh(part.mesh, path)
    result.update(
        {
            "part_index": part_index,
            "segment_index": part.segment_index,
            "local_part_index": local_part_index,
            "source_lineage": list(part.lineage),
            "source_base_part_count": len(part.lineage),
            "convex": True,
        }
    )
    return result


def build_candidate(row: dict[str, Any]) -> dict[str, Any]:
    """Derive one immutable global-K candidate from its threshold base."""
    protocol = _load_protocol()
    frozen = [
        value
        for value in protocol["candidate_family"]
        if value["candidate_id"] == row["candidate_id"]
    ]
    if frozen != [row]:
        raise RuntimeError("candidate row differs from frozen v4 family")
    root = CANDIDATE_ROOT / row["candidate_id"]
    manifest_path = root / "manifest.json"
    base_manifest_path = _base_root(float(row["threshold_m"])) / "manifest.json"
    parameters = {
        "method": row["method"],
        "threshold_m": row["threshold_m"],
        "max_convex_hull": row["max_hulls"],
        "max_ch_vertex": row["max_vertices"],
        "base_decomposition": protocol["base_decomposition"],
        "reducer": protocol["reducer"],
        "topology": protocol["topology"],
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
        raise RuntimeError("non-resumable attempt2-v4 candidate manifest")
    started = time.perf_counter()
    base_parts = _load_base_parts(float(row["threshold_m"]))
    base_hulls = sum(len(parts) for parts in base_parts.values())
    reduced, trace = reduce_segmented_parts(
        base_parts,
        target_hulls=int(row["max_hulls"]),
    )
    parts = []
    per_segment = []
    for segment_index, group in reduced.items():
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
    if not SEGMENT_COUNT <= len(parts) <= int(row["max_hulls"]):
        raise RuntimeError("attempt2-v4 final global hull budget violated")
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v4_candidate",
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
        "segment_manifest": protocol["topology"]["segment_manifest"],
        "oracle_cleaned_mesh": {
            "path": protocol["oracle"]["path"],
            "sha256": protocol["oracle"]["sha256"],
        },
        "parameters": parameters,
        "method": row["method"],
        "max_hulls": row["max_hulls"],
        "base_hull_count": base_hulls,
        "hull_count": len(parts),
        "merge_count": len(trace),
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
    atomic_json(manifest_path, payload)
    print(
        "E182_SEGMENTED_COACD_V4=PASS "
        f"candidate={row['candidate_id']} base={base_hulls} final={len(parts)} "
        f"merges={len(trace)} wall={payload['build_wall_seconds']:.3f}"
    )
    return payload


def build_candidates() -> dict[str, Any]:
    """Derive all nine candidates after the three shared bases are complete."""
    protocol = _load_protocol()
    base_summary = json.loads(
        (ATTEMPT_ROOT / "base_summary.json").read_text(encoding="utf-8")
    )
    if (
        base_summary.get("status") != "BUILD_PASS"
        or base_summary.get("base_count") != 3
    ):
        raise RuntimeError("attempt2-v4 bases are incomplete")
    manifests = [build_candidate(row) for row in protocol["candidate_family"]]
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v4_build",
        "status": "BUILD_PASS",
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "candidate_count": len(manifests),
        "candidate_ids": [manifest["candidate_id"] for manifest in manifests],
        "actual_hulls": [manifest["hull_count"] for manifest in manifests],
        "total_wall_seconds": float(
            base_summary["total_wall_seconds"]
            + sum(manifest["build_wall_seconds"] for manifest in manifests)
        ),
    }
    if payload["candidate_count"] != 9:
        raise RuntimeError("attempt2-v4 candidate family incomplete")
    atomic_json(ATTEMPT_ROOT / "build_summary.json", payload)
    return payload


def build_fixture() -> dict[str, Any]:
    """Freeze the bucket003-only nine-candidate fixture before any v4 score."""
    protocol = _load_protocol()
    summary = json.loads(
        (ATTEMPT_ROOT / "build_summary.json").read_text(encoding="utf-8")
    )
    if summary.get("status") != "BUILD_PASS" or summary.get("candidate_count") != 9:
        raise RuntimeError("attempt2-v4 build summary incomplete")
    original = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    cases = [case for case in original["cases"] if case["case_id"] == CASE_ID]
    if len(cases) != 1:
        raise RuntimeError("bucket003 dev case changed")
    rows = []
    for frozen in protocol["candidate_family"]:
        manifest_path = CANDIDATE_ROOT / frozen["candidate_id"] / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "BUILD_PASS":
            raise RuntimeError("attempt2-v4 fixture candidate is not BUILD_PASS")
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
        "stage": "S2_task_query_fixture_attempt2_segmented_x2y4_v4_global_budget",
        "status": "FROZEN",
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "freeze_timing": "AFTER_V4_BUILD_BEFORE_V4_SCORE",
        "attempt2_v4_protocol": {
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
            raise RuntimeError("existing attempt2-v4 fixture differs")
        return existing
    atomic_json(FIXTURE_PATH, payload)
    return payload


def parse_args() -> argparse.Namespace:
    """Parse explicit v4 parent/child stages."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "freeze-protocol",
            "build-base-child",
            "build-bases",
            "build-candidates",
            "build-fixture",
        ),
    )
    parser.add_argument("--threshold-m", type=float)
    parser.add_argument("--segment-index", type=int)
    return parser.parse_args()


def main() -> int:
    """Execute one attempt2-v4 stage."""
    args = parse_args()
    if args.command == "freeze-protocol":
        payload = freeze_protocol()
        print(f"E182_ATTEMPT2_V4_PROTOCOL={payload['status']}")
        return 0
    if args.command == "build-base-child":
        if args.threshold_m is None or args.segment_index is None:
            raise RuntimeError("base child requires threshold and segment")
        build_base_child(args.threshold_m, args.segment_index)
        return 0
    if args.command == "build-bases":
        payload = build_bases()
        print(
            f"E182_ATTEMPT2_V4_BASES={payload['status']} count={payload['base_count']}"
        )
        return 0
    if args.command == "build-candidates":
        payload = build_candidates()
        print(
            f"E182_ATTEMPT2_V4_BUILD={payload['status']} "
            f"candidates={payload['candidate_count']}"
        )
        return 0
    payload = build_fixture()
    print(
        f"E182_ATTEMPT2_V4_FIXTURE={payload['status']} "
        f"candidates={payload['candidate_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
