#!/usr/bin/env python3
"""Freeze and search partial convex merges of the four segment3 base parts."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import evaluate_task_queries as core
import numpy as np
import search_segment_hybrid_v6 as v6
import trimesh
from build_task_query_fixture import DEFAULT_OUTPUT_ROOT
from e182_common import atomic_json, relative_to_repo, repo_path, sha256_file
from search_segment_hybrid_v6 import OUTPUT_ROOT as V6_OUTPUT_ROOT
from search_segment_hybrid_v6 import (
    P_SCORE_DENOMINATOR,
    SEGMENT_COUNT,
    THRESHOLDS_M,
    _contact_metrics,
    _load_parent_candidate,
    _static_p_queries,
    parent_source_rows,
)
from search_segment_hybrid_v6 import PROTOCOL_PATH as V6_PROTOCOL_PATH
from search_segment_hybrid_v6 import SEARCH_PATH as V6_SEARCH_PATH

OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "attempt2_segmented_x2y4_v7_segment3_partition"
PROTOCOL_PATH = OUTPUT_ROOT / "v7_search_protocol_manifest.json"
SEARCH_PATH = OUTPUT_ROOT / "segment3_partition_search_results.json"

OBJECT_KEY = "bucket003"
CASE_ID = "bucket003_20231018_001_p1"
TARGET_HULLS = (16, 32)
TARGET_SEGMENT_INDEX = 3
TARGET_THRESHOLD_INDEX = 2
TARGET_THRESHOLD_M = 0.020
P_FLOOR = 0.70
TOP_PARTITIONS_PER_K = 3


def canonical_partitions(values: tuple[int, ...]) -> list[tuple[tuple[int, ...], ...]]:
    """Return every set partition once with sorted blocks and deterministic order."""
    if not values:
        return [()]
    first, *remaining = values
    results: set[tuple[tuple[int, ...], ...]] = set()
    for partition in canonical_partitions(tuple(remaining)):
        results.add(tuple(sorted(((first,), *partition))))
        for index in range(len(partition)):
            blocks = list(partition)
            blocks[index] = tuple(sorted((first, *blocks[index])))
            results.add(tuple(sorted(blocks)))
    return sorted(results, key=lambda partition: (len(partition), partition))


def _v6_array_path(max_hulls: int) -> Path:
    return (
        V6_OUTPUT_ROOT
        / "p_segment_attribution"
        / f"K{max_hulls:02d}_segment_contacts.npz"
    )


def _parent_v6_evidence() -> dict[str, Any]:
    """Bind the complete v6 0/6561 result and all three contact arrays."""
    search = json.loads(V6_SEARCH_PATH.read_text(encoding="utf-8"))
    if (
        search.get("status") != "COMPLETE"
        or search.get("selected_candidate_count") != 0
        or search.get("construction_eligible") is not False
        or search.get("heldout_access") != "NOT_ACCESSED_DEV3_ONLY"
    ):
        raise RuntimeError("v6 result does not authorize v7")
    counts = {
        f"K{int(group['max_hulls'])}": int(group["p_floor_passing_count"])
        for group in search["groups"]
    }
    if counts != {"K8": 0, "K16": 0, "K32": 0}:
        raise RuntimeError("v6 passing counts changed")
    arrays = []
    for max_hulls in (8, 16, 32):
        path = _v6_array_path(max_hulls)
        arrays.append(
            {
                "max_hulls": max_hulls,
                "path": relative_to_repo(path),
                "sha256": sha256_file(path),
            }
        )
    return {
        "status": "V6_COMPLETE_NO_SEGMENT_THRESHOLD_HYBRID",
        "selected_candidate_count": 0,
        "p_floor_passing_counts": counts,
        "search": {
            "path": relative_to_repo(V6_SEARCH_PATH),
            "sha256": sha256_file(V6_SEARCH_PATH),
        },
        "protocol": {
            "path": relative_to_repo(V6_PROTOCOL_PATH),
            "sha256": sha256_file(V6_PROTOCOL_PATH),
        },
        "contact_arrays": arrays,
    }


def best_near_miss_context(max_hulls: int) -> dict[str, Any]:
    """Select the frozen best budget-feasible v6 assignment without relaxing P."""
    if max_hulls not in TARGET_HULLS:
        raise ValueError("v7 only searches K16/K32 contexts")
    with np.load(_v6_array_path(max_hulls), allow_pickle=False) as values:
        oracle = np.asarray(values["oracle_contact"], dtype=bool)
        contacts = np.asarray(values["segment_contacts"], dtype=bool)
        counts = np.asarray(values["segment_hull_counts"], dtype=np.int32)
    rows = []
    segment_indices = np.arange(SEGMENT_COUNT, dtype=np.int32)
    for assignment in itertools.product(range(len(THRESHOLDS_M)), repeat=SEGMENT_COUNT):
        choice = np.asarray(assignment, dtype=np.int32)
        actual_hulls = int(counts[choice, segment_indices].sum())
        if actual_hulls > max_hulls:
            continue
        candidate_contact = np.any(contacts[choice, segment_indices], axis=0)
        metrics = _contact_metrics(candidate_contact, oracle)
        precision = float(metrics["precision"])
        recall = float(metrics["recall"])
        p_score = max(
            (1.0 - precision) / P_SCORE_DENOMINATOR,
            (1.0 - recall) / P_SCORE_DENOMINATOR,
        )
        rows.append(
            {
                "max_hulls": max_hulls,
                "actual_hulls": actual_hulls,
                "threshold_indices_by_segment": list(assignment),
                "thresholds_m_by_segment": [
                    THRESHOLDS_M[index] for index in assignment
                ],
                **metrics,
                "p_score": p_score,
            }
        )
    if not rows:
        raise RuntimeError("v7 context has no budget-feasible assignment")
    rows.sort(
        key=lambda row: (
            row["p_score"],
            row["phantom_contact_count"] + row["missed_contact_count"],
            row["actual_hulls"],
            row["threshold_indices_by_segment"],
        )
    )
    return rows[0]


def _target_parent(max_hulls: int) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [
        row
        for row in parent_source_rows()
        if int(row["max_hulls"]) == max_hulls
        and float(row["threshold_m"]) == TARGET_THRESHOLD_M
    ]
    if len(rows) != 1:
        raise RuntimeError("v7 segment3 parent identity changed")
    return rows[0], _load_parent_candidate(rows[0])


def _target_parts(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    parts = [
        part
        for part in candidate["parts"]
        if int(part["segment_index"]) == TARGET_SEGMENT_INDEX
    ]
    parts.sort(key=lambda part: int(part["local_part_index"]))
    if len(parts) != 4 or [part["source_lineage"] for part in parts] != [
        [0],
        [1],
        [2],
        [3],
    ]:
        raise RuntimeError("v7 target segment3 is not four unchanged base parts")
    return parts


def source_dependencies() -> list[dict[str, str]]:
    """Bind every source module whose runtime math is used by v7 search."""
    return [
        {
            "path": relative_to_repo(Path(core.__file__).resolve()),
            "sha256": sha256_file(Path(core.__file__).resolve()),
            "role": "UNCHANGED_P_CONTACT_MATH",
        },
        {
            "path": relative_to_repo(Path(v6.__file__).resolve()),
            "sha256": sha256_file(Path(v6.__file__).resolve()),
            "role": "V6_RUNTIME_QUERY_CONTACT_PARENT_HELPERS",
        },
        {
            "path": relative_to_repo(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
            "role": "V7_PARTITION_SEARCH",
        },
    ]


def freeze_protocol() -> dict[str, Any]:
    """Freeze K contexts and the full 15-partition search before geometry queries."""
    if SEARCH_PATH.exists():
        raise RuntimeError("cannot freeze v7 after partition search")
    evidence = _parent_v6_evidence()
    contexts = [best_near_miss_context(max_hulls) for max_hulls in TARGET_HULLS]
    parents = []
    for max_hulls in TARGET_HULLS:
        row, candidate = _target_parent(max_hulls)
        parts = _target_parts(candidate)
        parents.append(
            {
                "max_hulls": max_hulls,
                "candidate_id": row["candidate_id"],
                "manifest": row["manifest"],
                "segment3_parts": [
                    {
                        "path": part["path"],
                        "sha256": part["sha256"],
                        "local_part_index": part["local_part_index"],
                    }
                    for part in parts
                ],
            }
        )
    partitions = canonical_partitions((0, 1, 2, 3))
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v7_segment3_partition_protocol",
        "status": "FROZEN_BEFORE_V7_PARTITION_CONTACT_SEARCH",
        "selection_eligible": False,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "reason": "V6_SHOWS_SEGMENT3_PARTIAL_MERGE_IS_THE_ONLY_REMAINING_P_LEVER",
        "object_key": OBJECT_KEY,
        "case_id": CASE_ID,
        "parent_v6_evidence": evidence,
        "target": {
            "segment_index": TARGET_SEGMENT_INDEX,
            "source_threshold_index": TARGET_THRESHOLD_INDEX,
            "source_threshold_m": TARGET_THRESHOLD_M,
            "source_part_count": 4,
            "target_hulls": list(TARGET_HULLS),
            "contexts": contexts,
            "parents": parents,
        },
        "partition_contract": {
            "canonical_partition_count": len(partitions),
            "partitions": [
                [list(block) for block in partition] for partition in partitions
            ],
            "block_geometry": "CONVEX_HULL_OF_ALL_SOURCE_PART_VERTICES",
            "singleton_geometry": "UNCHANGED_SOURCE_PART",
            "cross_segment_merge": False,
            "actual_hulls": "CONTEXT_HULLS_MINUS_4_PLUS_PARTITION_BLOCK_COUNT",
        },
        "selection_contract": {
            "p_precision_floor": P_FLOOR,
            "p_recall_floor": P_FLOOR,
            "p_score": "MAX((1-PRECISION)/0.30,(1-RECALL)/0.30)",
            "sort": [
                "p_score",
                "phantom_plus_missed",
                "actual_hulls",
                "merge_added_volume_m3",
                "partition",
            ],
            "retain_per_K": TOP_PARTITIONS_PER_K,
            "full_role": "FORBIDDEN_UNTIL_BUILT_AND_UNCHANGED_FULL_P_R_G_SCREEN_PASS",
        },
        "source_dependencies": source_dependencies(),
    }
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("existing v7 protocol differs")
        return existing
    atomic_json(PROTOCOL_PATH, payload)
    return payload


def _load_protocol() -> dict[str, Any]:
    """Require exact frozen v7 source, v6 evidence, and source parts."""
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_V7_PARTITION_CONTACT_SEARCH":
        raise RuntimeError("v7 protocol is not frozen")
    evidence = protocol["parent_v6_evidence"]
    for entry in (
        evidence["search"],
        evidence["protocol"],
        *evidence["contact_arrays"],
    ):
        if entry["sha256"] != sha256_file(repo_path(entry["path"])):
            raise RuntimeError(f"v7 parent evidence changed: {entry['path']}")
    for parent in protocol["target"]["parents"]:
        if parent["manifest"]["sha256"] != sha256_file(
            repo_path(parent["manifest"]["path"])
        ):
            raise RuntimeError("v7 parent manifest changed")
        for part in parent["segment3_parts"]:
            if part["sha256"] != sha256_file(repo_path(part["path"])):
                raise RuntimeError("v7 source segment3 part changed")
    for source in protocol["source_dependencies"]:
        if source["sha256"] != sha256_file(repo_path(source["path"])):
            raise RuntimeError(f"v7 source changed: {source['path']}")
    return protocol


def _load_target_meshes(candidate: dict[str, Any]) -> list[trimesh.Trimesh]:
    meshes = []
    for part in _target_parts(candidate):
        path = repo_path(part["path"])
        if part["sha256"] != sha256_file(path):
            raise RuntimeError("v7 target part SHA changed")
        meshes.append(
            trimesh.load(path, force="mesh", process=False, maintain_order=True)
        )
    return meshes


def _block_mesh(
    source_meshes: list[trimesh.Trimesh], block: tuple[int, ...]
) -> trimesh.Trimesh:
    if len(block) == 1:
        return source_meshes[block[0]].copy()
    points = np.vstack([source_meshes[index].vertices for index in block])
    return trimesh.convex.convex_hull(points)


def _block_added_volume(
    source_meshes: list[trimesh.Trimesh],
    block: tuple[int, ...],
    merged: trimesh.Trimesh,
) -> float:
    if len(block) == 1:
        return 0.0
    union = core.build_exact_union_mesh([source_meshes[index] for index in block])
    raw = float(merged.volume) - float(union.volume)
    tolerance = max(1e-9, 1e-6 * float(union.volume))
    if raw < -tolerance:
        raise RuntimeError("v7 block convex volume below exact union")
    return max(0.0, raw)


def _mesh_contact(mesh: trimesh.Trimesh, queries: list[dict[str, Any]]) -> np.ndarray:
    scene = core._raycasting_scene(mesh)
    chunks = []
    for query in queries:
        clearance = (
            core._scene_signed_distance(scene, query["points"])
            - query["radii"][None, :]
        )
        chunks.append(clearance.min(axis=1) <= 0.0)
    return np.concatenate(chunks)


def search_partitions() -> dict[str, Any]:
    """Evaluate all 15 segment3 partitions in the two frozen near-miss contexts."""
    protocol = _load_protocol()
    queries, oracle_contact = _static_p_queries()
    groups = []
    selected = []
    partition_family = [
        tuple(tuple(block) for block in partition)
        for partition in protocol["partition_contract"]["partitions"]
    ]
    for max_hulls in TARGET_HULLS:
        context = next(
            row
            for row in protocol["target"]["contexts"]
            if row["max_hulls"] == max_hulls
        )
        with np.load(_v6_array_path(max_hulls), allow_pickle=False) as values:
            contacts = np.asarray(values["segment_contacts"], dtype=bool)
        assignment = np.asarray(context["threshold_indices_by_segment"], dtype=np.int32)
        other_segments = [
            index for index in range(SEGMENT_COUNT) if index != TARGET_SEGMENT_INDEX
        ]
        other_contact = np.any(
            contacts[assignment[other_segments], other_segments], axis=0
        )
        row, candidate = _target_parent(max_hulls)
        source_meshes = _load_target_meshes(candidate)
        rows = []
        for partition in partition_family:
            block_meshes = [_block_mesh(source_meshes, block) for block in partition]
            segment_contact = np.any(
                np.asarray([_mesh_contact(mesh, queries) for mesh in block_meshes]),
                axis=0,
            )
            candidate_contact = other_contact | segment_contact
            metrics = _contact_metrics(candidate_contact, oracle_contact)
            actual_hulls = int(context["actual_hulls"] - 4 + len(partition))
            added_volume = float(
                sum(
                    _block_added_volume(source_meshes, block, mesh)
                    for block, mesh in zip(partition, block_meshes, strict=True)
                )
            )
            precision = float(metrics["precision"])
            recall = float(metrics["recall"])
            p_score = max(
                (1.0 - precision) / P_SCORE_DENOMINATOR,
                (1.0 - recall) / P_SCORE_DENOMINATOR,
            )
            rows.append(
                {
                    "max_hulls": max_hulls,
                    "actual_hulls": actual_hulls,
                    "context_threshold_indices_by_segment": context[
                        "threshold_indices_by_segment"
                    ],
                    "source_candidate_id": row["candidate_id"],
                    "partition": [list(block) for block in partition],
                    "partition_block_count": len(partition),
                    "merge_added_volume_m3": added_volume,
                    **metrics,
                    "p_score": p_score,
                    "p_floor_pass": precision >= P_FLOOR and recall >= P_FLOOR,
                }
            )
        rows.sort(
            key=lambda value: (
                value["p_score"],
                value["phantom_contact_count"] + value["missed_contact_count"],
                value["actual_hulls"],
                value["merge_added_volume_m3"],
                value["partition"],
            )
        )
        passing = [
            row
            for row in rows
            if row["p_floor_pass"] and row["actual_hulls"] <= max_hulls
        ]
        retained = passing[:TOP_PARTITIONS_PER_K]
        for rank, result in enumerate(retained, start=1):
            result["rank_within_K"] = rank
            result["candidate_id"] = f"seg3part_v7_k{max_hulls:02d}_r{rank:02d}"
            selected.append(dict(result))
        groups.append(
            {
                "max_hulls": max_hulls,
                "context": context,
                "partition_count": len(rows),
                "p_floor_passing_count": len(passing),
                "retained_count": len(retained),
                "best_overall": rows[0],
                "retained": retained,
                "all_partitions": rows,
            }
        )
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v7_segment3_partition_search",
        "status": "COMPLETE",
        "selection_eligible": False,
        "construction_eligible": bool(selected),
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "protocol": {
            "path": relative_to_repo(PROTOCOL_PATH),
            "sha256": sha256_file(PROTOCOL_PATH),
        },
        "groups": groups,
        "selected_candidate_count": len(selected),
        "selected_candidates": selected,
    }
    atomic_json(SEARCH_PATH, payload)
    return payload


def parse_args() -> argparse.Namespace:
    """Parse v7 stages."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze-protocol", "search"))
    return parser.parse_args()


def main() -> int:
    """Execute one v7 stage."""
    args = parse_args()
    if args.command == "freeze-protocol":
        payload = freeze_protocol()
        print(f"E182_V7_SEARCH_PROTOCOL={payload['status']}")
        return 0
    payload = search_partitions()
    print(
        f"E182_V7_PARTITION_SEARCH={payload['status']} "
        f"selected={payload['selected_candidate_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
