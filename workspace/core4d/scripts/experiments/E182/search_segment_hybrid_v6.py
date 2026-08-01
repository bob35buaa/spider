#!/usr/bin/env python3
"""Freeze and search task-conditioned per-segment threshold hybrids for E182 v6."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import evaluate_task_queries as core
import numpy as np
import trimesh
from build_segmented_coacd_attempt2 import _candidate_identity_sha256
from build_segmented_coacd_attempt2_v5 import (
    FIXTURE_PATH as V5_FIXTURE_PATH,
)
from build_task_query_fixture import DEFAULT_OUTPUT_ROOT
from e182_common import atomic_json, relative_to_repo, repo_path, sha256_file
from evaluate_task_queries_v5 import install_v5_authority_adapter, load_v5_fixture
from run_task_query_screen_v5 import PROTOCOL_PATH as V5_SCREEN_PROTOCOL_PATH
from run_task_query_screen_v5 import SCREEN_ROOT as V5_SCREEN_ROOT

OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "attempt2_segmented_x2y4_v6_segment_hybrid"
PROTOCOL_PATH = OUTPUT_ROOT / "v6_search_protocol_manifest.json"
ATTRIBUTION_ROOT = OUTPUT_ROOT / "p_segment_attribution"
SEARCH_PATH = OUTPUT_ROOT / "hybrid_search_results.json"

OBJECT_KEY = "bucket003"
CASE_ID = "bucket003_20231018_001_p1"
SEGMENT_COUNT = 8
THRESHOLDS_M = (0.005, 0.010, 0.020)
TARGET_HULLS = (8, 16, 32)
P_FLOOR = 0.70
TOP_HYBRIDS_PER_K = 3
P_SCORE_DENOMINATOR = 0.30

V5_AGGREGATE_PATH = V5_SCREEN_ROOT / "screen_aggregate.json"
V5_TABLE_PATH = V5_SCREEN_ROOT / "screen_candidates.tsv"


def parent_source_rows() -> list[dict[str, Any]]:
    """Return the immutable v5 parent rows ordered by K then threshold."""
    fixture, _ = load_v5_fixture(V5_FIXTURE_PATH)
    rows = [dict(row) for row in fixture["candidates"]]
    rows.sort(key=lambda row: (int(row["max_hulls"]), float(row["threshold_m"])))
    if len(rows) != 9:
        raise RuntimeError("v6 parent candidate count changed")
    return rows


def _parent_screen_evidence() -> dict[str, Any]:
    """Bind the complete 0/9 v5 screen and every source result/manifest."""
    aggregate = json.loads(V5_AGGREGATE_PATH.read_text(encoding="utf-8"))
    if (
        aggregate.get("status") != "COMPLETE"
        or aggregate.get("candidate_count") != 9
        or aggregate.get("launch_floor_pass_count") != 0
        or aggregate.get("selected_group_count") != 0
        or aggregate.get("heldout_access") != "NOT_ACCESSED_DEV3_ONLY"
    ):
        raise RuntimeError("v5 screen does not authorize v6 hybrid search")
    rows = parent_source_rows()
    candidates = []
    for row in rows:
        manifest_path = repo_path(row["manifest"]["path"])
        result_path = (
            V5_SCREEN_ROOT / "results" / OBJECT_KEY / f"{row['candidate_id']}.json"
        )
        if (
            row["manifest"]["sha256"] != sha256_file(manifest_path)
            or not result_path.is_file()
        ):
            raise RuntimeError("v6 parent manifest/result evidence changed")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("status") != "COMPLETE"
            or result.get("selection_eligible") is not True
            or result.get("launch_floor", {}).get("status") != "FAIL"
        ):
            raise RuntimeError("v6 parent candidate result changed")
        candidates.append(
            {
                "candidate_id": row["candidate_id"],
                "max_hulls": int(row["max_hulls"]),
                "actual_hulls": int(row["actual_hulls"]),
                "threshold_m": float(row["threshold_m"]),
                "manifest": row["manifest"],
                "result": {
                    "path": relative_to_repo(result_path),
                    "sha256": sha256_file(result_path),
                },
            }
        )
    return {
        "status": "V5_SCREEN_COMPLETE_NO_LAUNCHABLE_GROUP",
        "candidate_count": aggregate["candidate_count"],
        "launch_floor_pass_count": aggregate["launch_floor_pass_count"],
        "selected_group_count": aggregate["selected_group_count"],
        "aggregate": {
            "path": relative_to_repo(V5_AGGREGATE_PATH),
            "sha256": sha256_file(V5_AGGREGATE_PATH),
        },
        "candidate_table": {
            "path": relative_to_repo(V5_TABLE_PATH),
            "sha256": sha256_file(V5_TABLE_PATH),
        },
        "screen_protocol": {
            "path": relative_to_repo(V5_SCREEN_PROTOCOL_PATH),
            "sha256": sha256_file(V5_SCREEN_PROTOCOL_PATH),
        },
        "candidates": candidates,
    }


def freeze_protocol() -> dict[str, Any]:
    """Freeze attribution, enumeration, and selection before segment-level queries."""
    if list(ATTRIBUTION_ROOT.glob("*")) or SEARCH_PATH.exists():
        raise RuntimeError("cannot freeze v6 search after attribution/search")
    evidence = _parent_screen_evidence()
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v6_hybrid_search_protocol",
        "status": "FROZEN_BEFORE_V6_SEGMENT_ATTRIBUTION_OR_SEARCH",
        "selection_eligible": False,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "reason": "V5_P_ONLY_THRESHOLD_TRADEOFF_WITH_COMPLEMENTARY_PRECISION_RECALL",
        "object_key": OBJECT_KEY,
        "case_id": CASE_ID,
        "parent_screen_evidence": evidence,
        "attribution_contract": {
            "pose_sources": ["reference", "e178_final"],
            "pose_count": 882,
            "robot_geom_consumer": "P_collision",
            "contact": "MIN_RADIUS_ADJUSTED_CLEARANCE_LE_ZERO",
            "segment_contact": "EXACT_UNION_OF_PARENT_PARTS_WITH_SAME_SEGMENT_INDEX",
            "candidate_contact_reconstruction": "OR_OVER_EIGHT_SEGMENTS",
            "required_validation": "EXACT_COUNTS_VS_FORMAL_V5_RESULT_FOR_ALL_9_PARENTS",
        },
        "enumeration_contract": {
            "thresholds_m": list(THRESHOLDS_M),
            "segment_count": SEGMENT_COUNT,
            "assignments_per_K": len(THRESHOLDS_M) ** SEGMENT_COUNT,
            "target_hulls": list(TARGET_HULLS),
            "actual_hulls": "SUM_SELECTED_PARENT_SEGMENT_HULL_COUNTS",
            "budget_gate": "ACTUAL_HULLS_LE_K",
            "p_precision_floor": P_FLOOR,
            "p_recall_floor": P_FLOOR,
            "p_score": "MAX((1-PRECISION)/0.30,(1-RECALL)/0.30)",
            "sort": [
                "p_score",
                "phantom_plus_missed",
                "actual_hulls",
                "threshold_indices_by_segment",
            ],
            "retain_per_K": TOP_HYBRIDS_PER_K,
            "full_role": "FORBIDDEN_UNTIL_BUILT_AND_UNCHANGED_FULL_P_R_G_SCREEN_PASS",
        },
        "construction_contract": {
            "parts": "IMMUTABLE_PARENT_PARTS_SELECTED_BY_SEGMENT_AND_THRESHOLD",
            "new_geometry_operation": "NONE",
            "cross_segment_merge": False,
            "heldout_dependency": False,
        },
        "fixture": {
            "path": relative_to_repo(V5_FIXTURE_PATH),
            "sha256": sha256_file(V5_FIXTURE_PATH),
        },
        "source_dependencies": [
            {
                "path": relative_to_repo(Path(core.__file__).resolve()),
                "sha256": sha256_file(Path(core.__file__).resolve()),
                "role": "UNCHANGED_P_CONTACT_MATH",
            },
            {
                "path": relative_to_repo(Path(__file__)),
                "sha256": sha256_file(Path(__file__)),
                "role": "V6_ATTRIBUTION_ENUMERATION",
            },
        ],
    }
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("existing v6 search protocol differs")
        return existing
    atomic_json(PROTOCOL_PATH, payload)
    return payload


def _load_protocol() -> dict[str, Any]:
    """Require exact frozen v6 source and all v5 parent evidence."""
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_V6_SEGMENT_ATTRIBUTION_OR_SEARCH":
        raise RuntimeError("v6 search protocol is not frozen")
    if protocol["fixture"]["sha256"] != sha256_file(V5_FIXTURE_PATH):
        raise RuntimeError("v6 fixture SHA changed")
    evidence = protocol["parent_screen_evidence"]
    checks = (
        evidence["aggregate"],
        evidence["candidate_table"],
        evidence["screen_protocol"],
    )
    for entry in checks:
        if entry["sha256"] != sha256_file(repo_path(entry["path"])):
            raise RuntimeError(f"v6 parent evidence changed: {entry['path']}")
    for candidate in evidence["candidates"]:
        for key in ("manifest", "result"):
            entry = candidate[key]
            if entry["sha256"] != sha256_file(repo_path(entry["path"])):
                raise RuntimeError(
                    f"v6 parent candidate evidence changed: {entry['path']}"
                )
    for source in protocol["source_dependencies"]:
        if source["sha256"] != sha256_file(repo_path(source["path"])):
            raise RuntimeError(f"v6 source changed: {source['path']}")
    return protocol


def _contact_metrics(
    candidate_contact: np.ndarray, oracle_contact: np.ndarray
) -> dict[str, int | float]:
    candidate = np.asarray(candidate_contact, dtype=bool).reshape(-1)
    oracle = np.asarray(oracle_contact, dtype=bool).reshape(-1)
    if candidate.shape != oracle.shape:
        raise ValueError("hybrid contact shapes differ")
    tp = int((candidate & oracle).sum())
    phantom = int((candidate & ~oracle).sum())
    missed = int((~candidate & oracle).sum())
    tn = int((~candidate & ~oracle).sum())
    precision = tp / (tp + phantom) if tp + phantom else 0.0
    recall = tp / (tp + missed) if tp + missed else 0.0
    return {
        "pose_count": len(candidate),
        "true_positive_count": tp,
        "true_negative_count": tn,
        "phantom_contact_count": phantom,
        "missed_contact_count": missed,
        "precision": precision,
        "recall": recall,
    }


def enumerate_hybrids(
    *,
    oracle_contact: np.ndarray,
    segment_contacts: np.ndarray,
    segment_hull_counts: np.ndarray,
    max_hulls: int,
) -> list[dict[str, Any]]:
    """Enumerate all threshold assignments and return P-floor-passing rows."""
    oracle = np.asarray(oracle_contact, dtype=bool).reshape(-1)
    contacts = np.asarray(segment_contacts, dtype=bool)
    counts = np.asarray(segment_hull_counts, dtype=np.int32)
    expected_shape = (len(THRESHOLDS_M), SEGMENT_COUNT, len(oracle))
    if contacts.shape != expected_shape:
        raise ValueError(f"segment contacts shape changed: {contacts.shape}")
    if counts.shape != (len(THRESHOLDS_M), SEGMENT_COUNT) or np.any(counts < 1):
        raise ValueError("segment hull-count matrix changed")
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
        if precision < P_FLOOR or recall < P_FLOOR:
            continue
        p_score = max(
            (1.0 - precision) / P_SCORE_DENOMINATOR,
            (1.0 - recall) / P_SCORE_DENOMINATOR,
        )
        rows.append(
            {
                "max_hulls": int(max_hulls),
                "actual_hulls": actual_hulls,
                "threshold_indices_by_segment": list(assignment),
                "thresholds_m_by_segment": [
                    THRESHOLDS_M[index] for index in assignment
                ],
                **metrics,
                "p_score": p_score,
            }
        )
    rows.sort(
        key=lambda row: (
            row["p_score"],
            row["phantom_contact_count"] + row["missed_contact_count"],
            row["actual_hulls"],
            row["threshold_indices_by_segment"],
        )
    )
    return rows


def _load_parent_candidate(row: dict[str, Any]) -> dict[str, Any]:
    path = repo_path(row["manifest"]["path"])
    if row["manifest"]["sha256"] != sha256_file(path):
        raise RuntimeError("parent candidate manifest SHA changed")
    return json.loads(path.read_text(encoding="utf-8"))


def _segment_scenes(candidate: dict[str, Any]) -> list[Any]:
    """Build one exact union scene per immutable parent segment."""
    grouped: dict[int, list[trimesh.Trimesh]] = {
        index: [] for index in range(SEGMENT_COUNT)
    }
    for part in candidate["parts"]:
        path = repo_path(part["path"])
        if part["sha256"] != sha256_file(path):
            raise RuntimeError("parent part SHA changed")
        grouped[int(part["segment_index"])].append(
            trimesh.load(path, force="mesh", process=False, maintain_order=True)
        )
    scenes = []
    for segment_index in range(SEGMENT_COUNT):
        parts = grouped[segment_index]
        if not parts:
            raise RuntimeError("parent segment is empty")
        union = parts[0] if len(parts) == 1 else core.build_exact_union_mesh(parts)
        scenes.append(core._raycasting_scene(union))
    return scenes


def _static_p_queries() -> tuple[list[dict[str, Any]], np.ndarray]:
    """Materialize only the 882 frozen static P pose queries and oracle contacts."""
    install_v5_authority_adapter()
    fixture, _ = load_v5_fixture(V5_FIXTURE_PATH)
    case = fixture["cases"][0]
    prg_path = repo_path(case["prg_manifest"]["path"])
    if case["prg_manifest"]["sha256"] != sha256_file(prg_path):
        raise RuntimeError("v6 PRG manifest changed")
    prg = json.loads(prg_path.read_text(encoding="utf-8"))
    static_entries = [
        entry for entry in prg["chunks"] if entry["source_family"] != "cem_on_a"
    ]
    parent = parent_source_rows()[0]
    _, _, oracle = core._load_candidate_and_oracle(parent)
    oracle_scene = core._raycasting_scene(oracle)
    queries = []
    oracle_contacts = []
    for entry in static_entries:
        chunk_path = core._tape_chunk_path(entry)
        if entry["sha256"] != sha256_file(chunk_path):
            raise RuntimeError("v6 static P chunk changed")
        with np.load(chunk_path, allow_pickle=False) as chunk:
            full = np.arange(len(chunk["point_geom_id"]), dtype=np.int32)
            points = core.materialize_selected_query_points(chunk, full)
            radii = np.asarray(chunk["point_radius_m"], dtype=np.float64)
            p_mask = np.isin(
                np.asarray(chunk["point_geom_id"], dtype=np.int32),
                prg["consumer_geom_ids"]["P_collision"],
            )
            p_points = points[:, p_mask]
            p_radii = radii[p_mask]
        clearance = (
            core._scene_signed_distance(oracle_scene, p_points) - p_radii[None, :]
        )
        oracle_contacts.append(clearance.min(axis=1) <= 0.0)
        queries.append(
            {
                "source_family": entry["source_family"],
                "source_chunk_index": int(entry["source_chunk_index"]),
                "points": p_points,
                "radii": p_radii,
            }
        )
    oracle_contact = np.concatenate(oracle_contacts)
    if len(oracle_contact) != 882:
        raise RuntimeError("v6 static P pose count changed")
    return queries, oracle_contact


def _candidate_segment_contacts(
    candidate: dict[str, Any], queries: list[dict[str, Any]]
) -> np.ndarray:
    """Return (8,882) contact bits for one parent candidate."""
    scenes = _segment_scenes(candidate)
    by_segment = []
    for scene in scenes:
        chunks = []
        for query in queries:
            clearance = (
                core._scene_signed_distance(scene, query["points"])
                - query["radii"][None, :]
            )
            chunks.append(clearance.min(axis=1) <= 0.0)
        by_segment.append(np.concatenate(chunks))
    result = np.asarray(by_segment, dtype=bool)
    if result.shape != (SEGMENT_COUNT, 882):
        raise RuntimeError("v6 segment contact shape changed")
    return result


def analyze_and_search() -> dict[str, Any]:
    """Attribute all parents, validate exact reconstruction, and search 3^8 hybrids."""
    protocol = _load_protocol()
    queries, oracle_contact = _static_p_queries()
    groups = []
    selected_rows = []
    for max_hulls in TARGET_HULLS:
        sources = [
            row for row in parent_source_rows() if int(row["max_hulls"]) == max_hulls
        ]
        sources.sort(key=lambda row: float(row["threshold_m"]))
        if len(sources) != len(THRESHOLDS_M):
            raise RuntimeError("v6 parent K group changed")
        contacts = []
        hull_counts = []
        validations = []
        for row in sources:
            candidate = _load_parent_candidate(row)
            segment_contacts = _candidate_segment_contacts(candidate, queries)
            reconstructed = np.any(segment_contacts, axis=0)
            metrics = _contact_metrics(reconstructed, oracle_contact)
            result_entry = next(
                entry
                for entry in protocol["parent_screen_evidence"]["candidates"]
                if entry["candidate_id"] == row["candidate_id"]
            )
            result = json.loads(repo_path(result_entry["result"]["path"]).read_text())
            formal = result["p_pose_contact"]
            count_keys = (
                "pose_count",
                "true_positive_count",
                "true_negative_count",
                "phantom_contact_count",
                "missed_contact_count",
            )
            if any(int(metrics[key]) != int(formal[key]) for key in count_keys):
                raise RuntimeError("segment OR does not reproduce formal P counts")
            contacts.append(segment_contacts)
            per_segment = sorted(
                candidate["per_segment"], key=lambda value: value["segment_index"]
            )
            hull_counts.append([int(value["hull_count"]) for value in per_segment])
            validations.append(
                {
                    "candidate_id": row["candidate_id"],
                    "threshold_m": float(row["threshold_m"]),
                    "formal_result": result_entry["result"],
                    "reconstructed_contact": metrics,
                    "exact_count_match": True,
                }
            )
        contacts_array = np.asarray(contacts, dtype=bool)
        hull_counts_array = np.asarray(hull_counts, dtype=np.int32)
        arrays_path = ATTRIBUTION_ROOT / f"K{max_hulls:02d}_segment_contacts.npz"
        core._atomic_npz(
            arrays_path,
            oracle_contact=oracle_contact,
            segment_contacts=contacts_array,
            segment_hull_counts=hull_counts_array,
            thresholds_m=np.asarray(THRESHOLDS_M, dtype=np.float64),
        )
        passing = enumerate_hybrids(
            oracle_contact=oracle_contact,
            segment_contacts=contacts_array,
            segment_hull_counts=hull_counts_array,
            max_hulls=max_hulls,
        )
        retained = passing[:TOP_HYBRIDS_PER_K]
        for rank, row in enumerate(retained, start=1):
            row["rank_within_K"] = rank
            row["candidate_id"] = f"segmix_v6_k{max_hulls:02d}_r{rank:02d}"
            row["source_candidate_ids_by_segment"] = [
                sources[index]["candidate_id"]
                for index in row["threshold_indices_by_segment"]
            ]
            selected_rows.append(dict(row))
        groups.append(
            {
                "max_hulls": max_hulls,
                "source_candidates": [row["candidate_id"] for row in sources],
                "segment_contact_arrays": {
                    "path": relative_to_repo(arrays_path),
                    "sha256": sha256_file(arrays_path),
                },
                "parent_validations": validations,
                "assignment_count": len(THRESHOLDS_M) ** SEGMENT_COUNT,
                "p_floor_passing_count": len(passing),
                "retained_count": len(retained),
                "retained": retained,
            }
        )
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v6_hybrid_search",
        "status": "COMPLETE",
        "selection_eligible": False,
        "construction_eligible": bool(selected_rows),
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "protocol": {
            "path": relative_to_repo(PROTOCOL_PATH),
            "sha256": sha256_file(PROTOCOL_PATH),
        },
        "oracle_contact_pose_count": int(oracle_contact.sum()),
        "pose_count": len(oracle_contact),
        "groups": groups,
        "selected_candidate_count": len(selected_rows),
        "selected_candidates": selected_rows,
        "selected_candidate_identity_sha256": _candidate_identity_sha256(
            [
                {
                    "object_key": OBJECT_KEY,
                    "candidate_id": row["candidate_id"],
                    "candidate_asset_sha256": "NOT_BUILT",
                    "manifest": {"path": "NOT_BUILT", "sha256": "NOT_BUILT"},
                    "max_hulls": row["max_hulls"],
                    "actual_hulls": row["actual_hulls"],
                    "threshold_m": -1.0,
                    "max_vertices": 256,
                }
                for row in selected_rows
            ]
        ),
    }
    atomic_json(SEARCH_PATH, payload)
    return payload


def parse_args() -> argparse.Namespace:
    """Parse v6 search stages."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze-protocol", "analyze-search"))
    return parser.parse_args()


def main() -> int:
    """Execute one v6 search stage."""
    args = parse_args()
    if args.command == "freeze-protocol":
        payload = freeze_protocol()
        print(f"E182_V6_SEARCH_PROTOCOL={payload['status']}")
        return 0
    payload = analyze_and_search()
    print(
        f"E182_V6_HYBRID_SEARCH={payload['status']} "
        f"selected={payload['selected_candidate_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
