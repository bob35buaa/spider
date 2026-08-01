#!/usr/bin/env python3
"""Derive v5 global-K candidates with audited boolean-volume tolerance."""

from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path
from typing import Any

from build_segmented_coacd_attempt2 import (
    DEFAULT_FIXTURE,
    SEGMENT_COUNT,
    _candidate_asset_sha256,
    _candidate_identity_sha256,
)
from build_segmented_coacd_attempt2_v4 import (
    ATTEMPT_ROOT as V4_ATTEMPT_ROOT,
)
from build_segmented_coacd_attempt2_v4 import (
    CANDIDATE_ROOT as V4_CANDIDATE_ROOT,
)
from build_segmented_coacd_attempt2_v4 import (
    FINAL_MAX_VERTICES,
    ReducerPart,
    _base_root,
    _checked_reducer_part,
    _export_reduced_part,
    _load_base_parts,
    _merged_part,
)
from build_segmented_coacd_attempt2_v4 import (
    PROTOCOL_PATH as V4_PROTOCOL_PATH,
)
from build_task_query_fixture import DEFAULT_OUTPUT_ROOT
from e182_common import atomic_json, relative_to_repo, repo_path, sha256_file
from evaluate_task_queries import build_exact_union_mesh

ATTEMPT_ROOT = DEFAULT_OUTPUT_ROOT / "attempt2_segmented_x2y4_v5_numeric_tolerance"
PROTOCOL_PATH = ATTEMPT_ROOT / "attempt2_v5_protocol_manifest.json"
CANDIDATE_ROOT = ATTEMPT_ROOT / "candidates"
FIXTURE_PATH = ATTEMPT_ROOT / "query_fixture_manifest.json"
SCREEN_ROOT = ATTEMPT_ROOT / "screen"
V4_BASE_SUMMARY_PATH = V4_ATTEMPT_ROOT / "base_summary.json"

OBJECT_KEY = "bucket003"
CASE_ID = "bucket003_20231018_001_p1"
THRESHOLDS_M = (0.005, 0.010, 0.020)
TARGET_HULLS = (8, 16, 32)
ABSOLUTE_VOLUME_TOLERANCE_M3 = 1e-9
RELATIVE_VOLUME_TOLERANCE = 1e-6
COST_ROUND_DIGITS = 15
V4_OLD_ABSOLUTE_TOLERANCE_M3 = 1e-10
V4_BUILDER_PATH = Path(_load_base_parts.__code__.co_filename).resolve()
EXACT_UNION_SOURCE_PATH = Path(build_exact_union_mesh.__code__.co_filename).resolve()


def _threshold_tag(threshold_m: float) -> str:
    return f"t{round(threshold_m * 1000):03d}"


def _candidate_id(threshold_m: float, max_hulls: int) -> str:
    return (
        f"segx2y4_gmerge5_{_threshold_tag(threshold_m)}_"
        f"k{max_hulls:02d}_v{FINAL_MAX_VERTICES:03d}"
    )


def candidate_family() -> list[dict[str, Any]]:
    """Return the unchanged 3×K family with v5-specific identities."""
    return [
        {
            "candidate_id": _candidate_id(threshold_m, max_hulls),
            "method": "ISOLATED_COACD_GLOBAL_BUDGET_MERGE_V2",
            "threshold_m": threshold_m,
            "max_hulls": max_hulls,
            "max_vertices": FINAL_MAX_VERTICES,
        }
        for threshold_m in THRESHOLDS_M
        for max_hulls in TARGET_HULLS
    ]


def _v4_failure_evidence() -> dict[str, Any]:
    """Quantify and bind the sole v4 negative-roundoff failure."""
    protocol = json.loads(V4_PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_V4_BUILD_OR_SCORE":
        raise RuntimeError("attempt2-v4 protocol changed")
    base_summary = json.loads(V4_BASE_SUMMARY_PATH.read_text(encoding="utf-8"))
    if base_summary.get("status") != "BUILD_PASS" or base_summary.get(
        "total_hulls"
    ) != [32, 29, 18]:
        raise RuntimeError("attempt2-v4 base summary changed")
    successful = sorted(V4_CANDIDATE_ROOT.glob("*/manifest.json"))
    if len(successful) != 3 or any(
        "t005" not in path.parent.name for path in successful
    ):
        raise RuntimeError("attempt2-v4 candidate prefix changed")
    if (V4_ATTEMPT_ROOT / "build_summary.json").exists():
        raise RuntimeError("attempt2-v4 unexpectedly has a complete build summary")
    groups = _load_base_parts(0.010)
    left = [part for part in groups[7] if part.lineage == (1,)]
    right = [part for part in groups[7] if part.lineage == (2,)]
    if len(left) != 1 or len(right) != 1:
        raise RuntimeError("attempt2-v4 failing pair identity changed")
    merged = _merged_part(left[0], right[0])
    exact_union = build_exact_union_mesh((left[0].mesh, right[0].mesh))
    union_volume = float(exact_union.volume)
    merged_volume = float(merged.mesh.volume)
    raw_added = merged_volume - union_volume
    relative = raw_added / union_volume
    old_tolerance = V4_OLD_ABSOLUTE_TOLERANCE_M3 * max(1.0, union_volume)
    new_tolerance = max(
        ABSOLUTE_VOLUME_TOLERANCE_M3,
        RELATIVE_VOLUME_TOLERANCE * union_volume,
    )
    if not raw_added < -old_tolerance or abs(raw_added) > new_tolerance:
        raise RuntimeError(
            "attempt2-v4 numeric failure no longer supports v5 tolerance"
        )
    return {
        "status": "V4_REDUCER_NUMERIC_TOLERANCE_FAILED",
        "successful_candidate_count": len(successful),
        "successful_candidates": [
            {
                "path": relative_to_repo(path),
                "sha256": sha256_file(path),
                "candidate_asset_sha256": json.loads(path.read_text(encoding="utf-8"))[
                    "candidate_asset_sha256"
                ],
            }
            for path in successful
        ],
        "failed_threshold_m": 0.010,
        "failed_target_hulls": 8,
        "segment_index": 7,
        "left_lineage": [1],
        "right_lineage": [2],
        "exact_pair_union_volume_m3": union_volume,
        "merged_convex_volume_m3": merged_volume,
        "raw_added_volume_m3": raw_added,
        "relative_difference": relative,
        "v4_tolerance_m3": old_tolerance,
        "v5_tolerance_m3": new_tolerance,
        "changed_contract": "ABSOLUTE_OR_RELATIVE_NUMERIC_TOLERANCE",
        "v4_protocol": {
            "path": relative_to_repo(V4_PROTOCOL_PATH),
            "sha256": sha256_file(V4_PROTOCOL_PATH),
        },
        "v4_base_summary": {
            "path": relative_to_repo(V4_BASE_SUMMARY_PATH),
            "sha256": sha256_file(V4_BASE_SUMMARY_PATH),
        },
    }


def freeze_protocol() -> dict[str, Any]:
    """Freeze the corrected reducer after base build but before v5 candidate/score."""
    if list(CANDIDATE_ROOT.glob("*/manifest.json")):
        raise RuntimeError("cannot freeze v5 after candidate build")
    if list((SCREEN_ROOT / "results").glob("*.json")):
        raise RuntimeError("cannot freeze v5 after score")
    v4_protocol = json.loads(V4_PROTOCOL_PATH.read_text(encoding="utf-8"))
    rows = candidate_family()
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v5_protocol",
        "status": "FROZEN_AFTER_V4_BASE_BEFORE_V5_CANDIDATE_OR_SCORE",
        "selection_eligible": False,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "reason": "V4_BOOLEAN_VOLUME_ROUNDOFF_EXCEEDED_OVERLY_STRICT_ABSOLUTE_TOLERANCE",
        "object_key": OBJECT_KEY,
        "case_id": CASE_ID,
        "candidate_family": rows,
        "candidate_count": len(rows),
        "base_decomposition": v4_protocol["base_decomposition"],
        "reducer": {
            **v4_protocol["reducer"],
            "name": "DETERMINISTIC_INTRA_SEGMENT_GREEDY_CONVEX_MERGE_V2",
            "negative_roundoff_tolerance": {
                "absolute_m3": ABSOLUTE_VOLUME_TOLERANCE_M3,
                "relative_to_exact_pair_union": RELATIVE_VOLUME_TOLERANCE,
                "effective": "MAX_ABSOLUTE_RELATIVE",
            },
            "negative_within_tolerance": "CLAMP_ADDED_VOLUME_TO_ZERO_AND_RECORD_RAW",
            "negative_outside_tolerance": "HARD_FAIL",
            "cost_round_digits": COST_ROUND_DIGITS,
        },
        "physics_contract": v4_protocol["physics_contract"],
        "topology": v4_protocol["topology"],
        "selection_contract": v4_protocol["selection_contract"],
        "v4_failure_evidence": _v4_failure_evidence(),
        "oracle": v4_protocol["oracle"],
        "original_fixture": {
            "path": relative_to_repo(DEFAULT_FIXTURE),
            "sha256": sha256_file(DEFAULT_FIXTURE),
        },
        "source_dependencies": [
            {
                "role": "V4_BASE_LOAD_MERGE_EXPORT",
                "path": relative_to_repo(V4_BUILDER_PATH),
                "sha256": sha256_file(V4_BUILDER_PATH),
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
            raise RuntimeError("existing attempt2-v5 protocol differs")
        return existing
    atomic_json(PROTOCOL_PATH, payload)
    return payload


def _load_protocol() -> dict[str, Any]:
    """Require exact v5 source, v4 base, prefix failure, and fixture authority."""
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_AFTER_V4_BASE_BEFORE_V5_CANDIDATE_OR_SCORE":
        raise RuntimeError("attempt2-v5 protocol is not frozen")
    checks = (
        (protocol["builder_source"], Path(__file__)),
        (protocol["v4_failure_evidence"]["v4_protocol"], V4_PROTOCOL_PATH),
        (protocol["v4_failure_evidence"]["v4_base_summary"], V4_BASE_SUMMARY_PATH),
        (protocol["original_fixture"], DEFAULT_FIXTURE),
    )
    for entry, path in checks:
        if entry["sha256"] != sha256_file(path):
            raise RuntimeError(f"attempt2-v5 authority SHA changed: {path}")
    for dependency in protocol["source_dependencies"]:
        path = repo_path(dependency["path"])
        if dependency["sha256"] != sha256_file(path):
            raise RuntimeError(f"attempt2-v5 source dependency changed: {path}")
    for candidate in protocol["v4_failure_evidence"]["successful_candidates"]:
        path = repo_path(candidate["path"])
        if candidate["sha256"] != sha256_file(path):
            raise RuntimeError(f"attempt2-v4 prefix candidate changed: {path}")
    return protocol


def _merge_cost(
    left: ReducerPart, right: ReducerPart
) -> tuple[float, float, float, ReducerPart, float]:
    merged = _merged_part(left, right)
    exact_union = build_exact_union_mesh((left.mesh, right.mesh))
    union_volume = float(exact_union.volume)
    raw_added = float(merged.mesh.volume) - union_volume
    tolerance = max(
        ABSOLUTE_VOLUME_TOLERANCE_M3,
        RELATIVE_VOLUME_TOLERANCE * union_volume,
    )
    if raw_added < -tolerance:
        raise RuntimeError(
            "convex merge volume is smaller than exact pair union beyond v5 tolerance"
        )
    return max(0.0, raw_added), raw_added, tolerance, merged, union_volume


def reduce_segmented_parts(
    parts_by_segment: dict[int, list[ReducerPart]],
    *,
    target_hulls: int,
) -> tuple[dict[int, list[ReducerPart]], list[dict[str, Any]]]:
    """Apply one auditable deterministic hierarchy with bounded numeric clamp."""
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
        choices = []
        for segment_index, parts in groups.items():
            for left, right in combinations(parts, 2):
                added, raw_added, tolerance, merged, union_volume = _merge_cost(
                    left, right
                )
                key = (
                    round(added, COST_ROUND_DIGITS),
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
                        added,
                        raw_added,
                        tolerance,
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
            added,
            raw_added,
            tolerance,
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
                "raw_added_volume_m3": raw_added,
                "tolerance_m3": tolerance,
                "negative_roundoff_clamped": raw_added < 0.0,
                "added_volume_m3": added,
            }
        )
    return groups, trace


def build_candidate(row: dict[str, Any]) -> dict[str, Any]:
    """Derive one v5 candidate from an immutable v4 CoACD base."""
    protocol = _load_protocol()
    frozen = [
        value
        for value in protocol["candidate_family"]
        if value["candidate_id"] == row["candidate_id"]
    ]
    if frozen != [row]:
        raise RuntimeError("candidate row differs from frozen v5 family")
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
        "physics_contract": protocol["physics_contract"],
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
        raise RuntimeError("non-resumable attempt2-v5 candidate manifest")
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
        raise RuntimeError("attempt2-v5 final global hull budget violated")
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v5_candidate",
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
    atomic_json(manifest_path, payload)
    print(
        "E182_SEGMENTED_COACD_V5=PASS "
        f"candidate={row['candidate_id']} base={base_hulls} final={len(parts)} "
        f"merges={len(trace)} clamps={payload['negative_roundoff_clamp_count']} "
        f"wall={payload['build_wall_seconds']:.3f}"
    )
    return payload


def build_candidates() -> dict[str, Any]:
    """Derive all nine v5 candidates without rerunning CoACD."""
    protocol = _load_protocol()
    manifests = [build_candidate(row) for row in protocol["candidate_family"]]
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v5_build",
        "status": "BUILD_PASS",
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "candidate_count": len(manifests),
        "candidate_ids": [manifest["candidate_id"] for manifest in manifests],
        "actual_hulls": [manifest["hull_count"] for manifest in manifests],
        "negative_roundoff_clamp_counts": [
            manifest["negative_roundoff_clamp_count"] for manifest in manifests
        ],
        "total_wall_seconds": float(
            sum(manifest["build_wall_seconds"] for manifest in manifests)
        ),
    }
    if payload["candidate_count"] != 9:
        raise RuntimeError("attempt2-v5 candidate family incomplete")
    atomic_json(ATTEMPT_ROOT / "build_summary.json", payload)
    return payload


def build_fixture() -> dict[str, Any]:
    """Freeze the bucket003-only v5 fixture before any score."""
    protocol = _load_protocol()
    summary = json.loads(
        (ATTEMPT_ROOT / "build_summary.json").read_text(encoding="utf-8")
    )
    if summary.get("status") != "BUILD_PASS" or summary.get("candidate_count") != 9:
        raise RuntimeError("attempt2-v5 build summary incomplete")
    original = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    cases = [case for case in original["cases"] if case["case_id"] == CASE_ID]
    if len(cases) != 1:
        raise RuntimeError("bucket003 dev case changed")
    rows = []
    for frozen in protocol["candidate_family"]:
        manifest_path = CANDIDATE_ROOT / frozen["candidate_id"] / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "BUILD_PASS":
            raise RuntimeError("attempt2-v5 fixture candidate is not BUILD_PASS")
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
        "stage": "S2_task_query_fixture_attempt2_segmented_x2y4_v5_numeric_tolerance",
        "status": "FROZEN",
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "freeze_timing": "AFTER_V5_BUILD_BEFORE_V5_SCORE",
        "attempt2_v5_protocol": {
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
            raise RuntimeError("existing attempt2-v5 fixture differs")
        return existing
    atomic_json(FIXTURE_PATH, payload)
    return payload


def parse_args() -> argparse.Namespace:
    """Parse explicit v5 stages."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("freeze-protocol", "build-candidates", "build-fixture"),
    )
    return parser.parse_args()


def main() -> int:
    """Execute one attempt2-v5 stage."""
    args = parse_args()
    if args.command == "freeze-protocol":
        payload = freeze_protocol()
        print(f"E182_ATTEMPT2_V5_PROTOCOL={payload['status']}")
        return 0
    if args.command == "build-candidates":
        payload = build_candidates()
        print(
            f"E182_ATTEMPT2_V5_BUILD={payload['status']} "
            f"candidates={payload['candidate_count']}"
        )
        return 0
    payload = build_fixture()
    print(
        f"E182_ATTEMPT2_V5_FIXTURE={payload['status']} "
        f"candidates={payload['candidate_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
