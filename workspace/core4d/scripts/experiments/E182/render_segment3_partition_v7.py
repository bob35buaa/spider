#!/usr/bin/env python3
"""Render frozen v7 segment3 partition signatures as task-contact diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import evaluate_task_queries as core
import numpy as np
import search_segment_hybrid_v6 as v6
import trimesh
from diagnose_p_contact import _point_classes, _render_2d, _render_3d
from e182_common import atomic_json, relative_to_repo, repo_path, sha256_file
from search_segment3_partition_v7 import (
    OUTPUT_ROOT as V7_OUTPUT_ROOT,
)
from search_segment3_partition_v7 import (
    PROTOCOL_PATH,
    SEARCH_PATH,
    TARGET_SEGMENT_INDEX,
    _block_mesh,
    _load_protocol,
    _load_target_meshes,
)

OUTPUT_ROOT = V7_OUTPUT_ROOT / "visual_diagnostic"
DIAGNOSTIC_K = 16
EXPECTED_SIGNATURES = ((18, 8, 9), (19, 10, 8))
SIGNATURE_LABELS = {
    (18, 8, 9): "precision_side",
    (19, 10, 8): "recall_side",
}


def _load_search() -> dict[str, Any]:
    """Load only the frozen, complete, construction-ineligible v7 result."""
    _load_protocol()
    payload = json.loads(SEARCH_PATH.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "COMPLETE"
        or payload.get("selected_candidate_count") != 0
        or payload.get("construction_eligible") is not False
        or payload.get("heldout_access") != "NOT_ACCESSED_DEV3_ONLY"
        or payload.get("protocol", {}).get("sha256") != sha256_file(PROTOCOL_PATH)
    ):
        raise RuntimeError("v7 result does not authorize diagnostic rendering")
    return payload


def contact_signature(row: dict[str, Any]) -> tuple[int, int, int]:
    """Return the P signature used to select the two explanatory rows."""
    return (
        int(row["true_positive_count"]),
        int(row["phantom_contact_count"]),
        int(row["missed_contact_count"]),
    )


def select_diagnostic_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Select one deterministic K16 row for each observed contact signature."""
    group = next(
        row for row in payload["groups"] if int(row["max_hulls"]) == DIAGNOSTIC_K
    )
    rows = []
    for signature in EXPECTED_SIGNATURES:
        matches = [
            dict(row)
            for row in group["all_partitions"]
            if contact_signature(row) == signature
        ]
        if not matches:
            raise RuntimeError(f"v7 diagnostic signature missing: {signature}")
        matches.sort(
            key=lambda row: (
                int(row["actual_hulls"]),
                float(row["merge_added_volume_m3"]),
                row["partition"],
            )
        )
        selected = matches[0]
        selected["diagnostic_label"] = SIGNATURE_LABELS[signature]
        rows.append(selected)
    return rows


def _source_row(max_hulls: int, threshold_index: int) -> dict[str, Any]:
    threshold_m = float(v6.THRESHOLDS_M[threshold_index])
    matches = [
        row
        for row in v6.parent_source_rows()
        if int(row["max_hulls"]) == max_hulls
        and np.isclose(float(row["threshold_m"]), threshold_m, atol=0.0, rtol=0.0)
    ]
    if len(matches) != 1:
        raise RuntimeError("v7 diagnostic parent identity changed")
    return matches[0]


def _load_part(part: dict[str, Any]) -> trimesh.Trimesh:
    path = repo_path(part["path"])
    if part["sha256"] != sha256_file(path):
        raise RuntimeError("v7 diagnostic source part SHA changed")
    return trimesh.load(path, force="mesh", process=False, maintain_order=True)


def hybrid_parts(row: dict[str, Any]) -> list[trimesh.Trimesh]:
    """Reconstruct the complete frozen K context and replace only segment3."""
    if int(row["max_hulls"]) != DIAGNOSTIC_K:
        raise ValueError("v7 diagnostic is frozen to K16")
    assignment = [int(value) for value in row["context_threshold_indices_by_segment"]]
    parts = []
    for segment_index, threshold_index in enumerate(assignment):
        if segment_index == TARGET_SEGMENT_INDEX:
            continue
        candidate = v6._load_parent_candidate(
            _source_row(DIAGNOSTIC_K, threshold_index)
        )
        entries = [
            part
            for part in candidate["parts"]
            if int(part["segment_index"]) == segment_index
        ]
        if not entries:
            raise RuntimeError("v7 diagnostic context segment is empty")
        parts.extend(_load_part(part) for part in entries)
    source_meshes = _load_target_meshes(
        v6._load_parent_candidate(
            _source_row(DIAGNOSTIC_K, int(v6.THRESHOLDS_M.index(0.020)))
        )
    )
    partition = tuple(
        tuple(int(value) for value in block) for block in row["partition"]
    )
    parts.extend(_block_mesh(source_meshes, block) for block in partition)
    if len(parts) != int(row["actual_hulls"]):
        raise RuntimeError("v7 diagnostic hull count differs from frozen search")
    if any(not part.is_convex or not part.is_watertight for part in parts):
        raise RuntimeError("v7 diagnostic contains a non-convex or open part")
    return parts


def _clearance(scenes: list[Any], points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    distances = np.asarray(
        [core._scene_signed_distance(scene, points) for scene in scenes]
    )
    return distances.min(axis=0) - radii[None, :]


def render_row(
    row: dict[str, Any],
    *,
    queries: list[dict[str, Any]],
    oracle: trimesh.Trimesh,
    output_root: Path,
) -> dict[str, Any]:
    """Recompute one signature, render its deepest phantom pose, and bind outputs."""
    parts = hybrid_parts(row)
    scenes = [core._raycasting_scene(part) for part in parts]
    oracle_scene = core._raycasting_scene(oracle)
    timelines = []
    representatives = []
    aggregate = {
        "pose_count": 0,
        "true_positive_count": 0,
        "true_negative_count": 0,
        "phantom_contact_count": 0,
        "missed_contact_count": 0,
    }
    for query in queries:
        candidate_clearance = _clearance(scenes, query["points"], query["radii"])
        oracle_clearance = (
            core._scene_signed_distance(oracle_scene, query["points"])
            - query["radii"][None, :]
        )
        candidate_min = candidate_clearance.min(axis=1)
        oracle_min = oracle_clearance.min(axis=1)
        candidate_contact = candidate_min <= 0.0
        oracle_contact = oracle_min <= 0.0
        tp = candidate_contact & oracle_contact
        phantom = candidate_contact & ~oracle_contact
        missed = ~candidate_contact & oracle_contact
        tn = ~candidate_contact & ~oracle_contact
        counts = {
            "pose_count": len(candidate_contact),
            "true_positive_count": int(tp.sum()),
            "true_negative_count": int(tn.sum()),
            "phantom_contact_count": int(phantom.sum()),
            "missed_contact_count": int(missed.sum()),
        }
        for key, value in counts.items():
            aggregate[key] += value
        timelines.append(
            {
                "source_family": query["source_family"],
                "candidate_contact": candidate_contact.tolist(),
                "oracle_contact": oracle_contact.tolist(),
                "counts": counts,
            }
        )
        for pose_index in np.flatnonzero(phantom):
            classes = _point_classes(
                candidate_clearance[pose_index], oracle_clearance[pose_index]
            )
            representatives.append(
                {
                    "source_family": query["source_family"],
                    "pose_index": int(pose_index),
                    "candidate_min_clearance_m": float(candidate_min[pose_index]),
                    "oracle_min_clearance_m": float(oracle_min[pose_index]),
                    "phantom_point_count": int(classes["phantom"].sum()),
                    "points": query["points"][pose_index],
                    "candidate_clearance": candidate_clearance[pose_index],
                    "oracle_clearance": oracle_clearance[pose_index],
                }
            )
    expected = {
        key: int(row[key])
        for key in (
            "pose_count",
            "true_positive_count",
            "true_negative_count",
            "phantom_contact_count",
            "missed_contact_count",
        )
    }
    if aggregate != expected:
        raise RuntimeError("v7 diagnostic does not reproduce frozen P counts")
    representative = min(
        representatives,
        key=lambda value: (
            value["candidate_min_clearance_m"],
            -value["phantom_point_count"],
            value["source_family"],
            value["pose_index"],
        ),
    )
    classes = _point_classes(
        representative["candidate_clearance"], representative["oracle_clearance"]
    )
    label = str(row["diagnostic_label"])
    pose_label = (
        f"{label} · {representative['source_family']} "
        f"pose={representative['pose_index']} · "
        f"C min={1000 * representative['candidate_min_clearance_m']:.1f}mm · "
        f"M min={1000 * representative['oracle_min_clearance_m']:.1f}mm"
    )
    three_d = output_root / f"K16_{label}_task_pose_3d.png"
    two_d = output_root / f"K16_{label}_task_pose_2d.png"
    visual_union = trimesh.util.concatenate(parts)
    _render_3d(
        candidate_id=f"v7_{label}",
        max_hulls=DIAGNOSTIC_K,
        parts=parts,
        oracle=oracle,
        points=representative["points"],
        classes=classes,
        pose_label=pose_label,
        output=three_d,
    )
    _render_2d(
        candidate_id=f"v7_{label}",
        max_hulls=DIAGNOSTIC_K,
        union=visual_union,
        oracle=oracle,
        points=representative["points"],
        classes=classes,
        timelines=timelines,
        pose_label=pose_label,
        output=two_d,
    )
    return {
        "diagnostic_label": label,
        "selection_role": "FROZEN_V7_SIGNATURE_DIAGNOSTIC_NOT_A_CANDIDATE",
        "partition": row["partition"],
        "actual_hulls": int(row["actual_hulls"]),
        "aggregate_contact": aggregate,
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "representative_pose": {
            key: value
            for key, value in representative.items()
            if key not in ("points", "candidate_clearance", "oracle_clearance")
        },
        "representative_point_classes": {
            key: int(mask.sum()) for key, mask in classes.items()
        },
        "three_d": {
            "path": relative_to_repo(three_d),
            "sha256": sha256_file(three_d),
        },
        "two_d": {
            "path": relative_to_repo(two_d),
            "sha256": sha256_file(two_d),
        },
    }


def main() -> int:
    """Render the two frozen v7 P signatures and write a diagnostic manifest."""
    payload = _load_search()
    rows = select_diagnostic_rows(payload)
    queries, _ = v6._static_p_queries()
    oracle_row = _source_row(DIAGNOSTIC_K, int(v6.THRESHOLDS_M.index(0.020)))
    _, _, oracle = core._load_candidate_and_oracle(oracle_row)
    results = [
        render_row(row, queries=queries, oracle=oracle, output_root=OUTPUT_ROOT)
        for row in rows
    ]
    manifest = {
        "experiment_id": "E182",
        "stage": "S2_v7_segment3_partition_visual_diagnostic",
        "status": "COMPLETE",
        "selection_eligible": False,
        "construction_eligible": False,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "search": {
            "path": relative_to_repo(SEARCH_PATH),
            "sha256": sha256_file(SEARCH_PATH),
        },
        "protocol": {
            "path": relative_to_repo(PROTOCOL_PATH),
            "sha256": sha256_file(PROTOCOL_PATH),
        },
        "source_dependencies": [
            {
                "path": relative_to_repo(Path(__file__)),
                "sha256": sha256_file(Path(__file__)),
                "role": "V7_VISUAL_DIAGNOSTIC",
            },
            {
                "path": relative_to_repo(
                    Path(_render_3d.__code__.co_filename).resolve()
                ),
                "sha256": sha256_file(Path(_render_3d.__code__.co_filename).resolve()),
                "role": "E182_P_RENDER_HELPERS",
            },
        ],
        "signature_count": len(results),
        "signatures": results,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUTPUT_ROOT / "visual_manifest.json", manifest)
    print(f"E182_V7_VISUAL_DIAGNOSTIC=COMPLETE signatures={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
