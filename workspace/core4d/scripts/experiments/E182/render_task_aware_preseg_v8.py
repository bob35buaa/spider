#!/usr/bin/env python3
"""Render frozen v8 static-P trade-offs as post-result diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import build_task_aware_preseg_v8 as v8
import evaluate_task_queries as core
import numpy as np
import trimesh
from diagnose_p_contact import _point_classes, _render_2d, _render_3d
from e182_common import atomic_json, relative_to_repo, repo_path, sha256_file

OUTPUT_ROOT = v8.ATTEMPT_ROOT / "visual_diagnostic"
EXPECTED_DIAGNOSTIC_IDS = (
    "taskpreseg_v8_p01_t010_k32_v256",
    "taskpreseg_v8_p00_t005_k32_v256",
    "taskpreseg_v8_p01_t020_k32_v256",
)


def load_static_rows() -> list[dict[str, Any]]:
    """Load and strictly validate all 12 frozen v8 static-P results."""
    protocol = v8._load_protocol()
    aggregate_path = v8.STATIC_P_ROOT / "static_p_aggregate.json"
    aggregate = v8._validate_static_p_aggregate(
        json.loads(aggregate_path.read_text(encoding="utf-8")),
        protocol=protocol,
    )
    if (
        aggregate["pass_count"] != 0
        or aggregate["full_prg_eligible"] is not False
        or aggregate["zero_pass_action"] != "STOP_V8_NO_PLANE_OR_FLOOR_CHANGES"
    ):
        raise RuntimeError("v8 visual diagnostic requires the frozen 0/12 result")
    rows = []
    for frozen in protocol["candidate_family"]:
        result_path = v8.STATIC_P_ROOT / "results" / f"{frozen['candidate_id']}.json"
        manifest_path = v8.CANDIDATE_ROOT / frozen["candidate_id"] / "manifest.json"
        rows.append(
            v8._validate_static_p_result(
                json.loads(result_path.read_text(encoding="utf-8")),
                row=frozen,
                manifest_path=manifest_path,
            )
        )
    return rows


def _metrics(row: dict[str, Any]) -> dict[str, Any]:
    return row["p_pose_contact"]


def select_diagnostic_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select three deterministic explanatory rows after the frozen 0/12 result."""
    if len(rows) != 12 or len({row["candidate_id"] for row in rows}) != 12:
        raise RuntimeError("v8 visual input family is incomplete")
    best_balance = min(
        rows,
        key=lambda row: (
            float(row["p_score"]),
            int(row["actual_hulls"]),
            int(row["plane_index"]),
            float(row["threshold_m"]),
            int(row["max_hulls"]),
            row["candidate_id"],
        ),
    )
    high_recall = min(
        rows,
        key=lambda row: (
            -int(_metrics(row)["true_positive_count"]),
            int(_metrics(row)["phantom_contact_count"]),
            int(row["actual_hulls"]),
            row["candidate_id"],
        ),
    )
    tp19_rows = [row for row in rows if int(_metrics(row)["true_positive_count"]) >= 19]
    if not tp19_rows:
        raise RuntimeError("v8 visual family no longer contains a TP19 work point")
    tp19_precision_best = min(
        tp19_rows,
        key=lambda row: (
            int(_metrics(row)["phantom_contact_count"]),
            -int(_metrics(row)["true_positive_count"]),
            int(row["actual_hulls"]),
            row["candidate_id"],
        ),
    )
    selected = []
    for label, source in (
        ("best_balance", best_balance),
        ("high_recall", high_recall),
        ("tp19_precision_best", tp19_precision_best),
    ):
        row = dict(source)
        row["diagnostic_label"] = label
        selected.append(row)
    identifiers = tuple(row["candidate_id"] for row in selected)
    if identifiers != EXPECTED_DIAGNOSTIC_IDS or len(set(identifiers)) != 3:
        raise RuntimeError(f"v8 diagnostic representative set changed: {identifiers}")
    return selected


def _frozen_row(candidate_id: str) -> dict[str, Any]:
    protocol = v8._load_protocol()
    matches = [
        row
        for row in protocol["candidate_family"]
        if row["candidate_id"] == candidate_id
    ]
    if len(matches) != 1:
        raise RuntimeError("v8 visual candidate is outside the frozen family")
    return matches[0]


def _candidate_payload(row: dict[str, Any]) -> dict[str, Any]:
    protocol = v8._load_protocol()
    frozen = _frozen_row(row["candidate_id"])
    path = v8.CANDIDATE_ROOT / row["candidate_id"] / "manifest.json"
    return v8._validate_candidate_manifest(
        json.loads(path.read_text(encoding="utf-8")),
        row=frozen,
        parameters=v8._candidate_parameters(frozen, protocol),
        root=path.parent,
        base_manifest_path=(
            v8._new_base_root(int(frozen["plane_index"]), float(frozen["threshold_m"]))
            / "manifest.json"
        ),
    )


def candidate_parts(row: dict[str, Any]) -> list[trimesh.Trimesh]:
    """Load the exact ordered convex parts of one frozen v8 candidate."""
    manifest = _candidate_payload(row)
    parts = []
    for part in manifest["parts"]:
        path = repo_path(part["path"])
        if part["sha256"] != sha256_file(path):
            raise RuntimeError("v8 visual candidate part SHA changed")
        parts.append(
            trimesh.load(path, force="mesh", process=False, maintain_order=True)
        )
    if len(parts) != int(row["actual_hulls"]):
        raise RuntimeError("v8 visual hull count differs from frozen static-P")
    if any(not part.is_convex or not part.is_watertight for part in parts):
        raise RuntimeError("v8 visual candidate contains invalid convex parts")
    return parts


def _candidate_clearance(
    scenes: list[Any], points: np.ndarray, radii: np.ndarray
) -> np.ndarray:
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
    """Recompute one frozen signature and render its deepest phantom pose."""
    manifest = _candidate_payload(row)
    parts = candidate_parts(row)
    scenes = v8._candidate_segment_scenes(manifest)
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
        candidate_clearance = _candidate_clearance(
            scenes, query["points"], query["radii"]
        )
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
    expected_metrics = _metrics(row)
    expected = {
        key: int(expected_metrics[key])
        for key in (
            "pose_count",
            "true_positive_count",
            "true_negative_count",
            "phantom_contact_count",
            "missed_contact_count",
        )
    }
    if aggregate != expected or not representatives:
        raise RuntimeError("v8 visual does not reproduce frozen P counts")
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
    three_d = output_root / f"{label}_task_pose_3d.png"
    two_d = output_root / f"{label}_task_pose_2d.png"
    _render_3d(
        candidate_id=row["candidate_id"],
        max_hulls=int(row["max_hulls"]),
        parts=parts,
        oracle=oracle,
        points=representative["points"],
        classes=classes,
        pose_label=pose_label,
        output=three_d,
    )
    _render_2d(
        candidate_id=row["candidate_id"],
        max_hulls=int(row["max_hulls"]),
        union=trimesh.util.concatenate(parts),
        oracle=oracle,
        points=representative["points"],
        classes=classes,
        timelines=timelines,
        pose_label=pose_label,
        output=two_d,
    )
    return {
        "diagnostic_label": label,
        "selection_role": "POST_RESULT_V8_TRADEOFF_DIAGNOSTIC_NOT_SELECTION",
        "candidate_id": row["candidate_id"],
        "plane_index": int(row["plane_index"]),
        "threshold_m": float(row["threshold_m"]),
        "max_hulls": int(row["max_hulls"]),
        "actual_hulls": int(row["actual_hulls"]),
        "aggregate_contact": aggregate,
        "precision": float(expected_metrics["precision"]),
        "recall": float(expected_metrics["recall"]),
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
    """Render three frozen v8 trade-off representatives and bind all outputs."""
    rows = select_diagnostic_rows(load_static_rows())
    queries, _, oracle = v8._oracle_and_queries()
    results = [
        render_row(row, queries=queries, oracle=oracle, output_root=OUTPUT_ROOT)
        for row in rows
    ]
    manifest = {
        "experiment_id": "E182",
        "stage": "S2_v8_task_aware_preseg_visual_diagnostic",
        "status": "COMPLETE",
        "selection_eligible": False,
        "construction_eligible": False,
        "full_prg_eligible": False,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "zero_pass_action": "STOP_V8_NO_PLANE_OR_FLOOR_CHANGES",
        "protocol": v8._artifact_reference(v8.PROTOCOL_PATH),
        "build_summary": v8._artifact_reference(v8.ATTEMPT_ROOT / "build_summary.json"),
        "static_p_aggregate": v8._artifact_reference(
            v8.STATIC_P_ROOT / "static_p_aggregate.json"
        ),
        "source_dependencies": [
            {
                "path": relative_to_repo(Path(__file__)),
                "sha256": sha256_file(Path(__file__)),
                "role": "V8_POST_RESULT_VISUAL_DIAGNOSTIC",
            },
            {
                "path": relative_to_repo(
                    Path(_render_3d.__code__.co_filename).resolve()
                ),
                "sha256": sha256_file(Path(_render_3d.__code__.co_filename).resolve()),
                "role": "E182_P_RENDER_HELPERS",
            },
        ],
        "diagnostic_count": len(results),
        "diagnostics": results,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUTPUT_ROOT / "visual_manifest.json", manifest)
    print(f"E182_V8_VISUAL_DIAGNOSTIC=COMPLETE diagnostics={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
