#!/usr/bin/env python3
"""Audit object-specific CoACD static-P contact over all E178 Full27 cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO_ROOT = Path(__file__).resolve().parents[5]
E182_SCRIPT_ROOT = REPO_ROOT / "workspace/core4d/scripts/experiments/E182"
if str(E182_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(E182_SCRIPT_ROOT))

import build_prg_query_tape as tape  # noqa: E402
import build_task_aware_preseg_v9 as v9  # noqa: E402
import evaluate_task_queries as core  # noqa: E402
from e182_common import (  # noqa: E402
    atomic_json,
    atomic_tsv,
    relative_to_repo,
    repo_path,
    sha256_file,
)

EXPERIMENT_ID = "E183"
AUTHORITY_PATH = (
    REPO_ROOT / "workspace/core4d/results/E182/authority/full27/manifest.tsv"
)
E178_SOURCE_PATH = (
    REPO_ROOT
    / "workspace/core4d/results/E178/s6_downstream/manifests"
    / "semantic_bucket_full_manifest.tsv"
)
EXPECTED_E178_SOURCE_SHA256 = (
    "de9a3d165301318049da208aad52b1f5bc4d3ed671631f0097958734a0f022a8"
)
E181_CANDIDATE_ROOT = REPO_ROOT / "workspace/core4d/results/E181/s2_coacd"
E181_ORACLE_ROOT = REPO_ROOT / "workspace/core4d/results/E181/s1_oracle"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "workspace/core4d/results/E183/full27_static_p"
PROTOCOL_NAME = "protocol_manifest.json"
OBJECT_KEYS = ("bucket003", "bucket004", "bucket007")
EXPECTED_OBJECT_COUNTS = {"bucket003": 9, "bucket004": 4, "bucket007": 14}
EXPECTED_CASE_COUNT = 27
EXPECTED_TOTAL_POSES = 14_542
EXPECTED_CANDIDATE_COUNT = 60
EXPECTED_CASE_CANDIDATE_ROWS = 540
P_FLOOR = 0.70
DEFAULT_WORKERS = 4
V9_REGRESSION = {
    "taskpreseg_v9_both_t005_k16_v256": (22, 24, 5),
    "taskpreseg_v9_both_t005_k32_v256": (22, 21, 5),
    "taskpreseg_v9_both_t010_k16_v256": (19, 17, 8),
    "taskpreseg_v9_both_t010_k32_v256": (19, 14, 8),
    "taskpreseg_v9_both_t020_k16_v256": (15, 10, 12),
    "taskpreseg_v9_both_t020_k32_v256": (15, 7, 12),
}


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _artifact(path: Path) -> dict[str, str]:
    return {"path": relative_to_repo(path), "sha256": sha256_file(path)}


def load_full27_rows() -> list[dict[str, str]]:
    """Join frozen E182 Full27 authority to the exact E178 runtime artifacts."""
    if sha256_file(E178_SOURCE_PATH) != EXPECTED_E178_SOURCE_SHA256:
        raise RuntimeError("E178 source manifest SHA changed")
    authority = _read_tsv(AUTHORITY_PATH)
    source = {row["case_id"]: row for row in _read_tsv(E178_SOURCE_PATH)}
    if len(authority) != EXPECTED_CASE_COUNT or len(source) < EXPECTED_CASE_COUNT:
        raise RuntimeError("Full27 authority count changed")
    joined: list[dict[str, str]] = []
    for row in authority:
        case_id = row["case_id"]
        source_row = source.get(case_id)
        if source_row is None:
            raise RuntimeError(f"missing E178 source row: {case_id}")
        if (
            source_row["object_key"] != row["object_key"]
            or source_row["trajectory"] != row["trajectory"]
            or source_row["cem_samples"] != "1024"
            or source_row["cem_opt_steps"] != "32"
            or source_row["cem_seed"] != "0"
        ):
            raise RuntimeError(f"E178/authority identity mismatch: {case_id}")
        trajectory = repo_path(row["trajectory"])
        scene = repo_path(row["source_e178_scene_act"])
        result = repo_path(source_row["result_npz"])
        config = repo_path(source_row["config_act"])
        if sha256_file(trajectory) != row["trajectory_sha256"]:
            raise RuntimeError(f"trajectory SHA changed: {case_id}")
        if sha256_file(scene) != row["source_e178_scene_sha256"]:
            raise RuntimeError(f"scene SHA changed: {case_id}")
        if not result.is_file() or not config.is_file():
            raise RuntimeError(f"E178 result/config missing: {case_id}")
        joined.append(
            {
                **row,
                "e178_result_npz": relative_to_repo(result),
                "e178_result_sha256": sha256_file(result),
                "e178_config_act": relative_to_repo(config),
                "e178_config_sha256": sha256_file(config),
            }
        )
    if len({row["case_id"] for row in joined}) != EXPECTED_CASE_COUNT:
        raise RuntimeError("Full27 case IDs are not unique")
    counts = {
        key: sum(row["object_key"] == key for row in joined) for key in OBJECT_KEYS
    }
    if counts != EXPECTED_OBJECT_COUNTS:
        raise RuntimeError(f"Full27 object distribution changed: {counts}")
    return joined


def _candidate_entry(path: Path, source: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "BUILD_PASS":
        raise RuntimeError(f"candidate is not BUILD_PASS: {path}")
    if core._candidate_asset_sha256(payload) != payload["candidate_asset_sha256"]:
        raise RuntimeError(f"candidate asset digest changed: {path}")
    for part in payload["parts"]:
        part_path = repo_path(part["path"])
        if sha256_file(part_path) != part["sha256"]:
            raise RuntimeError(f"candidate part SHA changed: {part_path}")
    parameters = payload.get("parameters", {})
    max_hulls = int(parameters.get("max_convex_hull", payload.get("max_hulls", 0)))
    threshold_m = float(parameters.get("threshold_m", payload.get("threshold_m")))
    max_vertices = int(
        parameters.get("max_ch_vertex", payload.get("max_part_vertex_count", 0))
    )
    key = f"{source}__{payload['object_key']}__{payload['candidate_id']}"
    return {
        "candidate_key": key,
        "source": source,
        "object_key": payload["object_key"],
        "candidate_id": payload["candidate_id"],
        "candidate_asset_sha256": payload["candidate_asset_sha256"],
        "manifest": _artifact(path),
        "threshold_m": threshold_m,
        "max_hulls": max_hulls,
        "actual_hulls": int(payload["hull_count"]),
        "max_vertices": max_vertices,
    }


def candidate_inventory() -> list[dict[str, Any]]:
    """Freeze E181 all-object family plus the six latest bucket003 v9 candidates."""
    rows: list[dict[str, Any]] = []
    for object_key in OBJECT_KEYS:
        paths = sorted((E181_CANDIDATE_ROOT / object_key).glob("t*/manifest.json"))
        if len(paths) != 18:
            raise RuntimeError(f"{object_key}: expected 18 E181 candidates")
        rows.extend(_candidate_entry(path, "E181") for path in paths)
    v9_paths = sorted(v9.CANDIDATE_ROOT.glob("*/manifest.json"))
    if len(v9_paths) != 6:
        raise RuntimeError("expected six E182-v9 candidates")
    rows.extend(_candidate_entry(path, "E182_V9") for path in v9_paths)
    rows.sort(key=lambda row: row["candidate_key"])
    if len(rows) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError(f"candidate inventory count changed: {len(rows)}")
    if len({row["candidate_key"] for row in rows}) != len(rows):
        raise RuntimeError("candidate keys are not unique")
    counts = {key: sum(row["object_key"] == key for row in rows) for key in OBJECT_KEYS}
    if counts != {"bucket003": 24, "bucket004": 18, "bucket007": 18}:
        raise RuntimeError(f"candidate object distribution changed: {counts}")
    return rows


def oracle_inventory() -> dict[str, dict[str, str]]:
    """Resolve the three immutable cleaned original-mesh oracle assets."""
    result: dict[str, dict[str, str]] = {}
    for object_key in OBJECT_KEYS:
        manifest_path = E181_ORACLE_ROOT / object_key / "oracle_manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        cleaned = payload["cleaned_mesh"]
        cleaned_path = repo_path(cleaned["path"])
        if sha256_file(cleaned_path) != cleaned["sha256"]:
            raise RuntimeError(f"oracle cleaned mesh SHA changed: {object_key}")
        result[object_key] = {
            "manifest_path": relative_to_repo(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "path": relative_to_repo(cleaned_path),
            "sha256": cleaned["sha256"],
        }
    return result


def protocol_payload() -> dict[str, Any]:
    """Build the complete pre-score protocol payload from current inputs."""
    rows = load_full27_rows()
    candidates = candidate_inventory()
    script_path = Path(__file__).resolve()
    return {
        "experiment_id": EXPERIMENT_ID,
        "stage": "full27_static_P_protocol",
        "status": "FROZEN_BEFORE_SCORE",
        "e182_v9_history": "IMMUTABLE_COMPLETE_NEGATIVE_RESULT",
        "selection_contract": "FULL27_COVERAGE_EVALUATION_ONLY_NO_RETUNING",
        "gpu_access": "FORBIDDEN_CPU_ONLY",
        "workers": DEFAULT_WORKERS,
        "p_floor": P_FLOOR,
        "query_contract": "ALL_REFERENCE_PLUS_E178_FINAL_OBJECT_LOCAL_P_POINTS",
        "contact_contract": "MIN_SIGNED_DISTANCE_MINUS_RADIUS_LE_ZERO",
        "authority": _artifact(AUTHORITY_PATH),
        "e178_source": _artifact(E178_SOURCE_PATH),
        "code": _artifact(script_path),
        "case_count": len(rows),
        "case_ids": [row["case_id"] for row in rows],
        "object_case_counts": EXPECTED_OBJECT_COUNTS,
        "case_input_digest": _digest_json(
            [
                {
                    "case_id": row["case_id"],
                    "object_key": row["object_key"],
                    "trajectory_sha256": row["trajectory_sha256"],
                    "scene_sha256": row["source_e178_scene_sha256"],
                    "result_sha256": row["e178_result_sha256"],
                    "config_sha256": row["e178_config_sha256"],
                }
                for row in rows
            ]
        ),
        "oracles": oracle_inventory(),
        "candidate_count": len(candidates),
        "candidate_inventory_digest": _digest_json(candidates),
        "candidates": candidates,
        "expected_total_poses": EXPECTED_TOTAL_POSES,
        "expected_case_candidate_rows": EXPECTED_CASE_CANDIDATE_ROWS,
        "v9_regression": {key: list(value) for key, value in V9_REGRESSION.items()},
    }


def freeze_protocol(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    """Freeze authority, candidate, oracle, code, and metric contracts once."""
    path = output_root / PROTOCOL_NAME
    payload = protocol_payload()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(
                "existing E183 protocol differs from current frozen inputs"
            )
        return existing
    if any(output_root.iterdir()) if output_root.exists() else False:
        raise RuntimeError("E183 result root must be empty before protocol freeze")
    atomic_json(path, payload)
    return payload


def load_protocol(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    """Load the protocol and revalidate it against all immutable sources."""
    path = output_root / PROTOCOL_NAME
    if not path.is_file():
        raise RuntimeError("E183 protocol is not frozen")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = protocol_payload()
    if payload != expected:
        raise RuntimeError("E183 protocol/source validation failed")
    return payload


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def _load_oracle_scene(entry: Mapping[str, str]) -> Any:
    mesh = core._load_checked_mesh(repo_path(entry["path"]), entry["sha256"])
    return core._raycasting_scene(mesh)


def _query_family(
    context: tape.CaseContext, qpos: np.ndarray, oracle_scene: Any
) -> tuple[np.ndarray, np.ndarray]:
    factored = tape.extract_query_chunk(context, qpos)
    points = tape.materialize_query_points(factored)
    p_mask = np.isin(
        np.asarray(factored["point_geom_id"], dtype=np.int32),
        context.consumer_geom_ids["P_collision"],
    )
    selected = np.asarray(points[:, p_mask], dtype=np.float32)
    radii = np.asarray(factored["point_radius_m"][p_mask], dtype=np.float32)
    if selected.shape[1] == 0 or not np.isfinite(selected).all():
        raise RuntimeError(f"{context.row['case_id']}: invalid static P points")
    clearance = core._scene_signed_distance(oracle_scene, selected) - radii[None, :]
    contact = clearance.min(axis=1) <= 0.0
    return selected, contact


def _validate_query_case(
    output_root: Path, case_id: str, expected_object: str | None = None
) -> dict[str, Any]:
    root = output_root / "query_tape" / case_id
    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("status") != "COMPLETE"
        or payload.get("case_id") != case_id
        or (
            expected_object is not None and payload.get("object_key") != expected_object
        )
    ):
        raise RuntimeError(f"query manifest identity mismatch: {case_id}")
    total = 0
    point_count = None
    for family in ("reference", "e178_final"):
        entry = payload["families"][family]
        path = root / entry["relative_path"]
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"query NPZ SHA mismatch: {path}")
        with np.load(path, allow_pickle=False) as values:
            points = values["points"]
            contact = values["oracle_contact"]
            radii = values["radii"]
            if (
                points.ndim != 3
                or points.shape[-1] != 3
                or contact.shape != (len(points),)
                or radii.shape != (points.shape[1],)
                or not np.isfinite(points).all()
                or not np.isfinite(radii).all()
            ):
                raise RuntimeError(f"query NPZ schema mismatch: {path}")
            total += len(points)
            point_count = points.shape[1] if point_count is None else point_count
            if point_count != points.shape[1]:
                raise RuntimeError(
                    f"P point inventory changed between families: {case_id}"
                )
    if total != int(payload["pose_count"]):
        raise RuntimeError(f"query pose count mismatch: {case_id}")
    return payload


def build_query_case(row: dict[str, str], output_root_string: str) -> dict[str, Any]:
    """Build or validate reference/final static-P points for one frozen case."""
    output_root = Path(output_root_string)
    root = output_root / "query_tape" / row["case_id"]
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        return _validate_query_case(output_root, row["case_id"], row["object_key"])
    started = time.perf_counter()
    context = tape.load_case_context(row)
    protocol = json.loads((output_root / PROTOCOL_NAME).read_text(encoding="utf-8"))
    oracle_entry = protocol["oracles"][row["object_key"]]
    oracle_scene = _load_oracle_scene(oracle_entry)
    families: dict[str, dict[str, Any]] = {}
    source_values = {
        "reference": (
            tape.load_reference_qpos(row, context),
            repo_path(row["trajectory"]),
        ),
        "e178_final": (
            tape.load_final_qpos(row, context),
            repo_path(row["e178_result_npz"]),
        ),
    }
    point_count = None
    oracle_count = 0
    for family, (qpos, source_path) in source_values.items():
        points, contact = _query_family(context, qpos, oracle_scene)
        radii = np.asarray(
            context.point_radius_m[
                np.isin(context.point_geom_id, context.consumer_geom_ids["P_collision"])
            ],
            dtype=np.float32,
        )
        output = root / f"{family}.npz"
        _atomic_npz(output, points=points, radii=radii, oracle_contact=contact)
        point_count = points.shape[1] if point_count is None else point_count
        if points.shape[1] != point_count:
            raise RuntimeError(f"{row['case_id']}: family point inventory mismatch")
        oracle_count += int(contact.sum())
        families[family] = {
            "relative_path": output.name,
            "sha256": sha256_file(output),
            "size_bytes": output.stat().st_size,
            "pose_count": len(points),
            "oracle_contact_count": int(contact.sum()),
            "source": relative_to_repo(source_path),
            "source_sha256": sha256_file(source_path),
        }
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "full27_static_P_query_tape",
        "status": "COMPLETE",
        "case_id": row["case_id"],
        "object_key": row["object_key"],
        "frame": "object_body_local",
        "point_contract": "E182_P_COLLISION_INTERSECTION_FULL_POINTS",
        "point_count": point_count,
        "pose_count": sum(entry["pose_count"] for entry in families.values()),
        "oracle_contact_count": oracle_count,
        "families": families,
        "wall_seconds": time.perf_counter() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    atomic_json(manifest_path, payload)
    return _validate_query_case(output_root, row["case_id"], row["object_key"])


def build_queries(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    workers: int = DEFAULT_WORKERS,
    case_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the selected static-P query cases with a bounded CPU process pool."""
    protocol = load_protocol(output_root)
    rows = load_full27_rows()
    if case_ids:
        requested = set(case_ids)
        rows = [row for row in rows if row["case_id"] in requested]
        if {row["case_id"] for row in rows} != requested:
            raise RuntimeError("unknown E183 case ID")
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        payloads = list(
            pool.map(
                build_query_case,
                rows,
                [str(output_root)] * len(rows),
            )
        )
    total_poses = sum(int(payload["pose_count"]) for payload in payloads)
    if not case_ids and total_poses != int(protocol["expected_total_poses"]):
        raise RuntimeError(f"full27 static pose total changed: {total_poses}")
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "full27_static_P_query_aggregate",
        "status": "COMPLETE",
        "case_count": len(payloads),
        "pose_count": total_poses,
        "oracle_contact_count": sum(
            int(payload["oracle_contact_count"]) for payload in payloads
        ),
        "workers": workers,
        "wall_seconds": time.perf_counter() - started,
        "max_worker_rss_mb": max(float(payload["max_rss_mb"]) for payload in payloads),
        "cases": [
            {
                "case_id": payload["case_id"],
                "object_key": payload["object_key"],
                "pose_count": payload["pose_count"],
                "oracle_contact_count": payload["oracle_contact_count"],
                "manifest": _artifact(
                    output_root / "query_tape" / payload["case_id"] / "manifest.json"
                ),
            }
            for payload in payloads
        ],
    }
    if not case_ids:
        atomic_json(output_root / "query_aggregate.json", summary)
    return summary


def contact_metrics(candidate: np.ndarray, oracle: np.ndarray) -> dict[str, Any]:
    """Compute the unchanged E182 confusion and 0.70 precision/recall gate."""
    metrics = v9.v8.v6._contact_metrics(candidate, oracle)
    metrics["gate_status"] = (
        "PASS"
        if float(metrics["precision"]) >= P_FLOOR
        and float(metrics["recall"]) >= P_FLOOR
        else "FAIL"
    )
    return metrics


def _load_candidate_scenes(
    entry: Mapping[str, Any],
) -> tuple[dict[str, Any], list[Any]]:
    manifest_path = repo_path(entry["manifest"]["path"])
    if sha256_file(manifest_path) != entry["manifest"]["sha256"]:
        raise RuntimeError(f"candidate manifest SHA mismatch: {entry['candidate_key']}")
    candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
    if entry["source"] == "E182_V9":
        return candidate, v9._candidate_segment_scenes(candidate)
    _, exact_union, _ = core._load_candidate_and_oracle(entry)
    return candidate, [core._raycasting_scene(exact_union)]


def _load_case_query_arrays(
    output_root: Path, case_id: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = output_root / "query_tape" / case_id
    payload = _validate_query_case(output_root, case_id)
    points = []
    oracle = []
    radii = None
    for family in ("reference", "e178_final"):
        path = root / payload["families"][family]["relative_path"]
        with np.load(path, allow_pickle=False) as values:
            points.append(np.asarray(values["points"], dtype=np.float32))
            oracle.append(np.asarray(values["oracle_contact"], dtype=bool))
            current = np.asarray(values["radii"], dtype=np.float32)
            if radii is None:
                radii = current
            elif not np.array_equal(radii, current):
                raise RuntimeError(f"case family radii changed: {case_id}")
    assert radii is not None
    return np.concatenate(points), radii, np.concatenate(oracle)


def _candidate_contacts(
    scenes: Sequence[Any], points: np.ndarray, radii: np.ndarray
) -> np.ndarray:
    contacts = []
    for scene in scenes:
        clearance = core._scene_signed_distance(scene, points) - radii[None, :]
        contacts.append(clearance.min(axis=1) <= 0.0)
    return np.any(np.asarray(contacts), axis=0)


def _merge_confusions(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row["true_positive_count"]) for row in metrics)
    phantom = sum(int(row["phantom_contact_count"]) for row in metrics)
    missed = sum(int(row["missed_contact_count"]) for row in metrics)
    tn = sum(int(row["true_negative_count"]) for row in metrics)
    precision = tp / (tp + phantom) if tp + phantom else 0.0
    recall = tp / (tp + missed) if tp + missed else 0.0
    return {
        "pose_count": tp + phantom + missed + tn,
        "true_positive_count": tp,
        "true_negative_count": tn,
        "phantom_contact_count": phantom,
        "missed_contact_count": missed,
        "precision": precision,
        "recall": recall,
        "gate_status": "PASS" if precision >= P_FLOOR and recall >= P_FLOOR else "FAIL",
    }


def _result_path(output_root: Path, entry: Mapping[str, Any]) -> Path:
    return output_root / "scores" / f"{entry['candidate_key']}.json"


def _validate_candidate_result(
    output_root: Path, entry: Mapping[str, Any], expected_cases: int
) -> dict[str, Any]:
    path = _result_path(output_root, entry)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("candidate_key") != entry["candidate_key"]
        or payload.get("object_key") != entry["object_key"]
        or len(payload.get("case_metrics", [])) != expected_cases
    ):
        raise RuntimeError(
            f"candidate result identity mismatch: {entry['candidate_key']}"
        )
    if payload["candidate_manifest"] != entry["manifest"]:
        raise RuntimeError(
            f"candidate result manifest mismatch: {entry['candidate_key']}"
        )
    merged = _merge_confusions(payload["case_metrics"])
    if payload["pooled"] != merged:
        raise RuntimeError(
            f"candidate pooled confusion mismatch: {entry['candidate_key']}"
        )
    return payload


def score_candidate(
    entry: dict[str, Any], rows: list[dict[str, str]], output_root_string: str
) -> dict[str, Any]:
    """Score one candidate across every matching-object case and write atomically."""
    output_root = Path(output_root_string)
    object_rows = [row for row in rows if row["object_key"] == entry["object_key"]]
    result_path = _result_path(output_root, entry)
    if result_path.exists():
        return _validate_candidate_result(output_root, entry, len(object_rows))
    started = time.perf_counter()
    _, scenes = _load_candidate_scenes(entry)
    case_metrics = []
    for row in object_rows:
        points, radii, oracle = _load_case_query_arrays(output_root, row["case_id"])
        candidate = _candidate_contacts(scenes, points, radii)
        metrics = contact_metrics(candidate, oracle)
        case_metrics.append(
            {
                "case_id": row["case_id"],
                "object_key": row["object_key"],
                **metrics,
            }
        )
    pooled = _merge_confusions(case_metrics)
    macro_precision = float(np.mean([row["precision"] for row in case_metrics]))
    macro_recall = float(np.mean([row["recall"] for row in case_metrics]))
    case_pass_count = sum(row["gate_status"] == "PASS" for row in case_metrics)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "full27_static_P_candidate_score",
        "status": "COMPLETE",
        **{
            key: entry[key]
            for key in (
                "candidate_key",
                "source",
                "object_key",
                "candidate_id",
                "threshold_m",
                "max_hulls",
                "actual_hulls",
                "max_vertices",
            )
        },
        "candidate_manifest": entry["manifest"],
        "case_count": len(case_metrics),
        "case_pass_count": case_pass_count,
        "case_pass_fraction": case_pass_count / len(case_metrics),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_gate_status": (
            "PASS" if macro_precision >= P_FLOOR and macro_recall >= P_FLOOR else "FAIL"
        ),
        "pooled": pooled,
        "all_case_coverage_status": (
            "PASS"
            if pooled["gate_status"] == "PASS"
            and macro_precision >= P_FLOOR
            and macro_recall >= P_FLOOR
            and case_pass_count == len(case_metrics)
            else "FAIL"
        ),
        "case_metrics": case_metrics,
        "wall_seconds": time.perf_counter() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    atomic_json(result_path, payload)
    return _validate_candidate_result(output_root, entry, len(object_rows))


def score_all(
    output_root: Path = DEFAULT_OUTPUT_ROOT, workers: int = DEFAULT_WORKERS
) -> list[dict[str, Any]]:
    """Score all 60 frozen candidates using the bounded CPU worker pool."""
    protocol = load_protocol(output_root)
    query_summary = json.loads((output_root / "query_aggregate.json").read_text())
    if query_summary.get("pose_count") != EXPECTED_TOTAL_POSES:
        raise RuntimeError("full27 query aggregate is incomplete")
    rows = load_full27_rows()
    candidates = protocol["candidates"]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(
            pool.map(
                score_candidate,
                candidates,
                [rows] * len(candidates),
                [str(output_root)] * len(candidates),
            )
        )
    runtime = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "full27_static_P_score_runtime",
        "status": "COMPLETE",
        "workers": workers,
        "candidate_count": len(results),
        "wall_seconds": time.perf_counter() - started,
        "sum_candidate_wall_seconds": sum(
            float(row["wall_seconds"]) for row in results
        ),
        "max_worker_rss_mb": max(float(row["max_rss_mb"]) for row in results),
        "gpu_access_count": 0,
    }
    atomic_json(output_root / "score_runtime.json", runtime)
    return results


def _float(value: float) -> str:
    return f"{float(value):.9f}"


def aggregate(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    """Validate 540-row closure and emit per-case and per-candidate tables."""
    protocol = load_protocol(output_root)
    rows = load_full27_rows()
    results = [
        _validate_candidate_result(
            output_root,
            entry,
            EXPECTED_OBJECT_COUNTS[entry["object_key"]],
        )
        for entry in protocol["candidates"]
    ]
    flat = []
    for result in results:
        for case in result["case_metrics"]:
            flat.append(
                {
                    "candidate_key": result["candidate_key"],
                    "source": result["source"],
                    "object_key": result["object_key"],
                    "candidate_id": result["candidate_id"],
                    "threshold_m": _float(result["threshold_m"]),
                    "max_hulls": str(result["max_hulls"]),
                    "actual_hulls": str(result["actual_hulls"]),
                    "max_vertices": str(result["max_vertices"]),
                    "case_id": case["case_id"],
                    "pose_count": str(case["pose_count"]),
                    "true_positive_count": str(case["true_positive_count"]),
                    "phantom_contact_count": str(case["phantom_contact_count"]),
                    "missed_contact_count": str(case["missed_contact_count"]),
                    "true_negative_count": str(case["true_negative_count"]),
                    "precision": _float(case["precision"]),
                    "recall": _float(case["recall"]),
                    "gate_status": case["gate_status"],
                }
            )
    if len(flat) != EXPECTED_CASE_CANDIDATE_ROWS:
        raise RuntimeError(f"candidate-case closure failed: {len(flat)}")
    if len({(row["candidate_key"], row["case_id"]) for row in flat}) != len(flat):
        raise RuntimeError("duplicate candidate-case score row")
    case_fields = list(flat[0])
    atomic_tsv(output_root / "case_candidate_metrics.tsv", flat, case_fields)
    summary_rows = []
    for result in results:
        pooled = result["pooled"]
        summary_rows.append(
            {
                "candidate_key": result["candidate_key"],
                "source": result["source"],
                "object_key": result["object_key"],
                "candidate_id": result["candidate_id"],
                "threshold_m": _float(result["threshold_m"]),
                "max_hulls": str(result["max_hulls"]),
                "actual_hulls": str(result["actual_hulls"]),
                "max_vertices": str(result["max_vertices"]),
                "case_count": str(result["case_count"]),
                "case_pass_count": str(result["case_pass_count"]),
                "case_pass_fraction": _float(result["case_pass_fraction"]),
                "macro_precision": _float(result["macro_precision"]),
                "macro_recall": _float(result["macro_recall"]),
                "macro_gate_status": result["macro_gate_status"],
                "pooled_precision": _float(pooled["precision"]),
                "pooled_recall": _float(pooled["recall"]),
                "pooled_gate_status": pooled["gate_status"],
                "pooled_tp": str(pooled["true_positive_count"]),
                "pooled_phantom": str(pooled["phantom_contact_count"]),
                "pooled_missed": str(pooled["missed_contact_count"]),
                "all_case_coverage_status": result["all_case_coverage_status"],
                "wall_seconds": _float(result["wall_seconds"]),
            }
        )
    atomic_tsv(
        output_root / "candidate_summary.tsv", summary_rows, list(summary_rows[0])
    )
    dev_case = "bucket003_20231018_001_p1"
    regression = {}
    for candidate_id, expected in V9_REGRESSION.items():
        result = next(
            row
            for row in results
            if row["source"] == "E182_V9" and row["candidate_id"] == candidate_id
        )
        metric = next(
            row for row in result["case_metrics"] if row["case_id"] == dev_case
        )
        actual = (
            int(metric["true_positive_count"]),
            int(metric["phantom_contact_count"]),
            int(metric["missed_contact_count"]),
        )
        regression[candidate_id] = {
            "expected": list(expected),
            "actual": list(actual),
            "status": "PASS" if actual == expected else "FAIL",
        }
    regression_status = (
        "PASS"
        if all(row["status"] == "PASS" for row in regression.values())
        else "FAIL"
    )
    best_by_object = {}
    for object_key in OBJECT_KEYS:
        group = [row for row in results if row["object_key"] == object_key]
        ranked = sorted(
            group,
            key=lambda row: (
                -row["case_pass_count"],
                -min(row["macro_precision"], row["macro_recall"]),
                -min(row["pooled"]["precision"], row["pooled"]["recall"]),
                row["actual_hulls"],
                row["candidate_key"],
            ),
        )
        best_by_object[object_key] = ranked[0]["candidate_key"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "full27_static_P_aggregate",
        "status": "COMPLETE" if regression_status == "PASS" else "INVALID_REGRESSION",
        "selection_contract": "FULL27_COVERAGE_EVALUATION_ONLY_NO_RETUNING",
        "gpu_access_count": 0,
        "case_count": len(rows),
        "candidate_count": len(results),
        "case_candidate_row_count": len(flat),
        "pose_count": json.loads((output_root / "query_aggregate.json").read_text())[
            "pose_count"
        ],
        "all_case_coverage_pass_count": sum(
            row["all_case_coverage_status"] == "PASS" for row in results
        ),
        "pooled_gate_pass_count": sum(
            row["pooled"]["gate_status"] == "PASS" for row in results
        ),
        "macro_gate_pass_count": sum(
            row["macro_gate_status"] == "PASS" for row in results
        ),
        "best_by_object": best_by_object,
        "v9_regression_status": regression_status,
        "v9_regression": regression,
        "protocol": _artifact(output_root / PROTOCOL_NAME),
        "query_aggregate": _artifact(output_root / "query_aggregate.json"),
        "score_runtime": _artifact(output_root / "score_runtime.json"),
        "case_table": _artifact(output_root / "case_candidate_metrics.tsv"),
        "candidate_table": _artifact(output_root / "candidate_summary.tsv"),
    }
    atomic_json(output_root / "aggregate.json", payload)
    return payload


def _load_candidate_parts(entry: Mapping[str, Any]) -> list[trimesh.Trimesh]:
    manifest = json.loads(repo_path(entry["manifest"]["path"]).read_text())
    return [
        core._load_checked_mesh(repo_path(part["path"]), part["sha256"])
        for part in manifest["parts"]
    ]


def _set_equal_bounds(axis: Any, points: np.ndarray) -> None:
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = (lower + upper) / 2
    radius = max(float((upper - lower).max()) * 0.55, 0.05)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.set_proj_type("ortho")


def render_visuals(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    """Render 3D geometry and 2D contact timelines for each object's best row."""
    aggregate_payload = json.loads((output_root / "aggregate.json").read_text())
    protocol = load_protocol(output_root)
    visual_root = output_root / "visual"
    visual_root.mkdir(parents=True, exist_ok=True)
    entries = {row["candidate_key"]: row for row in protocol["candidates"]}
    observations = []
    artifacts = []
    for object_key, candidate_key in aggregate_payload["best_by_object"].items():
        entry = entries[candidate_key]
        result = json.loads(_result_path(output_root, entry).read_text())
        worst = min(
            result["case_metrics"],
            key=lambda row: (
                min(float(row["precision"]), float(row["recall"])),
                -(int(row["phantom_contact_count"]) + int(row["missed_contact_count"])),
                row["case_id"],
            ),
        )
        points, radii, oracle_contact = _load_case_query_arrays(
            output_root, worst["case_id"]
        )
        _, scenes = _load_candidate_scenes(entry)
        candidate_contact = _candidate_contacts(scenes, points, radii)
        error_indices = np.flatnonzero(candidate_contact != oracle_contact)
        pose_index = int(error_indices[0]) if len(error_indices) else 0
        pose_points = points[pose_index]
        parts = _load_candidate_parts(entry)
        oracle_entry = protocol["oracles"][object_key]
        oracle_mesh = core._load_checked_mesh(
            repo_path(oracle_entry["path"]), oracle_entry["sha256"]
        )
        oracle_points, _ = trimesh.sample.sample_surface(oracle_mesh, 6000, seed=183)
        all_bounds = np.concatenate(
            [oracle_points, pose_points] + [part.vertices for part in parts], axis=0
        )
        fig = plt.figure(figsize=(10, 4.8), dpi=160)
        for panel, (azimuth, elevation) in enumerate(((-55, 23), (35, 20)), start=1):
            axis = fig.add_subplot(1, 2, panel, projection="3d")
            axis.scatter(*oracle_points.T, s=0.15, color="#1f77b4", alpha=0.28)
            for index, part in enumerate(parts):
                color = plt.get_cmap("tab20")(index % 20)
                axis.add_collection3d(
                    Poly3DCollection(
                        np.asarray(part.triangles),
                        facecolors=[(*color[:3], 0.10)],
                        edgecolors=[(*color[:3], 0.25)],
                        linewidths=0.12,
                    )
                )
            point_contact = []
            for scene in scenes:
                point_contact.append(
                    core._scene_signed_distance(scene, pose_points) - radii <= 0.0
                )
            candidate_points = np.any(np.asarray(point_contact), axis=0)
            oracle_scene = _load_oracle_scene(oracle_entry)
            oracle_points_contact = (
                core._scene_signed_distance(oracle_scene, pose_points) - radii <= 0.0
            )
            colors = np.full(len(pose_points), "#999999", dtype=object)
            colors[candidate_points & oracle_points_contact] = "#d88c00"
            colors[candidate_points & ~oracle_points_contact] = "#d000d0"
            colors[~candidate_points & oracle_points_contact] = "#00a6c7"
            axis.scatter(*pose_points.T, s=5, c=colors, depthshade=False)
            axis.view_init(elev=elevation, azim=azimuth)
            _set_equal_bounds(axis, all_bounds)
            axis.set_title(f"{object_key} view {panel}")
            axis.set_axis_off()
        fig.suptitle(
            f"{candidate_key}\nworst={worst['case_id']} pose={pose_index} "
            f"P/R={worst['precision']:.3f}/{worst['recall']:.3f}"
        )
        fig.tight_layout()
        path3d = visual_root / f"{object_key}_representative_3d.png"
        fig.savefig(path3d)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(12, 3.2), dpi=160)
        x = np.arange(len(candidate_contact))
        axis.step(
            x, oracle_contact.astype(int), where="mid", label="oracle", linewidth=1.2
        )
        axis.step(
            x,
            candidate_contact.astype(int) + 0.05,
            where="mid",
            label="candidate (+0.05)",
            linewidth=1.0,
        )
        phantom = candidate_contact & ~oracle_contact
        missed = ~candidate_contact & oracle_contact
        axis.scatter(
            x[phantom], np.full(phantom.sum(), 1.18), s=10, c="#d000d0", label="phantom"
        )
        axis.scatter(
            x[missed], np.full(missed.sum(), 1.30), s=10, c="#00a6c7", label="missed"
        )
        axis.set_ylim(-0.15, 1.45)
        axis.set_xlabel("pose index: reference then E178-final")
        axis.set_ylabel("contact")
        axis.set_title(f"{object_key}: {candidate_key} · {worst['case_id']}")
        axis.legend(ncol=4, fontsize=8, loc="upper right")
        fig.tight_layout()
        path2d = visual_root / f"{object_key}_representative_2d.png"
        fig.savefig(path2d)
        plt.close(fig)
        artifacts.extend([_artifact(path3d), _artifact(path2d)])
        observations.append(
            {
                "object_key": object_key,
                "candidate_key": candidate_key,
                "case_id": worst["case_id"],
                "pose_index": pose_index,
                "precision": worst["precision"],
                "recall": worst["recall"],
                "phantom_count": worst["phantom_contact_count"],
                "missed_count": worst["missed_contact_count"],
                "observation": "3D shows oracle blue, hull union translucent, phantom magenta and missed cyan; 2D shows whether errors cluster in reference or E178-final poses.",
            }
        )
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "full27_static_P_visual",
        "status": "COMPLETE",
        "artifacts": artifacts,
        "observations": observations,
    }
    atomic_json(visual_root / "visual_manifest.json", payload)
    return payload


def validate_all(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    """Validate protocol, query, score, regression, visualization, and GPU closure."""
    protocol = load_protocol(output_root)
    query = json.loads((output_root / "query_aggregate.json").read_text())
    aggregate_payload = json.loads((output_root / "aggregate.json").read_text())
    visual = json.loads((output_root / "visual/visual_manifest.json").read_text())
    checks = {
        "protocol_candidates": protocol["candidate_count"] == EXPECTED_CANDIDATE_COUNT,
        "query_cases": query["case_count"] == EXPECTED_CASE_COUNT,
        "query_poses": query["pose_count"] == EXPECTED_TOTAL_POSES,
        "score_rows": aggregate_payload["case_candidate_row_count"]
        == EXPECTED_CASE_CANDIDATE_ROWS,
        "v9_regression": aggregate_payload["v9_regression_status"] == "PASS",
        "visual_objects": len(visual["observations"]) == len(OBJECT_KEYS),
        "gpu_access_zero": aggregate_payload["gpu_access_count"] == 0,
    }
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "full27_static_P_validation",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    atomic_json(output_root / "validation.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError(f"E183 validation failed: {checks}")
    return payload


def parse_args() -> argparse.Namespace:
    """Parse one explicit resume-safe E183 stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "protocol",
            "query",
            "score",
            "aggregate",
            "visual",
            "validate",
            "all",
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--case-id", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    """Run the requested E183 stage and print a compact machine-readable status."""
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.stage in {"protocol", "all"}:
        payload = freeze_protocol(args.output_root)
        print(
            f"E183_PROTOCOL={payload['status']} candidates={payload['candidate_count']}"
        )
    if args.stage in {"query", "all"}:
        payload = build_queries(args.output_root, args.workers, args.case_id)
        print(
            f"E183_QUERY={payload['status']} cases={payload['case_count']} poses={payload['pose_count']}"
        )
    if args.stage in {"score", "all"}:
        payloads = score_all(args.output_root, args.workers)
        print(f"E183_SCORE=COMPLETE candidates={len(payloads)}")
    if args.stage in {"aggregate", "all"}:
        payload = aggregate(args.output_root)
        print(
            f"E183_AGGREGATE={payload['status']} rows={payload['case_candidate_row_count']} "
            f"coverage_pass={payload['all_case_coverage_pass_count']}"
        )
    if args.stage in {"visual", "all"}:
        payload = render_visuals(args.output_root)
        print(f"E183_VISUAL={payload['status']} objects={len(payload['observations'])}")
    if args.stage in {"validate", "all"}:
        payload = validate_all(args.output_root)
        print(f"E183_VALIDATE={payload['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
