#!/usr/bin/env python3
"""Freeze the pre-score progressive query-density fixture for E182 S2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from audit_gate1 import EXPECTED_CASE_IDS, REPO_ROOT
from e182_common import atomic_json, relative_to_repo, sha256_file
from run_query_tape_replay import DEFAULT_RESULT_ROOT as DEFAULT_S1_ROOT

DEFAULT_GATE1_AUDIT = DEFAULT_S1_ROOT / "gate1_audit.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "workspace/core4d/results/E182/s2_task_query_eval"
COACD_ROOT = REPO_ROOT / "workspace/core4d/results/E181/s2_coacd"
SCREEN_MESH_POINTS_PER_GEOM = 64
FINALIST_MESH_POINTS_PER_GEOM = 256
EXPECTED_POINTS_PER_POSE = 1669
TIER_CONTRACT = {
    "screen": "ALL_54_CANDIDATES_MESH64_ALL_POSES",
    "finalist": "BEST_PER_OBJECT_K_MESH256_ALL_POSES",
    "production_canary": "SELECTED_CANDIDATE_FULL_POINTS",
}


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write an uncompressed deterministic NPZ atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def _nested_geom_indices(
    point_geom_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep primitive points and nested mesh64/mesh256 subsets in stable order."""
    geom_ids = list(dict.fromkeys(np.asarray(point_geom_id, dtype=np.int64).tolist()))
    screen: list[int] = []
    finalist: list[int] = []
    primitive: list[int] = []
    for geom_id in geom_ids:
        full = np.flatnonzero(point_geom_id == geom_id)
        if len(full) <= 3:
            screen.extend(full.tolist())
            finalist.extend(full.tolist())
            primitive.extend(full.tolist())
            continue
        finalist_local = (
            np.linspace(
                0,
                len(full) - 1,
                min(FINALIST_MESH_POINTS_PER_GEOM, len(full)),
            )
            .round()
            .astype(np.int64)
        )
        finalist_for_geom = full[np.unique(finalist_local)]
        screen_local = (
            np.linspace(
                0,
                len(finalist_for_geom) - 1,
                min(SCREEN_MESH_POINTS_PER_GEOM, len(finalist_for_geom)),
            )
            .round()
            .astype(np.int64)
        )
        screen_for_geom = finalist_for_geom[np.unique(screen_local)]
        screen.extend(screen_for_geom.tolist())
        finalist.extend(finalist_for_geom.tolist())
    return (
        np.arange(len(point_geom_id), dtype=np.int32),
        np.asarray(sorted(screen), dtype=np.int32),
        np.asarray(sorted(finalist), dtype=np.int32),
        np.asarray(sorted(primitive), dtype=np.int32),
    )


def _candidate_inventory() -> list[dict[str, Any]]:
    """Freeze all 54 BUILD_PASS candidates without reading E181 fidelity scores."""
    inventory = []
    for object_key in ("bucket003", "bucket004", "bucket007"):
        paths = sorted((COACD_ROOT / object_key).glob("t*/manifest.json"))
        if len(paths) != 18:
            raise RuntimeError(f"{object_key}: expected 18 candidate manifests")
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "BUILD_PASS":
                raise RuntimeError(f"candidate is not BUILD_PASS: {path}")
            max_hulls = int(payload["parameters"]["max_convex_hull"])
            if max_hulls not in (8, 16, 32):
                raise RuntimeError(f"candidate is outside K8/K16/K32: {path}")
            inventory.append(
                {
                    "object_key": object_key,
                    "candidate_id": payload["candidate_id"],
                    "candidate_asset_sha256": payload["candidate_asset_sha256"],
                    "manifest": {
                        "path": relative_to_repo(path),
                        "sha256": sha256_file(path),
                    },
                    "max_hulls": max_hulls,
                    "actual_hulls": int(payload["hull_count"]),
                    "threshold_m": float(payload["parameters"]["threshold_m"]),
                    "max_vertices": int(payload["parameters"]["max_ch_vertex"]),
                }
            )
    if len(inventory) != 54:
        raise RuntimeError(f"candidate inventory mismatch: {len(inventory)}")
    return inventory


def build_task_query_fixture(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    gate1_path: Path = DEFAULT_GATE1_AUDIT,
    s1_root: Path = DEFAULT_S1_ROOT,
) -> dict[str, Any]:
    """Freeze nested point indices and all-pose query counts before candidate scores."""
    gate1 = json.loads(gate1_path.read_text(encoding="utf-8"))
    if gate1.get("status") != "PASS":
        raise RuntimeError("Gate1 must PASS before S2 fixture freeze")
    if tuple(gate1.get("case_ids", [])) != EXPECTED_CASE_IDS:
        raise RuntimeError("Gate1 dev3 case identity changed")
    if gate1.get("heldout_access") != "NOT_ACCESSED_DEV3_ONLY":
        raise RuntimeError("Gate1 heldout isolation changed")

    tape_root = s1_root / "prg_query_tape"
    cases = []
    for case_id in EXPECTED_CASE_IDS:
        manifest_path = tape_root / case_id / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "COMPLETE":
            raise RuntimeError(f"{case_id}: PRG tape is not COMPLETE")
        if manifest.get("points_per_pose") != EXPECTED_POINTS_PER_POSE:
            raise RuntimeError(f"{case_id}: point inventory changed")
        reference_entry = next(
            entry
            for entry in manifest["chunks"]
            if entry["source_family"] == "reference"
        )
        reference_path = tape_root / reference_entry["relative_path"]
        with np.load(reference_path, allow_pickle=False) as values:
            point_geom_id = np.asarray(values["point_geom_id"], dtype=np.int32)
            consumer_names = np.asarray(values["consumer_names"]).tolist()
            consumer_mask = np.asarray(values["point_consumer_mask"], dtype=bool)
        full, screen, finalist, primitive = _nested_geom_indices(point_geom_id)
        if len(full) != EXPECTED_POINTS_PER_POSE:
            raise RuntimeError(f"{case_id}: full point count changed")
        indices_path = output_root / "case_indices" / f"{case_id}.npz"
        _atomic_npz(
            indices_path,
            full_point_indices=full,
            screen_point_indices=screen,
            finalist_point_indices=finalist,
            primitive_point_indices=primitive,
        )
        source_pose_counts = {
            family: sum(
                int(entry["pose_count"])
                for entry in manifest["chunks"]
                if entry["source_family"] == family
            )
            for family in ("reference", "e178_final", "cem_on_a")
        }
        static_pose_count = (
            source_pose_counts["reference"] + source_pose_counts["e178_final"]
        )
        cem_pose_count = source_pose_counts["cem_on_a"]
        screen_counts = {
            name: int(consumer_mask[screen, index].sum())
            for index, name in enumerate(consumer_names)
            if name != "P_collision"
        }
        finalist_counts = {
            name: int(consumer_mask[finalist, index].sum())
            for index, name in enumerate(consumer_names)
            if name != "P_collision"
        }
        if not all(screen_counts.values()) or not all(finalist_counts.values()):
            raise RuntimeError(f"{case_id}: one R/G consumer lost all points")
        cases.append(
            {
                "case_id": case_id,
                "object_key": manifest["object_key"],
                "prg_manifest": {
                    "path": relative_to_repo(manifest_path),
                    "sha256": sha256_file(manifest_path),
                },
                "indices": {
                    "relative_path": indices_path.relative_to(output_root).as_posix(),
                    "sha256": sha256_file(indices_path),
                },
                "cem_pose_policy": "ALL_CHUNKS_SAMPLES_HORIZON",
                "source_pose_counts": source_pose_counts,
                "full_point_count": len(full),
                "screen_point_count": len(screen),
                "finalist_point_count": len(finalist),
                "primitive_point_count": len(primitive),
                "screen_consumer_point_counts": screen_counts,
                "finalist_consumer_point_counts": finalist_counts,
                "screen_query_count": static_pose_count * len(full)
                + cem_pose_count * len(screen),
                "finalist_query_count": static_pose_count * len(full)
                + cem_pose_count * len(finalist),
                "full_query_count": (static_pose_count + cem_pose_count) * len(full),
            }
        )

    candidates = _candidate_inventory()
    candidate_identity_sha256 = hashlib.sha256(
        json.dumps(candidates, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload = {
        "experiment_id": "E182",
        "stage": "S2_task_query_fixture",
        "status": "FROZEN",
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "freeze_timing": "BEFORE_ANY_E182_CANDIDATE_SCORE",
        "gate1": {
            "status": gate1["status"],
            "path": relative_to_repo(gate1_path),
            "sha256": sha256_file(gate1_path),
            "selection_set_sha256": gate1["selection_set_sha256"],
        },
        "tier_contract": TIER_CONTRACT,
        "mesh_points_per_geom": {
            "screen": SCREEN_MESH_POINTS_PER_GEOM,
            "finalist": FINALIST_MESH_POINTS_PER_GEOM,
            "production_canary": "ALL",
        },
        "pose_policy": "ALL_CEM_CHUNKS_64_SAMPLES_48_HORIZON",
        "static_pose_policy": "ALL_REFERENCE_AND_E178_FINAL_FULL_POINTS",
        "candidate_count": len(candidates),
        "candidate_identity_sha256": candidate_identity_sha256,
        "candidates": candidates,
        "cases": cases,
    }
    atomic_json(output_root / "query_fixture_manifest.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    """Parse fixture builder arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gate1", type=Path, default=DEFAULT_GATE1_AUDIT)
    return parser.parse_args()


def main() -> int:
    """Freeze the S2 fixture before any real candidate score is computed."""
    args = parse_args()
    payload = build_task_query_fixture(args.output_root, gate1_path=args.gate1)
    print(
        f"E182_S2_QUERY_FIXTURE={payload['status']} "
        f"cases={len(payload['cases'])} candidates={payload['candidate_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
