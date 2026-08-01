#!/usr/bin/env python3
"""Build attempt2-v3 with one fresh CoACD subprocess per exact segment."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import trimesh
from build_segmented_coacd_attempt2 import (
    DEFAULT_FIXTURE,
    SEGMENT_COUNT,
    _candidate_asset_sha256,
    _candidate_identity_sha256,
)
from build_segmented_coacd_attempt2_v2 import (
    CANDIDATE_ROOT as V2_CANDIDATE_ROOT,
)
from build_segmented_coacd_attempt2_v2 import (
    COACD_MAX_VERTICES,
    V1_SEGMENT_MANIFEST_PATH,
    _coacd_parts,
    _export_part_v2,
    _load_v1_segments,
)
from build_segmented_coacd_attempt2_v2 import (
    PROTOCOL_PATH as V2_PROTOCOL_PATH,
)
from build_task_query_fixture import DEFAULT_OUTPUT_ROOT
from e182_common import (
    REPO_ROOT,
    atomic_json,
    relative_to_repo,
    repo_path,
    sha256_file,
)

ATTEMPT_ROOT = DEFAULT_OUTPUT_ROOT / "attempt2_segmented_x2y4_v3_isolated"
PROTOCOL_PATH = ATTEMPT_ROOT / "attempt2_v3_protocol_manifest.json"
CHILD_ROOT = ATTEMPT_ROOT / "segment_children"
CANDIDATE_ROOT = ATTEMPT_ROOT / "candidates"
FIXTURE_PATH = ATTEMPT_ROOT / "query_fixture_manifest.json"
SCREEN_ROOT = ATTEMPT_ROOT / "screen"
V2_K8_MANIFEST_PATH = V2_CANDIDATE_ROOT / "segx2y4_hull_k08_v160/manifest.json"
V2_FAILED_CANDIDATE_ROOT = V2_CANDIDATE_ROOT / "segx2y4_coacd_t005_k16_v032"

OBJECT_KEY = "bucket003"
CASE_ID = "bucket003_20231018_001_p1"
THRESHOLDS_M = (0.005, 0.010, 0.020)
MAX_HULLS = (16, 32)
SEGMENT_PARALLEL_WORKERS = 4

for variable in ("OMP_NUM_THREADS", "TBB_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")


def candidate_family() -> list[dict[str, Any]]:
    """Return one reused K8 plus six isolated segmented-CoACD rows."""
    rows = [
        {
            "candidate_id": "segx2y4_hull_k08_v160",
            "method": "REUSE_V2_EXACT_SEGMENT_CONVEX_HULL",
            "threshold_m": 0.0,
            "max_hulls": 8,
            "per_segment_max_hulls": 1,
            "max_vertices": 160,
        }
    ]
    rows.extend(
        {
            "candidate_id": (
                f"segx2y4_iso_coacd_t{round(threshold_m * 1000):03d}_"
                f"k{max_hulls:02d}_v{COACD_MAX_VERTICES:03d}"
            ),
            "method": "ISOLATED_SEGMENTED_COACD",
            "threshold_m": threshold_m,
            "max_hulls": max_hulls,
            "per_segment_max_hulls": max_hulls // SEGMENT_COUNT,
            "max_vertices": COACD_MAX_VERTICES,
        }
        for threshold_m in THRESHOLDS_M
        for max_hulls in MAX_HULLS
    )
    return rows


def _v2_failure_evidence() -> dict[str, Any]:
    """Bind the v2 K8 success and same-process K16 prefix failure."""
    k8 = json.loads(V2_K8_MANIFEST_PATH.read_text(encoding="utf-8"))
    if k8.get("status") != "BUILD_PASS" or k8.get("hull_count") != 8:
        raise RuntimeError("attempt2-v2 K8 manifest changed")
    failed_manifest = V2_FAILED_CANDIDATE_ROOT / "manifest.json"
    parts = sorted((V2_FAILED_CANDIDATE_ROOT / "parts").glob("part_*.obj"))
    if failed_manifest.exists() or len(parts) != 6:
        raise RuntimeError("attempt2-v2 K16 partial prefix changed")
    return {
        "status": "K8_PASS_K16_SAME_PROCESS_BUILD_FAILED",
        "K8_manifest": {
            "path": relative_to_repo(V2_K8_MANIFEST_PATH),
            "sha256": sha256_file(V2_K8_MANIFEST_PATH),
            "candidate_asset_sha256": k8["candidate_asset_sha256"],
        },
        "failed_candidate_id": "segx2y4_coacd_t005_k16_v032",
        "signature": "segment 3 hull cap violated",
        "successful_prefix_part_count": len(parts),
        "successful_prefix_parts": [
            {"path": relative_to_repo(path), "sha256": sha256_file(path)}
            for path in parts
        ],
        "root_cause": "COACD_NATIVE_STATE_NOT_ISOLATED_ACROSS_SEGMENT_CALLS",
        "changed_execution_contract": "ONE_FRESH_PROCESS_PER_SEGMENT",
    }


def freeze_protocol() -> dict[str, Any]:
    """Freeze the isolated execution contract before any v3 child or score."""
    if list(CHILD_ROOT.glob("*/*/manifest.json")) or list(
        CANDIDATE_ROOT.glob("*/manifest.json")
    ):
        raise RuntimeError("cannot freeze v3 after child/candidate build")
    if list((SCREEN_ROOT / "results").glob("*.json")):
        raise RuntimeError("cannot freeze v3 after score")
    segments = _load_v1_segments()
    rows = candidate_family()
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v3_protocol",
        "status": "FROZEN_BEFORE_V3_BUILD_OR_SCORE",
        "selection_eligible": False,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "reason": "V2_MULTI_CALL_COACD_NATIVE_STATE_CHANGED_CAP_RESULT",
        "object_key": OBJECT_KEY,
        "case_id": CASE_ID,
        "candidate_family": rows,
        "candidate_count": len(rows),
        "topology": {
            "name": "REUSE_EXACT_MANIFOLD_X2_Y4_SOLID_PARTITION",
            "segment_count": SEGMENT_COUNT,
            "segment_manifest": {
                "path": relative_to_repo(V1_SEGMENT_MANIFEST_PATH),
                "sha256": sha256_file(V1_SEGMENT_MANIFEST_PATH),
            },
            "ordered_parts": "segment_index_then_local_part_index",
        },
        "execution_contract": {
            "coacd_calls_per_child": 1,
            "fresh_process_per_segment": True,
            "segment_parallel_workers": SEGMENT_PARALLEL_WORKERS,
            "candidates_parallel": False,
            "thread_env": {
                name: os.environ[name]
                for name in (
                    "OMP_NUM_THREADS",
                    "TBB_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            },
            "native_log_per_segment": True,
        },
        "selection_contract": {
            "launch_floor": "UNCHANGED_E182_P_R_G_V1",
            "score": "UNCHANGED_P_R_G_WORST_NORMALIZED_V1",
            "per_K": "launch_floor_then_score_then_actual_hulls_runtime_threshold_id",
            "full_role": "FORBIDDEN_UNTIL_PRODUCTION_FREEZE",
        },
        "v2_protocol": {
            "path": relative_to_repo(V2_PROTOCOL_PATH),
            "sha256": sha256_file(V2_PROTOCOL_PATH),
        },
        "v2_build_evidence": _v2_failure_evidence(),
        "oracle": segments["oracle"],
        "original_fixture": {
            "path": relative_to_repo(DEFAULT_FIXTURE),
            "sha256": sha256_file(DEFAULT_FIXTURE),
        },
        "builder_source": {
            "path": relative_to_repo(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
    }
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("existing attempt2-v3 protocol differs")
        return existing
    atomic_json(PROTOCOL_PATH, payload)
    return payload


def _load_protocol() -> dict[str, Any]:
    """Require exact v3 source, parent, segment, and fixture hashes."""
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_V3_BUILD_OR_SCORE":
        raise RuntimeError("attempt2-v3 protocol is not frozen")
    checks = (
        (protocol["builder_source"], Path(__file__)),
        (protocol["v2_protocol"], V2_PROTOCOL_PATH),
        (protocol["topology"]["segment_manifest"], V1_SEGMENT_MANIFEST_PATH),
        (protocol["original_fixture"], DEFAULT_FIXTURE),
    )
    for entry, path in checks:
        if entry["sha256"] != sha256_file(path):
            raise RuntimeError(f"attempt2-v3 authority SHA changed: {path}")
    return protocol


def _frozen_candidate(candidate_id: str) -> dict[str, Any]:
    """Resolve exactly one non-K8 frozen candidate row."""
    rows = [
        row
        for row in _load_protocol()["candidate_family"]
        if row["candidate_id"] == candidate_id
    ]
    if len(rows) != 1 or rows[0]["method"] != "ISOLATED_SEGMENTED_COACD":
        raise RuntimeError(f"candidate is not a frozen v3 CoACD row: {candidate_id}")
    return rows[0]


def build_segment_child(candidate_id: str, segment_index: int) -> dict[str, Any]:
    """Run exactly one CoACD call in this fresh child process."""
    row = _frozen_candidate(candidate_id)
    segments = _load_v1_segments()["segments"]
    if not 0 <= segment_index < len(segments):
        raise RuntimeError("segment index outside frozen partition")
    segment = segments[segment_index]
    root = CHILD_ROOT / candidate_id / f"segment_{segment_index:03d}"
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "BUILD_PASS"
            and existing.get("candidate_id") == candidate_id
            and existing.get("segment_sha256") == segment["sha256"]
            and all(
                sha256_file(repo_path(part["path"])) == part["sha256"]
                for part in existing["parts"]
            )
        ):
            return existing
        raise RuntimeError("non-resumable v3 child manifest")
    mesh = trimesh.load(
        repo_path(segment["path"]),
        force="mesh",
        process=False,
        maintain_order=True,
    )
    started = time.perf_counter()
    result = _coacd_parts(
        mesh,
        threshold_m=float(row["threshold_m"]),
        max_hulls=int(row["per_segment_max_hulls"]),
    )
    if not 1 <= len(result) <= int(row["per_segment_max_hulls"]):
        raise RuntimeError(
            f"{candidate_id}: segment {segment_index} hull cap violated "
            f"actual={len(result)} cap={row['per_segment_max_hulls']}"
        )
    parts = [
        _export_part_v2(
            vertices,
            faces,
            root / "parts" / f"part_{local_index:03d}.obj",
            part_index=local_index,
            segment_index=segment_index,
            local_part_index=local_index,
            max_vertices=int(row["max_vertices"]),
        )
        for local_index, (vertices, faces) in enumerate(result)
    ]
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v3_segment_child",
        "status": "BUILD_PASS",
        "selection_eligible": False,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "candidate_id": candidate_id,
        "segment_index": segment_index,
        "segment_path": segment["path"],
        "segment_sha256": segment["sha256"],
        "threshold_m": row["threshold_m"],
        "max_hulls": row["per_segment_max_hulls"],
        "actual_hulls": len(parts),
        "wall_seconds": time.perf_counter() - started,
        "parts": parts,
    }
    atomic_json(manifest_path, payload)
    print(
        "E182_V3_SEGMENT_CHILD=PASS "
        f"candidate={candidate_id} segment={segment_index} hulls={len(parts)} "
        f"wall={payload['wall_seconds']:.3f}"
    )
    return payload


def _run_segment_subprocess(candidate_id: str, segment_index: int) -> dict[str, Any]:
    """Launch one isolated segment child and attach its full native log."""
    root = CHILD_ROOT / candidate_id / f"segment_{segment_index:03d}"
    manifest_path = root / "manifest.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "build-segment",
        "--candidate-id",
        candidate_id,
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
    native_output = completed.stdout + completed.stderr
    log_path = root / "native.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(native_output, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{candidate_id}/segment{segment_index}: child RC={completed.returncode}; "
            f"log={log_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["native_log"] = {
        "path": relative_to_repo(log_path),
        "sha256": sha256_file(log_path),
    }
    atomic_json(manifest_path, manifest)
    print(completed.stdout.strip(), flush=True)
    return manifest


def build_candidate(row: dict[str, Any]) -> dict[str, Any]:
    """Build eight isolated segment children and assemble ordered candidate identity."""
    protocol = _load_protocol()
    if row not in protocol["candidate_family"] or row["max_hulls"] == 8:
        raise RuntimeError("candidate row outside frozen v3 CoACD family")
    manifest_path = CANDIDATE_ROOT / row["candidate_id"] / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "BUILD_PASS" and all(
            sha256_file(repo_path(part["path"])) == part["sha256"]
            for part in existing["parts"]
        ):
            return existing
        raise RuntimeError("non-resumable v3 candidate manifest")
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=SEGMENT_PARALLEL_WORKERS
    ) as executor:
        futures = [
            executor.submit(_run_segment_subprocess, row["candidate_id"], index)
            for index in range(SEGMENT_COUNT)
        ]
        children = [future.result() for future in futures]
    children.sort(key=lambda value: value["segment_index"])
    parts = []
    per_segment = []
    for child in children:
        first_part_index = len(parts)
        for local_index, child_part in enumerate(child["parts"]):
            part = dict(child_part)
            part["part_index"] = len(parts)
            part["segment_index"] = child["segment_index"]
            part["local_part_index"] = local_index
            parts.append(part)
        per_segment.append(
            {
                "segment_index": child["segment_index"],
                "segment_sha256": child["segment_sha256"],
                "first_part_index": first_part_index,
                "hull_count": child["actual_hulls"],
                "child_manifest": {
                    "path": relative_to_repo(
                        CHILD_ROOT
                        / row["candidate_id"]
                        / f"segment_{child['segment_index']:03d}/manifest.json"
                    ),
                    "sha256": sha256_file(
                        CHILD_ROOT
                        / row["candidate_id"]
                        / f"segment_{child['segment_index']:03d}/manifest.json"
                    ),
                },
            }
        )
    if len(parts) > int(row["max_hulls"]):
        raise RuntimeError("v3 assembled candidate exceeds total hull cap")
    parameters = {
        "method": row["method"],
        "threshold_m": row["threshold_m"],
        "max_convex_hull": row["max_hulls"],
        "per_segment_max_convex_hull": row["per_segment_max_hulls"],
        "max_ch_vertex": row["max_vertices"],
        "execution_contract": protocol["execution_contract"],
        "topology": protocol["topology"],
    }
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v3_build",
        "status": "BUILD_PASS",
        "selection_eligible": False,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "object_key": OBJECT_KEY,
        "candidate_id": row["candidate_id"],
        "protocol": {
            "path": relative_to_repo(PROTOCOL_PATH),
            "sha256": sha256_file(PROTOCOL_PATH),
        },
        "segment_manifest": protocol["topology"]["segment_manifest"],
        "oracle_cleaned_mesh": {
            "path": protocol["oracle"]["path"],
            "sha256": protocol["oracle"]["sha256"],
        },
        "parameters": parameters,
        "method": row["method"],
        "max_hulls": row["max_hulls"],
        "hull_count": len(parts),
        "max_part_vertex_count": max(part["vertex_count"] for part in parts),
        "total_vertex_count": sum(part["vertex_count"] for part in parts),
        "total_face_count": sum(part["face_count"] for part in parts),
        "build_wall_seconds": time.perf_counter() - started,
        "per_segment": per_segment,
        "parts": parts,
    }
    payload["candidate_asset_sha256"] = _candidate_asset_sha256(payload)
    atomic_json(manifest_path, payload)
    print(
        "E182_SEGMENTED_COACD_V3=PASS "
        f"candidate={row['candidate_id']} hulls={len(parts)} "
        f"wall={payload['build_wall_seconds']:.3f}"
    )
    return payload


def build_candidates() -> dict[str, Any]:
    """Build the six isolated CoACD rows and bind the reused K8 candidate."""
    protocol = _load_protocol()
    rows = [row for row in protocol["candidate_family"] if row["max_hulls"] != 8]
    manifests = [build_candidate(row) for row in rows]
    k8 = json.loads(V2_K8_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload = {
        "experiment_id": "E182",
        "stage": "S2_decomposition_attempt2_v3_build",
        "status": "BUILD_PASS",
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "protocol": {
            "path": relative_to_repo(PROTOCOL_PATH),
            "sha256": sha256_file(PROTOCOL_PATH),
        },
        "candidate_count": 1 + len(manifests),
        "reused_K8": {
            "path": relative_to_repo(V2_K8_MANIFEST_PATH),
            "sha256": sha256_file(V2_K8_MANIFEST_PATH),
            "candidate_asset_sha256": k8["candidate_asset_sha256"],
        },
        "built_candidate_ids": [manifest["candidate_id"] for manifest in manifests],
        "actual_hulls": [8, *[manifest["hull_count"] for manifest in manifests]],
        "total_wall_seconds": float(
            sum(manifest["build_wall_seconds"] for manifest in manifests)
        ),
    }
    if payload["candidate_count"] != 7:
        raise RuntimeError("attempt2-v3 candidate family incomplete")
    atomic_json(ATTEMPT_ROOT / "build_summary.json", payload)
    return payload


def build_fixture() -> dict[str, Any]:
    """Freeze a seven-row bucket003 fixture after isolated build closure."""
    protocol = _load_protocol()
    summary = json.loads(
        (ATTEMPT_ROOT / "build_summary.json").read_text(encoding="utf-8")
    )
    if summary.get("status") != "BUILD_PASS" or summary.get("candidate_count") != 7:
        raise RuntimeError("attempt2-v3 build summary incomplete")
    original = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    cases = [case for case in original["cases"] if case["case_id"] == CASE_ID]
    if len(cases) != 1:
        raise RuntimeError("bucket003 dev case changed")
    fixture_rows = []
    for row in protocol["candidate_family"]:
        manifest_path = (
            V2_K8_MANIFEST_PATH
            if row["max_hulls"] == 8
            else CANDIDATE_ROOT / row["candidate_id"] / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "BUILD_PASS":
            raise RuntimeError("attempt2-v3 fixture candidate is not BUILD_PASS")
        fixture_rows.append(
            {
                "object_key": OBJECT_KEY,
                "candidate_id": manifest["candidate_id"],
                "candidate_asset_sha256": manifest["candidate_asset_sha256"],
                "manifest": {
                    "path": relative_to_repo(manifest_path),
                    "sha256": sha256_file(manifest_path),
                },
                "max_hulls": int(row["max_hulls"]),
                "actual_hulls": int(manifest["hull_count"]),
                "threshold_m": float(row["threshold_m"]),
                "max_vertices": int(row["max_vertices"]),
            }
        )
    payload = {
        **original,
        "stage": "S2_task_query_fixture_attempt2_segmented_x2y4_v3_isolated",
        "status": "FROZEN",
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "freeze_timing": "AFTER_V3_BUILD_BEFORE_V3_SCORE",
        "attempt2_v3_protocol": {
            "path": relative_to_repo(PROTOCOL_PATH),
            "sha256": sha256_file(PROTOCOL_PATH),
        },
        "original_fixture": {
            "path": relative_to_repo(DEFAULT_FIXTURE),
            "sha256": sha256_file(DEFAULT_FIXTURE),
        },
        "cases": cases,
        "candidates": fixture_rows,
        "candidate_count": len(fixture_rows),
        "candidate_identity_sha256": _candidate_identity_sha256(fixture_rows),
    }
    if FIXTURE_PATH.exists():
        existing = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("existing attempt2-v3 fixture differs")
        return existing
    atomic_json(FIXTURE_PATH, payload)
    return payload


def parse_args() -> argparse.Namespace:
    """Parse explicit v3 parent/child stages."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "freeze-protocol",
            "build-segment",
            "build-candidates",
            "build-fixture",
        ),
    )
    parser.add_argument("--candidate-id")
    parser.add_argument("--segment-index", type=int)
    return parser.parse_args()


def main() -> int:
    """Execute one attempt2-v3 stage."""
    args = parse_args()
    if args.command == "freeze-protocol":
        payload = freeze_protocol()
        print(f"E182_ATTEMPT2_V3_PROTOCOL={payload['status']}")
        return 0
    if args.command == "build-segment":
        if args.candidate_id is None or args.segment_index is None:
            raise RuntimeError("build-segment requires candidate-id and segment-index")
        build_segment_child(args.candidate_id, args.segment_index)
        return 0
    if args.command == "build-candidates":
        payload = build_candidates()
        print(
            f"E182_ATTEMPT2_V3_BUILD={payload['status']} "
            f"candidates={payload['candidate_count']}"
        )
        return 0
    payload = build_fixture()
    print(
        f"E182_ATTEMPT2_V3_FIXTURE={payload['status']} "
        f"candidates={payload['candidate_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
