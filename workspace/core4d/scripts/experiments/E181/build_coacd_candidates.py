#!/usr/bin/env python3
"""Build the frozen 54-candidate E181 CoACD sweep with resume support."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TBB_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import coacd
import numpy as np
import trimesh
from build_authority import REPO_ROOT, relative_to_repo, sha256_file

DEFAULT_ORACLE_ROOT = REPO_ROOT / "workspace/core4d/results/E181/s1_oracle"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "workspace/core4d/results/E181/s2_coacd"
THRESHOLDS_M = (0.005, 0.010, 0.020)
MAX_CONVEX_HULLS = (8, 16, 32)
MAX_CH_VERTICES = (32, 64)
OBJECT_KEYS = ("bucket003", "bucket004", "bucket007")
COACD_PARAMETERS = {
    "preprocess_mode": "auto",
    "preprocess_resolution": 50,
    "resolution": 2000,
    "mcts_nodes": 20,
    "mcts_iterations": 150,
    "mcts_max_depth": 3,
    "pca": False,
    "merge": True,
    "decimate": True,
    "extrude": False,
    "extrude_margin_m": 0.01,
    "apx_mode": "ch",
    "seed": 1,
    "real_metric": True,
    "threading_contract": {
        "OMP_NUM_THREADS": "1",
        "TBB_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    },
}
CONCAVITY_WARNING = re.compile(
    r"Max concavity (?P<measured>[-+0-9.eE]+) exceeds the threshold "
    r"(?P<threshold>[-+0-9.eE]+) due to (?P<hulls>[0-9]+) convex hull limitation"
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic JSON without importing the Open3D oracle module."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def candidate_id(threshold_m: float, max_hulls: int, max_vertices: int) -> str:
    """Return the frozen lexical candidate identifier."""
    threshold_mm = round(threshold_m * 1000)
    return f"t{threshold_mm:03d}_k{max_hulls:02d}_v{max_vertices:03d}"


def candidate_asset_sha256(
    object_key: str,
    parameters: dict[str, Any],
    parts: list[dict[str, Any]],
) -> str:
    """Hash the object, full parameters, and ordered part payloads."""
    payload = {
        "object_key": object_key,
        "parameters": parameters,
        "parts": [
            {
                "part_index": part["part_index"],
                "sha256": part["sha256"],
                "vertex_count": part["vertex_count"],
                "face_count": part["face_count"],
            }
            for part in parts
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_oracle(
    oracle_root: Path,
    object_key: str,
) -> tuple[dict[str, Any], Path]:
    """Load a visually approved Gate A oracle mesh."""
    gate = json.loads((oracle_root / "gate_a_summary.json").read_text(encoding="utf-8"))
    if gate.get("status") != "PASS":
        raise RuntimeError("Gate A must be PASS before CoACD")
    manifest_path = oracle_root / object_key / "oracle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        raise RuntimeError(f"{object_key}: oracle manifest must be PASS")
    cleaned_path = REPO_ROOT / manifest["cleaned_mesh"]["path"]
    if sha256_file(cleaned_path) != manifest["cleaned_mesh"]["sha256"]:
        raise RuntimeError(f"{object_key}: cleaned mesh SHA changed")
    return manifest, cleaned_path


def full_parameters(
    threshold_m: float,
    max_hulls: int,
    max_vertices: int,
) -> dict[str, Any]:
    """Return all frozen CoACD parameters."""
    return {
        "threshold_m": threshold_m,
        "max_convex_hull": max_hulls,
        "max_ch_vertex": max_vertices,
        **COACD_PARAMETERS,
    }


def export_part(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_path: Path,
    *,
    part_index: int,
    max_vertices: int,
) -> dict[str, Any]:
    """Validate and deterministically export one convex part."""
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
        maintain_order=True,
    )
    coacd_vertex_count = int(len(mesh.vertices))
    mesh.remove_unreferenced_vertices()
    removed_unreferenced_vertex_count = coacd_vertex_count - int(len(mesh.vertices))
    if not mesh.is_convex:
        raise RuntimeError(f"part {part_index}: output is not convex")
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise RuntimeError(f"part {part_index}: topology is invalid")
    if len(mesh.vertices) > max_vertices:
        raise RuntimeError(
            f"part {part_index}: {len(mesh.vertices)} vertices exceed {max_vertices}"
        )
    if not np.isfinite(mesh.vertices).all() or float(mesh.volume) <= 0.0:
        raise RuntimeError(f"part {part_index}: non-finite or non-positive volume")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        trimesh.exchange.obj.export_obj(
            mesh,
            include_normals=False,
            include_color=False,
            include_texture=False,
            return_texture=False,
            write_texture=False,
            digits=17,
        ),
        encoding="utf-8",
    )
    reloaded = trimesh.load(
        output_path,
        force="mesh",
        process=False,
        maintain_order=True,
    )
    if not np.array_equal(reloaded.faces, mesh.faces):
        raise RuntimeError(f"part {part_index}: OBJ face round-trip changed")
    roundtrip_displacement_m = float(
        np.max(np.abs(reloaded.vertices - mesh.vertices), initial=0.0)
    )
    if roundtrip_displacement_m > 1e-15:
        raise RuntimeError(
            f"part {part_index}: OBJ round-trip displacement "
            f"{roundtrip_displacement_m}m exceeds 1e-15m"
        )
    return {
        "part_index": part_index,
        "path": relative_to_repo(output_path),
        "sha256": sha256_file(output_path),
        "coacd_vertex_count": coacd_vertex_count,
        "removed_unreferenced_vertex_count": removed_unreferenced_vertex_count,
        "roundtrip_vertex_displacement_max_m": roundtrip_displacement_m,
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "volume_m3": float(mesh.volume),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "convex": bool(mesh.is_convex),
        "bounds_m": mesh.bounds.tolist(),
    }


def build_one(
    object_key: str,
    threshold_m: float,
    max_hulls: int,
    max_vertices: int,
    oracle_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Run CoACD once and write a candidate manifest."""
    oracle, cleaned_path = load_oracle(oracle_root, object_key)
    identifier = candidate_id(threshold_m, max_hulls, max_vertices)
    candidate_root = output_root / object_key / identifier
    parts_root = candidate_root / "parts"
    parts_root.mkdir(parents=True, exist_ok=True)
    for stale in parts_root.glob("part_*.obj"):
        stale.unlink()
    mesh = trimesh.load(
        cleaned_path,
        force="mesh",
        process=False,
        maintain_order=True,
    )
    parameters = full_parameters(threshold_m, max_hulls, max_vertices)
    coacd.set_log_level("warn")
    started = time.perf_counter()
    result = coacd.run_coacd(
        coacd.Mesh(mesh.vertices, mesh.faces),
        threshold=threshold_m,
        max_convex_hull=max_hulls,
        preprocess_mode=parameters["preprocess_mode"],
        preprocess_resolution=parameters["preprocess_resolution"],
        resolution=parameters["resolution"],
        mcts_nodes=parameters["mcts_nodes"],
        mcts_iterations=parameters["mcts_iterations"],
        mcts_max_depth=parameters["mcts_max_depth"],
        pca=parameters["pca"],
        merge=parameters["merge"],
        decimate=parameters["decimate"],
        max_ch_vertex=max_vertices,
        extrude=parameters["extrude"],
        extrude_margin=parameters["extrude_margin_m"],
        apx_mode=parameters["apx_mode"],
        seed=parameters["seed"],
        real_metric=parameters["real_metric"],
    )
    wall_seconds = time.perf_counter() - started
    if len(result) > max_hulls:
        raise RuntimeError(f"{identifier}: {len(result)} hulls exceed cap {max_hulls}")
    parts = [
        export_part(
            vertices,
            faces,
            parts_root / f"part_{part_index:03d}.obj",
            part_index=part_index,
            max_vertices=max_vertices,
        )
        for part_index, (vertices, faces) in enumerate(result)
    ]
    manifest = {
        "experiment_id": "E181",
        "stage": "S2_coacd_build",
        "status": "BUILD_PASS",
        "object_key": object_key,
        "candidate_id": identifier,
        "oracle_cleaned_mesh": {
            "path": relative_to_repo(cleaned_path),
            "sha256": oracle["cleaned_mesh"]["sha256"],
        },
        "parameters": parameters,
        "hull_count": len(parts),
        "total_vertex_count": sum(part["vertex_count"] for part in parts),
        "total_face_count": sum(part["face_count"] for part in parts),
        "max_part_vertex_count": max(part["vertex_count"] for part in parts),
        "build_wall_seconds": wall_seconds,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "parts": parts,
    }
    manifest["candidate_asset_sha256"] = candidate_asset_sha256(
        object_key,
        parameters,
        parts,
    )
    atomic_json(candidate_root / "manifest.json", manifest)
    print(
        "E181_COACD_ONE=PASS "
        f"object={object_key} candidate={identifier} hulls={len(parts)} "
        f"wall={wall_seconds:.3f}"
    )
    return manifest


def is_resumable(
    manifest_path: Path,
    *,
    object_key: str,
    parameters: dict[str, Any],
    cleaned_sha256: str,
) -> bool:
    """Return true only for a complete candidate matching frozen authority."""
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "BUILD_PASS"
        or manifest.get("object_key") != object_key
        or manifest.get("parameters") != parameters
        or manifest.get("oracle_cleaned_mesh", {}).get("sha256") != cleaned_sha256
        or "coacd_native_log" not in manifest
        or "max_hull_merge_exceeded_threshold" not in manifest
    ):
        return False
    parts = manifest.get("parts", [])
    if len(parts) != manifest.get("hull_count"):
        return False
    return all(
        (REPO_ROOT / part["path"]).is_file()
        and sha256_file(REPO_ROOT / part["path"]) == part["sha256"]
        for part in parts
    )


def add_native_log_evidence(
    manifest_path: Path,
    native_output: str,
    log_path: Path,
) -> dict[str, Any]:
    """Attach native warning and log evidence after the isolated child exits."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(native_output, encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    match = CONCAVITY_WARNING.search(native_output)
    manifest["coacd_native_log"] = {
        "path": relative_to_repo(log_path),
        "sha256": sha256_file(log_path),
    }
    manifest["max_hull_merge_exceeded_threshold"] = match is not None
    manifest["coacd_reported_concavity"] = (
        {
            "measured_internal": float(match.group("measured")),
            "threshold_internal": float(match.group("threshold")),
            "hull_limit": int(match.group("hulls")),
        }
        if match
        else None
    )
    atomic_json(manifest_path, manifest)
    return manifest


def build_isolated_candidate(
    *,
    object_key: str,
    threshold_m: float,
    max_hulls: int,
    max_vertices: int,
    oracle_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Run one isolated child and attach its captured native log evidence."""
    identifier = candidate_id(threshold_m, max_hulls, max_vertices)
    candidate_root = output_root / object_key / identifier
    manifest_path = candidate_root / "manifest.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--one",
        object_key,
        str(threshold_m),
        str(max_hulls),
        str(max_vertices),
        "--oracle-root",
        str(oracle_root),
        "--output-root",
        str(output_root),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    native_output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        print(native_output, file=sys.stderr)
        raise RuntimeError(
            f"{object_key}/{identifier}: CoACD child failed with {completed.returncode}"
        )
    manifest = add_native_log_evidence(
        manifest_path,
        native_output,
        candidate_root / "coacd_native.log",
    )
    print(completed.stdout.strip(), flush=True)
    return manifest


def build_object(
    object_key: str,
    oracle_root: Path,
    output_root: Path,
    *,
    parallel_workers: int,
) -> dict[str, Any]:
    """Build or resume all 18 candidates for one object."""
    if parallel_workers < 1:
        raise ValueError("parallel_workers must be positive")
    oracle, _cleaned_path = load_oracle(oracle_root, object_key)
    manifests = []
    pending = []
    configurations = [
        (threshold_m, max_hulls, max_vertices)
        for threshold_m in THRESHOLDS_M
        for max_hulls in MAX_CONVEX_HULLS
        for max_vertices in MAX_CH_VERTICES
    ]
    for threshold_m, max_hulls, max_vertices in configurations:
        identifier = candidate_id(threshold_m, max_hulls, max_vertices)
        parameters = full_parameters(
            threshold_m,
            max_hulls,
            max_vertices,
        )
        manifest_path = output_root / object_key / identifier / "manifest.json"
        if is_resumable(
            manifest_path,
            object_key=object_key,
            parameters=parameters,
            cleaned_sha256=oracle["cleaned_mesh"]["sha256"],
        ):
            manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
            print(
                f"E181_COACD_RESUME=SKIP object={object_key} candidate={identifier}",
                flush=True,
            )
        else:
            pending.append((threshold_m, max_hulls, max_vertices))
    if pending:
        print(
            f"E181_COACD_PARALLEL object={object_key} pending={len(pending)} "
            f"workers={min(parallel_workers, len(pending))}",
            flush=True,
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=parallel_workers
        ) as executor:
            futures = [
                executor.submit(
                    build_isolated_candidate,
                    object_key=object_key,
                    threshold_m=threshold_m,
                    max_hulls=max_hulls,
                    max_vertices=max_vertices,
                    oracle_root=oracle_root,
                    output_root=output_root,
                )
                for threshold_m, max_hulls, max_vertices in pending
            ]
            manifests.extend(future.result() for future in futures)
    manifests.sort(key=lambda manifest: manifest["candidate_id"])
    summary = {
        "experiment_id": "E181",
        "stage": "S2_coacd_build",
        "status": "BUILD_PASS",
        "object_key": object_key,
        "candidate_count": len(manifests),
        "expected_candidate_count": 18,
        "candidate_ids": [manifest["candidate_id"] for manifest in manifests],
        "total_wall_seconds": float(
            sum(manifest["build_wall_seconds"] for manifest in manifests)
        ),
        "max_peak_rss_kib": max(manifest["peak_rss_kib"] for manifest in manifests),
        "candidate_parallel_workers": parallel_workers,
    }
    if summary["candidate_count"] != summary["expected_candidate_count"]:
        raise RuntimeError(f"{object_key}: incomplete candidate sweep")
    atomic_json(output_root / object_key / "build_summary.json", summary)
    print(f"E181_COACD_OBJECT=PASS object={object_key} candidates=18")
    return summary


def summarize(output_root: Path) -> dict[str, Any]:
    """Build the global 54-candidate summary after parallel object workers."""
    objects = {}
    for object_key in OBJECT_KEYS:
        path = output_root / object_key / "build_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        if (
            summary.get("status") != "BUILD_PASS"
            or summary.get("candidate_count") != 18
        ):
            raise RuntimeError(f"{object_key}: build summary is incomplete")
        objects[object_key] = summary
    summary = {
        "experiment_id": "E181",
        "stage": "S2_coacd_build",
        "status": "BUILD_PASS",
        "candidate_count": sum(item["candidate_count"] for item in objects.values()),
        "expected_candidate_count": 54,
        "objects": objects,
    }
    if summary["candidate_count"] != summary["expected_candidate_count"]:
        raise RuntimeError("global CoACD sweep is incomplete")
    atomic_json(output_root / "build_summary.json", summary)
    print("E181_COACD_BUILD=PASS objects=3 candidates=54")
    return summary


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--object", choices=OBJECT_KEYS, action="append")
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument(
        "--one",
        nargs=4,
        metavar=("OBJECT", "THRESHOLD_M", "MAX_HULLS", "MAX_VERTICES"),
    )
    args = parser.parse_args()
    if args.one:
        object_key, threshold_m, max_hulls, max_vertices = args.one
        build_one(
            object_key,
            float(threshold_m),
            int(max_hulls),
            int(max_vertices),
            args.oracle_root,
            args.output_root,
        )
        return 0
    if args.summarize:
        summarize(args.output_root)
        return 0
    selected_objects = args.object or list(OBJECT_KEYS)
    for object_key in selected_objects:
        build_object(
            object_key,
            args.oracle_root,
            args.output_root,
            parallel_workers=args.parallel_workers,
        )
    if set(selected_objects) == set(OBJECT_KEYS):
        summarize(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
