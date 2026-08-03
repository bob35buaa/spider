#!/usr/bin/env python3
"""Bake and two-stage validate E187's unlocked bucket007 2.5mm grid."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
E186_SCRIPTS = REPO / "workspace/core4d/scripts/experiments/E186"
sys.path.insert(0, str(E186_SCRIPTS))

import bake_canonical_grid_sdf as grid_v4  # noqa: E402

from spider.geometry.grid_sdf import CanonicalGridSDF, sha256_file  # noqa: E402

EXPERIMENT_ID = "E187"
OBJECT_KEY = "bucket007"
VOXEL_SIZE_M = 0.0025
MARGIN_M = 0.120
SMOKE_POINTS = 50_000
FORMAL_POINTS = 1_000_000
DEFAULT_OUTPUT_ROOT = (
    REPO
    / "workspace/core4d/results/E187/s2_canonical_grid_sdf/candidates/bucket007_2p5mm"
)


def relative(path: Path) -> str:
    """Render repository-relative artifact paths."""
    absolute = path.absolute()
    try:
        return absolute.relative_to(REPO.absolute()).as_posix()
    except ValueError:
        return str(absolute)


def json_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize stable JSON bytes."""
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def write_immutable(path: Path, payload: bytes) -> None:
    """Create one artifact or verify byte-identical content."""
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError(f"immutable E187 grid artifact mismatch: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def npy_bytes(values: np.ndarray) -> bytes:
    """Serialize a canonical NumPy payload without pickle."""
    stream = io.BytesIO()
    np.save(stream, values, allow_pickle=False)
    return stream.getvalue()


def canonical_digest(value: Any) -> str:
    """Hash JSON-compatible scientific content."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def exact_authority() -> tuple[dict[str, Any], list[Any], Any, list[Any], Any]:
    """Load E186's frozen collider and rebuild the exact-C authority."""
    lock = grid_v4.load_collider_lock()
    collider = lock["objects"][OBJECT_KEY]
    parts = grid_v4._load_parts(collider)
    union = grid_v4.build_exact_union_mesh(parts)
    volumes = grid_v4._convex_union_halfspaces(parts)
    scene = grid_v4._scene(union)
    return collider, parts, union, volumes, scene


def candidate_payload(output_root: Path) -> dict[str, Any]:
    """Bake or verify the single immutable 2.5mm grid payload."""
    candidate_path = output_root / "grid_candidate_manifest.json"
    grid_path = output_root / "grid.npy"
    if candidate_path.exists():
        payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        if (
            payload.get("experiment_id") != EXPERIMENT_ID
            or payload.get("object_key") != OBJECT_KEY
            or payload.get("grid", {}).get("voxel_size_m") != VOXEL_SIZE_M
        ):
            raise RuntimeError("existing E187 candidate identity mismatch")
        CanonicalGridSDF.load(candidate_path, allow_pending=True)
        return payload

    collider, _, union, volumes, scene = exact_authority()
    bounds = np.asarray(union.bounds, dtype=np.float64)
    origin, shape = grid_v4.grid_geometry(bounds, VOXEL_SIZE_M, MARGIN_M)
    values = grid_v4.bake_values(scene, volumes, origin, shape, VOXEL_SIZE_M)
    write_immutable(grid_path, npy_bytes(values))
    payload = {
        "schema": "spider_canonical_grid_sdf_v1",
        "experiment_id": EXPERIMENT_ID,
        "stage": "A1_S2_BUCKET007_2P5MM_CANDIDATE",
        "status": "GRID_PENDING_VALIDATION",
        "object_key": OBJECT_KEY,
        "source": {
            "reuse_policy": "E186_FROZEN_COLLIDER_NEW_E187_GRID",
            "collider_lock": relative(grid_v4.COLLIDER_LOCK),
            "collider_lock_sha256": grid_v4.EXPECTED_COLLIDER_LOCK_SHA256,
            "candidate_key": collider["candidate_key"],
            "candidate_asset_sha256": collider["candidate_asset_sha256"],
            "ordered_parts_sha256": collider["ordered_parts_sha256"],
            "ordered_part_sha256": [
                part["sha256"] for part in collider["ordered_parts"]
            ],
            "e186_grid_builder": relative(Path(grid_v4.__file__).resolve()),
            "e186_grid_builder_sha256": sha256_file(Path(grid_v4.__file__).resolve()),
        },
        "sign_convention": "negative_inside",
        "signed_distance_backend": (
            "open3d_unsigned_union_surface_plus_convex_part_halfspaces_v3"
        ),
        "outside_rule": "object_aabb_distance_lower_bound",
        "object_aabb_m": {"min": bounds[0].tolist(), "max": bounds[1].tolist()},
        "grid": {
            "path": relative(grid_path),
            "sha256": sha256_file(grid_path),
            "dtype": "float32",
            "shape": list(shape),
            "origin_object_m": origin.tolist(),
            "voxel_size_m": VOXEL_SIZE_M,
            "margin_m": MARGIN_M,
            "size_bytes": grid_path.stat().st_size,
            "interpolation": "trilinear",
        },
        "validation": {"epsilon_grid_m": 0.0, "status": "PENDING"},
        "builder": {
            "path": relative(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    payload["scientific_payload_sha256"] = canonical_digest(payload)
    write_immutable(candidate_path, json_bytes(payload))
    return payload


def validate_phase(output_root: Path, phase: str) -> dict[str, Any]:
    """Run smoke or formal validation against the exact same grid payload."""
    if phase not in {"smoke", "formal"}:
        raise ValueError(f"unsupported validation phase: {phase}")
    candidate = candidate_payload(output_root)
    candidate_path = output_root / "grid_candidate_manifest.json"
    smoke_path = output_root / "smoke50k.json"
    formal_path = output_root / "formal1m.json"
    if phase == "formal":
        if not smoke_path.is_file():
            raise RuntimeError("formal validation requires smoke50k evidence")
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        if smoke.get("status") != "PASS" or smoke.get(
            "candidate_manifest_sha256"
        ) != sha256_file(candidate_path):
            raise RuntimeError("smoke50k evidence is not valid for this candidate")

    _, _, union, volumes, scene = exact_authority()
    point_count = SMOKE_POINTS if phase == "smoke" else FORMAL_POINTS
    validation = grid_v4.validate_grid(
        candidate_path,
        union,
        volumes,
        scene,
        validation_points=point_count,
    )
    cuda = validation["cpu_cuda_parity"]
    phase_pass = bool(
        validation["status"] == "PASS"
        and validation["validation_point_count"] == point_count
        and cuda["available"]
        and cuda["checked"]
        and cuda["status"] == "PASS"
        and validation["minkowski_support_coverage"]["minimum_actual_padding_m"]
        >= 0.110
    )
    report = {
        "schema": "e187_grid_candidate_validation_v1",
        "experiment_id": EXPERIMENT_ID,
        "stage": f"A1_S2_{phase.upper()}",
        "object_key": OBJECT_KEY,
        "voxel_size_m": VOXEL_SIZE_M,
        "candidate_manifest": relative(candidate_path),
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "grid_sha256": candidate["grid"]["sha256"],
        "validation": validation,
        "status": "PASS" if phase_pass else "FAIL",
    }
    report["scientific_payload_sha256"] = canonical_digest(report)
    report_path = smoke_path if phase == "smoke" else formal_path
    write_immutable(report_path, json_bytes(report))
    if not phase_pass:
        raise RuntimeError(f"bucket007 2.5mm {phase} validation failed")

    if phase == "formal":
        final_manifest = {
            **candidate,
            "stage": "A1_S2_BUCKET007_2P5MM_FROZEN",
            "status": "GRID_FROZEN",
            "validation": validation,
            "validation_evidence": {
                "smoke50k": {
                    "path": relative(smoke_path),
                    "sha256": sha256_file(smoke_path),
                },
                "formal1m": {
                    "path": relative(formal_path),
                    "sha256": sha256_file(formal_path),
                },
            },
        }
        final_manifest["scientific_payload_sha256"] = canonical_digest(
            {
                key: value
                for key, value in final_manifest.items()
                if key != "scientific_payload_sha256"
            }
        )
        manifest_path = output_root / "manifest.json"
        write_immutable(manifest_path, json_bytes(final_manifest))
        CanonicalGridSDF.load(
            manifest_path,
            expected_candidate_asset_sha256=candidate["source"][
                "candidate_asset_sha256"
            ],
        )
        report["production_manifest"] = {
            "path": relative(manifest_path),
            "sha256": sha256_file(manifest_path),
        }
    return report


def preflight(output_root: Path) -> dict[str, Any]:
    """Verify the frozen input and requested output identity without baking."""
    lock = grid_v4.load_collider_lock()
    collider = lock["objects"][OBJECT_KEY]
    return {
        "status": "PASS",
        "object_key": OBJECT_KEY,
        "voxel_size_m": VOXEL_SIZE_M,
        "margin_m": MARGIN_M,
        "candidate_asset_sha256": collider["candidate_asset_sha256"],
        "ordered_parts_sha256": collider["ordered_parts_sha256"],
        "output_root": relative(output_root),
    }


def parse_args() -> argparse.Namespace:
    """Parse the frozen bucket007 grid-candidate operation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("preflight", "smoke", "formal"), required=True
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    """Run the requested immutable E187 grid-candidate phase."""
    args = parse_args()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = REPO / output_root
    result = (
        preflight(output_root)
        if args.phase == "preflight"
        else validate_phase(output_root, args.phase)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
