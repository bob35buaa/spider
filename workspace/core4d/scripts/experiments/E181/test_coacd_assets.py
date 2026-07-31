#!/usr/bin/env python3
"""Direct-main tests and determinism evidence for E181 CoACD assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import trimesh
from build_authority import REPO_ROOT, sha256_file
from build_coacd_candidates import (
    DEFAULT_OUTPUT_ROOT,
    MAX_CH_VERTICES,
    MAX_CONVEX_HULLS,
    OBJECT_KEYS,
    THRESHOLDS_M,
    atomic_json,
    candidate_asset_sha256,
    candidate_id,
    full_parameters,
)

DEFAULT_DETERMINISM_EVIDENCE = (
    DEFAULT_OUTPUT_ROOT / "determinism_probe_t020_k08_v032.json"
)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def test_grid_contract() -> None:
    """The frozen parameter grid has exactly 18 unique candidates."""
    identifiers = {
        candidate_id(threshold, max_hulls, max_vertices)
        for threshold in THRESHOLDS_M
        for max_hulls in MAX_CONVEX_HULLS
        for max_vertices in MAX_CH_VERTICES
    }
    assert len(identifiers) == 18
    assert candidate_id(0.005, 8, 32) == "t005_k08_v032"
    assert candidate_id(0.020, 32, 64) == "t020_k32_v064"
    parameters = full_parameters(0.020, 8, 32)
    assert parameters["real_metric"] is True
    assert parameters["merge"] is True
    assert parameters["decimate"] is True
    assert parameters["seed"] == 1
    assert set(parameters["threading_contract"].values()) == {"1"}


def determinism_manifest(root: Path) -> dict[str, Any]:
    """Load the fixed determinism candidate from an isolated root."""
    path = root / "bucket003/t020_k08_v032/manifest.json"
    manifest = load_json(path)
    assert manifest["status"] == "BUILD_PASS"
    assert manifest["parameters"] == full_parameters(0.020, 8, 32)
    assert manifest["candidate_asset_sha256"] == candidate_asset_sha256(
        manifest["object_key"],
        manifest["parameters"],
        manifest["parts"],
    )
    return manifest


def record_determinism(
    roots: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    """Persist two-run exact part-SHA reproducibility evidence."""
    if len(roots) != 2:
        raise ValueError("determinism recording requires exactly two roots")
    manifests = [determinism_manifest(root) for root in roots]
    asset_shas = [manifest["candidate_asset_sha256"] for manifest in manifests]
    part_shas = [
        [part["sha256"] for part in manifest["parts"]] for manifest in manifests
    ]
    if len(set(asset_shas)) != 1 or part_shas[0] != part_shas[1]:
        raise RuntimeError("CoACD determinism probe SHA mismatch")
    evidence = {
        "experiment_id": "E181",
        "stage": "S2_coacd_determinism_probe",
        "status": "PASS",
        "object_key": "bucket003",
        "candidate_id": "t020_k08_v032",
        "run_count": 2,
        "candidate_asset_sha256": asset_shas[0],
        "ordered_part_sha256": part_shas[0],
        "parameters": manifests[0]["parameters"],
        "oracle_cleaned_mesh": manifests[0]["oracle_cleaned_mesh"],
        "run_metrics": [
            {
                "run_id": f"run_{index + 1}",
                "build_wall_seconds": manifest["build_wall_seconds"],
                "peak_rss_kib": manifest["peak_rss_kib"],
                "hull_count": manifest["hull_count"],
            }
            for index, manifest in enumerate(manifests)
        ],
    }
    atomic_json(output_path, evidence)
    return evidence


def audit_candidate(manifest_path: Path) -> None:
    """Audit one formal candidate against its manifest and hard build gates."""
    manifest = load_json(manifest_path)
    assert manifest["status"] == "BUILD_PASS"
    assert manifest["candidate_asset_sha256"] == candidate_asset_sha256(
        manifest["object_key"],
        manifest["parameters"],
        manifest["parts"],
    )
    assert manifest["hull_count"] <= manifest["parameters"]["max_convex_hull"]
    assert manifest["max_part_vertex_count"] <= manifest["parameters"]["max_ch_vertex"]
    native_log = REPO_ROOT / manifest["coacd_native_log"]["path"]
    assert native_log.is_file()
    assert sha256_file(native_log) == manifest["coacd_native_log"]["sha256"]
    for part in manifest["parts"]:
        path = REPO_ROOT / part["path"]
        assert path.is_file()
        assert sha256_file(path) == part["sha256"]
        assert part["vertex_count"] <= manifest["parameters"]["max_ch_vertex"]
        assert part["roundtrip_vertex_displacement_max_m"] <= 1e-15
        mesh = trimesh.load(
            path,
            force="mesh",
            process=False,
            maintain_order=True,
        )
        assert mesh.is_convex
        assert mesh.is_watertight
        assert mesh.is_winding_consistent
        assert len(mesh.vertices) == part["vertex_count"]
        assert len(mesh.faces) == part["face_count"]


def test_formal_sweep(output_root: Path) -> None:
    """All 54 formal candidates are complete and satisfy build gates."""
    summary = load_json(output_root / "build_summary.json")
    assert summary["status"] == "BUILD_PASS"
    assert summary["candidate_count"] == 54
    seen = set()
    for object_key in OBJECT_KEYS:
        object_summary = load_json(output_root / object_key / "build_summary.json")
        assert object_summary["candidate_count"] == 18
        for identifier in object_summary["candidate_ids"]:
            seen.add((object_key, identifier))
            audit_candidate(output_root / object_key / identifier / "manifest.json")
    assert len(seen) == 54


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--determinism-root",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--determinism-evidence",
        type=Path,
        default=DEFAULT_DETERMINISM_EVIDENCE,
    )
    args = parser.parse_args()
    test_grid_contract()
    print("PASS test_grid_contract")
    if args.determinism_root:
        evidence = record_determinism(
            args.determinism_root,
            args.determinism_evidence,
        )
        print(f"PASS record_determinism sha={evidence['candidate_asset_sha256']}")
        print("E181_COACD_DETERMINISM=PASS runs=2")
        return 0
    test_formal_sweep(args.output_root)
    print("PASS test_formal_sweep")
    evidence = load_json(DEFAULT_DETERMINISM_EVIDENCE)
    assert evidence["status"] == "PASS"
    assert evidence["run_count"] == 2
    print("PASS test_determinism_evidence")
    print("E181_COACD_TESTS=PASS count=3 candidates=54")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
