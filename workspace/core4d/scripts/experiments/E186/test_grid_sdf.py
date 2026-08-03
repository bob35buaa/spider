#!/usr/bin/env python3
"""Standalone unit contracts for canonical grid-SDF interpolation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import trimesh
from bake_canonical_grid_sdf import (
    DEFAULT_MARGIN_M,
    DEFAULT_VOXEL_SIZE_BY_OBJECT,
    MIN_MINKOWSKI_SUPPORT_MARGIN_M,
    _convex_union_contains,
    _convex_union_halfspaces,
    _scene,
    _signed_distance,
    grid_geometry,
)

from spider.geometry.grid_sdf import CanonicalGridSDF, GridSDFManifest, sha256_file


def _fixture(root: Path) -> CanonicalGridSDF:
    shape = (7, 8, 9)
    origin = np.array([-0.3, -0.4, -0.5], dtype=np.float32)
    voxel = 0.1
    indices = np.stack(
        np.meshgrid(*[np.arange(value) for value in shape], indexing="ij"), axis=-1
    )
    points = origin + indices * voxel
    values = (points[..., 0] + 2 * points[..., 1] - points[..., 2]).astype(np.float32)
    grid = root / "grid.npy"
    np.save(grid, values, allow_pickle=False)
    manifest = {
        "schema": "spider_canonical_grid_sdf_v1",
        "status": "GRID_FROZEN",
        "object_key": "fixture",
        "source": {
            "candidate_asset_sha256": "a" * 64,
            "ordered_parts_sha256": "b" * 64,
        },
        "sign_convention": "negative_inside",
        "outside_rule": "object_aabb_distance_lower_bound",
        "object_aabb_m": {"min": [-0.1, -0.2, -0.3], "max": [0.1, 0.2, 0.3]},
        "grid": {
            "path": str(grid),
            "sha256": sha256_file(grid),
            "dtype": "float32",
            "shape": list(shape),
            "origin_object_m": origin.tolist(),
            "voxel_size_m": voxel,
        },
        "validation": {"epsilon_grid_m": 0.002},
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return CanonicalGridSDF.load(path, expected_candidate_asset_sha256="a" * 64)


def test_grid_geometry_alignment() -> None:
    """Voxel-aligned bounds include the requested margin."""
    bounds = np.array([[-0.12, -0.21, -0.31], [0.22, 0.33, 0.44]])
    origin, shape = grid_geometry(bounds, 0.05, 0.10)
    maximum = origin + (np.asarray(shape) - 1) * 0.05
    assert np.all(origin <= bounds[0] - 0.10 + 1e-12)
    assert np.all(maximum >= bounds[1] + 0.10 - 1e-12)


def test_frozen_object_resolution_map() -> None:
    """Only bucket004 is refined after the preregistered S1 sensitivity."""
    assert DEFAULT_VOXEL_SIZE_BY_OBJECT == {
        "bucket003": 0.005,
        "bucket004": 0.0025,
        "bucket007": 0.005,
    }


def test_minkowski_reward_support_margin() -> None:
    """Production padding covers max geom radius plus active reward support."""
    assert MIN_MINKOWSKI_SUPPORT_MARGIN_M == 0.11
    assert DEFAULT_MARGIN_M == 0.12
    assert DEFAULT_MARGIN_M > MIN_MINKOWSKI_SUPPORT_MARGIN_M


def test_robust_signed_distance_box() -> None:
    """Unsigned magnitude plus convex half-spaces has the expected box sign."""
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    scene = _scene(mesh)
    volumes = _convex_union_halfspaces([mesh])
    points = np.array(
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [1.1, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    actual = _signed_distance(scene, volumes, points)
    assert np.allclose(actual, [-1.0, -0.1, 0.1, 2.0], atol=1e-6, rtol=0.0)


def test_overlapping_convex_union_contains() -> None:
    """Analytic occupancy is the OR of overlapping convex part volumes."""
    left = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    right = left.copy()
    right.apply_translation((1.5, 0.0, 0.0))
    volumes = _convex_union_halfspaces([left, right])
    points = np.array(
        [[-0.9, 0.0, 0.0], [0.75, 0.0, 0.0], [2.4, 0.0, 0.0], [0.0, 2.0, 0.0]]
    )
    assert _convex_union_contains(volumes, points).tolist() == [
        True,
        True,
        True,
        False,
    ]


def test_trilinear_exact_for_affine_field() -> None:
    """Trilinear interpolation is exact for an affine field."""
    with tempfile.TemporaryDirectory(prefix="grid_sdf_affine_") as directory:
        backend = _fixture(Path(directory))
        generator = torch.Generator().manual_seed(7)
        points = torch.rand((1000, 3), generator=generator)
        points = torch.tensor([-0.29, -0.39, -0.49]) + points * torch.tensor(
            [0.58, 0.68, 0.78]
        )
        expected = points[:, 0] + 2 * points[:, 1] - points[:, 2]
        actual = backend.query(points)
        assert torch.allclose(actual, expected, atol=2e-6, rtol=0.0)
        assert torch.allclose(
            backend.conservative_query(points),
            expected - 0.002,
            atol=2e-6,
            rtol=0.0,
        )


def test_outside_aabb_lower_bound() -> None:
    """Out-of-grid points return the conservative object-AABB distance."""
    with tempfile.TemporaryDirectory(prefix="grid_sdf_oob_") as directory:
        backend = _fixture(Path(directory))
        points = torch.tensor([[2.1, 0.0, 0.0], [0.0, -3.2, 0.0]])
        assert torch.allclose(backend.query(points), torch.tensor([2.0, 3.0]))


def test_in_bounds_mask_matches_query_domain() -> None:
    """The diagnostic mask uses the exact same inclusive grid domain as query."""
    with tempfile.TemporaryDirectory(prefix="grid_sdf_bounds_") as directory:
        backend = _fixture(Path(directory))
        origin = torch.tensor(backend.manifest.origin_object_m)
        maximum = origin + backend.manifest.voxel_size_m * (
            torch.tensor(backend.manifest.shape) - 1
        )
        points = torch.stack((origin, maximum, origin - 1e-4, maximum + 1e-4))
        assert backend.in_bounds_mask(points).tolist() == [True, True, False, False]


def test_manifest_identity_and_payload_tamper() -> None:
    """Manifest identity and grid payload changes fail closed."""
    with tempfile.TemporaryDirectory(prefix="grid_sdf_tamper_") as directory:
        root = Path(directory)
        backend = _fixture(root)
        try:
            GridSDFManifest.load(
                backend.manifest.path,
                expected_candidate_asset_sha256="c" * 64,
            )
        except RuntimeError as error:
            assert "candidate asset SHA" in str(error)
        else:
            raise AssertionError("wrong candidate identity was accepted")
        backend.manifest.grid_path.write_bytes(
            backend.manifest.grid_path.read_bytes() + b"x"
        )
        try:
            CanonicalGridSDF.load(backend.manifest.path)
        except RuntimeError as error:
            assert "payload missing or SHA mismatch" in str(error)
        else:
            raise AssertionError("tampered grid payload was accepted")


def test_pending_manifest_is_builder_only() -> None:
    """Runtime rejects pending grids unless a builder explicitly opts in."""
    with tempfile.TemporaryDirectory(prefix="grid_sdf_pending_") as directory:
        root = Path(directory)
        backend = _fixture(root)
        payload = json.loads(backend.manifest.path.read_text(encoding="utf-8"))
        payload["status"] = "GRID_PENDING_VALIDATION"
        backend.manifest.path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            CanonicalGridSDF.load(backend.manifest.path)
        except RuntimeError as error:
            assert "not frozen" in str(error)
        else:
            raise AssertionError("runtime accepted a pending grid")
        CanonicalGridSDF.load(backend.manifest.path, allow_pending=True)


def test_numpy_gate_payload_is_json_serializable() -> None:
    """All metric gates must be normalized to Python bool before freezing."""
    payload = {"gate": bool(np.float64(0.1) <= np.sqrt(3.0) * np.float64(0.1))}
    assert json.loads(json.dumps(payload)) == {"gate": True}


def main() -> int:
    """Run grid-SDF unit contracts directly."""
    tests = (
        test_grid_geometry_alignment,
        test_frozen_object_resolution_map,
        test_minkowski_reward_support_margin,
        test_robust_signed_distance_box,
        test_overlapping_convex_union_contains,
        test_trilinear_exact_for_affine_field,
        test_outside_aabb_lower_bound,
        test_in_bounds_mask_matches_query_domain,
        test_manifest_identity_and_payload_tamper,
        test_pending_manifest_is_builder_only,
        test_numpy_gate_payload_is_json_serializable,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E186_GRID_SDF_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
