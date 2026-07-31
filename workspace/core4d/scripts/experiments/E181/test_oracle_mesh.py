#!/usr/bin/env python3
"""Direct-main tests for E181 oracle cleanup and D_M."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import trimesh
from build_oracle_meshes import (
    DEFAULT_AUTHORITY,
    build_all,
    clean_mesh,
    component_inventory,
    validate_oracle,
)


def synthetic_main_with_fragment() -> trimesh.Trimesh:
    """Create a valid cube plus a tiny disconnected 12-face cube."""
    main = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    fragment = trimesh.creation.box(extents=(0.001, 0.001, 0.001))
    fragment.apply_translation((2.0, 0.0, 0.0))
    return trimesh.util.concatenate((main, fragment))


def test_cleanup_preserves_main() -> None:
    """Frozen cleanup removes only the numerical-zero fragment."""
    raw = synthetic_main_with_fragment()
    inventory = component_inventory(raw)
    assert len(inventory) == 2
    assert inventory[1]["face_count"] <= 16
    assert inventory[1]["numerical_zero_fragment"]
    cleaned, cleanup = clean_mesh(raw, inventory)
    assert cleaned.is_watertight
    assert cleaned.is_winding_consistent
    assert len(cleaned.faces) == len(trimesh.creation.box().faces)
    assert cleanup["retained_vertex_displacement_max_m"] == 0.0
    assert cleanup["retained_face_change_count"] == 0


def test_oracle_sign_contract() -> None:
    """D_M is finite, negative inside, and sign-consistent."""
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    metrics, fixtures = validate_oracle(
        mesh,
        seed=181,
        uniform_points=2_000,
        surface_points=500,
    )
    assert metrics["finite_fraction"] == 1.0
    assert metrics["sign_agreement_outside_1mm_band"] >= 0.9999
    inside = fixtures["contains"]
    assert np.all(fixtures["d_m_m"][inside] <= 0)
    assert np.all(fixtures["d_m_m"][~inside] >= 0)


def test_actual_oracles_small() -> None:
    """All actual buckets pass a reduced deterministic build."""
    with tempfile.TemporaryDirectory(prefix="e181_oracle_test_") as directory:
        summary = build_all(
            DEFAULT_AUTHORITY,
            Path(directory),
            uniform_points=2_000,
            surface_points=500,
        )
        assert summary["status"] == "PASS_NONVISUAL"
        assert summary["object_status"] == {
            "bucket003": "PASS_NONVISUAL",
            "bucket004": "PASS_NONVISUAL",
            "bucket007": "PASS_NONVISUAL",
        }


def main() -> int:
    """Run tests without pytest."""
    tests = (
        test_cleanup_preserves_main,
        test_oracle_sign_contract,
        test_actual_oracles_small,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E181_ORACLE_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
