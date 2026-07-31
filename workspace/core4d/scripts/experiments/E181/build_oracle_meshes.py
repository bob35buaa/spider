#!/usr/bin/env python3
"""Build minimally cleaned E181 oracle meshes and validate D_M."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import open3d as o3d
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from build_authority import REPO_ROOT, relative_to_repo, sha256_file

DEFAULT_AUTHORITY = (
    REPO_ROOT / "workspace/core4d/results/E181/s0_environment/authority_manifest.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "workspace/core4d/results/E181/s1_oracle"
MAX_FRAGMENT_FACES = 16
MAX_FRAGMENT_VOLUME_RATIO = 1e-6
SIGN_BAND_M = 0.001
SURFACE_OFFSET_M = 0.002
OPEN3D_RAY_SAMPLES = 5
OPEN3D_STABILITY_RAY_SAMPLES = 11
WINDING_AUDIT_POINTS = 2_000
OBJECT_SEEDS = {
    "bucket003": 181003,
    "bucket004": 181004,
    "bucket007": 181007,
}


def sha256_array(values: np.ndarray) -> str:
    """Return the SHA-256 of a contiguous array."""
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_authority(path: Path) -> dict[str, Any]:
    """Load the S0 authority evidence."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise RuntimeError(f"S0 authority is not PASS: {path}")
    if set(payload["object_meshes"]) != set(OBJECT_SEEDS):
        raise RuntimeError("S0 authority object set is not bucket003/004/007")
    return payload


def component_inventory(mesh: trimesh.Trimesh) -> list[dict[str, Any]]:
    """Return face-adjacency component metrics in deterministic order."""
    components = trimesh.graph.connected_components(
        mesh.face_adjacency,
        nodes=np.arange(len(mesh.faces)),
        min_len=1,
    )
    ordered = sorted(
        (np.asarray(component, dtype=np.int64) for component in components),
        key=lambda values: (-len(values), int(values.min())),
    )
    inventory = []
    for component_index, face_ids in enumerate(ordered):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            submesh = mesh.submesh([face_ids], append=True, repair=False)
            volume = float(submesh.volume)
            center_mass = np.asarray(submesh.center_mass, dtype=float)
        inventory.append(
            {
                "component_index": component_index,
                "face_ids": face_ids,
                "face_count": int(len(face_ids)),
                "vertex_count": int(len(submesh.vertices)),
                "area_m2": float(submesh.area),
                "signed_volume_m3": volume,
                "abs_volume_m3": abs(volume),
                "watertight": bool(submesh.is_watertight),
                "winding_consistent": bool(submesh.is_winding_consistent),
                "center_mass_m": (
                    center_mass.tolist() if np.isfinite(center_mass).all() else None
                ),
            }
        )
    main_volume = inventory[0]["abs_volume_m3"]
    if main_volume <= 0:
        raise RuntimeError("main component has non-positive absolute volume")
    for entry in inventory:
        entry["volume_ratio_to_main"] = entry["abs_volume_m3"] / main_volume
        entry["numerical_zero_fragment"] = (
            entry["component_index"] != 0
            and entry["face_count"] <= MAX_FRAGMENT_FACES
            and entry["volume_ratio_to_main"] <= MAX_FRAGMENT_VOLUME_RATIO
        )
    return inventory


def clean_mesh(
    raw_mesh: trimesh.Trimesh,
    inventory: list[dict[str, Any]],
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Remove only frozen numerical-zero fragments and zero-area faces."""
    unexpected_components = [
        entry for entry in inventory[1:] if not entry["numerical_zero_fragment"]
    ]
    if unexpected_components:
        raise RuntimeError(
            "non-main component does not satisfy frozen removal gate: "
            f"{unexpected_components}"
        )

    main_face_ids = np.asarray(inventory[0]["face_ids"], dtype=np.int64)
    scale = max(float(np.linalg.norm(raw_mesh.extents)), 1.0)
    zero_area_epsilon = np.finfo(np.float64).eps * scale * scale * 16.0
    zero_area_mask = raw_mesh.area_faces <= zero_area_epsilon
    retained_face_ids = main_face_ids[~zero_area_mask[main_face_ids]]
    removed_face_ids = np.setdiff1d(
        np.arange(len(raw_mesh.faces), dtype=np.int64),
        retained_face_ids,
        assume_unique=True,
    )
    retained_faces_original = raw_mesh.faces[retained_face_ids]
    retained_vertex_ids = np.unique(retained_faces_original.reshape(-1))
    vertex_map = np.full(len(raw_mesh.vertices), -1, dtype=np.int64)
    vertex_map[retained_vertex_ids] = np.arange(len(retained_vertex_ids))
    cleaned_faces = vertex_map[retained_faces_original]
    cleaned_vertices = raw_mesh.vertices[retained_vertex_ids].copy()
    cleaned = trimesh.Trimesh(
        vertices=cleaned_vertices,
        faces=cleaned_faces,
        process=False,
        validate=False,
    )
    if not np.array_equal(
        cleaned.vertices,
        raw_mesh.vertices[retained_vertex_ids],
    ):
        raise RuntimeError("retained vertex coordinates changed")
    if not np.array_equal(
        cleaned.faces,
        vertex_map[raw_mesh.faces[retained_face_ids]],
    ):
        raise RuntimeError("retained face topology changed")
    removed_vertex_ids = np.setdiff1d(
        np.arange(len(raw_mesh.vertices), dtype=np.int64),
        retained_vertex_ids,
        assume_unique=True,
    )
    cleanup = {
        "zero_area_epsilon_m2": zero_area_epsilon,
        "zero_area_face_ids": np.flatnonzero(zero_area_mask).tolist(),
        "retained_face_count": int(len(retained_face_ids)),
        "retained_vertex_count": int(len(retained_vertex_ids)),
        "removed_face_ids": removed_face_ids.tolist(),
        "removed_vertex_ids": removed_vertex_ids.tolist(),
        "retained_face_ids_sha256": sha256_array(retained_face_ids),
        "retained_vertex_ids_sha256": sha256_array(retained_vertex_ids),
        "retained_vertex_displacement_max_m": 0.0,
        "retained_face_change_count": 0,
    }
    return cleaned, cleanup


def export_cleaned_mesh(mesh: trimesh.Trimesh, output_path: Path) -> dict[str, Any]:
    """Export OBJ with round-trip exact geometry validation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = trimesh.exchange.obj.export_obj(
        mesh,
        include_normals=False,
        include_color=False,
        include_texture=False,
        digits=17,
    )
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, output_path)
    reloaded = trimesh.load(
        output_path,
        force="mesh",
        process=False,
        maintain_order=True,
    )
    if len(reloaded.vertices) != len(mesh.vertices):
        raise RuntimeError("OBJ round-trip changed vertex count")
    if len(reloaded.faces) != len(mesh.faces):
        raise RuntimeError("OBJ round-trip changed face count")
    vertex_displacement = float(
        np.max(np.abs(reloaded.vertices - mesh.vertices), initial=0.0)
    )
    face_change_count = int(np.count_nonzero(reloaded.faces != mesh.faces))
    if vertex_displacement != 0.0 or face_change_count != 0:
        raise RuntimeError(
            "OBJ round-trip changed retained geometry: "
            f"vertex displacement={vertex_displacement}, "
            f"face changes={face_change_count}"
        )
    return {
        "sha256": sha256_file(output_path),
        "roundtrip_vertex_displacement_max_m": vertex_displacement,
        "roundtrip_face_change_count": face_change_count,
    }


def validate_oracle(
    mesh: trimesh.Trimesh,
    *,
    seed: int,
    uniform_points: int,
    surface_points: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Validate signed distance and contains semantics for D_M."""
    rng = np.random.default_rng(seed)
    margin = np.maximum(mesh.extents * 0.25, 0.05)
    uniform = rng.uniform(
        mesh.bounds[0] - margin,
        mesh.bounds[1] + margin,
        size=(uniform_points, 3),
    )
    surface, face_ids = trimesh.sample.sample_surface(
        mesh,
        surface_points,
        seed=rng,
    )
    normals = mesh.face_normals[face_ids]
    plus = surface + SURFACE_OFFSET_M * normals
    minus = surface - SURFACE_OFFSET_M * normals
    queries = np.vstack((uniform, plus, minus))

    started = time.perf_counter()
    trimesh_signed_positive_inside = trimesh.proximity.signed_distance(mesh, queries)
    distance_m = np.abs(trimesh_signed_positive_inside)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)),
    )
    query_tensor = o3d.core.Tensor(queries.astype(np.float32))
    contains = (
        scene.compute_occupancy(
            query_tensor,
            nsamples=OPEN3D_RAY_SAMPLES,
        )
        .numpy()
        .astype(bool)
    )
    contains_stability = (
        scene.compute_occupancy(
            query_tensor,
            nsamples=OPEN3D_STABILITY_RAY_SAMPLES,
        )
        .numpy()
        .astype(bool)
    )
    d_m = np.where(contains, -distance_m, distance_m)
    outside_band = distance_m >= SIGN_BAND_M
    cross_backend_agreement = (
        trimesh_signed_positive_inside[outside_band] > 0
    ) == contains[outside_band]
    stability_agreement = contains[outside_band] == contains_stability[outside_band]

    outside_indices = np.flatnonzero(outside_band)
    winding_audit_count = min(WINDING_AUDIT_POINTS, len(outside_indices))
    winding_audit_indices = rng.choice(
        outside_indices,
        size=winding_audit_count,
        replace=False,
    )
    disagreement_indices = np.flatnonzero(
        outside_band & (contains != (trimesh_signed_positive_inside > 0))
    )
    winding_audit_indices = np.unique(
        np.concatenate((winding_audit_indices, disagreement_indices))
    )
    winding_numbers = exact_winding_numbers(mesh, queries[winding_audit_indices])
    winding_contains = np.abs(winding_numbers) > 0.5
    winding_agreement = contains[winding_audit_indices] == winding_contains
    elapsed = time.perf_counter() - started
    finite = np.isfinite(d_m)
    plus_d_m = d_m[uniform_points : uniform_points + surface_points]
    minus_d_m = d_m[uniform_points + surface_points :]
    metrics = {
        "query_count": int(len(queries)),
        "finite_count": int(finite.sum()),
        "finite_fraction": float(finite.mean()),
        "outside_1mm_band_count": int(outside_band.sum()),
        "sign_agreement_outside_1mm_band": float(winding_agreement.mean()),
        "sign_audit_method": (
            "open3d_multi_ray_vs_exact_solid_angle_winding_on_stratified_sample"
        ),
        "sign_audit_count": int(len(winding_audit_indices)),
        "sign_audit_disagreement_count": int(np.count_nonzero(~winding_agreement)),
        "winding_number_abs_min": float(np.min(np.abs(winding_numbers))),
        "winding_number_abs_max": float(np.max(np.abs(winding_numbers))),
        "cross_backend_sign_agreement_outside_1mm_band": float(
            cross_backend_agreement.mean()
        ),
        "cross_backend_disagreement_count": int(
            np.count_nonzero(~cross_backend_agreement)
        ),
        "open3d_ray_samples": OPEN3D_RAY_SAMPLES,
        "open3d_stability_ray_samples": OPEN3D_STABILITY_RAY_SAMPLES,
        "open3d_ray_sample_stability_outside_1mm_band": float(
            stability_agreement.mean()
        ),
        "surface_plus_outside_fraction": float((plus_d_m >= 0).mean()),
        "surface_minus_inside_fraction": float((minus_d_m <= 0).mean()),
        "query_wall_seconds": elapsed,
        "d_m_sign_convention": "negative_inside",
        "d_m_distance_backend": "trimesh_exact_nearest_triangle",
        "d_m_sign_backend": "open3d_raycasting_scene_multi_ray_occupancy",
        "d_m_min_m": float(np.min(d_m)),
        "d_m_max_m": float(np.max(d_m)),
    }
    fixture_count = min(10_000, len(queries))
    fixture_indices = rng.choice(
        len(queries),
        size=fixture_count,
        replace=False,
    )
    fixtures = {
        "points_object_m": queries[fixture_indices].astype(np.float64),
        "d_m_m": d_m[fixture_indices].astype(np.float64),
        "contains": contains[fixture_indices].astype(np.bool_),
    }
    return metrics, fixtures


def exact_winding_numbers(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    *,
    chunk_size: int = 64,
) -> np.ndarray:
    """Compute exact generalized winding numbers from triangle solid angles."""
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    values = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        query = np.asarray(points[start:stop], dtype=np.float64)[:, None, :]
        a = triangles[None, :, 0, :] - query
        b = triangles[None, :, 1, :] - query
        c = triangles[None, :, 2, :] - query
        length_a = np.linalg.norm(a, axis=2)
        length_b = np.linalg.norm(b, axis=2)
        length_c = np.linalg.norm(c, axis=2)
        numerator = np.einsum("bfi,bfi->bf", a, np.cross(b, c))
        denominator = (
            length_a * length_b * length_c
            + np.einsum("bfi,bfi->bf", a, b) * length_c
            + np.einsum("bfi,bfi->bf", b, c) * length_a
            + np.einsum("bfi,bfi->bf", c, a) * length_b
        )
        solid_angles = 2.0 * np.arctan2(numerator, denominator)
        values[start:stop] = solid_angles.sum(axis=1) / (4.0 * np.pi)
    return values


def render_component_overlay(
    raw_mesh: trimesh.Trimesh,
    cleaned_mesh: trimesh.Trimesh,
    removed_face_ids: list[int],
    output_path: Path,
) -> None:
    """Render raw/cleaned component projections."""
    removed_faces = raw_mesh.faces[np.asarray(removed_face_ids, dtype=np.int64)]
    removed_vertices = (
        raw_mesh.vertices[np.unique(removed_faces.reshape(-1))]
        if len(removed_faces)
        else np.empty((0, 3))
    )
    projections = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))
    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    for column, (axis_x, axis_y, label) in enumerate(projections):
        axes[0, column].scatter(
            raw_mesh.vertices[:, axis_x],
            raw_mesh.vertices[:, axis_y],
            s=2,
            c="#2563eb",
            alpha=0.45,
            label="retained/main",
        )
        if len(removed_vertices):
            axes[0, column].scatter(
                removed_vertices[:, axis_x],
                removed_vertices[:, axis_y],
                s=18,
                c="#dc2626",
                label="removed fragment",
            )
        axes[0, column].set_title(f"Raw components — {label}")
        axes[0, column].axis("equal")
        axes[0, column].legend(loc="best", fontsize=7)
        axes[1, column].scatter(
            cleaned_mesh.vertices[:, axis_x],
            cleaned_mesh.vertices[:, axis_y],
            s=2,
            c="#16a34a",
            alpha=0.5,
        )
        axes[1, column].set_title(f"Cleaned M* — {label}")
        axes[1, column].axis("equal")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def render_cross_sections(
    mesh: trimesh.Trimesh,
    output_path: Path,
    *,
    seed: int,
) -> None:
    """Render deterministic principal-axis near-center surface slices."""
    rng = np.random.default_rng(seed)
    samples, _ = trimesh.sample.sample_surface(mesh, 80_000, seed=rng)
    center = mesh.bounds.mean(axis=0)
    bands = np.maximum(mesh.extents * 0.01, 0.001)
    configurations = (
        (2, 0, 1, "XY @ center Z"),
        (1, 0, 2, "XZ @ center Y"),
        (0, 1, 2, "YZ @ center X"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, (slice_axis, axis_x, axis_y, label) in zip(
        axes,
        configurations,
        strict=True,
    ):
        mask = np.abs(samples[:, slice_axis] - center[slice_axis]) <= bands[slice_axis]
        selected = samples[mask]
        axis.scatter(
            selected[:, axis_x],
            selected[:, axis_y],
            s=1,
            c="#2563eb",
            alpha=0.6,
        )
        axis.set_title(f"{label} (n={len(selected)})")
        axis.axis("equal")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_object(
    object_key: str,
    raw_path: Path,
    output_root: Path,
    *,
    uniform_points: int,
    surface_points: int,
) -> dict[str, Any]:
    """Build and validate one cleaned oracle mesh."""
    raw_mesh = trimesh.load(
        raw_path,
        force="mesh",
        process=False,
        maintain_order=True,
    )
    inventory = component_inventory(raw_mesh)
    cleaned, cleanup = clean_mesh(raw_mesh, inventory)
    if not cleaned.is_watertight or not cleaned.is_winding_consistent:
        raise RuntimeError(
            f"{object_key}: cleaned mesh topology gate failed "
            f"watertight={cleaned.is_watertight}, "
            f"winding={cleaned.is_winding_consistent}"
        )
    object_root = output_root / object_key
    cleaned_path = object_root / "cleaned.obj"
    export_metrics = export_cleaned_mesh(cleaned, cleaned_path)
    oracle_metrics, fixtures = validate_oracle(
        cleaned,
        seed=OBJECT_SEEDS[object_key],
        uniform_points=uniform_points,
        surface_points=surface_points,
    )
    fixture_path = object_root / "oracle_validation.npz"
    np.savez_compressed(fixture_path, **fixtures)
    visual_root = object_root / "visual_evidence"
    render_component_overlay(
        raw_mesh,
        cleaned,
        cleanup["removed_face_ids"],
        visual_root / "component_overlay.png",
    )
    render_cross_sections(
        cleaned,
        visual_root / "principal_axis_cross_sections.png",
        seed=OBJECT_SEEDS[object_key] + 1,
    )

    removed_component_count = sum(
        bool(entry["numerical_zero_fragment"]) for entry in inventory
    )
    hard_pass = (
        cleanup["retained_vertex_displacement_max_m"] == 0.0
        and cleanup["retained_face_change_count"] == 0
        and export_metrics["roundtrip_vertex_displacement_max_m"] == 0.0
        and export_metrics["roundtrip_face_change_count"] == 0
        and cleaned.is_watertight
        and cleaned.is_winding_consistent
        and oracle_metrics["finite_fraction"] == 1.0
        and oracle_metrics["sign_agreement_outside_1mm_band"] >= 0.9999
        and oracle_metrics["open3d_ray_sample_stability_outside_1mm_band"] == 1.0
    )
    manifest = {
        "experiment_id": "E181",
        "object_key": object_key,
        "stage": "S1_oracle",
        "status": "PASS_NONVISUAL" if hard_pass else "FAIL",
        "visual_review": "PENDING",
        "raw_mesh": {
            "path": relative_to_repo(raw_path),
            "sha256": sha256_file(raw_path),
            "vertex_count": int(len(raw_mesh.vertices)),
            "face_count": int(len(raw_mesh.faces)),
            "watertight": bool(raw_mesh.is_watertight),
            "winding_consistent": bool(raw_mesh.is_winding_consistent),
            "signed_volume_m3": float(raw_mesh.volume),
            "bounds_m": raw_mesh.bounds.tolist(),
        },
        "cleaned_mesh": {
            "path": relative_to_repo(cleaned_path),
            "sha256": export_metrics["sha256"],
            "vertex_count": int(len(cleaned.vertices)),
            "face_count": int(len(cleaned.faces)),
            "watertight": bool(cleaned.is_watertight),
            "winding_consistent": bool(cleaned.is_winding_consistent),
            "signed_volume_m3": float(cleaned.volume),
            "bounds_m": cleaned.bounds.tolist(),
        },
        "cleanup_contract": {
            "max_fragment_faces": MAX_FRAGMENT_FACES,
            "max_fragment_volume_ratio": MAX_FRAGMENT_VOLUME_RATIO,
            "removed_component_count": removed_component_count,
            **cleanup,
            **export_metrics,
        },
        "component_inventory": [
            {key: value for key, value in entry.items() if key != "face_ids"}
            for entry in inventory
        ],
        "oracle_metrics": oracle_metrics,
        "oracle_fixture": {
            "path": relative_to_repo(fixture_path),
            "sha256": sha256_file(fixture_path),
            "query_count": int(len(fixtures["d_m_m"])),
        },
        "visual_evidence": {
            "component_overlay": relative_to_repo(
                visual_root / "component_overlay.png"
            ),
            "principal_axis_cross_sections": relative_to_repo(
                visual_root / "principal_axis_cross_sections.png"
            ),
        },
    }
    atomic_json(object_root / "oracle_manifest.json", manifest)
    atomic_json(
        object_root / "component_inventory.json",
        {
            "object_key": object_key,
            "raw_mesh_sha256": manifest["raw_mesh"]["sha256"],
            "components": manifest["component_inventory"],
            "removed_face_ids": cleanup["removed_face_ids"],
            "removed_vertex_ids": cleanup["removed_vertex_ids"],
        },
    )
    if not hard_pass:
        raise RuntimeError(f"{object_key}: Gate A non-visual checks failed")
    return manifest


def build_all(
    authority_path: Path = DEFAULT_AUTHORITY,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    uniform_points: int = 50_000,
    surface_points: int = 10_000,
) -> dict[str, Any]:
    """Build all three oracle meshes."""
    authority = load_authority(authority_path)
    manifests = {}
    for object_key in OBJECT_SEEDS:
        raw_path = REPO_ROOT / authority["object_meshes"][object_key]["path"]
        if sha256_file(raw_path) != authority["object_meshes"][object_key]["sha256"]:
            raise RuntimeError(f"{object_key}: raw mesh SHA changed after S0")
        manifests[object_key] = build_object(
            object_key,
            raw_path,
            output_root,
            uniform_points=uniform_points,
            surface_points=surface_points,
        )
    summary = {
        "experiment_id": "E181",
        "stage": "S1_oracle",
        "status": "PASS_NONVISUAL",
        "visual_review": "PENDING",
        "object_status": {
            object_key: manifest["status"] for object_key, manifest in manifests.items()
        },
        "objects": {
            object_key: {
                "raw_sha256": manifest["raw_mesh"]["sha256"],
                "cleaned_sha256": manifest["cleaned_mesh"]["sha256"],
                "removed_component_count": manifest["cleanup_contract"][
                    "removed_component_count"
                ],
                "sign_agreement_outside_1mm_band": manifest["oracle_metrics"][
                    "sign_agreement_outside_1mm_band"
                ],
                "finite_fraction": manifest["oracle_metrics"]["finite_fraction"],
            }
            for object_key, manifest in manifests.items()
        },
    }
    atomic_json(output_root / "gate_a_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, default=DEFAULT_AUTHORITY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--uniform-points", type=int, default=50_000)
    parser.add_argument("--surface-points", type=int, default=10_000)
    return parser.parse_args()


def main() -> int:
    """Build S1 oracle assets."""
    args = parse_args()
    summary = build_all(
        args.authority,
        args.output_root,
        uniform_points=args.uniform_points,
        surface_points=args.surface_points,
    )
    print(
        "E181_GATE_A_NONVISUAL=PASS "
        f"objects={len(summary['object_status'])} "
        "visual_review=PENDING"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
