#!/usr/bin/env python3
"""Quantify bucket surface-voxel fidelity/runtime proxies before E175 build."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


REPO = Path(__file__).resolve().parents[5]
E174_MANIFEST = (
    REPO
    / "workspace/core4d/results/E174/s6_downstream/manifests/"
    "cem_full_manifest.tsv"
)
DEFAULT_OUT = REPO / "workspace/core4d/results/E175/diagnostics"
DCV3_TEMPLATES = (
    REPO
    / "workspace/core4d/scripts/data_construction_v3/stages/s2_templates"
)
sys.path.insert(0, str(DCV3_TEMPLATES))

from build_or_audit_templates import (  # noqa: E402
    surface_voxel_collision_geoms,
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def resolve_object_mesh(scene: Path) -> Path:
    root = ET.parse(scene).getroot()
    object_body = next(
        body for body in root.iter("body") if body.get("name") == "object"
    )
    mesh_refs = [
        geom.get("mesh")
        for geom in object_body.iter("geom")
        if geom.get("mesh")
    ]
    assets = {
        mesh.get("name"): mesh.get("file")
        for mesh in root.findall("./asset/mesh")
    }
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    base = Path(meshdir) if Path(meshdir).is_absolute() else scene.parent / meshdir
    for ref in mesh_refs:
        raw = assets.get(ref)
        if not raw:
            continue
        path = (base / raw).resolve()
        if path.is_file():
            return path
    raise FileNotFoundError(f"object mesh not resolved from {scene}")


def proxy_boxes(
    mesh_path: Path, target_cells: int
) -> tuple[np.ndarray, np.ndarray, float]:
    xml, _ = surface_voxel_collision_geoms(
        mesh_path,
        "bucket",
        target_cells=target_cells,
        max_boxes=1000,
    )
    geoms = list(ET.fromstring(f"<body>{xml}</body>").findall("geom"))
    centers = np.asarray(
        [[float(value) for value in geom.get("pos", "").split()] for geom in geoms],
        dtype=np.float64,
    )
    half_sizes = np.asarray(
        [[float(value) for value in geom.get("size", "").split()] for geom in geoms],
        dtype=np.float64,
    )
    mesh = trimesh.load_mesh(mesh_path, process=False)
    pitch = float(np.max(mesh.extents)) / float(target_cells)
    return centers, half_sizes, pitch


def point_box_signed(
    points: np.ndarray, centers: np.ndarray, half_sizes: np.ndarray
) -> np.ndarray:
    """Return point×box signed distances for axis-aligned local boxes."""
    delta = np.abs(points[:, None, :] - centers[None, :, :])
    q = delta - half_sizes[None, :, :]
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=2)
    inside = np.minimum(np.max(q, axis=2), 0.0)
    return outside + inside


def nearest_box_surface(
    points: np.ndarray,
    centers: np.ndarray,
    half_sizes: np.ndarray,
    chunk: int = 2048,
) -> np.ndarray:
    output = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), chunk):
        signed = point_box_signed(
            points[start : start + chunk], centers, half_sizes
        )
        output[start : start + chunk] = np.min(np.abs(signed), axis=1)
    return output


def inside_proxy(
    points: np.ndarray,
    centers: np.ndarray,
    half_sizes: np.ndarray,
    chunk: int = 2048,
) -> np.ndarray:
    output = np.zeros(len(points), dtype=bool)
    for start in range(0, len(points), chunk):
        signed = point_box_signed(
            points[start : start + chunk], centers, half_sizes
        )
        output[start : start + chunk] = np.any(signed <= 0.0, axis=1)
    return output


def sample_proxy_external_surface(
    centers: np.ndarray,
    half_sizes: np.ndarray,
    pitch: float,
    rng: np.random.Generator,
    count: int,
) -> np.ndarray:
    """Sample external faces of the box union, filtering internal overlaps."""
    faces: list[tuple[int, int, int, float]] = []
    for box_id, size in enumerate(half_sizes):
        for axis in range(3):
            other = [dim for dim in range(3) if dim != axis]
            area = float(4.0 * size[other[0]] * size[other[1]])
            for sign in (-1, 1):
                faces.append((box_id, axis, sign, area))
    weights = np.asarray([face[3] for face in faces], dtype=np.float64)
    weights /= weights.sum()
    candidate_count = max(count * 3, count + 1000)
    chosen = rng.choice(
        len(faces), size=candidate_count, replace=True, p=weights
    )
    points = np.empty((candidate_count, 3), dtype=np.float64)
    outward = np.empty_like(points)
    for index, face_index in enumerate(chosen):
        box_id, axis, sign, _ = faces[int(face_index)]
        size = half_sizes[box_id]
        point = centers[box_id].copy()
        point[axis] += sign * size[axis]
        for dim in range(3):
            if dim != axis:
                point[dim] += rng.uniform(-size[dim], size[dim])
        points[index] = point
        outward[index] = 0.0
        outward[index, axis] = float(sign)
    probe = points + outward * max(pitch * 1.0e-4, 1.0e-7)
    external = ~inside_proxy(probe, centers, half_sizes)
    kept = points[external]
    if len(kept) < count:
        raise RuntimeError(
            f"external proxy samples {len(kept)} < requested {count}"
        )
    return kept[:count]


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else math.nan


def evaluate(
    object_key: str,
    mesh_path: Path,
    target_cells: int,
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"{mesh_path} is not a single Trimesh")
    centers, half_sizes, pitch = proxy_boxes(mesh_path, target_cells)
    rng = np.random.default_rng(seed)

    mesh_points, _ = trimesh.sample.sample_surface(
        mesh, sample_count, seed=rng
    )
    mesh_to_proxy = nearest_box_surface(
        np.asarray(mesh_points, dtype=np.float64), centers, half_sizes
    )

    proxy_points = sample_proxy_external_surface(
        centers, half_sizes, pitch, rng, sample_count
    )
    _, proxy_to_mesh, _ = trimesh.proximity.closest_point(mesh, proxy_points)
    proxy_to_mesh = np.asarray(proxy_to_mesh, dtype=np.float64)

    aabb_min = np.asarray(mesh.bounds[0], dtype=np.float64)
    aabb_max = np.asarray(mesh.bounds[1], dtype=np.float64)
    volume_points = rng.uniform(
        aabb_min, aabb_max, size=(sample_count, 3)
    )
    volume_inside_proxy = inside_proxy(
        volume_points, centers, half_sizes
    )
    _, volume_to_mesh, _ = trimesh.proximity.closest_point(
        mesh, volume_points
    )
    volume_to_mesh = np.asarray(volume_to_mesh, dtype=np.float64)
    phantom = volume_inside_proxy & (volume_to_mesh > 1.5 * pitch)

    proxy_min = np.min(centers - half_sizes, axis=0)
    proxy_max = np.max(centers + half_sizes, axis=0)
    return {
        "object_key": object_key,
        "target_cells": target_cells,
        "pitch_m": pitch,
        "box_count": len(centers),
        "robot_object_pair_count": 18 * len(centers),
        "mesh_to_proxy_p50_m": percentile(mesh_to_proxy, 50),
        "mesh_to_proxy_p90_m": percentile(mesh_to_proxy, 90),
        "mesh_to_proxy_p99_m": percentile(mesh_to_proxy, 99),
        "mesh_to_proxy_far_3cm_frac": float(
            np.mean(mesh_to_proxy > 0.03)
        ),
        "proxy_to_mesh_p50_m": percentile(proxy_to_mesh, 50),
        "proxy_to_mesh_p90_m": percentile(proxy_to_mesh, 90),
        "proxy_to_mesh_p99_m": percentile(proxy_to_mesh, 99),
        "proxy_to_mesh_far_3cm_frac": float(
            np.mean(proxy_to_mesh > 0.03)
        ),
        "proxy_occupancy_aabb_frac": float(np.mean(volume_inside_proxy)),
        "phantom_fill_aabb_frac": float(np.mean(phantom)),
        "aabb_min_error_m": float(np.max(np.abs(proxy_min - aabb_min))),
        "aabb_max_error_m": float(np.max(np.abs(proxy_max - aabb_max))),
        "elapsed_s": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=E174_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-cells", default="12,14,16")
    parser.add_argument("--sample-count", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=175)
    args = parser.parse_args()

    rows = read_tsv(args.manifest)
    by_object: dict[str, Path] = {}
    for row in rows:
        object_key = row["object_key"]
        if not object_key.startswith("bucket") or object_key in by_object:
            continue
        by_object[object_key] = resolve_object_mesh(REPO / row["scene_act"])
    if len(by_object) != 5:
        raise SystemExit(f"expected five bucket objects, got {by_object}")

    target_cells = [int(value) for value in args.target_cells.split(",")]
    output: list[dict[str, Any]] = []
    for object_key, mesh_path in sorted(by_object.items()):
        for cells in target_cells:
            row = evaluate(
                object_key,
                mesh_path,
                cells,
                args.sample_count,
                args.seed + cells,
            )
            row["mesh_path"] = str(mesh_path.relative_to(REPO))
            output.append(row)
            print(
                object_key,
                cells,
                f"boxes={row['box_count']}",
                f"m2p90={row['mesh_to_proxy_p90_m']:.4f}",
                f"p2m90={row['proxy_to_mesh_p90_m']:.4f}",
                f"phantom={row['phantom_fill_aabb_frac']:.4f}",
            )

    out_dir = args.out_dir
    write_tsv(out_dir / "bucket_voxel_resolution.tsv", output)
    summary = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "manifest": str(args.manifest),
        "manifest_rows": len(rows),
        "bucket_objects": len(by_object),
        "target_cells": target_cells,
        "sample_count": args.sample_count,
        "rows": len(output),
    }
    (out_dir / "bucket_voxel_resolution.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
