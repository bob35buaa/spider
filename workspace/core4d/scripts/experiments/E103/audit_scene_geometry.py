#!/usr/bin/env python3
"""Audit object mesh/collision geometry for CORE4D scenes.

This script resolves the object visual mesh path from scene.xml, computes the
mesh AABB, and compares it against object_collision box half-extents.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


DEFAULT_ROOT = Path("example_datasets/processed/core4d/unitree_g1/humanoid_object")
REPO = Path(__file__).resolve().parents[4]


def fnum(value: str | None) -> float:
    try:
        return float(value or "nan")
    except ValueError:
        return float("nan")


def fmt(values: np.ndarray | None, digits: int = 6) -> str:
    if values is None:
        return ""
    return " ".join(f"{float(v):.{digits}f}" for v in values)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_vec(text: str | None) -> np.ndarray | None:
    if not text:
        return None
    parts = text.split()
    if len(parts) < 3:
        return None
    try:
        return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)
    except ValueError:
        return None


def resolve_mesh(scene: Path, file_attr: str) -> Path:
    path = Path(file_attr)
    if path.is_absolute():
        return path
    candidates = [
        (scene.parent / path).resolve(),
        (REPO / path).resolve(),
    ]
    marker = "example_datasets/processed/"
    if marker in file_attr:
        suffix = file_attr[file_attr.index(marker) :]
        candidates.append((REPO / suffix).resolve())
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def parse_obj_extents(path: Path) -> np.ndarray | None:
    vertices: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                continue
    if not vertices:
        return None
    arr = np.asarray(vertices, dtype=np.float64)
    return arr.max(axis=0) - arr.min(axis=0)


def audit_scene(scene: Path) -> dict[str, str]:
    row: dict[str, str] = {
        "task": scene.parent.name,
        "scene_xml": str(scene),
        "status": "unknown",
        "mujoco_load_ok": "False",
        "mujoco_error": "",
    }
    try:
        model = mujoco.MjModel.from_xml_path(str(scene))
        row.update({"mujoco_load_ok": "True", "nq": str(model.nq), "nv": str(model.nv), "nu": str(model.nu)})
    except Exception as exc:  # noqa: BLE001
        row["mujoco_error"] = f"{type(exc).__name__}: {exc}"

    try:
        root = ET.parse(scene).getroot()
    except Exception as exc:  # noqa: BLE001
        row["status"] = "xml_parse_error"
        row["xml_error"] = f"{type(exc).__name__}: {exc}"
        return row

    visual_mesh_name = ""
    collision_size = None
    collision_type = ""
    object_mass = ""
    object_inertia = ""
    for geom in root.iter("geom"):
        if geom.get("name") == "object_visual":
            visual_mesh_name = geom.get("mesh", "")
        elif geom.get("name") == "object_collision":
            collision_type = geom.get("type", "")
            collision_size = parse_vec(geom.get("size"))

    for body in root.iter("body"):
        if body.get("name") == "object":
            for child in body:
                if child.tag == "inertial":
                    object_mass = child.get("mass", "")
                    object_inertia = child.get("diaginertia", "")
            break

    mesh_file = ""
    mesh_scale = ""
    for mesh in root.iter("mesh"):
        if mesh.get("name") == visual_mesh_name:
            mesh_file = mesh.get("file", "")
            mesh_scale = mesh.get("scale", "")
            break

    mesh_path = resolve_mesh(scene, mesh_file) if mesh_file else Path("")
    mesh_exists = bool(mesh_file) and mesh_path.is_file()
    extents = parse_obj_extents(mesh_path) if mesh_exists else None
    scale = parse_vec(mesh_scale) if mesh_scale else np.ones(3, dtype=np.float64)
    if extents is not None and scale is not None:
        extents = extents * scale
    expected_half = extents / 2.0 if extents is not None else None

    rel_error = None
    max_rel_error = float("nan")
    geometry_policy = ""
    if collision_size is not None and expected_half is not None:
        denom = np.maximum(np.abs(expected_half), 1e-9)
        rel_error = (collision_size - expected_half) / denom
        max_rel_error = float(np.max(np.abs(rel_error)))
        if max_rel_error <= 0.01:
            geometry_policy = "mesh_aabb_half_extents"
        elif max_rel_error <= 0.08:
            geometry_policy = "near_mesh_aabb_or_margin"
        else:
            geometry_policy = "geometry_review"

    status_parts: list[str] = []
    if row["mujoco_load_ok"] != "True":
        status_parts.append("mujoco_load_error")
    if not visual_mesh_name or not mesh_file:
        status_parts.append("missing_object_visual_mesh")
    if not mesh_exists:
        status_parts.append("missing_asset")
    if collision_size is None or collision_type != "box":
        status_parts.append("missing_or_nonbox_collision")
    if geometry_policy == "geometry_review":
        status_parts.append("geometry_review")
    if fnum(object_mass) > 20:
        status_parts.append("object_mass_policy_review")
    if not status_parts:
        status_parts.append("clean")

    row.update(
        {
            "object_visual_mesh": visual_mesh_name,
            "object_mesh_file": mesh_file,
            "object_mesh_path": str(mesh_path) if mesh_file else "",
            "object_mesh_exists": str(mesh_exists),
            "object_mesh_sha256": sha256(mesh_path) if mesh_exists else "",
            "object_mesh_scale": mesh_scale,
            "mesh_extents_m": fmt(extents),
            "expected_collision_half_extents_m": fmt(expected_half),
            "object_collision_type": collision_type,
            "object_collision_half_extents_m": fmt(collision_size),
            "collision_rel_error": fmt(rel_error, 4),
            "collision_max_rel_error": "" if math.isnan(max_rel_error) else f"{max_rel_error:.6f}",
            "geometry_policy": geometry_policy,
            "object_mass": object_mass,
            "object_diaginertia": object_inertia,
            "status": ";".join(status_parts),
        }
    )
    return row


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--core4d-root", type=Path, default=Path(""))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = [audit_scene(scene) for scene in sorted(args.root.glob("*/scene.xml"))]
    fields = [
        "task",
        "scene_xml",
        "status",
        "mujoco_load_ok",
        "mujoco_error",
        "nq",
        "nv",
        "nu",
        "object_visual_mesh",
        "object_mesh_file",
        "object_mesh_path",
        "object_mesh_exists",
        "object_mesh_sha256",
        "object_mesh_scale",
        "mesh_extents_m",
        "expected_collision_half_extents_m",
        "object_collision_type",
        "object_collision_half_extents_m",
        "collision_rel_error",
        "collision_max_rel_error",
        "geometry_policy",
        "object_mass",
        "object_diaginertia",
    ]
    write_tsv(args.out, rows, fields)

    review = sum(row["status"] != "clean" for row in rows)
    print(f"audited scenes={len(rows)} geometry_review_or_other={review}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
