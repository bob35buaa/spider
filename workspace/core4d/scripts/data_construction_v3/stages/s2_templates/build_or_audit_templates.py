#!/usr/bin/env python3
"""Build S2 source-template backlog and optionally create missing box templates."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
import sys

SCRIPT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "data_construction_v3")
for _path in (SCRIPT_ROOT / "lib", SCRIPT_ROOT / "state", SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from typing import Any

import mujoco
import numpy as np

from common import SCHEMA_VERSION, find_spider_repo, json_dumps, read_tsv, sha256_file, timestamp, write_json, write_tsv
from geometry import aabb_extents, load_obj_vertices


PERSONS = ("person1", "person2")
POLLUTED_MASS = 29.632
DEFAULT_MASS_KG = 5.0
NONBOX_PROXY_CATEGORIES = {"bucket", "board", "stick", "desk", "chair"}
SURFACE_VOXEL_PROXY_CATEGORIES = {"desk", "chair"}


def person_from_row(row: dict[str, str]) -> str:
    person = row.get("person", "")
    if person in PERSONS:
        return person
    idx = row.get("person_idx", "")
    if idx == "0":
        return "person1"
    if idx == "1":
        return "person2"
    raise ValueError(f"cannot infer person from row: {row}")


def source_task(object_key: str, person: str) -> str:
    return f"{object_key}_{person}"


def parse_vec(text: str | None) -> np.ndarray | None:
    if not text:
        return None
    parts = text.split()
    if len(parts) < 3:
        return None
    try:
        return np.asarray([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)
    except ValueError:
        return None


def fmt(values: np.ndarray | list[float], digits: int = 6) -> str:
    return " ".join(f"{float(v):.{digits}f}".rstrip("0").rstrip(".") for v in values)


def fnum(text: str | None) -> float:
    try:
        return float(text or "nan")
    except ValueError:
        return float("nan")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_obj_extents(path: Path) -> np.ndarray:
    return aabb_extents(load_obj_vertices(path))


def box_inertia(mass: float, half_extents: np.ndarray) -> np.ndarray:
    x, y, z = half_extents * 2.0
    return np.asarray(
        [
            mass / 12.0 * (y * y + z * z),
            mass / 12.0 * (x * x + z * z),
            mass / 12.0 * (x * x + y * y),
        ],
        dtype=np.float64,
    )


def resolve_mesh(spider_repo: Path, scene: Path, file_attr: str) -> Path:
    path = Path(file_attr)
    if path.is_absolute():
        return path
    candidates = [(scene.parent / path).resolve(), (spider_repo / path).resolve()]
    marker = "example_datasets/processed/"
    if marker in file_attr:
        candidates.append((spider_repo / file_attr[file_attr.index(marker) :]).resolve())
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def audit_scene(scene: Path, spider_repo: Path) -> dict[str, str]:
    row: dict[str, str] = {
        "task": scene.parent.name,
        "scene_xml": str(scene),
        "scene_exists": str(scene.is_file()),
        "mujoco_load_ok": "False",
        "mujoco_error": "",
        "inertial_status": "unknown",
        "geometry_status": "unknown",
        "template_status": "audit_fail",
    }
    if not scene.is_file():
        row.update({"inertial_status": "missing_scene", "geometry_status": "missing_scene", "template_status": "backlog"})
        return row

    try:
        model = mujoco.MjModel.from_xml_path(str(scene))
        row.update({"mujoco_load_ok": "True", "nq": str(model.nq), "nv": str(model.nv), "nu": str(model.nu)})
        site_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)]
        hand_sites = [name for name in site_names if name and "contact" in name and "hand" in name]
        row["hand_contact_site_count"] = str(len(hand_sites))
        row["hand_contact_sites"] = ",".join(hand_sites)
    except Exception as exc:  # noqa: BLE001 - audit should report failures.
        row["mujoco_error"] = f"{type(exc).__name__}: {exc}"

    try:
        root = ET.parse(scene).getroot()
    except Exception as exc:  # noqa: BLE001
        row.update(
            {
                "inertial_status": "xml_parse_error",
                "geometry_status": "xml_parse_error",
                "template_status": "audit_fail",
                "xml_error": f"{type(exc).__name__}: {exc}",
            }
        )
        return row

    robot_inertials: list[tuple[str, str, str]] = []
    object_inertial: tuple[str, str] | None = None
    for body in root.iter("body"):
        inertial = next((child for child in body if child.tag == "inertial"), None)
        if inertial is None:
            continue
        body_name = body.get("name", "")
        if body_name == "object":
            object_inertial = (inertial.get("mass", ""), inertial.get("diaginertia", ""))
        else:
            robot_inertials.append((body_name, inertial.get("mass", ""), inertial.get("diaginertia", "")))

    robot_pairs = sorted({(mass, inertia) for _, mass, inertia in robot_inertials})
    robot_masses = [fnum(mass) for _, mass, _ in robot_inertials]
    robot_polluted = any(abs(mass - POLLUTED_MASS) < 1e-3 for mass in robot_masses if not math.isnan(mass))
    inertial_parts: list[str] = []
    if row["mujoco_load_ok"] != "True":
        inertial_parts.append("mujoco_load_error")
    if not robot_inertials:
        inertial_parts.append("missing_robot_inertials")
    if object_inertial is None:
        inertial_parts.append("missing_object_inertial")
    if robot_polluted or (len(robot_pairs) == 1 and robot_inertials and abs(fnum(robot_pairs[0][0]) - POLLUTED_MASS) < 1e-3):
        inertial_parts.append("polluted_robot_inertial")
    elif len(robot_pairs) <= 5:
        inertial_parts.append("robot_inertial_low_diversity")
    object_mass = fnum(object_inertial[0]) if object_inertial else float("nan")
    if not math.isnan(object_mass) and object_mass > 20:
        inertial_parts.append("object_mass_policy_review")
    if not inertial_parts:
        inertial_parts.append("clean")

    visual_mesh_name = ""
    collision_size = None
    collision_type = ""
    mesh_file = ""
    mesh_scale = ""
    for geom in root.iter("geom"):
        if geom.get("name") == "object_visual":
            visual_mesh_name = geom.get("mesh", "")
        elif geom.get("name") == "object_collision":
            collision_type = geom.get("type", "")
            collision_size = parse_vec(geom.get("size"))
    for mesh in root.iter("mesh"):
        if mesh.get("name") == visual_mesh_name:
            mesh_file = mesh.get("file", "")
            mesh_scale = mesh.get("scale", "")
            break
    mesh_path = resolve_mesh(spider_repo, scene, mesh_file) if mesh_file else Path("")
    mesh_exists = bool(mesh_file) and mesh_path.is_file()
    extents = parse_obj_extents(mesh_path) if mesh_exists else None
    scale = parse_vec(mesh_scale) if mesh_scale else np.ones(3, dtype=np.float64)
    if extents is not None and scale is not None:
        extents = extents * scale
    expected_half = extents / 2.0 if extents is not None else None
    max_rel_error = float("nan")
    geometry_policy = ""
    if collision_size is not None and expected_half is not None:
        denom = np.maximum(np.abs(expected_half), 1e-9)
        max_rel_error = float(np.max(np.abs((collision_size - expected_half) / denom)))
        if max_rel_error <= 0.01:
            geometry_policy = "mesh_aabb_half_extents"
        elif max_rel_error <= 0.08:
            geometry_policy = "near_mesh_aabb_or_margin"
        else:
            geometry_policy = "geometry_review"

    geometry_parts: list[str] = []
    if row["mujoco_load_ok"] != "True":
        geometry_parts.append("mujoco_load_error")
    if not mesh_file or not visual_mesh_name:
        geometry_parts.append("missing_object_visual_mesh")
    if not mesh_exists:
        geometry_parts.append("missing_asset")
    if collision_size is None or collision_type != "box":
        geometry_parts.append("missing_or_nonbox_collision")
    if geometry_policy == "geometry_review":
        geometry_parts.append("geometry_review")
    if not geometry_parts:
        geometry_parts.append("clean")

    template_status = "clean" if inertial_parts == ["clean"] and geometry_parts == ["clean"] else "audit_fail"
    if row.get("hand_contact_site_count") not in {"2"}:
        template_status = "audit_fail"

    row.update(
        {
            "inertial_status": ";".join(inertial_parts),
            "geometry_status": ";".join(geometry_parts),
            "template_status": template_status,
            "robot_inertial_count": str(len(robot_inertials)),
            "robot_inertial_unique_pairs": str(len(robot_pairs)),
            "robot_polluted_mass_29_632": str(robot_polluted),
            "object_mass": "" if object_inertial is None else object_inertial[0],
            "object_diaginertia": "" if object_inertial is None else object_inertial[1],
            "object_visual_mesh": visual_mesh_name,
            "object_mesh_file": mesh_file,
            "object_mesh_path": str(mesh_path) if mesh_file else "",
            "object_mesh_exists": str(mesh_exists),
            "object_mesh_sha256": sha256_file(mesh_path) if mesh_exists else "",
            "mesh_extents_m": fmt(extents) if extents is not None else "",
            "object_collision_type": collision_type,
            "object_collision_half_extents_m": fmt(collision_size) if collision_size is not None else "",
            "expected_collision_half_extents_m": fmt(expected_half) if expected_half is not None else "",
            "collision_max_rel_error": "" if math.isnan(max_rel_error) else f"{max_rel_error:.6f}",
            "geometry_policy": geometry_policy,
            "updated_at": timestamp(),
            "schema_version": SCHEMA_VERSION,
        }
    )
    return row


def read_base_object_pos(base_text: str) -> str:
    match = re.search(r'<body name="object"([^>]*)>', base_text)
    if not match:
        raise ValueError("base object body not found")
    pos_match = re.search(r'pos="([^"]+)"', match.group(1))
    if not pos_match:
        raise ValueError("base object body pos not found")
    return pos_match.group(1)


def build_box_scene_xml(
    base_text: str,
    object_key: str,
    half_extents: np.ndarray,
    mass: float,
    mesh_file_attr: str,
    robot_meshdir_attr: str | None = None,
) -> str:
    mesh_file = f'    <mesh name="{object_key}" file="{mesh_file_attr}" scale="1 1 1" />'
    material = f'    <material name="{object_key}_material" rgba="0.40 0.50 0.60 1" />'
    scene = re.sub(r'    <mesh name="box023" file="[^"]+" scale="[^"]+" />', mesh_file, base_text, count=1)
    scene = re.sub(r'    <material name="box_material" rgba="[^"]+" />', material, scene, count=1)
    if robot_meshdir_attr is not None:
        scene = re.sub(r'meshdir="[^"]+"', f'meshdir="{robot_meshdir_attr}"', scene, count=1)
    pos = read_base_object_pos(base_text)
    inertia = box_inertia(mass, half_extents)
    object_body = (
        f'    <body name="object" pos="{pos}">\n'
        f'      <freejoint name="object_joint" />\n'
        f'      <inertial pos="0 0 0" mass="{mass:.3f}" diaginertia="{fmt(inertia, 8)}" />\n'
        f'      <geom name="object_visual" type="mesh" mesh="{object_key}" material="{object_key}_material" '
        f'group="2" contype="0" conaffinity="0" />\n'
        f'      <geom name="object_collision" type="box" size="{fmt(half_extents, 6)}" '
        f'rgba="0.40 0.50 0.60 0.3" group="3" contype="1" conaffinity="1" '
        f'friction="1 0.005 0.0001" condim="3" />\n'
        f'      <site name="trace_object" size="0.02" rgba="0 0 1 1" />\n'
        f'    </body>\n'
        f'  </worldbody>'
    )
    return re.sub(r'    <body name="object"[\s\S]*?    </body>\n  </worldbody>', object_body, scene, count=1)


def geom_box_xml(
    name: str,
    pos: np.ndarray | list[float],
    size: np.ndarray | list[float],
    rgba: str = "0.40 0.50 0.60 0.3",
) -> str:
    return (
        f'      <geom name="{name}" type="box" pos="{fmt(pos, 6)}" size="{fmt(size, 6)}" '
        f'rgba="{rgba}" group="3" contype="1" conaffinity="1" friction="1 0.005 0.0001" condim="3" />'
    )


def merge_occupied_voxels(
    occupied: np.ndarray,
    pitch: np.ndarray,
    transform: np.ndarray,
    max_boxes: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    occ = occupied.copy()
    boxes: list[tuple[np.ndarray, np.ndarray]] = []
    pitch = np.asarray(pitch, dtype=np.float64)
    origin = transform[:3, 3].astype(np.float64)
    axes_order = sorted(range(3), key=lambda axis: occ.shape[axis], reverse=True)
    while bool(occ.any()) and len(boxes) < max_boxes:
        start = np.argwhere(occ)[0]
        lo = start.copy()
        hi = start.copy()
        grown = True
        while grown:
            grown = False
            best: tuple[int, np.ndarray] | None = None
            best_gain = 0
            for axis in axes_order:
                candidate = hi.copy()
                candidate[axis] += 1
                if candidate[axis] >= occ.shape[axis]:
                    continue
                slab = [slice(lo[dim], hi[dim] + 1) for dim in range(3)]
                slab[axis] = slice(candidate[axis], candidate[axis] + 1)
                if bool(occ[tuple(slab)].all()):
                    gain = int(np.prod([hi[dim] - lo[dim] + 1 for dim in range(3) if dim != axis]))
                    if gain > best_gain:
                        best = (axis, candidate)
                        best_gain = gain
            if best is not None:
                axis, candidate = best
                hi[axis] = candidate[axis]
                grown = True
        block = tuple(slice(lo[dim], hi[dim] + 1) for dim in range(3))
        occ[block] = False
        center = origin + ((lo + hi) / 2.0) * pitch
        half_size = ((hi - lo + 1) / 2.0) * pitch
        boxes.append((center, half_size))
    boxes.sort(key=lambda item: float(np.prod(item[1])), reverse=True)
    return boxes, int(occ.sum())


def surface_voxel_collision_geoms(
    mesh_path: Path,
    category: str,
    rgba: str = "0.40 0.50 0.60 0.3",
    target_cells: int = 26,
    max_boxes: int = 180,
) -> tuple[str, str]:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("desk/chair surface voxel proxy requires trimesh") from exc

    mesh = trimesh.load_mesh(mesh_path, process=False)
    max_extent = float(np.max(mesh.extents))
    if not np.isfinite(max_extent) or max_extent <= 0:
        raise ValueError(f"invalid mesh extent for voxel proxy: {mesh_path}")
    voxels = mesh.voxelized(max_extent / float(target_cells))
    boxes, left_unmerged = merge_occupied_voxels(
        voxels.matrix.astype(bool),
        np.asarray(voxels.pitch, dtype=np.float64),
        np.asarray(voxels.transform, dtype=np.float64),
        max_boxes=max_boxes,
    )
    if left_unmerged:
        raise ValueError(f"voxel proxy exceeded max_boxes={max_boxes}; left_unmerged={left_unmerged}")
    geoms = []
    pitch = np.asarray(voxels.pitch, dtype=np.float64)
    for idx, (center, half_size) in enumerate(boxes):
        shrink = np.minimum(pitch * 0.12, half_size * 0.25)
        half_size = np.maximum(half_size - shrink, pitch * 0.22)
        name = "object_collision" if idx == 0 else f"object_collision_voxel_{idx:03d}"
        geoms.append(geom_box_xml(name, center, half_size, rgba=rgba))
    return "\n".join(geoms), f"{category}_surface_voxel_multibox_proxy_draft"


def object_collision_geoms(
    object_key: str,
    category: str,
    half_extents: np.ndarray,
    mesh_path: Path | None = None,
    rgba: str = "0.40 0.50 0.60 0.3",
) -> tuple[str, str]:
    if category in SURFACE_VOXEL_PROXY_CATEGORIES:
        if mesh_path is None:
            raise ValueError(f"{category} surface voxel proxy requires mesh_path")
        return surface_voxel_collision_geoms(mesh_path, category, rgba=rgba)
    if category == "bucket":
        wall = max(0.005, min(float(np.min(half_extents[:2])) * 0.08, 0.025))
        hz = float(half_extents[2])
        hx = float(half_extents[0])
        hy = float(half_extents[1])
        geoms = [
            (
                f'      <geom name="object_collision" type="box" '
                f'pos="0 0 {-hz + wall / 2.0:.6f}" size="{fmt([hx, hy, wall / 2.0], 6)}" '
                f'rgba="{rgba}" group="3" contype="1" conaffinity="1" friction="1 0.005 0.0001" condim="3" />'
            ),
            (
                f'      <geom name="object_collision_bucket_xneg" type="box" '
                f'pos="{-hx + wall / 2.0:.6f} 0 0" size="{fmt([wall / 2.0, hy, hz], 6)}" '
                f'rgba="{rgba}" group="3" contype="1" conaffinity="1" friction="1 0.005 0.0001" condim="3" />'
            ),
            (
                f'      <geom name="object_collision_bucket_xpos" type="box" '
                f'pos="{hx - wall / 2.0:.6f} 0 0" size="{fmt([wall / 2.0, hy, hz], 6)}" '
                f'rgba="{rgba}" group="3" contype="1" conaffinity="1" friction="1 0.005 0.0001" condim="3" />'
            ),
            (
                f'      <geom name="object_collision_bucket_yneg" type="box" '
                f'pos="0 {-hy + wall / 2.0:.6f} 0" size="{fmt([hx, wall / 2.0, hz], 6)}" '
                f'rgba="{rgba}" group="3" contype="1" conaffinity="1" friction="1 0.005 0.0001" condim="3" />'
            ),
            (
                f'      <geom name="object_collision_bucket_ypos" type="box" '
                f'pos="0 {hy - wall / 2.0:.6f} 0" size="{fmt([hx, wall / 2.0, hz], 6)}" '
                f'rgba="{rgba}" group="3" contype="1" conaffinity="1" friction="1 0.005 0.0001" condim="3" />'
            ),
        ]
        return "\n".join(geoms), "bucket_wall_proxy_aabb"
    return (
        f'      <geom name="object_collision" type="box" size="{fmt(half_extents, 6)}" '
        f'rgba="{rgba}" group="3" contype="1" conaffinity="1" friction="1 0.005 0.0001" condim="3" />',
        "mesh_aabb_box_proxy",
    )


def build_object_proxy_scene_xml(
    base_text: str,
    object_key: str,
    category: str,
    half_extents: np.ndarray,
    mass: float,
    mesh_file_attr: str,
    mesh_path: Path | None = None,
    robot_meshdir_attr: str | None = None,
) -> tuple[str, str]:
    mesh_file = f'    <mesh name="{object_key}" file="{mesh_file_attr}" scale="1 1 1" />'
    material = f'    <material name="{object_key}_material" rgba="0.40 0.50 0.60 1" />'
    scene = re.sub(r'    <mesh name="box023" file="[^"]+" scale="[^"]+" />', mesh_file, base_text, count=1)
    scene = re.sub(r'    <material name="box_material" rgba="[^"]+" />', material, scene, count=1)
    if robot_meshdir_attr is not None:
        scene = re.sub(r'meshdir="[^"]+"', f'meshdir="{robot_meshdir_attr}"', scene, count=1)
    pos = read_base_object_pos(base_text)
    inertia = box_inertia(mass, half_extents)
    collision_geoms, collision_policy = object_collision_geoms(object_key, category, half_extents, mesh_path=mesh_path)
    object_body = (
        f'    <body name="object" pos="{pos}">\n'
        f'      <freejoint name="object_joint" />\n'
        f'      <inertial pos="0 0 0" mass="{mass:.3f}" diaginertia="{fmt(inertia, 8)}" />\n'
        f'      <geom name="object_visual" type="mesh" mesh="{object_key}" material="{object_key}_material" '
        f'group="2" contype="0" conaffinity="0" />\n'
        f'{collision_geoms}\n'
        f'      <site name="trace_object" size="0.02" rgba="0 0 1 1" />\n'
        f'    </body>\n'
        f'  </worldbody>'
    )
    return re.sub(r'    <body name="object"[\s\S]*?    </body>\n  </worldbody>', object_body, scene, count=1), collision_policy


def write_task_info(
    task: str,
    object_key: str,
    person: str,
    raw_mesh: Path,
    asset_mesh: Path,
    task_dir: Path,
    base_scene: Path,
    half_extents: np.ndarray,
    mass: float,
    build_mode: str,
    object_category: str = "box",
    proxy_template: bool = False,
    manual_review_required: bool = False,
    collision_policy: str = "mesh_aabb_box",
) -> None:
    info = {
        "task": task,
        "data_construction_v3_built": True,
        "build_mode": build_mode,
        "base_scene": str(base_scene),
        "clean_robot_inertial_source": str(base_scene),
        "object_key": object_key,
        "object_category": object_category,
        "person": person,
        "raw_object_mesh": str(raw_mesh),
        "asset_path": str(asset_mesh),
        "mesh_sha256": sha256_file(asset_mesh),
        "extents_m": (half_extents * 2.0).tolist(),
        "half_extents_m": half_extents.tolist(),
        "mass_kg": mass,
        "proxy_template": proxy_template,
        "manual_review_required": manual_review_required,
        "collision_policy": collision_policy,
        "object_mass_policy": "assumed_uniform_5kg_v3_no_real_mass_source",
        "diaginertia": box_inertia(mass, half_extents).tolist(),
        "inertia_formula": "box: Ixx=m/12*(y^2+z^2), Iyy=m/12*(x^2+z^2), Izz=m/12*(x^2+y^2)",
        "source_template_pose_policy": "neutral placeholder from clean base; target scene must patch object pose from trimmed qpos first frame",
        "created_at": timestamp(),
        "schema_version": SCHEMA_VERSION,
    }
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task_info.json").write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_missing_box_template(
    *,
    spider_repo: Path,
    raw_root: Path,
    scene_root: Path,
    asset_root: Path,
    base_scene: Path,
    task: str,
    object_key: str,
    person: str,
    mass: float,
    apply: bool,
    overwrite: bool,
) -> dict[str, str]:
    raw_mesh = raw_root / "object_models" / "box" / f"{object_key}_m.obj"
    asset_mesh = asset_root / object_key / f"{object_key}_m.obj"
    task_dir = scene_root / task
    scene_path = task_dir / "scene.xml"
    row = {
        "task": task,
        "object_key": object_key,
        "person": person,
        "raw_mesh": str(raw_mesh),
        "asset_mesh": str(asset_mesh),
        "scene_xml": str(scene_path),
        "build_action": "dry_run",
        "build_status": "not_run",
        "build_error": "",
    }
    try:
        if not raw_mesh.is_file():
            raise FileNotFoundError(f"raw box mesh missing: {raw_mesh}")
        if scene_path.exists() and not overwrite:
            row.update({"build_action": "skip_existing", "build_status": "not_needed"})
            return row
        half_extents = parse_obj_extents(raw_mesh) / 2.0
        default_scene_root = spider_repo / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
        default_asset_root = spider_repo / "example_datasets/processed/core4d/assets/objects"
        if scene_root == default_scene_root.resolve() and asset_root == default_asset_root.resolve():
            mesh_file_attr = f"../../../../../example_datasets/processed/core4d/assets/objects/{object_key}/{object_key}_m.obj"
            robot_meshdir_attr = None
        else:
            mesh_file_attr = str(asset_mesh)
            robot_meshdir_attr = str(spider_repo / "spider/assets/robots/unitree_g1/meshes")
        scene_xml = build_box_scene_xml(
            base_scene.read_text(encoding="utf-8"),
            object_key,
            half_extents,
            mass,
            mesh_file_attr,
            robot_meshdir_attr,
        )
        if apply:
            asset_mesh.parent.mkdir(parents=True, exist_ok=True)
            if not asset_mesh.exists() or overwrite:
                shutil.copy2(raw_mesh, asset_mesh)
            task_dir.mkdir(parents=True, exist_ok=True)
            scene_path.write_text(scene_xml, encoding="utf-8")
            write_task_info(task, object_key, person, raw_mesh, asset_mesh, task_dir, base_scene, half_extents, mass, "apply")
            audit = audit_scene(scene_path, spider_repo)
            row.update({"build_action": "created_or_overwritten", "build_status": audit["template_status"]})
        else:
            row.update({"build_action": "would_create", "build_status": "dry_run_ok"})
    except Exception as exc:  # noqa: BLE001
        row.update({"build_status": "error", "build_error": f"{type(exc).__name__}: {exc}"})
    return row


def build_missing_nonbox_proxy_template(
    *,
    spider_repo: Path,
    raw_root: Path,
    scene_root: Path,
    asset_root: Path,
    base_scene: Path,
    task: str,
    object_key: str,
    object_category: str,
    person: str,
    mass: float,
    apply: bool,
    overwrite: bool,
) -> dict[str, str]:
    raw_mesh = raw_root / "object_models" / object_category / f"{object_key}_m.obj"
    asset_mesh = asset_root / object_key / f"{object_key}_m.obj"
    task_dir = scene_root / task
    scene_path = task_dir / "scene.xml"
    row = {
        "task": task,
        "object_key": object_key,
        "object_category": object_category,
        "person": person,
        "raw_mesh": str(raw_mesh),
        "asset_mesh": str(asset_mesh),
        "scene_xml": str(scene_path),
        "template_adapter": "nonbox_surface_voxel_review" if object_category in SURFACE_VOXEL_PROXY_CATEGORIES else "nonbox_proxy_aabb_review",
        "collision_policy": "",
        "proxy_template": "True",
        "build_action": "dry_run",
        "build_status": "not_run",
        "build_error": "",
    }
    try:
        if object_category not in NONBOX_PROXY_CATEGORIES:
            row.update({"template_adapter": "manual_complex_shape", "build_status": "manual_review_required"})
            return row
        if not raw_mesh.is_file():
            raise FileNotFoundError(f"raw non-box mesh missing: {raw_mesh}")
        if scene_path.exists() and not overwrite:
            row.update({"build_action": "skip_existing", "build_status": "review_required"})
            return row
        half_extents = parse_obj_extents(raw_mesh) / 2.0
        default_scene_root = spider_repo / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
        default_asset_root = spider_repo / "example_datasets/processed/core4d/assets/objects"
        if scene_root == default_scene_root.resolve() and asset_root == default_asset_root.resolve():
            mesh_file_attr = f"../../../../../example_datasets/processed/core4d/assets/objects/{object_key}/{object_key}_m.obj"
            robot_meshdir_attr = None
        else:
            mesh_file_attr = str(asset_mesh)
            robot_meshdir_attr = str(spider_repo / "spider/assets/robots/unitree_g1/meshes")
        scene_xml, collision_policy = build_object_proxy_scene_xml(
            base_scene.read_text(encoding="utf-8"),
            object_key,
            object_category,
            half_extents,
            mass,
            mesh_file_attr,
            raw_mesh,
            robot_meshdir_attr,
        )
        row["collision_policy"] = collision_policy
        if apply:
            asset_mesh.parent.mkdir(parents=True, exist_ok=True)
            if not asset_mesh.exists() or overwrite:
                shutil.copy2(raw_mesh, asset_mesh)
            task_dir.mkdir(parents=True, exist_ok=True)
            scene_path.write_text(scene_xml, encoding="utf-8")
            write_task_info(
                task,
                object_key,
                person,
                raw_mesh,
                asset_mesh,
                task_dir,
                base_scene,
                half_extents,
                mass,
                "apply_proxy_review",
                object_category=object_category,
                proxy_template=True,
                manual_review_required=True,
                collision_policy=collision_policy,
            )
            audit = audit_scene(scene_path, spider_repo)
            status = "review_required" if audit.get("mujoco_load_ok") == "True" else "error"
            row.update({"build_action": "created_or_overwritten", "build_status": status})
        else:
            row.update({"build_action": "would_create_proxy", "build_status": "dry_run_review_required"})
    except Exception as exc:  # noqa: BLE001
        row.update({"build_status": "error", "build_error": f"{type(exc).__name__}: {exc}"})
    return row


def required_templates_from_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_task: dict[str, dict[str, str]] = {}
    support: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        object_key = row.get("object_key", "").strip().lower()
        if not object_key:
            continue
        person = person_from_row(row)
        task = source_task(object_key, person)
        support[task].append(row.get("case_id") or f"{object_key}_{row.get('date','')}_{row.get('seq','')}_{person}")
        if task not in by_task:
            by_task[task] = {
                "source_scene_task": task,
                "object_key": object_key,
                "object_name": row.get("object_name", ""),
                "object_category": row.get("object_category", ""),
                "person": person,
                "required_by_count": "0",
                "required_by_cases": "",
            }
    out: list[dict[str, str]] = []
    for task, item in sorted(by_task.items()):
        cases = sorted(set(support[task]))
        item["required_by_count"] = str(len(cases))
        item["required_by_cases"] = ",".join(cases[:20])
        out.append(item)
    return out


def template_decision(required: dict[str, str], audit: dict[str, str], build: dict[str, str] | None) -> dict[str, str]:
    category = required.get("object_category", "")
    scene_exists = audit.get("scene_exists") == "True"
    status = audit.get("template_status", "backlog")
    decision = status
    action = "use_existing_clean_template" if status == "clean" else "template_backlog"
    notes: list[str] = []
    if category != "box":
        decision = "manual_review_required"
        action = "manual_template_review_required"
        notes.append("non_box_template_not_auto_released")
        if build:
            adapter = build.get("template_adapter", "")
            if adapter:
                notes.append(f"template_adapter={adapter}")
            if build.get("proxy_template"):
                notes.append("proxy_template_review_required")
    elif not scene_exists:
        decision = "backlog"
        action = "build_box_source_template"
    elif status != "clean":
        decision = "audit_fail"
        action = "fix_or_rebuild_template"
    if build:
        notes.append(f"build_status={build.get('build_status','')}")
        if category == "box" and build.get("build_status") == "clean":
            decision = "clean"
            action = "built_clean_template"
    return {
        **required,
        "scene_xml": audit.get("scene_xml", ""),
        "scene_exists": audit.get("scene_exists", "False"),
        "template_status": decision,
        "recommended_action": action,
        "inertial_status": audit.get("inertial_status", ""),
        "geometry_status": audit.get("geometry_status", ""),
        "mujoco_load_ok": audit.get("mujoco_load_ok", ""),
        "nq": audit.get("nq", ""),
        "nv": audit.get("nv", ""),
        "nu": audit.get("nu", ""),
        "hand_contact_site_count": audit.get("hand_contact_site_count", ""),
        "robot_inertial_unique_pairs": audit.get("robot_inertial_unique_pairs", ""),
        "robot_polluted_mass_29_632": audit.get("robot_polluted_mass_29_632", ""),
        "object_mass": audit.get("object_mass", ""),
        "object_mesh_path": audit.get("object_mesh_path", ""),
        "object_mesh_sha256": audit.get("object_mesh_sha256", ""),
        "object_collision_half_extents_m": audit.get("object_collision_half_extents_m", ""),
        "expected_collision_half_extents_m": audit.get("expected_collision_half_extents_m", ""),
        "collision_max_rel_error": audit.get("collision_max_rel_error", ""),
        "template_adapter": build.get("template_adapter", "box_aabb") if build else ("box_aabb" if category == "box" else "manual_complex_shape"),
        "proxy_template": build.get("proxy_template", "False") if build else "False",
        "collision_policy": build.get("collision_policy", audit.get("geometry_policy", "")) if build else audit.get("geometry_policy", ""),
        "build_status": build.get("build_status", "") if build else "",
        "build_action": build.get("build_action", "") if build else "",
        "build_error": build.get("build_error", "") if build else "",
        "notes": ",".join(notes),
        "schema_version": SCHEMA_VERSION,
        "updated_at": timestamp(),
    }


def markdown_summary(summary: dict[str, Any], rows: list[dict[str, str]]) -> str:
    lines = [
        "# S2 source template backlog/audit summary",
        "",
        f"- created_at: `{summary['created_at']}`",
        f"- required templates: `{summary['required_templates']}`",
        f"- apply_build: `{summary['apply_build']}`",
        "",
        "## template status counts",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for key, count in summary["template_status_counts"].items():
        lines.append(f"| `{key}` | {count} |")
    lines.extend(["", "## recommended action counts", "", "| action | count |", "|---|---:|"])
    for key, count in summary["recommended_action_counts"].items():
        lines.append(f"| `{key}` | {count} |")
    lines.extend(["", "## non-clean templates", "", "| task | status | action | notes |", "|---|---|---|---|"])
    for row in rows:
        if row["template_status"] != "clean":
            lines.append(f"| `{row['source_scene_task']}` | `{row['template_status']}` | `{row['recommended_action']}` | `{row['notes']}` |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tsv", type=Path, required=True, help="inventory.tsv or raw_contact candidate/pass TSV")
    parser.add_argument("--core4d-raw-root", type=Path, default=None)
    parser.add_argument("--spider-repo", type=Path, default=None)
    parser.add_argument("--scene-root", type=Path, default=None)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--base-scene", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--apply-build", action="store_true", help="create missing box templates and review-only non-box proxy templates")
    parser.add_argument("--overwrite-existing", action="store_true", help="allow replacing existing scene.xml")
    parser.add_argument("--mass-kg", type=float, default=DEFAULT_MASS_KG)
    args = parser.parse_args()

    raw_root = args.core4d_raw_root or (Path(os.environ["CORE4D_RAW_ROOT"]) if os.environ.get("CORE4D_RAW_ROOT") else None)
    if raw_root is None and args.apply_build:
        raise SystemExit("missing --core4d-raw-root or CORE4D_RAW_ROOT for --apply-build")
    spider_repo = (args.spider_repo or find_spider_repo()).resolve()
    scene_root = (args.scene_root or (spider_repo / "example_datasets/processed/core4d/unitree_g1/humanoid_object")).resolve()
    asset_root = (args.asset_root or (spider_repo / "example_datasets/processed/core4d/assets/objects")).resolve()
    base_scene = (args.base_scene or (scene_root / "box023_person1/scene.xml")).resolve()
    if not base_scene.is_file():
        raise SystemExit(f"base scene missing: {base_scene}")

    input_rows = read_tsv(args.input_tsv)
    required = required_templates_from_rows(input_rows)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    build_rows: list[dict[str, str]] = []
    build_by_task: dict[str, dict[str, str]] = {}
    if args.apply_build or raw_root is not None:
        for item in required:
            task = item["source_scene_task"]
            scene = scene_root / task / "scene.xml"
            if item["object_category"] == "box" and (not scene.exists() or args.overwrite_existing):
                build = build_missing_box_template(
                    spider_repo=spider_repo,
                    raw_root=raw_root.expanduser().resolve() if raw_root else Path(""),
                    scene_root=scene_root,
                    asset_root=asset_root,
                    base_scene=base_scene,
                    task=task,
                    object_key=item["object_key"],
                    person=item["person"],
                    mass=args.mass_kg,
                    apply=args.apply_build,
                    overwrite=args.overwrite_existing,
                )
                build_rows.append(build)
                build_by_task[task] = build
            elif item["object_category"] in NONBOX_PROXY_CATEGORIES and (not scene.exists() or args.overwrite_existing):
                build = build_missing_nonbox_proxy_template(
                    spider_repo=spider_repo,
                    raw_root=raw_root.expanduser().resolve() if raw_root else Path(""),
                    scene_root=scene_root,
                    asset_root=asset_root,
                    base_scene=base_scene,
                    task=task,
                    object_key=item["object_key"],
                    object_category=item["object_category"],
                    person=item["person"],
                    mass=args.mass_kg,
                    apply=args.apply_build,
                    overwrite=args.overwrite_existing,
                )
                build_rows.append(build)
                build_by_task[task] = build

    audit_rows: list[dict[str, str]] = []
    decision_rows: list[dict[str, str]] = []
    for item in required:
        task = item["source_scene_task"]
        audit = audit_scene(scene_root / task / "scene.xml", spider_repo)
        audit_rows.append(audit)
        decision_rows.append(template_decision(item, audit, build_by_task.get(task)))

    fields = list(decision_rows[0].keys()) if decision_rows else []
    write_tsv(out_dir / "template_backlog.tsv", decision_rows, fields)
    write_json(out_dir / "template_backlog.json", decision_rows)
    audit_fields = sorted({key for row in audit_rows for key in row})
    write_tsv(out_dir / "template_audit.tsv", audit_rows, audit_fields)
    write_json(out_dir / "template_audit.json", audit_rows)
    build_fields = sorted({key for row in build_rows for key in row}) if build_rows else [
        "task",
        "object_key",
        "person",
        "build_action",
        "build_status",
        "build_error",
    ]
    write_tsv(out_dir / "template_build.tsv", build_rows, build_fields)
    write_json(out_dir / "template_build.json", build_rows)

    summary = {
        "stage": "S2_template_backlog_audit",
        "created_at": timestamp(),
        "schema_version": SCHEMA_VERSION,
        "input_tsv": str(args.input_tsv),
        "scene_root": str(scene_root),
        "asset_root": str(asset_root),
        "base_scene": str(base_scene),
        "apply_build": args.apply_build,
        "overwrite_existing": args.overwrite_existing,
        "required_templates": len(decision_rows),
        "template_status_counts": dict(Counter(row["template_status"] for row in decision_rows)),
        "recommended_action_counts": dict(Counter(row["recommended_action"] for row in decision_rows)),
        "object_counts": dict(Counter(row["object_key"] for row in decision_rows)),
        "build_status_counts": dict(Counter(row.get("build_status", "") for row in build_rows)),
    }
    write_json(out_dir / "template_summary.json", summary)
    (out_dir / "template_summary.md").write_text(markdown_summary(summary, decision_rows), encoding="utf-8")
    print(json_dumps(summary))


if __name__ == "__main__":
    main()
