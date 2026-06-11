#!/usr/bin/env python3
"""Create review-only rough non-box source templates for E144.

This script intentionally does not approve templates. It creates auditable draft
scene.xml files so bucket/desk/chair rows can be rendered and manually reviewed.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[4]
DEFAULT_RUN_ROOT = REPO / "workspace/core4d/results/E144/E144_full_nonbox_raw_contact"
DEFAULT_SCENE_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
DEFAULT_ASSET_ROOT = REPO / "example_datasets/processed/core4d/assets/objects"
DEFAULT_BASE_SCENE = DEFAULT_SCENE_ROOT / "box023_person1/scene.xml"
DEFAULT_MASS_KG = 5.0

DRAFT_FIELDS = [
    "source_scene_task",
    "object_key",
    "object_category",
    "person",
    "required_by_count",
    "draft_action",
    "draft_status",
    "draft_policy",
    "draft_notes",
    "raw_mesh",
    "asset_mesh",
    "scene_xml",
    "backup_scene_xml",
    "mesh_extents_m",
    "half_extents_m",
    "mass_kg",
    "mujoco_load_ok",
    "inertial_status",
    "geometry_status",
    "template_status_after_draft",
    "object_collision_half_extents_m",
    "expected_collision_half_extents_m",
    "collision_max_rel_error",
    "build_error",
]


def load_template_module():
    path = REPO / "workspace/core4d/scripts/data_construction_v3/stages/s2_templates/build_or_audit_templates.py"
    spec = importlib.util.spec_from_file_location("dcv3_template_builder_for_e144", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        lines = [line for line in f if line.strip() and not line.startswith("#")]
        if not lines:
            return []
        return list(csv.DictReader(lines, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def fnum(text: str, default: float = DEFAULT_MASS_KG) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value) or value <= 0:
        return default
    return value


def mesh_file_attr(object_key: str) -> str:
    return f"../../../../../example_datasets/processed/core4d/assets/objects/{object_key}/{object_key}_m.obj"


def fmt(values: np.ndarray | list[float]) -> str:
    return " ".join(f"{float(value):.6g}" for value in values)


def parse_obj_bounds_and_normal_axis(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    vertices: list[list[float]] = []
    normal_area = np.zeros(3, dtype=float)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                if len(vertices) < 3:
                    continue
                raw_indices = []
                for token in line.split()[1:]:
                    idx_text = token.split("/")[0]
                    if not idx_text:
                        continue
                    idx = int(idx_text)
                    raw_indices.append(idx - 1 if idx > 0 else len(vertices) + idx)
                if len(raw_indices) < 3:
                    continue
                v0 = np.asarray(vertices[raw_indices[0]], dtype=float)
                for i in range(1, len(raw_indices) - 1):
                    v1 = np.asarray(vertices[raw_indices[i]], dtype=float)
                    v2 = np.asarray(vertices[raw_indices[i + 1]], dtype=float)
                    normal_area += np.abs(np.cross(v1 - v0, v2 - v0)) * 0.5
    if not vertices:
        raise ValueError(f"OBJ has no vertices: {path}")
    verts = np.asarray(vertices, dtype=float)
    normal_axis = int(np.argmax(normal_area)) if np.any(normal_area > 0) else 2
    return verts.min(axis=0), verts.max(axis=0), normal_axis


def geom_box(name: str, pos: list[float], size: list[float], rgba: str = "1 0.18 0.02 0.45") -> str:
    return (
        f'      <geom name="{name}" type="box" pos="{fmt(pos)}" size="{fmt(size)}" '
        f'rgba="{rgba}" group="3" contype="1" conaffinity="1" friction="1 0.005 0.0001" condim="3" />'
    )


def axis_box(name: str, center: np.ndarray, half_size: np.ndarray) -> str:
    return geom_box(name, center.tolist(), half_size.tolist())


def merge_occupied_voxels(occupied: np.ndarray, pitch: np.ndarray, transform: np.ndarray, max_boxes: int = 128) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    occ = occupied.copy()
    boxes: list[tuple[np.ndarray, np.ndarray]] = []
    pitch = np.asarray(pitch, dtype=float)
    origin = transform[:3, 3].astype(float)
    axes_order = sorted(range(3), key=lambda axis: occupied.shape[axis], reverse=True)
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


def surface_voxel_collision_geoms(mesh_path: Path, category: str, target_cells: int = 26, max_boxes: int = 180) -> tuple[str, str]:
    import trimesh

    mesh = trimesh.load_mesh(mesh_path, process=False)
    max_extent = float(np.max(mesh.extents))
    if not np.isfinite(max_extent) or max_extent <= 0:
        raise ValueError(f"invalid mesh extent for voxel proxy: {mesh_path}")
    pitch = max_extent / float(target_cells)
    voxels = mesh.voxelized(pitch)
    boxes, left_unmerged = merge_occupied_voxels(voxels.matrix.astype(bool), np.asarray(voxels.pitch, dtype=float), np.asarray(voxels.transform, dtype=float), max_boxes=max_boxes)
    if left_unmerged:
        raise ValueError(f"voxel proxy exceeded max_boxes={max_boxes}; left_unmerged={left_unmerged}")
    geoms = []
    for idx, (center, half_size) in enumerate(boxes):
        shrink = np.minimum(np.asarray(voxels.pitch, dtype=float) * 0.12, half_size * 0.25)
        half_size = np.maximum(half_size - shrink, np.asarray(voxels.pitch, dtype=float) * 0.22)
        name = "object_collision" if idx == 0 else f"object_collision_voxel_{idx:03d}"
        geoms.append(axis_box(name, center, half_size))
    return "\n".join(geoms), f"{category}_surface_voxel_multibox_proxy_draft"


def desk_collision_geoms(half_extents: np.ndarray, up_axis: int) -> tuple[str, str]:
    axes = [0, 1, 2]
    plane_axes = [axis for axis in axes if axis != up_axis]
    h = [float(value) for value in half_extents]
    eps = 0.005
    top_thick = max(eps, min(0.045, 0.14 * h[up_axis]))
    leg_w0 = max(eps, min(0.035, 0.10 * h[plane_axes[0]]))
    leg_w1 = max(eps, min(0.035, 0.10 * h[plane_axes[1]]))
    leg_h = max(eps, h[up_axis] - top_thick / 2.0)
    p0 = max(0.0, h[plane_axes[0]] - leg_w0)
    p1 = max(0.0, h[plane_axes[1]] - leg_w1)
    top_center = np.zeros(3, dtype=float)
    top_size = half_extents.copy()
    top_center[up_axis] = h[up_axis] - top_thick / 2.0
    top_size[up_axis] = top_thick / 2.0
    leg_size = np.zeros(3, dtype=float)
    leg_size[plane_axes[0]] = leg_w0
    leg_size[plane_axes[1]] = leg_w1
    leg_size[up_axis] = leg_h / 2.0
    leg_up = -h[up_axis] + leg_h / 2.0
    rail_thick = max(eps, min(0.025, 0.07 * min(h[plane_axes[0]], h[plane_axes[1]])))
    rail_up = -h[up_axis] + min(0.28 * (2.0 * h[up_axis]), leg_h * 0.45)
    geoms = [axis_box("object_collision", top_center, top_size)]
    for s0 in (-1.0, 1.0):
        for s1 in (-1.0, 1.0):
            center = np.zeros(3, dtype=float)
            center[plane_axes[0]] = s0 * p0
            center[plane_axes[1]] = s1 * p1
            center[up_axis] = leg_up
            geoms.append(
                axis_box(
                    f"object_collision_leg_{'p' if s0 > 0 else 'n'}{plane_axes[0]}_{'p' if s1 > 0 else 'n'}{plane_axes[1]}",
                    center,
                    leg_size,
                )
            )
    rail_size = np.zeros(3, dtype=float)
    rail_size[plane_axes[0]] = leg_w0
    rail_size[plane_axes[1]] = h[plane_axes[1]]
    rail_size[up_axis] = rail_thick / 2.0
    for s0, suffix in ((-1.0, "neg"), (1.0, "pos")):
        center = np.zeros(3, dtype=float)
        center[plane_axes[0]] = s0 * p0
        center[up_axis] = rail_up
        geoms.append(axis_box(f"object_collision_crossbar_axis{plane_axes[0]}_{suffix}", center, rail_size))
    axis_name = "xyz"[up_axis]
    return "\n".join(geoms), f"desk_multibox_{axis_name}up_top_legs_crossbars_draft"


def multi_box_collision_geoms(category: str, half_extents: np.ndarray, normal_axis: int = 2) -> tuple[str, str]:
    hx, hy, hz = [float(value) for value in half_extents]
    eps = 0.005
    if category == "desk":
        return desk_collision_geoms(half_extents, normal_axis)
    if category == "chair":
        seat_thick = max(eps, min(0.045, 0.12 * hz))
        leg_x = max(eps, min(0.035, 0.10 * hx))
        leg_y = max(eps, min(0.035, 0.10 * hy))
        seat_z = -0.10 * hz
        seat_hx = hx
        seat_hy = max(leg_y * 2.0, hy * 0.72)
        back_thick = max(eps, min(0.035, 0.08 * hy))
        back_h = max(eps, hz * 0.62)
        back_y = hy - back_thick
        back_z = max(-hz + back_h / 2.0, 0.22 * hz)
        leg_h = max(eps, seat_z - seat_thick / 2.0 + hz)
        x = max(0.0, hx - leg_x)
        y_front = -max(0.0, seat_hy - leg_y)
        y_back = max(0.0, seat_hy - leg_y)
        z_leg = -hz + leg_h / 2.0
        rail_thick = max(eps, min(0.025, 0.07 * min(hx, hy)))
        z_rail = -hz + leg_h * 0.45
        geoms = [
            geom_box("object_collision", [0, 0, seat_z], [seat_hx, seat_hy, seat_thick / 2.0]),
            geom_box("object_collision_back", [0, back_y, back_z], [hx, back_thick, back_h / 2.0]),
        ]
        for sx in (-1.0, 1.0):
            for y, yname in ((y_front, "front"), (y_back, "back")):
                geoms.append(
                    geom_box(
                        f"object_collision_leg_{'p' if sx > 0 else 'n'}x_{yname}",
                        [sx * x, y, z_leg],
                        [leg_x, leg_y, leg_h / 2.0],
                    )
                )
        geoms.extend(
            [
                geom_box("object_collision_side_rail_xneg", [-x, 0, z_rail], [leg_x, seat_hy, rail_thick / 2.0]),
                geom_box("object_collision_side_rail_xpos", [x, 0, z_rail], [leg_x, seat_hy, rail_thick / 2.0]),
            ]
        )
        return "\n".join(geoms), "chair_multibox_seat_back_legs_rails_draft"
    raise ValueError(f"unsupported multibox category: {category}")


def build_multibox_scene_xml(
    bot: Any,
    base_text: str,
    object_key: str,
    category: str,
    half_extents: np.ndarray,
    normal_axis: int,
    mass_kg: float,
    mesh_file: str,
) -> tuple[str, str]:
    import re

    mesh = f'    <mesh name="{object_key}" file="{mesh_file}" scale="1 1 1" />'
    material = f'    <material name="{object_key}_material" rgba="0.40 0.50 0.60 1" />'
    scene = re.sub(r'    <mesh name="box023" file="[^"]+" scale="[^"]+" />', mesh, base_text, count=1)
    scene = re.sub(r'    <material name="box_material" rgba="[^"]+" />', material, scene, count=1)
    pos = bot.read_base_object_pos(base_text)
    inertia = bot.box_inertia(mass_kg, half_extents)
    geoms, policy = multi_box_collision_geoms(category, half_extents, normal_axis)
    object_body = (
        f'    <body name="object" pos="{pos}">\n'
        f'      <freejoint name="object_joint" />\n'
        f'      <inertial pos="0 0 0" mass="{mass_kg:.3f}" diaginertia="{fmt(inertia)}" />\n'
        f'      <geom name="object_visual" type="mesh" mesh="{object_key}" material="{object_key}_material" '
        f'group="2" contype="0" conaffinity="0" />\n'
        f'{geoms}\n'
        f'      <site name="trace_object" size="0.02" rgba="0 0 1 1" />\n'
        f'    </body>\n'
        f'  </worldbody>'
    )
    return re.sub(r'    <body name="object"[\s\S]*?    </body>\n  </worldbody>', object_body, scene, count=1), policy


def build_surface_voxel_scene_xml(
    bot: Any,
    base_text: str,
    object_key: str,
    category: str,
    mesh_path: Path,
    half_extents: np.ndarray,
    mass_kg: float,
    mesh_file: str,
) -> tuple[str, str]:
    import re

    mesh = f'    <mesh name="{object_key}" file="{mesh_file}" scale="1 1 1" />'
    material = f'    <material name="{object_key}_material" rgba="0.40 0.50 0.60 1" />'
    scene = re.sub(r'    <mesh name="box023" file="[^"]+" scale="[^"]+" />', mesh, base_text, count=1)
    scene = re.sub(r'    <material name="box_material" rgba="[^"]+" />', material, scene, count=1)
    pos = bot.read_base_object_pos(base_text)
    inertia = bot.box_inertia(mass_kg, half_extents)
    geoms, policy = surface_voxel_collision_geoms(mesh_path, category)
    object_body = (
        f'    <body name="object" pos="{pos}">\n'
        f'      <freejoint name="object_joint" />\n'
        f'      <inertial pos="0 0 0" mass="{mass_kg:.3f}" diaginertia="{fmt(inertia)}" />\n'
        f'      <geom name="object_visual" type="mesh" mesh="{object_key}" material="{object_key}_material" '
        f'group="2" contype="0" conaffinity="0" />\n'
        f'{geoms}\n'
        f'      <site name="trace_object" size="0.02" rgba="0 0 1 1" />\n'
        f'    </body>\n'
        f'  </worldbody>'
    )
    return re.sub(r'    <body name="object"[\s\S]*?    </body>\n  </worldbody>', object_body, scene, count=1), policy


def build_one(
    *,
    bot: Any,
    row: dict[str, str],
    raw_root: Path,
    scene_root: Path,
    asset_root: Path,
    base_scene: Path,
    backup_root: Path,
    apply: bool,
    overwrite: bool,
    mass_kg: float,
) -> dict[str, Any]:
    task = row["source_scene_task"]
    object_key = row["object_key"]
    category = row["object_category"]
    person = row["person"]
    raw_mesh = raw_root / "object_models" / category / f"{object_key}_m.obj"
    asset_mesh = asset_root / object_key / f"{object_key}_m.obj"
    task_dir = scene_root / task
    scene_xml = task_dir / "scene.xml"
    backup = backup_root / task / "scene_before_e144_draft.xml"
    draft_policy = "bucket_wall_proxy_aabb" if category == "bucket" else f"{category}_multibox_proxy_draft"
    out: dict[str, Any] = {
        "source_scene_task": task,
        "object_key": object_key,
        "object_category": category,
        "person": person,
        "required_by_count": row.get("required_by_count", ""),
        "draft_action": "dry_run",
        "draft_status": "not_run",
        "draft_policy": draft_policy,
        "draft_notes": "review_only_not_auto_release",
        "raw_mesh": str(raw_mesh),
        "asset_mesh": str(asset_mesh),
        "scene_xml": str(scene_xml),
        "backup_scene_xml": "",
        "mass_kg": mass_kg,
        "build_error": "",
    }
    try:
        if category not in {"bucket", "desk", "chair"}:
            out.update({"draft_status": "skipped", "draft_notes": "unsupported_category_for_e144_draft"})
            return out
        if not raw_mesh.is_file():
            raise FileNotFoundError(f"raw mesh missing: {raw_mesh}")
        if scene_xml.exists() and not overwrite and category != "bucket":
            out.update({"draft_action": "skip_existing", "draft_status": "existing_scene_not_overwritten"})
            audit = bot.audit_scene(scene_xml, REPO)
            out.update({f: audit.get(f, "") for f in [
                "mujoco_load_ok",
                "inertial_status",
                "geometry_status",
                "object_collision_half_extents_m",
                "expected_collision_half_extents_m",
                "collision_max_rel_error",
            ]})
            out["template_status_after_draft"] = audit.get("template_status", "")
            return out
        mesh_min, mesh_max, normal_axis = parse_obj_bounds_and_normal_axis(raw_mesh)
        half_extents = (mesh_max - mesh_min) / 2.0
        base_text = base_scene.read_text(encoding="utf-8")
        if category == "bucket":
            scene_text, collision_policy = bot.build_object_proxy_scene_xml(
                base_text,
                object_key,
                "bucket",
                half_extents,
                mass_kg,
                mesh_file_attr(object_key),
            )
        else:
            scene_text, collision_policy = build_surface_voxel_scene_xml(
                bot,
                base_text,
                object_key,
                category,
                raw_mesh,
                half_extents,
                mass_kg,
                mesh_file_attr(object_key),
            )
        out["draft_policy"] = collision_policy
        out["mesh_extents_m"] = fmt(half_extents * 2.0)
        out["half_extents_m"] = fmt(half_extents)
        if apply:
            asset_mesh.parent.mkdir(parents=True, exist_ok=True)
            if not asset_mesh.exists() or overwrite:
                shutil.copy2(raw_mesh, asset_mesh)
            if scene_xml.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                if not backup.exists():
                    shutil.copy2(scene_xml, backup)
                out["backup_scene_xml"] = str(backup)
            task_dir.mkdir(parents=True, exist_ok=True)
            scene_xml.write_text(scene_text, encoding="utf-8")
            bot.write_task_info(
                task,
                object_key,
                person,
                raw_mesh,
                asset_mesh,
                task_dir,
                base_scene,
                half_extents,
                mass_kg,
                "e144_apply_draft_proxy_review",
                object_category=category,
                proxy_template=True,
                manual_review_required=True,
                collision_policy=out["draft_policy"],
            )
            audit = bot.audit_scene(scene_xml, REPO)
            out.update({f: audit.get(f, "") for f in [
                "mujoco_load_ok",
                "inertial_status",
                "geometry_status",
                "object_collision_half_extents_m",
                "expected_collision_half_extents_m",
                "collision_max_rel_error",
            ]})
            out["template_status_after_draft"] = audit.get("template_status", "")
            out["draft_action"] = "created_or_overwritten"
            out["draft_status"] = "draft_needs_review" if audit.get("mujoco_load_ok") == "True" else "draft_audit_error"
        else:
            out.update({"draft_action": "would_create_or_overwrite", "draft_status": "dry_run_ok"})
    except Exception as exc:  # noqa: BLE001
        out.update({"draft_status": "error", "build_error": f"{type(exc).__name__}: {exc}"})
    return out


def markdown_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# E144 Nonbox Template Draft Summary",
        "",
        f"- rows: `{summary['rows']}`",
        f"- apply: `{summary['apply']}`",
        "",
        "## Status Counts",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for status, count in summary["draft_status_counts"].items():
        lines.append(f"| `{status}` | {count} |")
    lines.extend(["", "## Drafts", "", "| task | category | policy | status | notes |", "|---|---|---|---|---|"])
    for row in rows:
        lines.append(
            f"| `{row['source_scene_task']}` | `{row['object_category']}` | `{row['draft_policy']}` | "
            f"`{row['draft_status']}` | `{row['draft_notes'] or row['build_error']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--core4d-raw-root", type=Path, default=None)
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--base-scene", type=Path, default=DEFAULT_BASE_SCENE)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--mass-kg", type=float, default=DEFAULT_MASS_KG)
    args = parser.parse_args()

    raw_root = args.core4d_raw_root
    if raw_root is None:
        raw_root_text = os.environ.get("CORE4D_RAW_ROOT", "")
        raw_root = Path(raw_root_text) if raw_root_text else None
    if raw_root is None:
        raise SystemExit("missing --core4d-raw-root or CORE4D_RAW_ROOT")
    raw_root = raw_root.expanduser().resolve()
    scene_root = args.scene_root.expanduser().resolve()
    asset_root = args.asset_root.expanduser().resolve()
    base_scene = args.base_scene.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    if not base_scene.is_file():
        raise SystemExit(f"base scene missing: {base_scene}")

    review_rows = read_tsv(run_root / "s2_templates/nonbox_template_review.tsv")
    backlog_by_task = {
        row["source_scene_task"]: row
        for row in read_tsv(run_root / "s2_templates/template_backlog.tsv")
    }
    targets = []
    for review in review_rows:
        if review.get("review_decision") == "approve_clean":
            continue
        task = review["source_scene_task"]
        row = {**backlog_by_task.get(task, {}), **review}
        targets.append(row)
    targets.sort(key=lambda row: (row.get("object_category", ""), row.get("object_key", ""), row.get("person", "")))

    bot = load_template_module()
    backup_root = run_root / "s2_templates/draft_backups"
    rows = [
        build_one(
            bot=bot,
            row=row,
            raw_root=raw_root,
            scene_root=scene_root,
            asset_root=asset_root,
            base_scene=base_scene,
            backup_root=backup_root,
            apply=args.apply,
            overwrite=args.overwrite_existing,
            mass_kg=fnum(row.get("object_mass", ""), args.mass_kg),
        )
        for row in targets
    ]

    out_dir = run_root / "s2_templates/draft_proxy"
    write_tsv(out_dir / "draft_proxy_manifest.tsv", rows, DRAFT_FIELDS)
    write_json(out_dir / "draft_proxy_manifest.json", rows)
    summary = {
        "rows": len(rows),
        "apply": args.apply,
        "overwrite_existing": args.overwrite_existing,
        "draft_status_counts": dict(Counter(row["draft_status"] for row in rows)),
        "object_category_counts": dict(Counter(row["object_category"] for row in rows)),
        "draft_policy_counts": dict(Counter(row["draft_policy"] for row in rows)),
        "manifest": str(out_dir / "draft_proxy_manifest.tsv"),
        "note": "Drafts are review-only and must not be treated as clean_reviewed without visual/template review.",
    }
    write_json(out_dir / "draft_proxy_summary.json", summary)
    (out_dir / "draft_proxy_summary.md").write_text(markdown_summary(summary, rows), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
