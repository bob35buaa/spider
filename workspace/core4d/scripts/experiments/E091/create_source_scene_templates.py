#!/usr/bin/env python3
"""Create source scene templates for E091 medium-box objects."""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
DEFAULT_CORE4D_ROOT = Path(
    "/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real"
)
DEFAULT_TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
DEFAULT_ASSET_ROOT = REPO / "example_datasets/processed/core4d/assets/objects"
DEFAULT_V2_ROOT = Path("/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2")
DEFAULT_HOLOSOMA_RETARGET_ROOT = Path(
    "/home/ubuntu/Workspace/holosoma/src/holosoma_retargeting/holosoma_retargeting"
)

OBJECT_META = {
    "box026": {
        "object_name": "Box026",
        "base_template": "box021_person1",
        "base_g1_object": "Box021",
        "mesh_rel": "box/box026_m.obj",
        "mass_kg": 5.0,
        "rgba": "0.35 0.52 0.74 1",
        "collision_rgba": "0.35 0.52 0.74 0.3",
    },
    "box004": {
        "object_name": "box004",
        "base_template": "box023_person1",
        "base_g1_object": "Box023",
        "mesh_rel": "box/box004_m.obj",
        "mass_kg": 5.0,
        "rgba": "0.67 0.46 0.28 1",
        "collision_rgba": "0.67 0.46 0.28 0.3",
    },
    "box022": {
        "object_name": "Box022",
        "base_template": "box021_person1",
        "base_g1_object": "Box021",
        "mesh_rel": "box/box022_m.obj",
        "mass_kg": 5.0,
        "rgba": "0.50 0.55 0.40 1",
        "collision_rgba": "0.50 0.55 0.40 0.3",
    },
}


def parse_obj_extents(path: Path) -> np.ndarray:
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
        raise ValueError(f"No vertices found in {path}")
    arr = np.asarray(vertices, dtype=np.float64)
    return arr.max(axis=0) - arr.min(axis=0)


def fmt(values: np.ndarray, digits: int = 6) -> str:
    return " ".join(f"{float(v):.{digits}f}" for v in values)


def box_inertia(mass: float, extents: np.ndarray) -> np.ndarray:
    x, y, z = [float(v) for v in extents]
    return np.array(
        [
            mass / 12.0 * (y * y + z * z),
            mass / 12.0 * (x * x + z * z),
            mass / 12.0 * (x * x + y * y),
        ],
        dtype=np.float64,
    )


def find_named(root: ET.Element, tag: str, name: str) -> ET.Element:
    for elem in root.iter(tag):
        if elem.get("name") == name:
            return elem
    raise ValueError(f"No <{tag} name='{name}'> found")


def find_asset(root: ET.Element) -> ET.Element:
    for elem in root.iter("asset"):
        return elem
    raise ValueError("No <asset> found")


def object_visual_mesh_name(root: ET.Element) -> str:
    geom = find_named(root, "geom", "object_visual")
    mesh = geom.get("mesh")
    if not mesh:
        raise ValueError("object_visual has no mesh attribute")
    return mesh


def write_simple_urdf(path: Path, object_name: str) -> None:
    text = f"""<?xml version=\"1.0\" ?>
<robot name=\"{object_name}\">
  <dynamics damping=\"0.5\" friction=\"0.9\"/>
  <link name=\"{object_name}_link\">
    <inertial>
      <mass value=\"1.0\"/>
      <origin xyz=\"0 0 0\"/>
      <inertia ixx=\"0.01\" ixy=\"0\" ixz=\"0\" iyy=\"0.01\" iyz=\"0\" izz=\"0.01\"/>
    </inertial>
    <visual>
      <origin rpy=\"0 0 0\" xyz=\"0 0 0\"/>
      <geometry>
        <mesh filename=\"{object_name}.obj\" scale=\"1.0 1.0 1.0\"/>
      </geometry>
      <material name=\"mat\">
        <color rgba=\"0.7 0.8 0.9 0.7\"/>
      </material>
    </visual>
    <collision name=\"{object_name}\">
      <origin rpy=\"0 0 0\" xyz=\"0 0 0\"/>
      <geometry>
        <mesh filename=\"{object_name}.obj\" scale=\"1.0 1.0 1.0\"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
    path.write_text(text, encoding="utf-8")


def sync_retarget_object_model(object_key: str, core4d_root: Path, retarget_root: Path) -> Path:
    meta = OBJECT_META[object_key]
    object_name = str(meta["object_name"])
    src = core4d_root / "object_models" / str(meta["mesh_rel"])
    if not src.is_file():
        raise FileNotFoundError(src)
    out_dir = retarget_root / "models" / object_name
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out_dir / f"{object_name}.obj")
    write_simple_urdf(out_dir / f"{object_name}.urdf", object_name)
    return out_dir


def patch_g1_object_xml(path: Path, object_key: str) -> None:
    meta = OBJECT_META[object_key]
    object_name = str(meta["object_name"])
    old_mesh_name = ""
    tree = ET.parse(path)
    root = tree.getroot()
    for mesh in root.iter("mesh"):
        file_attr = mesh.get("file", "")
        if file_attr.startswith("../../") and file_attr.endswith(".obj"):
            old_mesh_name = mesh.get("name", "")
            mesh.set("name", f"{object_name}_mesh")
            mesh.set("file", f"../../{object_name}/{object_name}.obj")
            mesh.set("scale", "1 1 1")
            break
    if not old_mesh_name:
        raise ValueError(f"No object mesh asset found in {path}")

    for body in root.iter("body"):
        object_geom = None
        for child in body:
            if child.tag == "geom" and child.get("mesh") == old_mesh_name:
                object_geom = child
                break
        if object_geom is None:
            continue
        body.set("name", f"{object_name}_link")
        object_geom.set("name", object_name)
        object_geom.set("mesh", f"{object_name}_mesh")
        object_geom.set("rgba", str(meta["rgba"]).replace(" 1", " 0.7"))
        break
    else:
        raise ValueError(f"No object geom referencing {old_mesh_name} found in {path}")

    tree.write(path, encoding="unicode")


def create_g1_object_xml(object_key: str, retarget_root: Path, force: bool) -> Path:
    meta = OBJECT_META[object_key]
    object_name = str(meta["object_name"])
    base_object = str(meta["base_g1_object"])
    base = retarget_root / "models/g1" / f"g1_29dof_w_{base_object}.xml"
    out = retarget_root / "models/g1" / f"g1_29dof_w_{object_name}.xml"
    if not base.is_file():
        raise FileNotFoundError(base)
    if force or not out.is_file():
        shutil.copy2(base, out)
    patch_g1_object_xml(out, object_key)
    mujoco.MjModel.from_xml_path(str(out))
    return out


def ensure_asset(object_key: str, core4d_root: Path, asset_root: Path) -> tuple[Path, np.ndarray]:
    meta = OBJECT_META[object_key]
    src = core4d_root / "object_models" / str(meta["mesh_rel"])
    if not src.is_file():
        raise FileNotFoundError(src)
    dst_dir = asset_root / object_key
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{object_key}_m.obj"
    if not dst.is_file() or src.stat().st_size != dst.stat().st_size:
        shutil.copy2(src, dst)
    return dst, parse_obj_extents(dst)


def patch_scene(scene_path: Path, object_key: str, extents: np.ndarray) -> dict[str, Any]:
    meta = OBJECT_META[object_key]
    mass = float(meta["mass_kg"])
    half_extents = extents / 2.0
    inertia = box_inertia(mass, extents)

    tree = ET.parse(scene_path)
    root = tree.getroot()

    old_mesh_name = object_visual_mesh_name(root)
    mesh_elem = find_named(root, "mesh", old_mesh_name)
    mesh_elem.set("name", object_key)
    mesh_elem.set(
        "file",
        f"../../../../../example_datasets/processed/core4d/assets/objects/{object_key}/{object_key}_m.obj",
    )
    mesh_elem.set("scale", "1 1 1")

    material_name = f"{object_key}_material"
    material = None
    for elem in root.iter("material"):
        if elem.get("name") == find_named(root, "geom", "object_visual").get("material"):
            material = elem
            break
    if material is None:
        material = ET.SubElement(find_named(root, "asset", None), "material")
    material.set("name", material_name)
    material.set("rgba", str(meta["rgba"]))

    visual = find_named(root, "geom", "object_visual")
    visual.set("mesh", object_key)
    visual.set("material", material_name)

    collision = find_named(root, "geom", "object_collision")
    collision.set("size", fmt(half_extents, 6))
    collision.set("rgba", str(meta["collision_rgba"]))

    body = find_named(root, "body", "object")
    inertial = None
    for child in body:
        if child.tag == "inertial":
            inertial = child
            break
    if inertial is None:
        inertial = ET.SubElement(body, "inertial")
    inertial.set("pos", "0 0 0")
    inertial.set("mass", f"{mass:.3f}")
    inertial.set("diaginertia", fmt(inertia, 8))

    tree.write(scene_path, encoding="unicode")
    return {
        "object_key": object_key,
        "mass_kg": mass,
        "extents_m": extents.tolist(),
        "half_extents_m": half_extents.tolist(),
        "diaginertia": inertia.tolist(),
    }


def create_template(
    task: str,
    core4d_root: Path,
    task_root: Path,
    asset_root: Path,
    retarget_root: Path,
    force: bool,
) -> dict[str, Any]:
    parts = task.split("_")
    if len(parts) != 2 or parts[1] not in {"person1", "person2"}:
        raise ValueError(f"Expected task like box026_person2, got {task}")
    object_key, person = parts
    if object_key not in OBJECT_META:
        raise KeyError(f"Unsupported object key {object_key}")

    asset_path, extents = ensure_asset(object_key, core4d_root, asset_root)
    retarget_model_dir = sync_retarget_object_model(object_key, core4d_root, retarget_root)
    g1_xml_path = create_g1_object_xml(object_key, retarget_root, force)
    base = task_root / str(OBJECT_META[object_key]["base_template"]) / "scene.xml"
    if not base.is_file():
        raise FileNotFoundError(base)
    out_dir = task_root / task
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_path = out_dir / "scene.xml"
    if force or not scene_path.is_file():
        shutil.copy2(base, scene_path)
    patch = patch_scene(scene_path, object_key, extents)
    model = mujoco.MjModel.from_xml_path(str(scene_path))

    info = {
        "task": task,
        "object_key": object_key,
        "person": person,
        "scene_path": str(scene_path),
        "base_template": str(OBJECT_META[object_key]["base_template"]),
        "base_scene": str(base),
        "asset_path": str(asset_path),
        "retarget_model_dir": str(retarget_model_dir),
        "retarget_g1_xml_path": str(g1_xml_path),
        "force": force,
        "mujoco_load_ok": True,
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        **patch,
    }
    (out_dir / "task_info.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, help="Comma-separated source scene tasks, e.g. box026_person2")
    parser.add_argument("--core4d-root", type=Path, default=DEFAULT_CORE4D_ROOT)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--holosoma-retarget-root", type=Path, default=DEFAULT_HOLOSOMA_RETARGET_ROOT)
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    rows = [
        create_template(
            task,
            args.core4d_root,
            args.task_root,
            args.asset_root,
            args.holosoma_retarget_root,
            args.force,
        )
        for task in tasks
    ]
    out = args.v2_root / "results/template_preflight/source_scene_templates_created.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for row in rows:
        print(f"{row['task']}: scene OK nq={row['nq']} nv={row['nv']} nu={row['nu']} mass={row['mass_kg']}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
