#!/usr/bin/env python3
"""Rebuild E103 canonical CORE4D box source templates from a clean base scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import trimesh


REPO = Path(__file__).resolve().parents[4]
HUMANOID_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
ASSET_ROOT = REPO / "example_datasets/processed/core4d/assets/objects"
DEFAULT_RAW_ROOT = Path(
    "/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real"
)
BASE_TASK = "box023_person1"
BASE_SCENE = HUMANOID_ROOT / BASE_TASK / "scene.xml"
SNAPSHOT_ROOT = REPO / "workspace/core4d/results/E103/pre_rebuild_scene_snapshot"
VALIDATION_PATH = REPO / "workspace/core4d/results/E103/rebuilt_source_template_validation.tsv"


@dataclass(frozen=True)
class ObjectSpec:
    key: str
    object_name: str
    mesh_rel: str
    material_rgba: str
    collision_rgba: str


OBJECTS = {
    "box021": ObjectSpec(
        key="box021",
        object_name="Box021",
        mesh_rel="box/box021_m.obj",
        material_rgba="0.40 0.50 0.60 1",
        collision_rgba="0.40 0.50 0.60 0.3",
    ),
    "box022": ObjectSpec(
        key="box022",
        object_name="Box022",
        mesh_rel="box/box022_m.obj",
        material_rgba="0.46 0.56 0.68 1",
        collision_rgba="0.46 0.56 0.68 0.3",
    ),
    "box026": ObjectSpec(
        key="box026",
        object_name="Box026",
        mesh_rel="box/box026_m.obj",
        material_rgba="0.35 0.52 0.74 1",
        collision_rgba="0.35 0.52 0.74 0.3",
    ),
}

TASKS = {
    f"{key}_{person}": (spec, person)
    for key, spec in OBJECTS.items()
    for person in ("person1", "person2")
}


def fmt(values: np.ndarray | list[float], digits: int = 6) -> str:
    return " ".join(f"{float(v):.{digits}f}".rstrip("0").rstrip(".") for v in values)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_asset(spec: ObjectSpec, raw_root: Path, apply: bool) -> tuple[Path, Path, bool]:
    dest = ASSET_ROOT / spec.key / f"{spec.key}_m.obj"
    src = raw_root / "object_models" / spec.mesh_rel
    if not src.exists():
        raise FileNotFoundError(f"raw object mesh missing: {src}")
    copied = False
    if not dest.exists() and apply:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied = True
    return src, dest, copied


def mesh_half_extents(path: Path) -> np.ndarray:
    mesh = trimesh.load(path, force="mesh")
    if mesh.is_empty:
        raise ValueError(f"empty mesh: {path}")
    return np.asarray(mesh.bounding_box.extents, dtype=float) / 2.0


def box_inertia(mass: float, half_extents: np.ndarray) -> np.ndarray:
    a, b, c = half_extents * 2.0
    return np.array(
        [
            mass / 12.0 * (b * b + c * c),
            mass / 12.0 * (a * a + c * c),
            mass / 12.0 * (a * a + b * b),
        ],
        dtype=float,
    )


def read_base_pose(base_text: str) -> tuple[str, str | None]:
    match = re.search(r'<body name="object"([^>]*)>', base_text)
    if not match:
        raise ValueError("base object body not found")
    attrs = match.group(1)
    pos_match = re.search(r'pos="([^"]+)"', attrs)
    quat_match = re.search(r'quat="([^"]+)"', attrs)
    if not pos_match:
        raise ValueError("base object pos not found")
    return pos_match.group(1), quat_match.group(1) if quat_match else None


def snapshot_task(task: str) -> Path:
    src_dir = HUMANOID_ROOT / task
    dst_dir = SNAPSHOT_ROOT / task
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not src_dir.exists():
        marker = dst_dir / "MISSING_SOURCE_TEMPLATE.txt"
        if not marker.exists():
            marker.write_text(f"{task} did not exist before E103 rebuild.\n", encoding="utf-8")
        return dst_dir

    for src in src_dir.iterdir():
        dst = dst_dir / src.name
        if dst.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    return dst_dir


def build_scene_xml(base_text: str, spec: ObjectSpec, half_extents: np.ndarray, mass: float) -> str:
    mesh_file = (
        f'    <mesh name="{spec.key}" '
        f'file="../../../../../example_datasets/processed/core4d/assets/objects/{spec.key}/{spec.key}_m.obj" '
        f'scale="1 1 1" />'
    )
    material = f'    <material name="{spec.key}_material" rgba="{spec.material_rgba}" />'

    scene = re.sub(
        r'    <mesh name="box023" file="[^"]+" scale="[^"]+" />',
        mesh_file,
        base_text,
        count=1,
    )
    scene = re.sub(
        r'    <material name="box_material" rgba="[^"]+" />',
        material,
        scene,
        count=1,
    )

    pos, _quat = read_base_pose(base_text)
    inertia = box_inertia(mass, half_extents)
    object_body = (
        f'    <body name="object" pos="{pos}">\n'
        f'      <freejoint name="object_joint" />\n'
        f'      <inertial pos="0 0 0" mass="{mass:.3f}" diaginertia="{fmt(inertia, 8)}" />\n'
        f'      <geom name="object_visual" type="mesh" mesh="{spec.key}" '
        f'material="{spec.key}_material" group="2" contype="0" conaffinity="0" />\n'
        f'      <geom name="object_collision" type="box" size="{fmt(half_extents, 6)}" '
        f'rgba="{spec.collision_rgba}" group="3" contype="1" conaffinity="1" '
        f'friction="1 0.005 0.0001" condim="3" />\n'
        f'      <site name="trace_object" size="0.02" rgba="0 0 1 1" />\n'
        f'    </body>\n'
        f'  </worldbody>'
    )
    scene = re.sub(
        r'    <body name="object"[\s\S]*?    </body>\n  </worldbody>',
        object_body,
        scene,
        count=1,
    )
    return scene


def validate_scene(scene_path: Path) -> dict[str, object]:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    site_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        for i in range(model.nsite)
    ]
    hand_site_ids = [
        i for i, name in enumerate(site_names) if name and "contact" in name and "hand" in name
    ]
    if model.nq != 43 or model.nv != 41 or model.nu != 29:
        raise AssertionError(f"unexpected nq/nv/nu: {model.nq}/{model.nv}/{model.nu}")
    if len(hand_site_ids) != 2:
        raise AssertionError(f"unexpected hand contact sites: {hand_site_ids}")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object") < 0:
        raise AssertionError("object body missing")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision") < 0:
        raise AssertionError("object_collision geom missing")
    return {
        "mujoco_load_ok": True,
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "contact_site_ids": hand_site_ids,
        "contact_site_names": [site_names[i] for i in hand_site_ids],
    }


def write_task_info(
    task: str,
    spec: ObjectSpec,
    person: str,
    raw_mesh: Path,
    asset_mesh: Path,
    half_extents: np.ndarray,
    mass: float,
    snapshot_dir: Path,
    validation: dict[str, object],
) -> None:
    info = {
        "task": task,
        "e103_rebuilt": True,
        "base_template": BASE_TASK,
        "base_scene": str(BASE_SCENE),
        "clean_robot_inertial_source": str(BASE_SCENE),
        "snapshot_before_rebuild": str(snapshot_dir),
        "object_key": spec.key,
        "object_name": spec.object_name,
        "person": person,
        "object_model_rel": spec.mesh_rel,
        "raw_object_mesh": str(raw_mesh),
        "asset_path": str(asset_mesh),
        "mesh_sha256": sha256(asset_mesh),
        "extents_m": (half_extents * 2.0).tolist(),
        "half_extents_m": half_extents.tolist(),
        "mass_kg": mass,
        "object_mass_policy": "assumed_uniform_5kg_E103_no_real_mass_source",
        "diaginertia": box_inertia(mass, half_extents).tolist(),
        "inertia_formula": "box: Ixx=m/12*(y^2+z^2), Iyy=m/12*(x^2+z^2), Izz=m/12*(x^2+y^2)",
        "source_template_pose_policy": "neutral_placeholder_from_box023_person1; target scenes must be patched from trimmed qpos first frame",
        "scene_act_policy": "not generated for canonical source template; regenerate after target trajectory_kinematic.npz is produced",
        "runtime_artifact_policy": "old trajectory/scene_act artifacts were moved to E103 pre_rebuild snapshot or stale runtime artifact archive",
        "object_initial_pos": [float(v) for v in read_base_pose(BASE_SCENE.read_text(encoding="utf-8"))[0].split()],
        "object_initial_quat": [1.0, 0.0, 0.0, 0.0],
        **validation,
    }
    out = HUMANOID_ROOT / task / "task_info.json"
    out.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--mass", type=float, default=5.0)
    parser.add_argument("--apply", action="store_true", help="write rebuilt scene.xml/task_info.json")
    parser.add_argument("--tasks", nargs="*", default=sorted(TASKS))
    parser.add_argument("--validation-out", type=Path, default=VALIDATION_PATH)
    args = parser.parse_args()

    base_text = BASE_SCENE.read_text(encoding="utf-8")
    results: list[dict[str, object]] = []

    for task in args.tasks:
        if task not in TASKS:
            raise KeyError(f"unknown task {task}; expected one of {sorted(TASKS)}")
        spec, person = TASKS[task]
        raw_mesh, asset_mesh, asset_copied = ensure_asset(spec, args.raw_root, args.apply)
        half_extents_source = asset_mesh if asset_mesh.exists() else raw_mesh
        half_extents = mesh_half_extents(half_extents_source)
        scene_xml = build_scene_xml(base_text, spec, half_extents, args.mass)

        task_dir = HUMANOID_ROOT / task
        snapshot_dir = SNAPSHOT_ROOT / task
        validation: dict[str, object] = {"mujoco_load_ok": False}
        if args.apply:
            snapshot_dir = snapshot_task(task)
            task_dir.mkdir(parents=True, exist_ok=True)
            scene_path = task_dir / "scene.xml"
            scene_path.write_text(scene_xml, encoding="utf-8")
            validation = validate_scene(scene_path)
            write_task_info(
                task=task,
                spec=spec,
                person=person,
                raw_mesh=raw_mesh,
                asset_mesh=asset_mesh,
                half_extents=half_extents,
                mass=args.mass,
                snapshot_dir=snapshot_dir,
                validation=validation,
            )
        results.append(
            {
                "task": task,
                "object_key": spec.key,
                "person": person,
                "asset_mesh": str(asset_mesh),
                "asset_copied": str(asset_copied),
                "half_extents_m": fmt(half_extents, 8),
                "mass_kg": f"{args.mass:.3f}",
                "diaginertia": fmt(box_inertia(args.mass, half_extents), 8),
                "snapshot_dir": str(snapshot_dir),
                "mujoco_load_ok": str(validation.get("mujoco_load_ok", False)),
                "nq": str(validation.get("nq", "")),
                "nv": str(validation.get("nv", "")),
                "nu": str(validation.get("nu", "")),
                "contact_site_ids": ",".join(str(i) for i in validation.get("contact_site_ids", [])),
            }
        )

    args.validation_out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(results[0])
    with args.validation_out.open("w", encoding="utf-8") as f:
        f.write("\t".join(fields) + "\n")
        for row in results:
            f.write("\t".join(str(row[field]) for field in fields) + "\n")
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"{mode}: wrote {args.validation_out}")


if __name__ == "__main__":
    main()
