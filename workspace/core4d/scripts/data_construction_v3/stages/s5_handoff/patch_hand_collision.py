#!/usr/bin/env python3
"""Patch CEM scenes for robot hand-collision variants.

The adapter never overwrites the source scene. For `rubber_hull` it writes a
sidecar scene whose `lh`/`rh` geoms reuse the existing rubber-hand visual meshes
as convex mesh collision geoms.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco

SCRIPT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "data_construction_v3")
for _path in (SCRIPT_ROOT / "lib", SCRIPT_ROOT / "state", SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common import (  # noqa: E402
    DEFAULT_HAND_COLLISION_VARIANT_ID,
    HAND_COLLISION_VARIANT_SPECS,
    HAND_COLLISION_VARIANTS,
    SCHEMA_VERSION,
    find_spider_repo,
    json_dumps,
    timestamp,
)
from common import sha256_file, stable_params_json, write_json, write_tsv  # noqa: E402


FIELDS = [
    "case_id",
    "hand_collision_variant_id",
    "status",
    "base_scene_act",
    "patched_scene_act",
    "installed_scene_act",
    "scene_name",
    "base_scene_sha256",
    "patched_scene_sha256",
    "lh_type",
    "rh_type",
    "lh_rbound",
    "rh_rbound",
    "patch_params_json",
    "schema_version",
    "updated_at",
    "notes",
]


def rel(path: Path, repo: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo.resolve()))
    except ValueError:
        return str(path.resolve())


def indent(elem: ET.Element) -> None:
    ET.indent(elem, space="  ")


def remove_hand_collision_default(root: ET.Element) -> bool:
    removed = False
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "default" and child.get("class") == "hand_collision":
                parent.remove(child)
                removed = True
    return removed


def patch_mesh_assets(root: ET.Element, spec: dict[str, str]) -> None:
    asset = root.find("asset")
    if asset is None:
        raise ValueError("scene has no <asset> section")
    for mesh_name in (spec["left_mesh"], spec["right_mesh"]):
        found = False
        for mesh in asset.findall("mesh"):
            if mesh.get("name") == mesh_name:
                mesh.set("maxhullvert", spec["maxhullvert"])
                found = True
                break
        if not found:
            raise ValueError(f"mesh asset {mesh_name!r} not found")


def replace_hand_geom(root: ET.Element, name: str, mesh: str, pos: str) -> None:
    for parent in root.iter():
        for idx, child in enumerate(list(parent)):
            if child.tag == "geom" and child.get("name") == name:
                new_geom = ET.Element(
                    "geom",
                    {
                        "name": name,
                        "class": "collision",
                        "type": "mesh",
                        "mesh": mesh,
                        "pos": pos,
                    },
                )
                parent.remove(child)
                parent.insert(idx, new_geom)
                return
    raise ValueError(f"hand geom {name!r} not found")


def validate_scene(path: Path) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(path))
    out: dict[str, Any] = {}
    for name in ("lh", "rh"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            raise ValueError(f"{name} geom missing after patch")
        geom_type = int(model.geom_type[gid])
        type_name = mujoco.mjtGeom(geom_type).name.replace("mjGEOM_", "").lower()
        rbound = float(model.geom_rbound[gid])
        out[f"{name}_type"] = type_name
        out[f"{name}_rbound"] = rbound
        if rbound <= 0.0:
            raise ValueError(f"{name} rbound is not positive: {rbound}")
    return out


def patch_scene(
    *,
    base_scene_act: Path,
    out_dir: Path,
    case_id: str,
    hand_collision_variant_id: str,
    scene_name: str,
    install_dir: Path | None,
    repo: Path,
) -> dict[str, Any]:
    if hand_collision_variant_id not in HAND_COLLISION_VARIANTS:
        raise ValueError(f"unsupported hand_collision_variant_id={hand_collision_variant_id}")
    if not base_scene_act.is_file():
        raise FileNotFoundError(base_scene_act)

    spec = HAND_COLLISION_VARIANT_SPECS[hand_collision_variant_id]
    out_dir.mkdir(parents=True, exist_ok=True)
    patched_scene = out_dir / f"{scene_name}.xml"
    installed_scene = ""
    notes = []

    if hand_collision_variant_id == DEFAULT_HAND_COLLISION_VARIANT_ID:
        shutil.copy2(base_scene_act, patched_scene)
        notes.append("noop sphere5cm copy")
    elif hand_collision_variant_id == "rubber_hull":
        tree = ET.parse(base_scene_act)
        root = tree.getroot()
        removed_default = remove_hand_collision_default(root)
        patch_mesh_assets(root, spec)
        replace_hand_geom(root, "lh", spec["left_mesh"], spec["left_pos"])
        replace_hand_geom(root, "rh", spec["right_mesh"], spec["right_pos"])
        indent(root)
        tree.write(patched_scene, encoding="unicode", xml_declaration=False)
        notes.append(f"removed_hand_collision_default={removed_default}")
    else:
        raise ValueError(f"variant not implemented: {hand_collision_variant_id}")

    validate_path = patched_scene
    if install_dir is not None:
        install_dir.mkdir(parents=True, exist_ok=True)
        installed_path = install_dir / f"{scene_name}.xml"
        shutil.copy2(patched_scene, installed_path)
        installed_scene = rel(installed_path, repo)
        validate_path = installed_path

    validation = validate_scene(validate_path)
    if hand_collision_variant_id == "rubber_hull":
        for side in ("lh", "rh"):
            if validation[f"{side}_type"] != "mesh":
                raise ValueError(f"{side} is not mesh after rubber_hull patch: {validation[f'{side}_type']}")

    params = {
        "variant_spec": spec,
        "scene_name": scene_name,
        "install_dir": str(install_dir) if install_dir else "",
    }
    row = {
        "case_id": case_id,
        "hand_collision_variant_id": hand_collision_variant_id,
        "status": "pass",
        "base_scene_act": rel(base_scene_act, repo),
        "patched_scene_act": rel(patched_scene, repo),
        "installed_scene_act": installed_scene,
        "scene_name": scene_name,
        "base_scene_sha256": sha256_file(base_scene_act),
        "patched_scene_sha256": sha256_file(patched_scene),
        "lh_type": validation["lh_type"],
        "rh_type": validation["rh_type"],
        "lh_rbound": f"{validation['lh_rbound']:.8g}",
        "rh_rbound": f"{validation['rh_rbound']:.8g}",
        "patch_params_json": stable_params_json(params),
        "schema_version": SCHEMA_VERSION,
        "updated_at": timestamp(),
        "notes": ";".join(notes),
    }
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-scene-act", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--hand-collision-variant-id", default="rubber_hull", choices=sorted(HAND_COLLISION_VARIANTS))
    parser.add_argument("--scene-name", default="")
    parser.add_argument("--install-dir", type=Path, default=None)
    parser.add_argument("--spider-repo", type=Path, default=None)
    args = parser.parse_args()

    repo = (args.spider_repo or find_spider_repo()).resolve()
    scene_name = args.scene_name or f"scene_act_{args.hand_collision_variant_id}"
    row = patch_scene(
        base_scene_act=(args.base_scene_act if args.base_scene_act.is_absolute() else repo / args.base_scene_act).resolve(),
        out_dir=(args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir).resolve(),
        case_id=args.case_id,
        hand_collision_variant_id=args.hand_collision_variant_id,
        scene_name=scene_name,
        install_dir=(args.install_dir if args.install_dir is None or args.install_dir.is_absolute() else repo / args.install_dir),
        repo=repo,
    )
    write_tsv((args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir) / "hand_collision_adapter_manifest.tsv", [row], FIELDS)
    write_json((args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir) / "hand_collision_adapter_manifest.json", [row])
    print(json_dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
