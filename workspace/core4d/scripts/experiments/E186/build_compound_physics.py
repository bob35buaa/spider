#!/usr/bin/env python3
"""Build and validate the frozen E186 compound-convex physics sidecars."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import os
import shutil
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[5]
RESULTS = REPO / "workspace/core4d/results/E186"
KEEP22 = RESULTS / "s5_handoff/keep22_authority_manifest.tsv"
COLLIDER_LOCK = RESULTS / "s0_environment/collider_lock.json"
E178_MANIFEST = (
    REPO / "workspace/core4d/results/E178/s6_downstream/manifests/"
    "semantic_bucket_full_manifest.tsv"
)
GRID_ROOT = RESULTS / "s1_canonical_grid_sdf_v4"
OUTPUT_ROOT = RESULTS / "s2_compound_physics"
OVERRIDE_DIR = REPO / "examples/config/override"
SCENE_NAME = "scene_act_E186_coacd_compound"
METHOD_ID = "E186_object_specific_coacd_compound_grid_sdf_r1"
HAND_GEOMS = ("lh", "rh")
LOWER_BODY_GEOMS = (
    "left_hip_collision",
    "right_hip_collision",
    "left_thigh_collision",
    "right_thigh_collision",
    "left_shin_collision",
    "right_shin_collision",
    "left_linkage_brace_collision",
    "right_linkage_brace_collision",
    "lf0",
    "lf1",
    "lf2",
    "lf3",
    "rf0",
    "rf1",
    "rf2",
    "rf3",
)
ROBOT_OBJECT_GEOMS = HAND_GEOMS + LOWER_BODY_GEOMS
EXPECTED_CASES_BY_OBJECT = {"bucket003": 5, "bucket004": 4, "bucket007": 13}


def now() -> str:
    """Return an ISO timestamp for persisted evidence."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def repo_path(value: str | Path) -> Path:
    """Resolve a repository-relative path."""
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def relative(path: str | Path) -> str:
    """Serialize repository paths without host-specific prefixes."""
    lexical = Path(path)
    if not lexical.is_absolute():
        return str(lexical)
    lexical = lexical.absolute()
    try:
        return str(lexical.relative_to(REPO.absolute()))
    except ValueError:
        return str(lexical)


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a tab-separated manifest."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a stable tab-separated manifest."""
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


def write_json(path: Path, payload: Any) -> None:
    """Write deterministic JSON evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def find_object_body(root: ET.Element) -> ET.Element:
    """Return the unique named object body."""
    matches = [body for body in root.iter("body") if body.get("name") == "object"]
    if len(matches) != 1:
        raise ValueError(f"expected one object body, got {len(matches)}")
    return matches[0]


def is_object_collision_name(name: str) -> bool:
    """Recognize legacy and E186 object collision geom names."""
    return name == "object_collision" or name.startswith("object_collision_")


def object_collision_geoms(object_body: ET.Element) -> list[ET.Element]:
    """Return direct child collision geoms in XML order."""
    return [
        geom
        for geom in object_body.findall("geom")
        if is_object_collision_name(geom.get("name", ""))
    ]


def axis_stripped_signature(root: ET.Element) -> tuple[Any, ...]:
    """Compare scene semantics outside the intended collision/pair/asset axes."""
    clone = copy.deepcopy(root)
    for body in clone.iter("body"):
        for geom in list(body.findall("geom")):
            if is_object_collision_name(geom.get("name", "")):
                body.remove(geom)
    asset = clone.find("asset")
    if asset is not None:
        for mesh in list(asset.findall("mesh")):
            if mesh.get("name", "").startswith("E186_object_part_"):
                asset.remove(mesh)
    contact = clone.find("contact")
    if contact is not None:
        for pair in list(contact.findall("pair")):
            if is_object_collision_name(
                pair.get("geom1", "")
            ) or is_object_collision_name(pair.get("geom2", "")):
                contact.remove(pair)

    def signature(element: ET.Element) -> tuple[Any, ...]:
        return (
            element.tag,
            tuple(sorted(element.attrib.items())),
            (element.text or "").strip(),
            tuple(signature(child) for child in element),
        )

    return signature(clone)


def mesh_file_reference(scene: Path, asset_path: Path, root: ET.Element) -> str:
    """Create a portable mesh file path under the scene compiler meshdir."""
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    mesh_root = Path(meshdir)
    if not mesh_root.is_absolute():
        mesh_root = scene.parent / mesh_root
    return os.path.relpath(asset_path.resolve(), mesh_root.resolve())


def replace_collision_axis(
    root: ET.Element,
    source_scene: Path,
    collider: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Replace E178 boxes and their pairs with the frozen convex-part union."""
    object_body = find_object_body(root)
    previous = object_collision_geoms(object_body)
    if not previous:
        raise ValueError("source scene has no object collision geoms")
    previous_names = {geom.get("name", "") for geom in previous}
    template_geom = dict(previous[0].attrib)
    children = list(object_body)
    insert_at = min(children.index(geom) for geom in previous)
    for geom in previous:
        object_body.remove(geom)

    asset_section = root.find("asset")
    if asset_section is None:
        asset_section = ET.Element("asset")
        root.insert(0, asset_section)
    task_asset_dir = (
        source_scene.parent
        / "e186_compound_assets"
        / str(collider["object_key"])
        / str(collider["candidate_id"])
    )
    task_asset_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    asset_rows: list[dict[str, Any]] = []
    for index, part in enumerate(collider["ordered_parts"]):
        source_part = repo_path(part["path"])
        if sha256(source_part) != part["sha256"]:
            raise RuntimeError(f"frozen part SHA mismatch: {source_part}")
        target_part = task_asset_dir / f"part_{index:03d}.obj"
        shutil.copy2(source_part, target_part)
        if sha256(target_part) != part["sha256"]:
            raise RuntimeError(f"copied part SHA mismatch: {target_part}")
        mesh_name = f"E186_object_part_{index:03d}"
        geom_name = f"object_collision_{index:03d}"
        ET.SubElement(
            asset_section,
            "mesh",
            {
                "name": mesh_name,
                "file": mesh_file_reference(source_scene, target_part, root),
            },
        )
        attributes = dict(template_geom)
        for key in ("pos", "quat", "euler", "axisangle", "size", "fromto"):
            attributes.pop(key, None)
        attributes.update({"name": geom_name, "type": "mesh", "mesh": mesh_name})
        object_body.insert(insert_at + index, ET.Element("geom", attributes))
        names.append(geom_name)
        asset_rows.append(
            {
                "part_index": index,
                "geom_name": geom_name,
                "mesh_name": mesh_name,
                "source_part": relative(source_part),
                "task_asset": relative(target_part),
                "sha256": part["sha256"],
                "vertex_count": part["vertex_count"],
                "face_count": part["face_count"],
            }
        )

    contact = root.find("contact")
    if contact is None:
        raise ValueError("source scene has no contact section")
    pair_templates: dict[str, dict[str, str]] = {}
    floor_template: dict[str, str] | None = None
    for pair in list(contact.findall("pair")):
        first = pair.get("geom1", "")
        second = pair.get("geom2", "")
        touches_previous = first in previous_names or second in previous_names
        if not touches_previous:
            continue
        other = second if first in previous_names else first
        if other in ROBOT_OBJECT_GEOMS and other not in pair_templates:
            pair_templates[other] = dict(pair.attrib)
        if other == "floor" and floor_template is None:
            floor_template = dict(pair.attrib)
        contact.remove(pair)
    missing_templates = sorted(set(ROBOT_OBJECT_GEOMS) - set(pair_templates))
    if missing_templates or floor_template is None:
        raise ValueError(
            f"missing E178 pair templates: robot={missing_templates}, "
            f"floor={floor_template is None}"
        )

    for index, object_name in enumerate(names):
        floor_attributes = dict(floor_template)
        floor_attributes.update(
            {
                "name": f"E186_object_floor_{index:03d}",
                "geom1": object_name,
                "geom2": "floor",
            }
        )
        contact.append(ET.Element("pair", floor_attributes))
        for robot_name in ROBOT_OBJECT_GEOMS:
            attributes = dict(pair_templates[robot_name])
            attributes.update(
                {
                    "name": f"E186_{robot_name}_obj{index:03d}",
                    "geom1": robot_name,
                    "geom2": object_name,
                }
            )
            contact.append(ET.Element("pair", attributes))
    return names, asset_rows


def compiled_contract(
    source_scene: Path,
    effective_scene: Path,
    expected_names: list[str],
) -> dict[str, Any]:
    """Compile CPU/MJWarp and verify pair, inertia, and friction contracts."""
    started = time.perf_counter()
    source_model = mujoco.MjModel.from_xml_path(str(source_scene))
    model = mujoco.MjModel.from_xml_path(str(effective_scene))
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    source_object_body_id = mujoco.mj_name2id(
        source_model, mujoco.mjtObj.mjOBJ_BODY, "object"
    )
    if object_body_id < 0 or source_object_body_id < 0:
        raise ValueError("compiled scene missing object body")
    np.testing.assert_array_equal(
        model.body_mass[object_body_id], source_model.body_mass[source_object_body_id]
    )
    np.testing.assert_array_equal(
        model.body_inertia[object_body_id],
        source_model.body_inertia[source_object_body_id],
    )

    object_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in expected_names
    ]
    if any(gid < 0 for gid in object_ids):
        raise ValueError("compiled scene missing compound object geoms")
    if any(
        int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_MESH)
        for gid in object_ids
    ):
        raise ValueError("compiled compound contains a non-mesh geom")
    source_primary = mujoco.mj_name2id(
        source_model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision"
    )
    for gid in object_ids:
        np.testing.assert_array_equal(
            model.geom_friction[gid], source_model.geom_friction[source_primary]
        )
        if int(model.geom_condim[gid]) != int(source_model.geom_condim[source_primary]):
            raise AssertionError("object geom condim drift")

    robot_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ROBOT_OBJECT_GEOMS
    }
    expected_pairs = {
        (robot_ids[robot], object_id)
        for robot in ROBOT_OBJECT_GEOMS
        for object_id in object_ids
    }
    observed_pairs: list[tuple[int, int]] = []
    for pair_id in range(model.npair):
        first = int(model.pair_geom1[pair_id])
        second = int(model.pair_geom2[pair_id])
        normalized = (first, second)
        if first in object_ids and second in robot_ids.values():
            normalized = (second, first)
        if normalized in expected_pairs:
            observed_pairs.append(normalized)
    counts = Counter(observed_pairs)
    if set(counts) != expected_pairs or any(count != 1 for count in counts.values()):
        raise AssertionError("compiled robot-object pair matrix is not exact 18×K")

    model_wp = mjwarp.put_model(model)
    warp_contract = {
        "warp_ngeom": int(model_wp.ngeom),
        "warp_nmesh": int(model_wp.nmesh),
        "warp_npair": int(model_wp.npair),
    }
    del model_wp
    gc.collect()
    wp.synchronize()
    return {
        "cpu_ngeom": int(model.ngeom),
        "cpu_nmesh": int(model.nmesh),
        "cpu_npair": int(model.npair),
        "object_geom_count": len(object_ids),
        "robot_object_pair_count": len(observed_pairs),
        "expected_robot_object_pair_count": len(expected_pairs),
        "object_mass": float(model.body_mass[object_body_id]),
        "object_inertia": " ".join(
            f"{value:.17g}" for value in model.body_inertia[object_body_id]
        ),
        "compile_seconds": time.perf_counter() - started,
        **warp_contract,
    }


def write_override(
    keep: dict[str, str],
    e178: dict[str, str],
    grid_manifest_path: Path,
    grid_manifest: dict[str, Any],
) -> Path:
    """Write the explicit E186 config delta over the matching E178 case."""
    output = OVERRIDE_DIR / f"core4d_E186_{keep['case_id']}_coacdCompound.yaml"
    payload = {
        "defaults": [e178["override_id"], "_self_"],
        "scene_name": SCENE_NAME,
        "object_collision_sdf_mode": "compound",
        "object_collision_sdf_batch_groups": True,
        "object_distance_backend": "grid_sdf",
        "object_distance_manifest": relative(grid_manifest_path),
        "object_distance_expected_asset_sha256": keep["collider_asset_sha256"],
        "object_distance_error_bound_m": float(
            grid_manifest["validation"]["epsilon_grid_m"]
        ),
    }
    header = (
        "# @package _global_\n# Auto-generated by E186 build_compound_physics.py.\n"
    )
    output.write_text(
        header + yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    return output


def audit_override(e178_override_id: str, override: Path) -> None:
    """Assert the Hydra config changes only the pre-registered E186 axes."""
    config_dir = (REPO / "examples/config").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        baseline = OmegaConf.to_container(
            compose(config_name="default", overrides=[f"+override={e178_override_id}"]),
            resolve=True,
        )
        candidate = OmegaConf.to_container(
            compose(config_name="default", overrides=[f"+override={override.stem}"]),
            resolve=True,
        )
    assert isinstance(baseline, dict) and isinstance(candidate, dict)
    allowed = {
        "scene_name",
        "object_collision_sdf_mode",
        "object_collision_sdf_batch_groups",
        "object_distance_backend",
        "object_distance_manifest",
        "object_distance_expected_asset_sha256",
        "object_distance_error_bound_m",
    }
    drift = [
        key
        for key in sorted(set(baseline) | set(candidate))
        if key not in allowed and baseline.get(key) != candidate.get(key)
    ]
    if drift:
        raise AssertionError(f"override drift outside E186 axes: {drift}")


def build_case(
    keep: dict[str, str],
    e178: dict[str, str],
    collider: dict[str, Any],
    *,
    overwrite: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build one sidecar and all associated compile evidence."""
    case_id = keep["case_id"]
    object_key = keep["object_key"]
    source_scene = repo_path(keep["source_e178_scene_act"])
    if sha256(source_scene) != keep["source_e178_scene_sha256"]:
        raise RuntimeError(f"E178 source scene SHA mismatch: {case_id}")
    source_root = ET.parse(source_scene).getroot()
    root = copy.deepcopy(source_root)
    names, asset_rows = replace_collision_axis(root, source_scene, collider)
    if len(names) != int(collider["actual_hulls"]):
        raise AssertionError(f"{case_id}: actual hull count drift")
    if axis_stripped_signature(source_root) != axis_stripped_signature(root):
        raise AssertionError(f"{case_id}: scene drift outside E186 axes")

    effective_scene = source_scene.with_name(f"{SCENE_NAME}.xml")
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    if effective_scene.exists() and not overwrite:
        raise FileExistsError(effective_scene)
    tree.write(effective_scene, encoding="utf-8", xml_declaration=True)
    contract = compiled_contract(source_scene, effective_scene, names)

    grid_manifest_path = GRID_ROOT / object_key / "manifest.json"
    grid_manifest = json.loads(grid_manifest_path.read_text(encoding="utf-8"))
    if (
        grid_manifest["source"]["candidate_asset_sha256"]
        != keep["collider_asset_sha256"]
    ):
        raise RuntimeError(f"{case_id}: grid/collider SHA mismatch")
    override = write_override(keep, e178, grid_manifest_path, grid_manifest)
    audit_override(e178["override_id"], override)

    snapshot = OUTPUT_ROOT / "scene_snapshot" / case_id
    snapshot.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_scene, snapshot / source_scene.name)
    shutil.copy2(effective_scene, snapshot / effective_scene.name)
    snapshot_assets = snapshot / "compound_assets"
    snapshot_assets.mkdir(exist_ok=True)
    for asset in asset_rows:
        shutil.copy2(
            repo_path(asset["task_asset"]),
            snapshot_assets / Path(asset["task_asset"]).name,
        )

    row = {
        **keep,
        "spider_method_id": METHOD_ID,
        "source_e178_override_id": e178["override_id"],
        "source_e178_override_sha256": e178["override_sha256"],
        "scene_act": relative(effective_scene),
        "scene_name": SCENE_NAME,
        "effective_scene_sha256": sha256(effective_scene),
        "override_id": override.stem,
        "override_path": relative(override),
        "override_sha256": sha256(override),
        "object_collision_sdf_mode": "compound",
        "object_distance_backend": "grid_sdf",
        "object_distance_manifest": relative(grid_manifest_path),
        "object_distance_manifest_sha256": sha256(grid_manifest_path),
        "object_distance_error_bound_m": grid_manifest["validation"]["epsilon_grid_m"],
        "physics_status": "COMPOUND_CPU_MJWARP_PASS",
        "rg_status": "GRID_BACKEND_READY_AUDIT_PENDING",
        "updated_at": now(),
        **contract,
    }
    return row, [
        {"case_id": case_id, "object_key": object_key, **asset} for asset in asset_rows
    ]


def build(*, overwrite: bool) -> dict[str, Any]:
    """Build all frozen keep22 sidecars and aggregate their contracts."""
    keep_rows = read_tsv(KEEP22)
    if len(keep_rows) != 22 or len({row["case_id"] for row in keep_rows}) != 22:
        raise ValueError("keep22 authority is not 22 unique rows")
    distribution = Counter(row["object_key"] for row in keep_rows)
    if dict(distribution) != EXPECTED_CASES_BY_OBJECT:
        raise ValueError(f"keep22 object distribution drift: {dict(distribution)}")
    lock = json.loads(COLLIDER_LOCK.read_text(encoding="utf-8"))
    if lock.get("status") != "FROZEN":
        raise RuntimeError("collider lock is not frozen")
    e178_by_case = {row["case_id"]: row for row in read_tsv(E178_MANIFEST)}

    scene_rows: list[dict[str, Any]] = []
    asset_rows: list[dict[str, Any]] = []
    for keep in keep_rows:
        case_id = keep["case_id"]
        if case_id not in e178_by_case:
            raise KeyError(f"keep case missing from E178: {case_id}")
        collider = lock["objects"][keep["object_key"]]
        row, assets = build_case(
            keep, e178_by_case[case_id], collider, overwrite=overwrite
        )
        scene_rows.append(row)
        asset_rows.extend(assets)
        print(
            f"PASS {case_id} K={row['object_geom_count']} "
            f"pairs={row['robot_object_pair_count']} "
            f"compile={row['compile_seconds']:.3f}s"
        )

    write_tsv(OUTPUT_ROOT / "compound_scene_manifest.tsv", scene_rows)
    write_tsv(OUTPUT_ROOT / "compound_asset_manifest.tsv", asset_rows)
    aggregate = {
        "schema": "e186_compound_physics_v1",
        "experiment_id": "E186",
        "status": "PASS",
        "generated_at": now(),
        "case_count": len(scene_rows),
        "object_counts": dict(distribution),
        "hulls_by_object": {
            key: lock["objects"][key]["actual_hulls"] for key in distribution
        },
        "robot_geom_count": len(ROBOT_OBJECT_GEOMS),
        "pair_counts_by_object": {
            key: len(ROBOT_OBJECT_GEOMS) * int(lock["objects"][key]["actual_hulls"])
            for key in distribution
        },
        "cpu_compile_pass": len(scene_rows),
        "mjwarp_compile_pass": len(scene_rows),
        "collider_set_sha256": lock["collider_set_sha256"],
        "scene_manifest_sha256": sha256(OUTPUT_ROOT / "compound_scene_manifest.tsv"),
        "asset_manifest_sha256": sha256(OUTPUT_ROOT / "compound_asset_manifest.tsv"),
    }
    write_json(OUTPUT_ROOT / "aggregate.json", aggregate)
    return aggregate


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(overwrite=args.overwrite), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
