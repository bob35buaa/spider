#!/usr/bin/env python3
"""Build E169-scoped lower-body/object collision sidecars and snapshots."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from e169_common import (
    CASES,
    LOWER_BODY_GEOMS,
    NO_PHYSICS_SCENE_NAME,
    PHYSICS_SCENE_NAME,
    RESULTS,
    e168_rows,
    rel,
    repo_path,
    sha256,
    write_tsv,
)


PAIR_PREFIX = "E169_"
PAIR_SUFFIX = "_object"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def tree_signature(element: ET.Element, *, ignore_e169_pairs: bool = False) -> Any:
    children = []
    for child in element:
        if (
            ignore_e169_pairs
            and child.tag == "pair"
            and child.get("name", "").startswith(PAIR_PREFIX)
        ):
            continue
        children.append(tree_signature(child, ignore_e169_pairs=ignore_e169_pairs))
    text = (element.text or "").strip()
    return element.tag, tuple(sorted(element.attrib.items())), text, tuple(children)


def convert_reference_to_scene(qpos: np.ndarray, scene_xml: Path) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    if qpos.ndim == 3:
        qpos = qpos[:, 0, :]
    if qpos.shape[1] == model.nq:
        return qpos.astype(np.float64, copy=True)
    nq_robot = model.nq - 6
    if qpos.shape[1] < nq_robot + 7:
        raise ValueError(f"cannot convert qpos {qpos.shape} to nq={model.nq}")
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if object_id < 0:
        raise ValueError(f"scene has no object body: {scene_xml}")
    convention = "XYZ"
    meta = scene_xml.with_name("scene_act_meta.json")
    if meta.is_file():
        convention = str(json.loads(meta.read_text(encoding="utf-8")).get("euler_convention", "XYZ"))
    body_pos = model.body_pos[object_id]
    body_quat = model.body_quat[object_id]
    body_rot = R.from_quat([body_quat[1], body_quat[2], body_quat[3], body_quat[0]])
    object_pos = qpos[:, nq_robot : nq_robot + 3]
    object_quat = qpos[:, nq_robot + 3 : nq_robot + 7]
    object_slide = body_rot.inv().apply(object_pos - body_pos[np.newaxis, :])
    object_xyzw = np.column_stack(
        [object_quat[:, 1], object_quat[:, 2], object_quat[:, 3], object_quat[:, 0]]
    )
    object_euler = (body_rot.inv() * R.from_quat(object_xyzw)).as_euler(convention)
    converted = np.zeros((qpos.shape[0], model.nq), dtype=np.float64)
    converted[:, :nq_robot] = qpos[:, :nq_robot]
    converted[:, nq_robot : nq_robot + 3] = object_slide
    converted[:, nq_robot + 3 : nq_robot + 6] = object_euler
    return converted


def min_lowerbody_distance(model: mujoco.MjModel, qpos: np.ndarray) -> float:
    data = mujoco.MjData(model)
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")
    geom_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in LOWER_BODY_GEOMS
    ]
    if object_id < 0 or any(geom_id < 0 for geom_id in geom_ids):
        raise ValueError("lower-body/object geoms missing during distance audit")
    minimum = float("inf")
    fromto = np.zeros(6, dtype=np.float64)
    for frame in qpos:
        data.qpos[:] = frame
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for geom_id in geom_ids:
            distance = mujoco.mj_geomDistance(
                model, data, geom_id, object_id, 10.0, fromto
            )
            minimum = min(minimum, float(distance))
    return minimum


def build_case(case_id: str, source: dict[str, str], *, overwrite: bool) -> dict[str, Any]:
    base_scene = repo_path(source["scene_act"])
    if not base_scene.is_file():
        raise FileNotFoundError(base_scene)
    tree = ET.parse(base_scene)
    root = tree.getroot()
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for pair in list(contact.findall("pair")):
        if pair.get("name", "").startswith(PAIR_PREFIX):
            contact.remove(pair)
    all_geom_names = {geom.get("name") for geom in root.iter("geom") if geom.get("name")}
    missing_geoms = sorted(set(LOWER_BODY_GEOMS) - all_geom_names)
    if "object_collision" not in all_geom_names or missing_geoms:
        raise ValueError(
            f"{case_id}: missing object/lower geoms object={'object_collision' not in all_geom_names} lower={missing_geoms}"
        )
    for geom_name in LOWER_BODY_GEOMS:
        ET.SubElement(
            contact,
            "pair",
            {
                "name": f"{PAIR_PREFIX}{geom_name}{PAIR_SUFFIX}",
                "geom1": geom_name,
                "geom2": "object_collision",
                "solref": "0.008 1",
                "margin": "0",
                "gap": "0",
                "condim": "1",
            },
        )
    physical_scene = base_scene.with_name(f"{PHYSICS_SCENE_NAME}.xml")
    if physical_scene.exists() and not overwrite:
        existing_root = ET.parse(physical_scene).getroot()
        if tree_signature(existing_root) != tree_signature(root):
            raise FileExistsError(f"refusing to overwrite different E169 scene: {physical_scene}")
    else:
        ET.indent(tree, space="  ")
        tree.write(physical_scene, encoding="utf-8", xml_declaration=True)

    base_root = ET.parse(base_scene).getroot()
    physical_root = ET.parse(physical_scene).getroot()
    if tree_signature(base_root) != tree_signature(physical_root, ignore_e169_pairs=True):
        raise AssertionError(f"{case_id}: physical sidecar changes more than E169 pairs")
    pairs = [
        pair
        for pair in physical_root.findall("./contact/pair")
        if pair.get("name", "").startswith(PAIR_PREFIX)
    ]
    if len(pairs) != len(LOWER_BODY_GEOMS):
        raise AssertionError(f"{case_id}: expected 16 E169 pairs, got {len(pairs)}")
    for pair in pairs:
        if pair.get("condim") != "1" or pair.get("solref") != "0.008 1":
            raise AssertionError(f"{case_id}: invalid E169 pair contract: {pair.attrib}")

    model = mujoco.MjModel.from_xml_path(str(physical_scene))
    pair_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_PAIR, pair_id)
        for pair_id in range(model.npair)
    }
    expected_names = {f"{PAIR_PREFIX}{name}{PAIR_SUFFIX}" for name in LOWER_BODY_GEOMS}
    if not expected_names.issubset(pair_names):
        raise AssertionError(f"{case_id}: compiled model is missing E169 pairs")

    trajectory = repo_path(source["trajectory"])
    with np.load(trajectory, allow_pickle=True) as data:
        reference = np.asarray(data["qpos"], dtype=np.float64)
    converted = convert_reference_to_scene(reference, physical_scene)
    audit_frames = converted[: min(5, len(converted))]
    qpos0_min = min_lowerbody_distance(model, model.qpos0[np.newaxis, :])
    reference_min = min_lowerbody_distance(model, audit_frames)
    if min(qpos0_min, reference_min) < -0.005:
        raise AssertionError(
            f"{case_id}: initial/reference lower-body overlap exceeds 5mm: qpos0={qpos0_min:.6f} ref={reference_min:.6f}"
        )

    snapshot_base = RESULTS / "scene_snapshot/no_physics" / case_id
    snapshot_physics = RESULTS / "scene_snapshot/lowerbody_physics" / case_id
    snapshot_base.mkdir(parents=True, exist_ok=True)
    snapshot_physics.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_scene, snapshot_base / base_scene.name)
    shutil.copy2(physical_scene, snapshot_physics / physical_scene.name)
    for source_key in ("config_act", "result_npz", "video"):
        source_path = repo_path(source[source_key])
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

    return {
        "case_id": case_id,
        "status": "pass",
        "base_scene": rel(base_scene),
        "base_scene_sha256": sha256(base_scene),
        "physical_scene": rel(physical_scene),
        "physical_scene_sha256": sha256(physical_scene),
        "trajectory": rel(trajectory),
        "trajectory_sha256": sha256(trajectory),
        "contact_mask": source["contact_mask"],
        "contact_mask_sha256": sha256(source["contact_mask"]),
        "e168_result_npz": source["result_npz"],
        "e168_result_sha256": sha256(source["result_npz"]),
        "e168_config_act": source["config_act"],
        "e168_config_sha256": sha256(source["config_act"]),
        "e168_video": source["video"],
        "e168_video_sha256": sha256(source["video"]),
        "compiled_pair_count": len(expected_names),
        "qpos0_min_lowerbody_object_distance_m": qpos0_min,
        "reference_first5_min_lowerbody_object_distance_m": reference_min,
        "semantic_diff_only_e169_pairs": True,
        "created_at": now(),
    }


def build_all(*, overwrite: bool = False) -> list[dict[str, Any]]:
    sources = e168_rows()
    rows = [build_case(case_id, sources[case_id], overwrite=overwrite) for case_id in CASES]
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_tsv(RESULTS / "scene_snapshot/scene_audit.tsv", rows)
    summary = {
        "created_at": now(),
        "status": "pass",
        "cases": len(rows),
        "pair_count_per_physics_scene": len(LOWER_BODY_GEOMS),
        "no_physics_scene_name": NO_PHYSICS_SCENE_NAME,
        "physics_scene_name": PHYSICS_SCENE_NAME,
        "rows": rows,
    }
    (RESULTS / "scene_snapshot/scene_snapshot_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    rows = build_all(overwrite=args.overwrite)
    for row in rows:
        print(
            f"{row['case_id']}: pairs={row['compiled_pair_count']} "
            f"qpos0_min={row['qpos0_min_lowerbody_object_distance_m']:.6f} "
            f"ref_min={row['reference_first5_min_lowerbody_object_distance_m']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
