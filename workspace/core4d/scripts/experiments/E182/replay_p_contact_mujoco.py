#!/usr/bin/env python3
"""Validate E182 P point-contact predictions in true compound-convex MuJoCo scenes."""

from __future__ import annotations

import argparse
import copy
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from build_task_query_fixture import DEFAULT_OUTPUT_ROOT
from e182_common import atomic_json, relative_to_repo, repo_path, sha256_file
from evaluate_task_queries import (
    _load_candidate_and_oracle,
    _raycasting_scene,
    _resolve_case_and_candidate,
    _scene_signed_distance,
    _tape_chunk_path,
    materialize_selected_query_points,
)

DEFAULT_DIAGNOSTIC = (
    DEFAULT_OUTPUT_ROOT / "p_contact_diagnostic/diagnostic_manifest.json"
)
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "p_contact_mujoco_replay"
SHORT_ROLLOUT_STEPS = 5


def _find_named_body(root: ET.Element, name: str) -> ET.Element:
    """Resolve exactly one named body."""
    matches = [body for body in root.iter("body") if body.get("name") == name]
    if len(matches) != 1:
        raise RuntimeError(f"expected one body {name!r}, got {len(matches)}")
    return matches[0]


def _object_collision_geoms(object_body: ET.Element) -> list[ET.Element]:
    """Return the full E178 object-proxy collision geom set."""
    return [
        geom
        for geom in object_body.findall("geom")
        if (geom.get("name") or "") == "object_collision"
        or (geom.get("name") or "").startswith("object_collision_")
    ]


def _atomic_xml(path: Path, root: ET.Element) -> None:
    """Write an indented experiment sidecar atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _pair_physics_attributes(pair: ET.Element) -> dict[str, str]:
    """Return pair attributes that must agree across legacy object proxy parts."""
    return {
        key: value
        for key, value in pair.attrib.items()
        if key not in {"name", "geom1", "geom2"}
    }


def _absolutize_compiler_directories(root: ET.Element, base_scene: Path) -> None:
    """Preserve asset resolution when an MJCF sidecar moves directories."""
    compiler = root.find("compiler")
    if compiler is None:
        raise RuntimeError("base scene has no compiler section")
    for attribute in ("assetdir", "meshdir", "texturedir"):
        value = compiler.get(attribute)
        if value and not Path(value).is_absolute():
            compiler.set(attribute, str((base_scene.parent / value).resolve()))


def build_compound_sidecar(
    *,
    base_scene: Path,
    candidate_manifest: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Replace all E178 proxy boxes/pairs with ordered convex mesh geoms/pairs."""
    tree = ET.parse(base_scene)
    root = tree.getroot()
    _absolutize_compiler_directories(root, base_scene)
    object_body = _find_named_body(root, "object")
    old_geoms = _object_collision_geoms(object_body)
    if not old_geoms:
        raise RuntimeError("base scene has no object collision geoms")
    old_names = {str(geom.get("name")) for geom in old_geoms}
    children = list(object_body)
    insert_at = min(children.index(geom) for geom in old_geoms)
    template_geom = copy.deepcopy(old_geoms[0])
    for geom in old_geoms:
        object_body.remove(geom)

    asset = root.find("asset")
    if asset is None:
        raise RuntimeError("base scene has no asset section")
    new_names = []
    mesh_assets = []
    for part_index, part in enumerate(candidate_manifest["parts"]):
        mesh_name = f"E182_{candidate_manifest['candidate_id']}_part_{part_index:03d}"
        geom_name = (
            "object_collision"
            if part_index == 0
            else f"object_collision_coacd_{part_index:03d}"
        )
        mesh_path = repo_path(part["path"]).resolve()
        if sha256_file(mesh_path) != part["sha256"]:
            raise RuntimeError(f"candidate part SHA changed: {mesh_path}")
        asset.append(
            ET.Element(
                "mesh",
                {
                    "name": mesh_name,
                    "file": str(mesh_path),
                    "scale": "1 1 1",
                },
            )
        )
        mesh_assets.append(mesh_name)
        geom = copy.deepcopy(template_geom)
        geom.attrib.pop("pos", None)
        geom.attrib.pop("quat", None)
        geom.attrib.pop("size", None)
        geom.set("name", geom_name)
        geom.set("type", "mesh")
        geom.set("mesh", mesh_name)
        geom.set("rgba", "0.95 0.22 0.02 0.24")
        object_body.insert(insert_at + part_index, geom)
        new_names.append(geom_name)

    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    pair_templates: dict[str, ET.Element] = {}
    remove_pairs = []
    for pair in contact.findall("pair"):
        geom1 = pair.get("geom1") or ""
        geom2 = pair.get("geom2") or ""
        if geom1 not in old_names and geom2 not in old_names:
            continue
        other = geom2 if geom1 in old_names else geom1
        if other in pair_templates and _pair_physics_attributes(
            pair_templates[other]
        ) != _pair_physics_attributes(pair):
            raise RuntimeError(
                f"inconsistent explicit pair physics attributes for {other!r}"
            )
        pair_templates.setdefault(other, copy.deepcopy(pair))
        remove_pairs.append(pair)
    for pair in remove_pairs:
        contact.remove(pair)
    if not pair_templates:
        raise RuntimeError("base scene has no explicit object collision pairs")
    pair_count = 0
    for other_geom, template in sorted(pair_templates.items()):
        for part_index, object_geom in enumerate(new_names):
            pair = copy.deepcopy(template)
            pair.set("name", f"E182_{other_geom}_obj{part_index:03d}")
            if template.get("geom1") in old_names:
                pair.set("geom1", object_geom)
                pair.set("geom2", other_geom)
            else:
                pair.set("geom1", other_geom)
                pair.set("geom2", object_geom)
            contact.append(pair)
            pair_count += 1

    _atomic_xml(output_path, root)
    model = mujoco.MjModel.from_xml_path(str(output_path))
    loaded_object_geoms = [
        name
        for name in new_names
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0
    ]
    if len(loaded_object_geoms) != len(new_names):
        raise RuntimeError("one or more CoACD object geoms failed to load")
    return {
        "path": relative_to_repo(output_path),
        "sha256": sha256_file(output_path),
        "base_scene": relative_to_repo(base_scene),
        "base_scene_sha256": sha256_file(base_scene),
        "candidate_asset_sha256": candidate_manifest["candidate_asset_sha256"],
        "object_geom_names": new_names,
        "mesh_asset_names": mesh_assets,
        "object_geom_count": len(new_names),
        "explicit_pair_count": pair_count,
        "model_ngeom": int(model.ngeom),
        "model_npair": int(model.npair),
    }


def _physics_contact_summary(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_geom_ids: set[int],
    object_geom_ids: set[int],
) -> dict[str, Any]:
    """Extract true MuJoCo robot–compound-object contacts after mj_forward."""
    distances = []
    pairs = []
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        left = int(contact.geom1)
        right = int(contact.geom2)
        crosses = (left in robot_geom_ids and right in object_geom_ids) or (
            right in robot_geom_ids and left in object_geom_ids
        )
        if not crosses:
            continue
        distances.append(float(contact.dist))
        pairs.append((left, right))
    return {
        "contact": bool(distances),
        "contact_count": len(distances),
        "min_contact_distance_m": min(distances) if distances else None,
        "geom_id_pairs": pairs,
    }


def _binary_confusion(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Return exact boolean confusion counts and precision/recall."""
    predicted = np.asarray(predicted, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if predicted.shape != target.shape:
        raise ValueError("binary confusion shapes do not match")
    tp = int((predicted & target).sum())
    fp = int((predicted & ~target).sum())
    fn = int((~predicted & target).sum())
    tn = int((~predicted & ~target).sum())
    return {
        "count": int(predicted.size),
        "true_positive_count": tp,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "true_negative_count": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def replay_candidate(
    diagnostic_row: dict[str, Any],
    *,
    fixture: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Compile one sidecar and compare point-C contact with true MuJoCo contact."""
    case, fixture_candidate = _resolve_case_and_candidate(
        fixture, diagnostic_row["case_id"], diagnostic_row["candidate_id"]
    )
    candidate, exact_union, oracle = _load_candidate_and_oracle(fixture_candidate)
    prg_manifest_path = repo_path(case["prg_manifest"]["path"])
    prg = json.loads(prg_manifest_path.read_text(encoding="utf-8"))
    base_scene = repo_path(prg["model"])
    sidecar_path = (
        output_root
        / "sidecars"
        / f"{diagnostic_row['case_id']}__{diagnostic_row['candidate_id']}.xml"
    )
    sidecar = build_compound_sidecar(
        base_scene=base_scene,
        candidate_manifest=candidate,
        output_path=sidecar_path,
    )
    model = mujoco.MjModel.from_xml_path(str(sidecar_path))
    data = mujoco.MjData(model)
    robot_geom_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in prg["consumer_geom_names"]["P_collision"]
    }
    object_geom_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in sidecar["object_geom_names"]
    }
    if min(robot_geom_ids | object_geom_ids) < 0:
        raise RuntimeError("sidecar lost one or more P/object collision geoms")
    union_scene = _raycasting_scene(exact_union)
    oracle_scene = _raycasting_scene(oracle)

    point_candidate_all = []
    point_oracle_all = []
    physics_all = []
    physics_min_distance = []
    source_rows = []
    qpos_by_identity: dict[tuple[str, int], np.ndarray] = {}
    for entry in prg["chunks"]:
        if entry["source_family"] == "cem_on_a":
            continue
        chunk_path = _tape_chunk_path(entry)
        with np.load(chunk_path, allow_pickle=False) as chunk:
            qpos = np.asarray(chunk["qpos"], dtype=np.float64)
            full = np.arange(len(chunk["point_geom_id"]), dtype=np.int32)
            points = materialize_selected_query_points(chunk, full)
            radii = np.asarray(chunk["point_radius_m"], dtype=np.float64)
            p_mask = np.isin(
                np.asarray(chunk["point_geom_id"], dtype=np.int32),
                prg["consumer_geom_ids"]["P_collision"],
            )
            p_points = points[:, p_mask]
            p_radii = radii[p_mask]
            candidate_min = (
                _scene_signed_distance(union_scene, p_points) - p_radii[None, :]
            ).min(axis=1)
            oracle_min = (
                _scene_signed_distance(oracle_scene, p_points) - p_radii[None, :]
            ).min(axis=1)
            point_candidate = candidate_min <= 0.0
            point_oracle = oracle_min <= 0.0
            physics = np.zeros(len(qpos), dtype=bool)
            minimum = np.full(len(qpos), np.nan, dtype=np.float64)
            contact_counts = np.zeros(len(qpos), dtype=np.int32)
            for pose_index, pose in enumerate(qpos):
                data.qpos[:] = pose
                data.qvel[:] = 0.0
                if model.nu:
                    data.ctrl[:] = 0.0
                mujoco.mj_forward(model, data)
                contact = _physics_contact_summary(
                    model, data, robot_geom_ids, object_geom_ids
                )
                physics[pose_index] = contact["contact"]
                contact_counts[pose_index] = contact["contact_count"]
                if contact["min_contact_distance_m"] is not None:
                    minimum[pose_index] = contact["min_contact_distance_m"]
                qpos_by_identity[(entry["source_family"], pose_index)] = pose.copy()
            point_candidate_all.append(point_candidate)
            point_oracle_all.append(point_oracle)
            physics_all.append(physics)
            physics_min_distance.append(minimum)
            source_rows.append(
                {
                    "source_family": entry["source_family"],
                    "pose_count": len(qpos),
                    "point_candidate_contact_count": int(point_candidate.sum()),
                    "point_oracle_contact_count": int(point_oracle.sum()),
                    "mujoco_contact_count": int(physics.sum()),
                    "mujoco_contact_event_count": int(contact_counts.sum()),
                }
            )
    point_candidate = np.concatenate(point_candidate_all)
    point_oracle = np.concatenate(point_oracle_all)
    physics = np.concatenate(physics_all)
    minimum = np.concatenate(physics_min_distance)
    point_phantom = point_candidate & ~point_oracle
    physics_on_point_phantom = physics[point_phantom]

    representative = diagnostic_row["representative_pose"]
    representative_pose = qpos_by_identity[
        (representative["source_family"], int(representative["pose_index"]))
    ]
    data.qpos[:] = representative_pose
    data.qvel[:] = 0.0
    if model.nu:
        data.ctrl[:] = 0.0
    rollout_contacts = []
    initial_qpos = data.qpos.copy()
    for step in range(SHORT_ROLLOUT_STEPS + 1):
        mujoco.mj_forward(model, data)
        contact = _physics_contact_summary(model, data, robot_geom_ids, object_geom_ids)
        rollout_contacts.append(
            {
                "step": step,
                "contact": contact["contact"],
                "contact_count": contact["contact_count"],
                "min_contact_distance_m": contact["min_contact_distance_m"],
            }
        )
        if step < SHORT_ROLLOUT_STEPS:
            mujoco.mj_step(model, data)
    return {
        "case_id": diagnostic_row["case_id"],
        "object_key": diagnostic_row["object_key"],
        "candidate_id": diagnostic_row["candidate_id"],
        "max_hulls": diagnostic_row["max_hulls"],
        "actual_hulls": diagnostic_row["actual_hulls"],
        "selection_role": "DIAGNOSTIC_ONLY_NOT_FINALIST",
        "sidecar": sidecar,
        "pose_count": int(len(physics)),
        "point_C_vs_mujoco_C": _binary_confusion(point_candidate, physics),
        "point_M_vs_mujoco_C": _binary_confusion(point_oracle, physics),
        "point_phantom_pose_count": int(point_phantom.sum()),
        "point_phantom_confirmed_by_mujoco_count": int(physics_on_point_phantom.sum()),
        "point_phantom_confirmed_by_mujoco_fraction": float(
            physics_on_point_phantom.mean() if len(physics_on_point_phantom) else 0.0
        ),
        "mujoco_contact_pose_count": int(physics.sum()),
        "mujoco_min_contact_distance_m": float(np.nanmin(minimum))
        if np.isfinite(minimum).any()
        else None,
        "source_rows": source_rows,
        "representative_short_rollout": {
            "source_family": representative["source_family"],
            "pose_index": representative["pose_index"],
            "steps": SHORT_ROLLOUT_STEPS,
            "contacts": rollout_contacts,
            "qpos_linf_delta": float(np.max(np.abs(data.qpos - initial_qpos))),
        },
    }


def main() -> int:
    """Replay all three bucket003 best-failed candidates in true MuJoCo scenes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    diagnostic = json.loads(args.diagnostic.read_text(encoding="utf-8"))
    if (
        diagnostic.get("status") != "COMPLETE"
        or diagnostic.get("selection_eligible") is not False
        or diagnostic.get("heldout_access") != "NOT_ACCESSED_DEV3_ONLY"
    ):
        raise RuntimeError("P visual diagnostic contract changed")
    fixture_path = repo_path(
        "workspace/core4d/results/E182/s2_task_query_eval/query_fixture_manifest.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    results = [
        replay_candidate(row, fixture=fixture, output_root=args.output_root)
        for row in diagnostic["candidates"]
    ]
    payload = {
        "experiment_id": "E182",
        "stage": "S2_bucket003_true_MuJoCo_P_contact_replay",
        "status": "COMPLETE",
        "selection_eligible": False,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "diagnostic": {
            "path": relative_to_repo(args.diagnostic),
            "sha256": sha256_file(args.diagnostic),
        },
        "fixture": {
            "path": relative_to_repo(fixture_path),
            "sha256": sha256_file(fixture_path),
        },
        "candidate_count": len(results),
        "short_rollout_steps": SHORT_ROLLOUT_STEPS,
        "candidates": results,
    }
    atomic_json(args.output_root / "mujoco_replay_manifest.json", payload)
    print(f"E182_P_MUJOCO_REPLAY=COMPLETE candidates={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
