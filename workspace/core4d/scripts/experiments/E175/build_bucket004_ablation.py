#!/usr/bin/env python3
"""Build the E175 five-arm bucket004 proxy/pair/PRG ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation


REPO = Path(__file__).resolve().parents[5]
RESULTS = REPO / "workspace/core4d/results/E175"
E174_MANIFEST = (
    REPO
    / "workspace/core4d/results/E174/s6_downstream/manifests/"
    "cem_full_manifest.tsv"
)
OVERRIDE_DIR = REPO / "examples/config/override"
CASE_IDS = (
    "bucket004_20231002_021_p1",
    "bucket004_20231003_1_012_p1",
    "bucket004_20231002_021_p2",
    "bucket004_20231003_1_012_p2",
)
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
HAND_GEOMS = ("lh", "rh")
ROBOT_GEOMS = LOWER_BODY_GEOMS + HAND_GEOMS


@dataclass(frozen=True)
class Arm:
    arm_id: str
    proxy_variant: str
    scene_key: str
    physics_pair_mode: str
    sdf_mode: str


ARMS = (
    Arm("A", "wall", "wall_primary", "primary", "primary"),
    Arm("B", "wall", "wall_union", "union", "primary"),
    Arm("C", "wall", "wall_union", "union", "union"),
    Arm("D", "rim_inner", "rimInner_union", "union", "union"),
    Arm("E", "solid_hull", "solid_union", "union", "union"),
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def rel(path: str | Path) -> str:
    value = Path(path)
    try:
        return str(value.resolve().relative_to(REPO.resolve()))
    except (OSError, ValueError):
        return str(path)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def vector(raw: str) -> np.ndarray:
    return np.asarray([float(value) for value in raw.split()], dtype=np.float64)


def format_vector(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def find_object_body(root: ET.Element) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == "object":
            return body
    raise ValueError("missing object body")


def collision_geoms(object_body: ET.Element) -> list[ET.Element]:
    return [
        geom
        for geom in object_body.findall("geom")
        if (geom.get("name") or "").startswith("object_collision")
    ]


def geom_style(source: ET.Element) -> dict[str, str]:
    return {
        key: source.get(key, value)
        for key, value in {
            "type": "box",
            "rgba": "0.40 0.50 0.60 0.3",
            "group": "3",
            "contype": "1",
            "conaffinity": "1",
            "friction": "1 0.005 0.0001",
            "condim": "3",
        }.items()
    }


def add_box(
    object_body: ET.Element,
    source: ET.Element,
    *,
    name: str,
    pos: np.ndarray,
    size: np.ndarray,
) -> None:
    attrs = {
        "name": name,
        **geom_style(source),
        "pos": format_vector(pos),
        "size": format_vector(size),
    }
    ET.SubElement(object_body, "geom", attrs)


def add_rim_inner_shell(object_body: ET.Element) -> None:
    by_name = {
        geom.get("name", ""): geom for geom in collision_geoms(object_body)
    }
    primary = by_name["object_collision"]
    xwall = by_name["object_collision_bucket_xpos"]
    ywall = by_name["object_collision_bucket_ypos"]
    xwall_pos = vector(xwall.get("pos", ""))
    xwall_size = vector(xwall.get("size", ""))
    ywall_pos = vector(ywall.get("pos", ""))
    ywall_size = vector(ywall.get("size", ""))
    outer_x = xwall_pos[0] + xwall_size[0]
    inner_x = xwall_pos[0] - xwall_size[0]
    outer_y = ywall_pos[1] + ywall_size[1]
    inner_y = ywall_pos[1] - ywall_size[1]
    top_z = xwall_size[2]

    rim_inset = 0.020
    rim_half_z = 0.003
    rim_center_z = top_z + 0.001
    x_band_min = inner_x - rim_inset
    x_band_center = (outer_x + x_band_min) / 2.0
    x_band_half = (outer_x - x_band_min) / 2.0
    y_band_min = inner_y - rim_inset
    y_band_center = (outer_y + y_band_min) / 2.0
    y_band_half = (outer_y - y_band_min) / 2.0
    for sign, label in ((-1.0, "neg"), (1.0, "pos")):
        add_box(
            object_body,
            primary,
            name=f"object_collision_rim_x{label}",
            pos=np.asarray(
                [sign * x_band_center, 0.0, rim_center_z]
            ),
            size=np.asarray([x_band_half, outer_y, rim_half_z]),
        )
        add_box(
            object_body,
            primary,
            name=f"object_collision_rim_y{label}",
            pos=np.asarray(
                [0.0, sign * y_band_center, rim_center_z]
            ),
            size=np.asarray(
                [max(x_band_min, 0.001), y_band_half, rim_half_z]
            ),
        )

    # Existing wall boxes already expose an inner face. These 3-mm extensions
    # make the requested inner shell explicit while keeping the change thin.
    shell_thickness = 0.003
    shell_half = shell_thickness / 2.0
    shell_half_z = max(top_z - 0.006, 0.001)
    for sign, label in ((-1.0, "neg"), (1.0, "pos")):
        add_box(
            object_body,
            primary,
            name=f"object_collision_inner_x{label}",
            pos=np.asarray(
                [sign * (inner_x - shell_half), 0.0, 0.0]
            ),
            size=np.asarray([shell_half, inner_y, shell_half_z]),
        )
        add_box(
            object_body,
            primary,
            name=f"object_collision_inner_y{label}",
            pos=np.asarray(
                [0.0, sign * (inner_y - shell_half), 0.0]
            ),
            size=np.asarray(
                [max(inner_x - shell_thickness, 0.001), shell_half, shell_half_z]
            ),
        )


def visual_mesh_bounds(base_scene: Path) -> tuple[np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(base_scene))
    gid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "object_visual"
    )
    mesh_id = int(model.geom_dataid[gid])
    start = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    vertices = np.asarray(
        model.mesh_vert[start : start + count], dtype=np.float64
    )
    quat = model.geom_quat[gid]
    rotation = Rotation.from_quat(
        [quat[1], quat[2], quat[3], quat[0]]
    )
    vertices = rotation.apply(vertices) + model.geom_pos[gid]
    return vertices.min(axis=0), vertices.max(axis=0)


def replace_with_solid(
    object_body: ET.Element,
    base_scene: Path,
) -> None:
    old = collision_geoms(object_body)
    primary = next(
        geom for geom in old if geom.get("name") == "object_collision"
    )
    style = geom_style(primary)
    for geom in old:
        object_body.remove(geom)
    lower, upper = visual_mesh_bounds(base_scene)
    center = (lower + upper) / 2.0
    half = (upper - lower) / 2.0
    ET.SubElement(
        object_body,
        "geom",
        {
            "name": "object_collision",
            **style,
            "pos": format_vector(center),
            "size": format_vector(half),
        },
    )


def replace_robot_object_pairs(
    root: ET.Element,
    object_names: list[str],
    *,
    pair_mode: str,
    pair_prefix: str,
) -> None:
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    object_set = set(object_names)
    for pair in list(contact.findall("pair")):
        geom1 = pair.get("geom1", "")
        geom2 = pair.get("geom2", "")
        if (
            geom1 in ROBOT_GEOMS
            and geom2.startswith("object_collision")
        ) or (
            geom2 in ROBOT_GEOMS
            and geom1.startswith("object_collision")
        ):
            contact.remove(pair)
    paired_objects = (
        ["object_collision"] if pair_mode == "primary" else object_names
    )
    for robot in ROBOT_GEOMS:
        for object_name in paired_objects:
            attrs = {
                "name": (
                    f"{pair_prefix}_{robot}_{object_name.replace('object_collision', 'obj')}"
                ),
                "geom1": robot,
                "geom2": object_name,
                "solref": "0.008 1",
                "margin": "0",
                "gap": "0",
                "condim": "4" if robot in HAND_GEOMS else "1",
            }
            if robot in HAND_GEOMS:
                attrs["friction"] = "2 1"
            ET.SubElement(contact, "pair", attrs)


def compiled_pair_coverage(
    model: mujoco.MjModel,
    object_names: list[str],
) -> dict[str, set[str]]:
    object_set = set(object_names)
    output: dict[str, set[str]] = defaultdict(set)
    for pair_id in range(model.npair):
        names = [
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                int(model.pair_geom1[pair_id]),
            ),
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                int(model.pair_geom2[pair_id]),
            ),
        ]
        for robot in ROBOT_GEOMS:
            if robot in names:
                other = names[1] if names[0] == robot else names[0]
                if other in object_set:
                    output[robot].add(str(other))
    return output


def min_lowerbody_union_first5(
    model: mujoco.MjModel,
    reference_qpos: np.ndarray,
    object_names: list[str],
) -> float:
    lower_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in LOWER_BODY_GEOMS
    ]
    object_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in object_names
    ]
    data = mujoco.MjData(model)
    fromto = np.zeros(6, dtype=np.float64)
    minimum = math.inf
    for qpos in reference_qpos[: min(5, len(reference_qpos))]:
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for lower_id in lower_ids:
            for object_id in object_ids:
                minimum = min(
                    minimum,
                    float(
                        mujoco.mj_geomDistance(
                            model,
                            data,
                            lower_id,
                            object_id,
                            10.0,
                            fromto,
                        )
                    ),
                )
    return minimum


def build_scene(
    base_scene: Path,
    output: Path,
    *,
    scene_key: str,
    proxy_variant: str,
    pair_mode: str,
    reference_qpos: np.ndarray,
) -> dict[str, Any]:
    tree = ET.parse(base_scene)
    root = tree.getroot()
    object_body = find_object_body(root)
    if proxy_variant == "rim_inner":
        add_rim_inner_shell(object_body)
    elif proxy_variant == "solid_hull":
        replace_with_solid(object_body, base_scene)
    elif proxy_variant != "wall":
        raise ValueError(f"unknown proxy variant: {proxy_variant}")
    names = [
        geom.get("name", "") for geom in collision_geoms(object_body)
    ]
    if not names or names[0] != "object_collision":
        raise ValueError(f"invalid collision geom ordering: {names}")
    replace_robot_object_pairs(
        root,
        names,
        pair_mode=pair_mode,
        pair_prefix=f"E175_{scene_key}",
    )
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)

    model = mujoco.MjModel.from_xml_path(str(output))
    coverage = compiled_pair_coverage(model, names)
    expected = {"object_collision"} if pair_mode == "primary" else set(names)
    failures = [
        robot
        for robot in ROBOT_GEOMS
        if coverage.get(robot, set()) != expected
    ]
    if failures:
        raise AssertionError(f"pair coverage mismatch: {failures}")
    for name in names:
        gid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, name
        )
        if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            raise AssertionError(f"non-box proxy geom: {name}")
    for hand in HAND_GEOMS:
        gid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, hand
        )
        if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            raise AssertionError(f"{hand} is not rubber mesh")
    return {
        "scene_act": rel(output),
        "scene_sha256": sha256(output),
        "object_geom_names": names,
        "object_geom_count": len(names),
        "robot_object_pair_count": len(ROBOT_GEOMS) * len(expected),
        "reference_first5_union_min_distance_m": (
            min_lowerbody_union_first5(
                model, reference_qpos, names
            )
        ),
    }


def write_override(
    case_id: str,
    arm: Arm,
    scene_name: str,
    e174_override_id: str,
) -> Path:
    override_id = f"core4d_E175_{case_id}_{arm.arm_id}"
    output = OVERRIDE_DIR / f"{override_id}.yaml"
    payload = {
        "defaults": [e174_override_id, "_self_"],
        "scene_name": scene_name,
        "object_collision_sdf_mode": arm.sdf_mode,
    }
    output.write_text(
        "# @package _global_\n"
        "# Auto-generated by E175 build_bucket004_ablation.py.\n"
        + yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return output


def audit_override(
    output: Path,
    e174_override_id: str,
    arm: Arm,
    scene_name: str,
) -> None:
    config_dir = str((REPO / "examples/config").resolve())
    with initialize_config_dir(
        version_base=None, config_dir=config_dir
    ):
        baseline = OmegaConf.to_container(
            compose(
                config_name="default",
                overrides=[f"+override={e174_override_id}"],
            ),
            resolve=True,
        )
        candidate = OmegaConf.to_container(
            compose(
                config_name="default",
                overrides=[f"+override={output.stem}"],
            ),
            resolve=True,
        )
    allowed = {"scene_name", "object_collision_sdf_mode"}
    for key in sorted(set(baseline) | set(candidate)):
        baseline_value = baseline.get(
            key, "primary" if key == "object_collision_sdf_mode" else None
        )
        if key not in allowed and baseline_value != candidate.get(key):
            raise AssertionError(f"override drift outside axis: {key}")
    if candidate.get("scene_name") != scene_name:
        raise AssertionError("scene_name override mismatch")
    if candidate.get("object_collision_sdf_mode") != arm.sdf_mode:
        raise AssertionError("SDF mode override mismatch")


def artifact_paths(
    case_id: str,
    arm_id: str,
    stage: str,
) -> dict[str, Any]:
    variant = f"E175_{case_id}_{arm_id}_{stage}"
    root = f"workspace/core4d/results/E175/s6_downstream/cem/{stage}"
    return {
        "variant": variant,
        "result_npz": f"{root}/{variant}.npz",
        "outdir_npz": f"{root}/{variant}_outdir/trajectory_mjwp_act.npz",
        "config_act": f"{root}/{variant}_outdir/config_act.yaml",
        "video": (
            "workspace/core4d/results/E175/s6_downstream/render/"
            f"{stage}/{variant}.mp4"
        ),
        "log": f"logs/E175/cem/{stage}/{variant}.log",
        "cem_samples": 64 if stage == "canary" else 1024,
        "cem_opt_steps": 4 if stage == "canary" else 32,
        "cem_seed": 0,
        "execution_mode": (
            "canary" if stage == "canary" else "production"
        ),
        "status": (
            "READY_FOR_CANARY"
            if stage == "canary"
            else "READY_FOR_FULL"
        ),
    }


def local_authorities(
    e174_row: dict[str, str],
) -> tuple[Path, Path, Path]:
    task_dir = (REPO / e174_row["scene_act"]).parent
    target_scene = task_dir / "scene.xml"
    trajectory = task_dir / "0/trajectory_kinematic.npz"
    config = yaml.safe_load(
        (REPO / e174_row["config_act"]).read_text(encoding="utf-8")
    )
    contact_mask = REPO / str(config["contact_hdmi_mask_path"])
    for path in (target_scene, trajectory, contact_mask):
        if not path.is_file():
            raise FileNotFoundError(path)
    return target_scene, trajectory, contact_mask


def geometry_signature(scene: Path) -> list[tuple[str, tuple[float, ...], tuple[float, ...]]]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    object_bid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "object"
    )
    output = []
    for gid in range(model.ngeom):
        name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            or ""
        )
        if (
            int(model.geom_bodyid[gid]) == object_bid
            and name.startswith("object_collision")
        ):
            output.append(
                (
                    name,
                    tuple(float(x) for x in model.geom_pos[gid]),
                    tuple(float(x) for x in model.geom_size[gid, :3]),
                )
            )
    return output


def build() -> dict[str, Any]:
    e174_rows = {
        row["case_id"]: row
        for row in read_tsv(E174_MANIFEST)
        if row["case_id"] in CASE_IDS
    }
    if set(e174_rows) != set(CASE_IDS):
        raise ValueError(
            f"bucket004 case set mismatch: {sorted(e174_rows)}"
        )
    manifest_rows: dict[str, list[dict[str, Any]]] = {
        "canary": [],
        "full": [],
    }
    variant_rows: list[dict[str, Any]] = []
    scene_audit: list[dict[str, Any]] = []
    built_scenes: dict[tuple[str, str], dict[str, Any]] = {}

    for case_ordinal, case_id in enumerate(CASE_IDS, 1):
        source = e174_rows[case_id]
        base_scene = REPO / source["scene_act"]
        target_scene, trajectory, contact_mask = local_authorities(source)
        with np.load(REPO / source["outdir_npz"], allow_pickle=True) as data:
            reference_qpos = np.asarray(
                data["qpos"][:, 1, :], dtype=np.float64
            )
        task_dir = base_scene.parent
        for scene_key, proxy_variant, pair_mode in {
            (
                arm.scene_key,
                arm.proxy_variant,
                arm.physics_pair_mode,
            )
            for arm in ARMS
        }:
            scene_name = f"scene_act_E175_{scene_key}"
            output = task_dir / f"{scene_name}.xml"
            result = build_scene(
                base_scene,
                output,
                scene_key=scene_key,
                proxy_variant=proxy_variant,
                pair_mode=pair_mode,
                reference_qpos=reference_qpos,
            )
            built_scenes[(case_id, scene_key)] = result
            snapshot_dir = (
                RESULTS
                / "scene_snapshot/bucket004_ablation"
                / case_id
            )
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output, snapshot_dir / output.name)
            scene_audit.append(
                {
                    "case_id": case_id,
                    "scene_key": scene_key,
                    "proxy_variant": proxy_variant,
                    "physics_pair_mode": pair_mode,
                    **result,
                }
            )
            model = mujoco.MjModel.from_xml_path(str(output))
            for geom_name in result["object_geom_names"]:
                gid = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
                )
                variant_rows.append(
                    {
                        "case_id": case_id,
                        "scene_key": scene_key,
                        "proxy_variant": proxy_variant,
                        "physics_pair_mode": pair_mode,
                        "scene_act": rel(output),
                        "scene_sha256": result["scene_sha256"],
                        "geom_name": geom_name,
                        "geom_type": "box",
                        "geom_pos": format_vector(model.geom_pos[gid]),
                        "geom_size": format_vector(
                            model.geom_size[gid, :3]
                        ),
                    }
                )
        if geometry_signature(
            task_dir / "scene_act_E175_wall_primary.xml"
        ) != geometry_signature(
            task_dir / "scene_act_E175_wall_union.xml"
        ):
            raise AssertionError("A/B wall geometry drift")

        for arm_ordinal, arm in enumerate(ARMS, 1):
            scene_name = f"scene_act_E175_{arm.scene_key}"
            scene = built_scenes[(case_id, arm.scene_key)]
            override = write_override(
                case_id, arm, scene_name, source["override_id"]
            )
            audit_override(
                override, source["override_id"], arm, scene_name
            )
            base = {
                "ordinal": (case_ordinal - 1) * len(ARMS) + arm_ordinal,
                "case_id": case_id,
                "object_key": "bucket004",
                "arm_id": arm.arm_id,
                "cell_id": arm.arm_id,
                "proxy_variant": arm.proxy_variant,
                "physics_pair_mode": arm.physics_pair_mode,
                "object_collision_sdf_mode": arm.sdf_mode,
                "retarget_variant_id": "omnirt_v1",
                "selected_retarget_variant_id": "omnirt_v1",
                "target_variant_id": "ref_fk",
                "hand_collision_variant_id": "rubber_hull",
                "spider_method_id": "E175_bucket004_proxy_pair_sdf_ablation",
                "source_config_id": source["override_id"],
                "p_enabled": "true",
                "r_enabled": "true",
                "g_enabled": "true",
                "contact_mask_label": "3cm",
                "target_task": source["target_task"],
                "target_scene": rel(target_scene),
                "trajectory": rel(trajectory),
                "contact_mask": rel(contact_mask),
                "assigned_gpu": "0",
                "gpu_id": "",
                "override_id": override.stem,
                "override_path": rel(override),
                "override_sha256": sha256(override),
                "scene_act": scene["scene_act"],
                "scene_name": scene_name,
                "base_scene_sha256": sha256(base_scene),
                "effective_scene_sha256": scene["scene_sha256"],
                "trajectory_sha256": sha256(trajectory),
                "contact_mask_sha256": sha256(contact_mask),
                "reference_first5_union_min_distance_m": scene[
                    "reference_first5_union_min_distance_m"
                ],
                "failure_mode": "",
                "blocker_detail": "",
                "updated_at": now(),
            }
            for stage in ("canary", "full"):
                row = dict(base)
                row.update(artifact_paths(case_id, arm.arm_id, stage))
                manifest_rows[stage].append(row)

    output_root = RESULTS / "s6_downstream/manifests"
    write_tsv(
        output_root / "bucket004_ablation_canary_manifest.tsv",
        manifest_rows["canary"],
    )
    write_tsv(
        output_root / "bucket004_ablation_full_manifest.tsv",
        manifest_rows["full"],
    )
    write_tsv(
        RESULTS / "scene_snapshot/bucket004_ablation/scene_audit.tsv",
        scene_audit,
    )
    write_tsv(
        RESULTS
        / "scene_snapshot/bucket004_ablation/"
        "scene_variant_manifest.tsv",
        variant_rows,
    )
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    summary = {
        "status": "pass",
        "created_at": now(),
        "git_commit": git_commit,
        "cases": len(CASE_IDS),
        "arms": len(ARMS),
        "canary_cells": len(manifest_rows["canary"]),
        "full_cells": len(manifest_rows["full"]),
        "unique_scenes": len(built_scenes),
        "scene_audit_rows": len(scene_audit),
        "scene_variant_geom_rows": len(variant_rows),
        "runtime_schedule": "single_visible_gpu_serial",
    }
    (output_root / "bucket004_ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
