#!/usr/bin/env python3
"""Build E175 39-case surface-voxel + multi-geom production authority."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[5]
RESULTS = REPO / "workspace/core4d/results/E175"
E174_MANIFEST = (
    REPO
    / "workspace/core4d/results/E174/s6_downstream/manifests/"
    "cem_full_manifest.tsv"
)
OVERRIDE_DIR = REPO / "examples/config/override"
SCENE_NAME = "scene_act_E175_surfaceVoxel_multiGeom"
METHOD_ID = "E175_nonbox_surfaceVoxel_multiGeom_union_r1"
BUCKET_TARGET_CELLS = 16
BUCKET_MAX_BOXES = 180

DCV3_TEMPLATES = (
    REPO
    / "workspace/core4d/scripts/data_construction_v3/stages/s2_templates"
)
sys.path.insert(0, str(DCV3_TEMPLATES))
from build_or_audit_templates import (  # noqa: E402
    surface_voxel_collision_geoms,
)
from restore_stage2b_inputs import runner_reference  # noqa: E402


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
EXPECTED_OBJECT_COUNTS = {
    "bucket003": 9,
    "bucket004": 4,
    "bucket007": 14,
    "bucket009": 1,
    "bucket010": 2,
    "desk007": 9,
}
CANARY_CASES = (
    "bucket003_20231018_001_p1",
    "bucket004_20231002_021_p1",
    "bucket007_20231020_055_p1",
    "bucket009_20231002_056_p2",
    "bucket010_20231003_2_055_p2",
    "desk007_20231030_034_p2",
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return REPO / path


def rel(raw: str | Path) -> str:
    path = Path(raw)
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.absolute().relative_to(REPO.absolute()))
    except ValueError:
        return str(raw)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def find_object_body(root: ET.Element) -> ET.Element:
    matches = [
        body for body in root.iter("body") if body.get("name") == "object"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one object body, got {len(matches)}")
    return matches[0]


def object_collision_elements(object_body: ET.Element) -> list[ET.Element]:
    return [
        geom
        for geom in object_body.findall("geom")
        if (geom.get("name") or "") == "object_collision"
        or (geom.get("name") or "").startswith("object_collision_")
    ]


def resolve_object_mesh(scene: Path) -> Path:
    root = ET.parse(scene).getroot()
    object_body = find_object_body(root)
    refs = [
        geom.get("mesh")
        for geom in object_body.findall("geom")
        if geom.get("mesh")
    ]
    assets = {
        mesh.get("name"): mesh
        for mesh in root.findall("./asset/mesh")
        if mesh.get("name")
    }
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    base = Path(meshdir) if Path(meshdir).is_absolute() else scene.parent / meshdir
    for ref in refs:
        asset = assets.get(ref)
        if asset is None or not asset.get("file"):
            continue
        scale = [
            float(value)
            for value in asset.get("scale", "1 1 1").split()
        ]
        if scale != [1.0, 1.0, 1.0]:
            raise ValueError(
                f"E175 bucket voxel builder requires unit mesh scale: "
                f"{scene}:{ref}:{scale}"
            )
        path = (base / str(asset.get("file"))).resolve()
        if path.is_file():
            return path
    raise FileNotFoundError(f"object mesh not resolved from {scene}")


def build_bucket_proxy_xml(mesh: Path) -> tuple[str, int]:
    xml, policy = surface_voxel_collision_geoms(
        mesh,
        "bucket",
        rgba="1 0.18 0.02 0.45",
        target_cells=BUCKET_TARGET_CELLS,
        max_boxes=BUCKET_MAX_BOXES,
    )
    if policy != "bucket_surface_voxel_multibox_proxy_draft":
        raise AssertionError(f"unexpected bucket collision policy: {policy}")
    count = len(ET.fromstring(f"<body>{xml}</body>").findall("geom"))
    if not 1 <= count <= BUCKET_MAX_BOXES:
        raise AssertionError(f"bucket proxy box count={count}")
    return xml, count


def replace_bucket_proxy(
    object_body: ET.Element, proxy_xml: str
) -> list[str]:
    previous = object_collision_elements(object_body)
    if not previous:
        raise ValueError("base scene has no object_collision geoms")
    children = list(object_body)
    insert_at = min(children.index(geom) for geom in previous)
    for geom in previous:
        object_body.remove(geom)
    generated = list(
        ET.fromstring(f"<body>{proxy_xml}</body>").findall("geom")
    )
    for offset, geom in enumerate(generated):
        object_body.insert(insert_at + offset, copy.deepcopy(geom))
    names = [str(geom.get("name")) for geom in generated]
    if len(names) != len(set(names)) or names[0] != "object_collision":
        raise AssertionError("invalid generated bucket collision names")
    return names


def object_collision_names(object_body: ET.Element) -> list[str]:
    names = [
        str(geom.get("name"))
        for geom in object_collision_elements(object_body)
    ]
    if not names or names[0] != "object_collision":
        raise ValueError(f"invalid object collision set: {names[:3]}")
    if len(names) != len(set(names)):
        raise ValueError("duplicate object collision geom names")
    return names


def is_robot_object_pair(pair: ET.Element) -> bool:
    first = pair.get("geom1", "")
    second = pair.get("geom2", "")
    return (
        first in ROBOT_OBJECT_GEOMS
        and (
            second == "object_collision"
            or second.startswith("object_collision_")
        )
    ) or (
        second in ROBOT_OBJECT_GEOMS
        and (
            first == "object_collision"
            or first.startswith("object_collision_")
        )
    )


def replace_robot_object_pairs(
    root: ET.Element, object_names: list[str]
) -> int:
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for pair in list(contact.findall("pair")):
        if is_robot_object_pair(pair):
            contact.remove(pair)

    for object_index, object_name in enumerate(object_names):
        for hand in HAND_GEOMS:
            ET.SubElement(
                contact,
                "pair",
                {
                    "name": f"E175_{hand}_obj{object_index:03d}",
                    "geom1": hand,
                    "geom2": object_name,
                    "solref": "0.008 1",
                    "friction": "2 1",
                    "condim": "4",
                },
            )
        for body_geom in LOWER_BODY_GEOMS:
            ET.SubElement(
                contact,
                "pair",
                {
                    "name": (
                        f"E175_{body_geom}_obj{object_index:03d}"
                    ),
                    "geom1": body_geom,
                    "geom2": object_name,
                    "solref": "0.008 1",
                    "margin": "0",
                    "gap": "0",
                    "condim": "1",
                },
            )
    return len(ROBOT_OBJECT_GEOMS) * len(object_names)


def stripped_signature(root: ET.Element) -> Any:
    """Signature excluding the two intended E175 axes."""

    clone = copy.deepcopy(root)
    for body in clone.iter("body"):
        for geom in list(body.findall("geom")):
            name = geom.get("name", "")
            if name == "object_collision" or name.startswith(
                "object_collision_"
            ):
                body.remove(geom)
    contact = clone.find("contact")
    if contact is not None:
        for pair in list(contact.findall("pair")):
            if is_robot_object_pair(pair):
                contact.remove(pair)

    def signature(element: ET.Element) -> Any:
        return (
            element.tag,
            tuple(sorted(element.attrib.items())),
            tuple(signature(child) for child in element),
        )

    return signature(clone)


def compiled_contract(scene: Path) -> dict[str, Any]:
    started = time.perf_counter()
    model = mujoco.MjModel.from_xml_path(str(scene))
    object_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "object"
    )
    if object_body_id < 0:
        raise ValueError("compiled scene missing object body")
    object_ids: list[int] = []
    object_names: list[str] = []
    for gid in range(model.ngeom):
        name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            or ""
        )
        if (
            int(model.geom_bodyid[gid]) == object_body_id
            and (
                name == "object_collision"
                or name.startswith("object_collision_")
            )
        ):
            object_ids.append(gid)
            object_names.append(name)
            if int(model.geom_type[gid]) != int(
                mujoco.mjtGeom.mjGEOM_BOX
            ):
                raise ValueError(f"non-box E175 object geom: {name}")

    robot_ids = {
        name: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, name
        )
        for name in ROBOT_OBJECT_GEOMS
    }
    missing_robot = [name for name, gid in robot_ids.items() if gid < 0]
    if missing_robot:
        raise ValueError(f"compiled scene missing robot geoms: {missing_robot}")

    observed: list[tuple[str, str]] = []
    object_by_id = dict(zip(object_ids, object_names))
    robot_by_id = {gid: name for name, gid in robot_ids.items()}
    for pair_id in range(model.npair):
        first = int(model.pair_geom1[pair_id])
        second = int(model.pair_geom2[pair_id])
        if first in robot_by_id and second in object_by_id:
            observed.append(
                (robot_by_id[first], object_by_id[second])
            )
        elif second in robot_by_id and first in object_by_id:
            observed.append(
                (robot_by_id[second], object_by_id[first])
            )
    expected = {
        (robot, object_name)
        for robot in ROBOT_OBJECT_GEOMS
        for object_name in object_names
    }
    counts = Counter(observed)
    missing = sorted(expected - set(counts))
    duplicates = sorted(pair for pair, count in counts.items() if count != 1)
    extras = sorted(set(counts) - expected)
    if missing or duplicates or extras:
        raise AssertionError(
            "compiled pair matrix mismatch: "
            f"missing={missing[:5]} duplicates={duplicates[:5]} "
            f"extras={extras[:5]}"
        )

    geom_rows = []
    for index, (gid, name) in enumerate(zip(object_ids, object_names)):
        geom_rows.append(
            {
                "geom_index": index,
                "geom_name": name,
                "geom_type": "box",
                "pos": " ".join(f"{value:.9g}" for value in model.geom_pos[gid]),
                "quat": " ".join(
                    f"{value:.9g}" for value in model.geom_quat[gid]
                ),
                "size": " ".join(
                    f"{value:.9g}" for value in model.geom_size[gid, :3]
                ),
            }
        )
    return {
        "model": model,
        "object_ids": object_ids,
        "object_names": object_names,
        "object_geom_count": len(object_ids),
        "compiled_robot_object_pair_count": len(observed),
        "expected_robot_object_pair_count": len(expected),
        "compile_seconds": time.perf_counter() - started,
        "geom_rows": geom_rows,
    }


def reference_first5_diagnostic(
    model: mujoco.MjModel,
    object_ids: list[int],
    reference: np.ndarray,
) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim != 2:
        raise ValueError(f"invalid converted reference: {reference.shape}")
    if reference.shape[1] != model.nq:
        raise ValueError(
            f"reference nq={reference.shape[1]} != scene nq={model.nq}"
        )
    lower_ids = [
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, name
        )
        for name in LOWER_BODY_GEOMS
    ]
    data = mujoco.MjData(model)
    fromto = np.zeros(6, dtype=np.float64)
    minimum = float("inf")
    penetrating_rows = 0
    frame_any = 0
    frames = reference[: min(5, len(reference))]
    for frame in frames:
        data.qpos[:] = frame
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        any_penetration = False
        for lower_id in lower_ids:
            distance = min(
                float(
                    mujoco.mj_geomDistance(
                        model,
                        data,
                        lower_id,
                        object_id,
                        10.0,
                        fromto,
                    )
                )
                for object_id in object_ids
            )
            minimum = min(minimum, distance)
            penetrating_rows += int(distance < 0.0)
            any_penetration |= distance < 0.0
        frame_any += int(any_penetration)
    denominator = max(1, len(frames) * len(lower_ids))
    return {
        "reference_first5_union_min_distance_m": minimum,
        "reference_first5_lowerbody_penetration_frac": (
            penetrating_rows / denominator
        ),
        "reference_first5_any_penetration_frame_frac": (
            frame_any / max(1, len(frames))
        ),
    }


def local_authorities(
    e174_row: dict[str, str]
) -> tuple[Path, Path, Path]:
    task_dir = repo_path(e174_row["scene_act"]).parent
    target_scene = task_dir / "scene.xml"
    trajectory = task_dir / "0/trajectory_kinematic.npz"
    e174_config = repo_path(e174_row["config_act"])
    config = yaml.safe_load(e174_config.read_text(encoding="utf-8"))
    contact_mask = repo_path(str(config["contact_hdmi_mask_path"]))
    for path in (target_scene, trajectory, contact_mask):
        if not path.is_file():
            raise FileNotFoundError(path)
    return target_scene, trajectory, contact_mask


def semantic_xml_signature(element: ET.Element) -> tuple[Any, ...]:
    """Compare XML meaning without indentation or attribute-order noise."""
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        (element.text or "").strip(),
        tuple(semantic_xml_signature(child) for child in element),
    )


def write_override(e174_row: dict[str, str]) -> Path:
    override_id = f"core4d_E175_{e174_row['case_id']}_multiGeom"
    output = OVERRIDE_DIR / f"{override_id}.yaml"
    payload = {
        "defaults": [e174_row["override_id"], "_self_"],
        "scene_name": SCENE_NAME,
        "object_collision_sdf_mode": "union",
    }
    header = (
        "# @package _global_\n"
        "# Auto-generated by E175 build_nonbox_multigeom_production.py.\n"
    )
    output.write_text(
        header + yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return output


def audit_override(
    e174_override_id: str, e175_override: Path
) -> tuple[bool, str]:
    config_dir = (REPO / "examples/config").resolve()
    with initialize_config_dir(
        version_base=None, config_dir=str(config_dir)
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
                overrides=[f"+override={e175_override.stem}"],
            ),
            resolve=True,
        )
    assert isinstance(baseline, dict) and isinstance(candidate, dict)
    allowed = {"scene_name", "object_collision_sdf_mode"}
    failures = [
        f"non_axis_drift:{key}"
        for key in sorted(set(baseline) | set(candidate))
        if key not in allowed and baseline.get(key) != candidate.get(key)
    ]
    if candidate.get("scene_name") != SCENE_NAME:
        failures.append("axis_mismatch:scene_name")
    if candidate.get("object_collision_sdf_mode") != "union":
        failures.append("axis_mismatch:object_collision_sdf_mode")
    return not failures, ";".join(failures)


def artifact_paths(case_id: str, stage: str) -> dict[str, Any]:
    is_canary = stage == "canary"
    suffix = "canary" if is_canary else "full"
    variant = f"E175_{case_id}_multiGeom_{suffix}"
    root = f"workspace/core4d/results/E175/s6_downstream/cem/{stage}"
    return {
        "variant": variant,
        "result_npz": f"{root}/{variant}.npz",
        "outdir_npz": (
            f"{root}/{variant}_outdir/trajectory_mjwp_act.npz"
        ),
        "config_act": f"{root}/{variant}_outdir/config_act.yaml",
        "video": (
            "workspace/core4d/results/E175/s6_downstream/render/"
            f"{stage}/{variant}.mp4"
        ),
        "log": f"logs/E175/cem/{stage}/{variant}.log",
        "cem_samples": 64 if is_canary else 1024,
        "cem_opt_steps": 4 if is_canary else 32,
        "cem_seed": 0,
        "execution_mode": "canary" if is_canary else "production",
        "status": "READY_FOR_CANARY" if is_canary else "READY_FOR_FULL",
    }


def build_scene(
    e174_row: dict[str, str],
    bucket_proxy_cache: dict[str, tuple[str, int]],
    trajectory: Path,
    *,
    overwrite: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_id = e174_row["case_id"]
    object_key = e174_row["object_key"]
    source_scene = repo_path(e174_row["scene_act"])
    source_root = ET.parse(source_scene).getroot()
    root = copy.deepcopy(source_root)
    object_body = find_object_body(root)
    mesh = resolve_object_mesh(source_scene)

    if object_key.startswith("bucket"):
        mesh_digest = sha256(mesh)
        if mesh_digest not in bucket_proxy_cache:
            bucket_proxy_cache[mesh_digest] = build_bucket_proxy_xml(mesh)
        proxy_xml, expected_count = bucket_proxy_cache[mesh_digest]
        object_names = replace_bucket_proxy(object_body, proxy_xml)
        if len(object_names) != expected_count:
            raise AssertionError("cached bucket proxy count drift")
        proxy_variant = f"bucket_surface_voxel_{BUCKET_TARGET_CELLS}"
        collision_policy = "bucket_surface_voxel_multibox_proxy_draft"
    elif object_key == "desk007":
        object_names = object_collision_names(object_body)
        proxy_variant = "desk_surface_voxel_26_existing"
        collision_policy = "desk_surface_voxel_multibox_proxy_draft"
    else:
        raise ValueError(f"unexpected E175 object: {object_key}")

    xml_pair_count = replace_robot_object_pairs(root, object_names)
    if stripped_signature(source_root) != stripped_signature(root):
        raise AssertionError("scene drift outside proxy/pair axes")

    output = source_scene.with_name(f"{SCENE_NAME}.xml")
    tree = ET.ElementTree(root)
    if output.exists() and not overwrite:
        existing = ET.parse(output).getroot()
        if semantic_xml_signature(existing) != semantic_xml_signature(root):
            raise FileExistsError(f"different E175 scene exists: {output}")
    else:
        ET.indent(tree, space="  ")
        tree.write(output, encoding="utf-8", xml_declaration=True)

    compiled = compiled_contract(output)
    if compiled["compiled_robot_object_pair_count"] != xml_pair_count:
        raise AssertionError("XML/compiled pair count drift")
    converted_reference = runner_reference(
        trajectory,
        repo_path(e174_row["config_act"]),
        output,
    )["qpos"]
    diagnostic = reference_first5_diagnostic(
        compiled["model"],
        compiled["object_ids"],
        converted_reference,
    )

    snapshot = RESULTS / "scene_snapshot/nonbox_multigeom" / case_id
    snapshot.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_scene, snapshot / source_scene.name)
    shutil.copy2(output, snapshot / output.name)

    scene = {
        "case_id": case_id,
        "object_key": object_key,
        "proxy_variant": proxy_variant,
        "collision_policy": collision_policy,
        "mesh_path": rel(mesh),
        "mesh_sha256": sha256(mesh),
        "source_scene_act": rel(source_scene),
        "source_scene_sha256": sha256(source_scene),
        "scene_act": rel(output),
        "scene_name": SCENE_NAME,
        "effective_scene_sha256": sha256(output),
        "object_geom_count": compiled["object_geom_count"],
        "object_collision_geom_names": ",".join(
            compiled["object_names"]
        ),
        "compiled_robot_object_pair_count": compiled[
            "compiled_robot_object_pair_count"
        ],
        "expected_robot_object_pair_count": compiled[
            "expected_robot_object_pair_count"
        ],
        "scene_compile_seconds": compiled["compile_seconds"],
        **diagnostic,
    }
    geom_rows = [
        {
            "case_id": case_id,
            "object_key": object_key,
            "proxy_variant": proxy_variant,
            "scene_act": rel(output),
            **row,
        }
        for row in compiled["geom_rows"]
    ]
    return scene, geom_rows


def manifest_row(
    source: dict[str, str],
    scene: dict[str, Any],
    override: Path,
    target_scene: Path,
    trajectory: Path,
    contact_mask: Path,
    stage: str,
    assigned_gpu: int,
) -> dict[str, Any]:
    row: dict[str, Any] = dict(source)
    row.update(
        {
            "source_e174_status": source.get("status", ""),
            "source_e174_scene_act": source["scene_act"],
            "source_e174_scene_sha256": source[
                "effective_scene_sha256"
            ],
            "source_e174_override_id": source["override_id"],
            "source_e174_override_sha256": source["override_sha256"],
            "source_e174_result_npz": source["result_npz"],
            "source_e174_outdir_npz": source["outdir_npz"],
            "spider_method_id": METHOD_ID,
            "cell_id": "production_multiGeom",
            "proxy_variant": scene["proxy_variant"],
            "collision_policy": scene["collision_policy"],
            "physics_pair_mode": "union",
            "object_collision_sdf_mode": "union",
            "object_geom_count": scene["object_geom_count"],
            "object_collision_geom_names": scene[
                "object_collision_geom_names"
            ],
            "compiled_robot_object_pair_count": scene[
                "compiled_robot_object_pair_count"
            ],
            "expected_robot_object_pair_count": scene[
                "expected_robot_object_pair_count"
            ],
            "reference_first5_union_min_distance_m": scene[
                "reference_first5_union_min_distance_m"
            ],
            "reference_first5_lowerbody_penetration_frac": scene[
                "reference_first5_lowerbody_penetration_frac"
            ],
            "reference_first5_any_penetration_frame_frac": scene[
                "reference_first5_any_penetration_frame_frac"
            ],
            "target_scene": rel(target_scene),
            "trajectory": rel(trajectory),
            "contact_mask": rel(contact_mask),
            "assigned_gpu": assigned_gpu,
            "gpu_id": "",
            "override_id": override.stem,
            "override_path": rel(override),
            "override_sha256": sha256(override),
            "scene_act": scene["scene_act"],
            "scene_name": SCENE_NAME,
            "base_scene_sha256": scene["source_scene_sha256"],
            "effective_scene_sha256": scene[
                "effective_scene_sha256"
            ],
            "trajectory_sha256": sha256(trajectory),
            "contact_mask_sha256": sha256(contact_mask),
            "failure_mode": "",
            "blocker_detail": "",
            "updated_at": now(),
        }
    )
    row.update(artifact_paths(source["case_id"], stage))
    return row


def build(
    *, overwrite: bool, require_review: bool
) -> dict[str, Any]:
    sources = read_tsv(E174_MANIFEST)
    case_ids = [row["case_id"] for row in sources]
    distribution = Counter(row["object_key"] for row in sources)
    if len(sources) != 39 or len(case_ids) != len(set(case_ids)):
        raise ValueError(
            f"E174 authority must be 39 unique rows, got {len(sources)}"
        )
    if dict(distribution) != EXPECTED_OBJECT_COUNTS:
        raise ValueError(
            f"E174 object distribution drift: {dict(distribution)}"
        )
    if not set(CANARY_CASES).issubset(case_ids):
        raise ValueError("canary case missing from E174 authority")

    bucket_proxy_cache: dict[str, tuple[str, int]] = {}
    scenes: dict[str, dict[str, Any]] = {}
    scene_audit: list[dict[str, Any]] = []
    geom_manifest: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    canary_rows: list[dict[str, Any]] = []

    for ordinal, source in enumerate(sources, start=1):
        target_scene, trajectory, contact_mask = local_authorities(source)
        scene, geom_rows = build_scene(
            source,
            bucket_proxy_cache,
            trajectory,
            overwrite=overwrite,
        )
        scenes[source["case_id"]] = scene
        geom_manifest.extend(geom_rows)
        override = write_override(source)
        override_ok, override_detail = audit_override(
            source["override_id"], override
        )
        if not override_ok:
            raise AssertionError(
                f"{source['case_id']} override parity: {override_detail}"
            )
        scene_audit.append(
            {
                **scene,
                "override_id": override.stem,
                "override_sha256": sha256(override),
                "override_parity": "pass",
                "status": "pass",
            }
        )
        full_rows.append(
            manifest_row(
                source,
                scene,
                override,
                target_scene,
                trajectory,
                contact_mask,
                "full",
                (ordinal - 1) % 4,
            )
        )

    source_by_case = {row["case_id"]: row for row in sources}
    for ordinal, case_id in enumerate(CANARY_CASES):
        source = source_by_case[case_id]
        scene = scenes[case_id]
        override = OVERRIDE_DIR / f"core4d_E175_{case_id}_multiGeom.yaml"
        target_scene, trajectory, contact_mask = local_authorities(source)
        canary_rows.append(
            manifest_row(
                source,
                scene,
                override,
                target_scene,
                trajectory,
                contact_mask,
                "canary",
                ordinal % 4,
            )
        )

    manifest_root = RESULTS / "s6_downstream/manifests"
    write_tsv(
        manifest_root / "nonbox_multigeom_full_manifest.tsv", full_rows
    )
    write_tsv(
        manifest_root / "nonbox_multigeom_canary_manifest.tsv",
        canary_rows,
    )
    write_tsv(
        RESULTS / "scene_snapshot/nonbox_multigeom/scene_audit.tsv",
        scene_audit,
    )
    write_tsv(
        RESULTS / "scene_snapshot/nonbox_multigeom/"
        "scene_variant_manifest.tsv",
        geom_manifest,
    )

    review_path = (
        RESULTS
        / "scene_snapshot/nonbox_multigeom/proxy_visual_review.tsv"
    )
    previous_reviews = (
        {
            row["object_key"]: row
            for row in read_tsv(review_path)
        }
        if review_path.is_file()
        else {}
    )
    review_rows = []
    for object_key in EXPECTED_OBJECT_COUNTS:
        exemplar = next(
            row for row in scene_audit if row["object_key"] == object_key
        )
        previous = previous_reviews.get(object_key, {})
        review_rows.append(
            {
                "object_key": object_key,
                "source_scene_task": Path(
                    exemplar["scene_act"]
                ).parent.name,
                "object_category": (
                    "bucket"
                    if object_key.startswith("bucket")
                    else "desk"
                ),
                "template_status": "manual_review_required",
                "recommended_action": "review_mesh_collision_overlay",
                "proxy_variant": exemplar["proxy_variant"],
                "proxy_policy": exemplar["collision_policy"],
                "mesh_path": exemplar["mesh_path"],
                "scene_act": exemplar["scene_act"],
                "scene_xml": exemplar["scene_act"],
                "object_geom_count": exemplar["object_geom_count"],
                "review_decision": previous.get(
                    "review_decision", "PENDING_CODEX_REVIEW"
                ),
                "reviewer": previous.get("reviewer", ""),
                "reviewed_at": previous.get("reviewed_at", ""),
                "evidence_path": previous.get("evidence_path", ""),
                "notes": previous.get("notes", ""),
            }
        )
    write_tsv(review_path, review_rows)
    if require_review:
        pending = [
            row["object_key"]
            for row in review_rows
            if row["review_decision"] != "approve_clean"
        ]
        if pending:
            raise ValueError(f"proxy visual review pending: {pending}")

    summary = {
        "created_at": now(),
        "source_manifest": rel(E174_MANIFEST),
        "source_rows": len(sources),
        "full_rows": len(full_rows),
        "canary_rows": len(canary_rows),
        "case_set_equal": set(case_ids)
        == {row["case_id"] for row in full_rows},
        "object_distribution": dict(distribution),
        "bucket_target_cells": BUCKET_TARGET_CELLS,
        "bucket_max_boxes": BUCKET_MAX_BOXES,
        "unique_bucket_proxy_meshes": len(bucket_proxy_cache),
        "scene_audit_pass": sum(
            row["status"] == "pass" for row in scene_audit
        ),
        "review_pending": [
            row["object_key"]
            for row in review_rows
            if row["review_decision"] != "approve_clean"
        ],
        "status": "pass",
    }
    write_json(
        manifest_root / "nonbox_multigeom_manifest_summary.json",
        summary,
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-review", action="store_true")
    args = parser.parse_args()
    summary = build(
        overwrite=args.overwrite, require_review=args.require_review
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
