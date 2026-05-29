#!/usr/bin/env python3
"""Generate E029 D6-equivalent dynamic-support scenes and references."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e029_common as common  # noqa: E402

E016_DIR = common.REPO / "workspace/core4d_collab_retarget/scripts/E016"
if str(E016_DIR) not in sys.path:
    sys.path.insert(0, str(E016_DIR))
import generate_e016_assets as e016_assets  # noqa: E402


RESULT_ROOT = common.E029_RESULTS / "d6"
MANIFEST = RESULT_ROOT / "manifest.tsv"
PREFLIGHT = common.E029_RESULTS / "preflight/axis_contact_summary.csv"
SUPPORT_BODY = "support_dynamic_anchor"
WELD_NAME_PREFIX = "e029_support_weld"

PROFILES = {
    "d6_locked": {
        "support_dynamic_mass": 2.0,
        "support_dynamic_pos_kp": 400.0,
        "support_dynamic_pos_kd": -1.0,
        "support_dynamic_rot_kp": 60.0,
        "support_dynamic_rot_kd": -1.0,
        "support_dynamic_force_clamp": 250.0,
        "support_dynamic_torque_clamp": 80.0,
        "weld_solref_timeconst": 0.02,
        "weld_solimp_1": 0.9,
        "weld_solimp_2": 0.98,
        "weld_solimp_width": 0.001,
        "full_queue": "sanity",
    },
    "d6_compliant": {
        "support_dynamic_mass": 2.0,
        "support_dynamic_pos_kp": 200.0,
        "support_dynamic_pos_kd": -1.0,
        "support_dynamic_rot_kp": 20.0,
        "support_dynamic_rot_kd": -1.0,
        "support_dynamic_force_clamp": 120.0,
        "support_dynamic_torque_clamp": 30.0,
        "weld_solref_timeconst": 0.05,
        "weld_solimp_1": 0.75,
        "weld_solimp_2": 0.95,
        "weld_solimp_width": 0.005,
        "full_queue": "local",
    },
}


FIELDNAMES = [
    "variant",
    "source_e028_variant",
    "source_task",
    "derived_task",
    "person_idx",
    "profile",
    "queue",
    "scene_name",
    "data_relpath",
    "support_endpoint_policy",
    "height_local_axis",
    "selected_side_face",
    "selected_side_frac",
    "selected_side_margin",
    "endpoint_confidence",
    "support_proxy_point_local_x",
    "support_proxy_point_local_y",
    "support_proxy_point_local_z",
    "selected_side_centroid_x",
    "selected_side_centroid_y",
    "selected_side_centroid_z",
    "object_half_x",
    "object_half_y",
    "object_half_z",
    "support_dynamic_mass",
    "support_dynamic_pos_kp",
    "support_dynamic_pos_kd",
    "support_dynamic_rot_kp",
    "support_dynamic_rot_kd",
    "support_dynamic_force_clamp",
    "support_dynamic_torque_clamp",
    "weld_solref_timeconst",
    "weld_solimp_1",
    "weld_solimp_2",
    "weld_solimp_width",
    "hold_contact_rew_scale",
    "hold_contact_sigma",
    "hold_contact_start_eval_time",
    "hold_contact_end_eval_time",
    "hold_contact_require_ref_contact",
    "mask_path_source",
    "mask_slug",
]


def _read_preflight(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["variant"]: row for row in csv.DictReader(f)}


def _parse_point(text: str) -> np.ndarray:
    return np.asarray([float(v) for v in text.split(",")], dtype=np.float64)


def _indent(elem: ET.Element, level: int = 0) -> None:
    spacer = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = spacer + "  "
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = spacer
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = spacer


def _remove_named_children(parent: ET.Element, tag: str, names: set[str]) -> None:
    for child in list(parent.findall(tag)):
        if child.get("name") in names:
            parent.remove(child)


def _joint(parent: ET.Element, name: str, joint_type: str, axis: str) -> None:
    ET.SubElement(
        parent,
        "joint",
        {
            "name": name,
            "type": joint_type,
            "axis": axis,
            "limited": "false",
            "damping": "0",
            "armature": "0.001",
        },
    )


def _variant_stem(e028_variant: str) -> str:
    stem = e028_variant.removeprefix("E028_").removesuffix("_canonical_t02")
    return f"E029_{stem}"


def _endpoint(row: dict[str, str], preflight: dict[str, str]) -> tuple[np.ndarray, str]:
    half = common.object_half(row)
    side = preflight["selected_side_face"]
    if side not in common.SIDE_FACES:
        # B3 修复 (exp_diagnostic_v2 §2)：原回退是无条件 "+x"；B1 修复后
        # row["anchor_face"] 可能是 ±z，此时仍需选一个 SIDE。优先级：
        # 1. row["anchor_face"] 本身已是 SIDE → 用它
        # 2. preflight 提供了 selected_side_face 备选则用之
        # 3. 否则 raise，显式告知 caller 该 case 需要人工选 side
        anchor_face = row.get("anchor_face", "")
        if anchor_face in common.SIDE_FACES:
            side = anchor_face
        elif preflight.get("selected_side_face") in common.SIDE_FACES:
            side = preflight["selected_side_face"]
        else:
            raise ValueError(
                f"_endpoint: cannot pick SIDE face for row={row.get('source_task','?')!r} "
                f"anchor_face={anchor_face!r} preflight selected_side_face="
                f"{preflight.get('selected_side_face')!r}. "
                "Either rerun preflight or specify side manually."
            )
    centroid = _parse_point(preflight["selected_side_centroid"])
    if not np.all(np.isfinite(centroid)):
        centroid = _parse_point(preflight["selected_robust_centroid"])
    if not np.all(np.isfinite(centroid)):
        centroid = common.anchor_local(row)
    point = np.clip(centroid, -0.95 * half, 0.95 * half)
    side_axis = common.AXES.index(side[1])
    point[side_axis] = half[side_axis] if side[0] == "+" else -half[side_axis]
    confidence = "axis_review"
    if (
        preflight.get("height_local_axis") == "y"
        and float(preflight.get("selected_side_margin", "0")) >= 0.12
        and float(preflight.get("selected_side_frac", "0")) >= 0.4
    ):
        confidence = "axis_remap_side_ok"
    return point.astype(np.float64), confidence


def build_manifest(
    *,
    manifest_path: Path,
    candidates_path: Path,
    preflight_path: Path,
    out_path: Path,
) -> list[dict[str, str]]:
    preflight = _read_preflight(preflight_path)
    rows: list[dict[str, str]] = []
    for e028 in common.candidate_rows(manifest_path=manifest_path, candidates_path=candidates_path):
        pf = preflight[e028["variant"]]
        point, confidence = _endpoint(e028, pf)
        centroid = _parse_point(pf["selected_side_centroid"])
        half = common.object_half(e028)
        mask_slug = Path(e028["mask_path_source"]).parent.name
        for profile, params in PROFILES.items():
            variant = f"{_variant_stem(e028['variant'])}_{profile}"
            derived_task = f"{e028['source_task']}_freejoint_legobj_e029d6"
            scene_name = f"scene_e029_{profile}_{e028['source_task']}"
            data_relpath = (
                f"workspace/core4d_collab_retarget/results/E029/d6/data/"
                f"{variant}/trajectory_kinematic.npz"
            )
            row = {
                "variant": variant,
                "source_e028_variant": e028["variant"],
                "source_task": e028["source_task"],
                "derived_task": derived_task,
                "person_idx": e028["person_idx"],
                "profile": profile,
                "queue": str(params["full_queue"]),
                "scene_name": scene_name,
                "data_relpath": data_relpath,
                "support_endpoint_policy": "axis_remap_selected_side_centroid",
                "height_local_axis": pf["height_local_axis"],
                "selected_side_face": pf["selected_side_face"],
                "selected_side_frac": pf["selected_side_frac"],
                "selected_side_margin": pf["selected_side_margin"],
                "endpoint_confidence": confidence,
                "support_proxy_point_local_x": f"{point[0]:.8g}",
                "support_proxy_point_local_y": f"{point[1]:.8g}",
                "support_proxy_point_local_z": f"{point[2]:.8g}",
                "selected_side_centroid_x": f"{centroid[0]:.8g}",
                "selected_side_centroid_y": f"{centroid[1]:.8g}",
                "selected_side_centroid_z": f"{centroid[2]:.8g}",
                "object_half_x": f"{half[0]:.8g}",
                "object_half_y": f"{half[1]:.8g}",
                "object_half_z": f"{half[2]:.8g}",
                "hold_contact_rew_scale": "0.0",
                "hold_contact_sigma": "0.05",
                "hold_contact_start_eval_time": e028.get("hold_contact_start_eval_time", "0.64"),
                "hold_contact_end_eval_time": e028.get("hold_contact_end_eval_time", "4.08"),
                "hold_contact_require_ref_contact": e028.get("hold_contact_require_ref_contact", "true"),
                "mask_path_source": e028["mask_path_source"],
                "mask_slug": mask_slug,
            }
            for key, value in params.items():
                if key == "full_queue":
                    continue
                row[key] = f"{float(value):.8g}"
            rows.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def read_manifest(path: Path = MANIFEST) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _point_local(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [
            float(row["support_proxy_point_local_x"]),
            float(row["support_proxy_point_local_y"]),
            float(row["support_proxy_point_local_z"]),
        ],
        dtype=np.float64,
    )


def generate_scene(row: dict[str, str]) -> Path:
    task_dir = common.BASE / row["derived_task"]
    src = task_dir / "scene.xml"
    tree = ET.parse(src)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{src} missing worldbody")
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")

    weld_name = f"{WELD_NAME_PREFIX}_{row['profile']}"
    _remove_named_children(worldbody, "body", {SUPPORT_BODY, "support_weld_anchor"})
    _remove_named_children(
        equality,
        "weld",
        {weld_name, "e028_support_weld", "e018b_support_weld", "e018_support_weld", "object_weld"},
    )

    object_idx = -1
    for idx, child in enumerate(list(worldbody)):
        if child.tag == "body" and child.get("name") == "object":
            object_idx = idx
            break
    if object_idx < 0:
        raise ValueError(f"{src} missing top-level object body")

    mass = float(row["support_dynamic_mass"])
    inertia = max(0.01 * mass, 0.005)
    support = ET.Element("body", {"name": SUPPORT_BODY, "pos": "0 0 0"})
    _joint(support, "support_dyn_tx", "slide", "1 0 0")
    _joint(support, "support_dyn_ty", "slide", "0 1 0")
    _joint(support, "support_dyn_tz", "slide", "0 0 1")
    _joint(support, "support_dyn_rx", "hinge", "1 0 0")
    _joint(support, "support_dyn_ry", "hinge", "0 1 0")
    _joint(support, "support_dyn_rz", "hinge", "0 0 1")
    ET.SubElement(
        support,
        "inertial",
        {
            "pos": "0 0 0",
            "mass": f"{mass:.8g}",
            "diaginertia": f"{inertia:.8g} {inertia:.8g} {inertia:.8g}",
        },
    )
    ET.SubElement(
        support,
        "geom",
        {
            "name": "support_dynamic_anchor_geom",
            "type": "sphere",
            "size": "0.04",
            "rgba": "0.0 0.45 1.0 0.45",
            "group": "4",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(support, "site", {"name": "trace_support_dynamic_anchor", "size": "0.025"})
    worldbody.insert(object_idx, support)

    point = _point_local(row)
    ET.SubElement(
        equality,
        "weld",
        {
            "name": weld_name,
            "body1": "object",
            "body2": SUPPORT_BODY,
            "relpose": f"{point[0]:.8g} {point[1]:.8g} {point[2]:.8g} 1 0 0 0",
            "solref": f"{float(row['weld_solref_timeconst']):.8g} 1",
            "solimp": (
                f"{float(row['weld_solimp_1']):.8g} "
                f"{float(row['weld_solimp_2']):.8g} "
                f"{float(row['weld_solimp_width']):.8g}"
            ),
        },
    )
    _indent(root)
    out = task_dir / f"{row['scene_name']}.xml"
    tree.write(out, encoding="unicode", xml_declaration=False)
    text = out.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        out.write_text(text + "\n", encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(str(out))
    if (model.nq, model.nv, model.nu) != (49, 47, 29):
        raise ValueError(f"{out} changed dims: nq/nv/nu={model.nq}/{model.nv}/{model.nu}")
    support_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, SUPPORT_BODY)
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    equality_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
    if support_id < 0 or object_id < 0 or equality_id < 0:
        raise ValueError(f"{out} missing support/object/equality")
    if model.body_mocapid[support_id] >= 0:
        raise ValueError(f"{out} support body is mocap")
    if int(model.body_jntnum[support_id]) != 6:
        raise ValueError(f"{out} support joint count={model.body_jntnum[support_id]}")
    support_jadr = int(model.body_jntadr[support_id])
    object_jadr = int(model.body_jntadr[object_id])
    support_qadr = int(model.jnt_qposadr[support_jadr])
    support_dadr = int(model.jnt_dofadr[support_jadr])
    object_qadr = int(model.jnt_qposadr[object_jadr])
    object_dadr = int(model.jnt_dofadr[object_jadr])
    if support_qadr != 36 or support_dadr != 35:
        raise ValueError(f"{out} support addresses q/d={support_qadr}/{support_dadr}")
    if object_qadr != model.nq - 7 or object_dadr != model.nv - 6:
        raise ValueError(f"{out} object not last q/d={object_qadr}/{object_dadr}")
    return out


def _source_ref_dt(task_dir: Path) -> float:
    info = task_dir / "task_info.json"
    if not info.is_file():
        return 0.02
    data = json.loads(info.read_text(encoding="utf-8"))
    return float(data.get("ref_dt", 0.02))


def _support_qpos_from_object(qpos: np.ndarray, point_local: np.ndarray) -> np.ndarray:
    obj_pos = qpos[:, -7:-4].astype(np.float64)
    obj_quat = qpos[:, -4:].astype(np.float64)
    support_pos = obj_pos + np.asarray([common.quat_apply(q, point_local) for q in obj_quat])
    quat_xyzw = np.column_stack([obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3], obj_quat[:, 0]])
    support_euler = R.from_quat(quat_xyzw).as_euler("XYZ")
    return np.concatenate([support_pos, support_euler], axis=1)


def _finite_velocity(values: np.ndarray, dt: float) -> np.ndarray:
    vel = np.zeros_like(values)
    if len(values) > 1:
        vel[1:] = (values[1:] - values[:-1]) / max(dt, 1e-8)
        vel[0] = vel[1]
    return vel


def generate_data(row: dict[str, str]) -> Path:
    task_dir = common.BASE / row["derived_task"]
    src = task_dir / "0/trajectory_kinematic.npz"
    data = np.load(src, allow_pickle=True)
    qpos = data["qpos"].astype(np.float64)
    qvel = data["qvel"].astype(np.float64)
    if qpos.shape[1] != 43 or qvel.shape[1] != 41:
        raise ValueError(f"{src} expected 43/41 qpos/qvel, got {qpos.shape}/{qvel.shape}")
    support_qpos = _support_qpos_from_object(qpos, _point_local(row))
    support_qvel = _finite_velocity(support_qpos, _source_ref_dt(task_dir))
    qpos_aug = np.zeros((qpos.shape[0], 49), dtype=qpos.dtype)
    qvel_aug = np.zeros((qvel.shape[0], 47), dtype=qvel.dtype)
    qpos_aug[:, :36] = qpos[:, :36]
    qpos_aug[:, 36:42] = support_qpos
    qpos_aug[:, 42:] = qpos[:, -7:]
    qvel_aug[:, :35] = qvel[:, :35]
    qvel_aug[:, 35:41] = support_qvel
    qvel_aug[:, 41:] = qvel[:, -6:]
    out = common.REPO / row["data_relpath"]
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: data[key] for key in data.files if key not in {"qpos", "qvel"}}
    np.savez(out, qpos=qpos_aug, qvel=qvel_aug, **payload)
    return out


def prepare_assets(rows: list[dict[str, str]], *, force: bool) -> None:
    prepared: set[str] = set()
    for row in rows:
        if row["derived_task"] not in prepared:
            task_dir = e016_assets.copy_freejoint_case(row["source_task"], row["derived_task"], force=force)
            e016_assets.patch_leg_object_pairs(task_dir / "scene.xml", row["source_task"], row["derived_task"])
            prepared.add(row["derived_task"])
        scene = generate_scene(row)
        data = generate_data(row)
        print(
            f"{row['variant']}: scene={common.rel_or_abs(scene)} data={common.rel_or_abs(data)} "
            f"endpoint=[{row['support_proxy_point_local_x']},{row['support_proxy_point_local_y']},"
            f"{row['support_proxy_point_local_z']}] confidence={row['endpoint_confidence']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--source-manifest", type=Path, default=common.MANIFEST)
    parser.add_argument("--candidates", type=Path, default=common.CANDIDATES)
    parser.add_argument("--preflight", type=Path, default=PREFLIGHT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reuse-manifest", action="store_true")
    args = parser.parse_args()

    if args.reuse_manifest and args.manifest.is_file():
        rows = read_manifest(args.manifest)
    else:
        rows = build_manifest(
            manifest_path=args.source_manifest,
            candidates_path=args.candidates,
            preflight_path=args.preflight,
            out_path=args.manifest,
        )
        print(f"Wrote {common.rel_or_abs(args.manifest)} rows={len(rows)}")
    prepare_assets(rows, force=args.force)


if __name__ == "__main__":
    main()
