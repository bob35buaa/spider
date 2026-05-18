#!/usr/bin/env python3
"""Generate E015 dynamic-support scene XMLs and augmented reference data."""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


REPO = Path(__file__).resolve().parents[4]
BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
VARIANTS = REPO / "workspace/core4d_collab_retarget/scripts/E015/variants.tsv"

FIELDNAMES = [
    "variant",
    "source_task",
    "mask_slug",
    "person_idx",
    "queue",
    "role",
    "scene_name",
    "data_relpath",
    "support_proxy_point_local_x",
    "support_proxy_point_local_y",
    "support_proxy_point_local_z",
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
]

SUPPORT_BODY = "support_dynamic_anchor"
WELD_NAME = "e015_support_weld"


def read_variants(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(
            csv.DictReader(
                (line for line in f if line.strip() and not line.startswith("#")),
                delimiter="\t",
                fieldnames=FIELDNAMES,
            )
        )


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


def _quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = q.astype(np.float64)
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8, None)
    qvec = q[..., 1:]
    t = 2.0 * np.cross(qvec, v)
    return v + q[..., :1] * t + np.cross(qvec, t)


def _point_local(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [
            float(row["support_proxy_point_local_x"]),
            float(row["support_proxy_point_local_y"]),
            float(row["support_proxy_point_local_z"]),
        ],
        dtype=np.float64,
    )


def _remove_named_children(parent: ET.Element, tag: str, names: set[str]) -> None:
    for child in list(parent.findall(tag)):
        if child.get("name") in names:
            parent.remove(child)


def _joint(
    parent: ET.Element,
    name: str,
    joint_type: str,
    axis: str,
    damping: float,
    armature: float,
) -> None:
    ET.SubElement(
        parent,
        "joint",
        {
            "name": name,
            "type": joint_type,
            "axis": axis,
            "limited": "false",
            "damping": f"{damping:.8g}",
            "armature": f"{armature:.8g}",
        },
    )


def generate_scene(row: dict[str, str]) -> Path:
    task_dir = BASE / row["source_task"]
    src = task_dir / "scene.xml"
    if not src.is_file():
        raise FileNotFoundError(src)

    tree = ET.parse(src)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{src} missing worldbody")
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")

    _remove_named_children(worldbody, "body", {SUPPORT_BODY, "support_weld_anchor"})
    _remove_named_children(equality, "weld", {WELD_NAME, "e014_support_weld", "object_weld"})

    object_body = None
    object_idx = -1
    for idx, child in enumerate(list(worldbody)):
        if child.tag == "body" and child.get("name") == "object":
            object_body = child
            object_idx = idx
            break
    if object_body is None:
        raise ValueError(f"{src} missing top-level object body")

    mass = float(row["support_dynamic_mass"])
    inertia = max(0.01 * mass, 0.005)
    support = ET.Element("body", {"name": SUPPORT_BODY, "pos": "0 0 0"})
    _joint(support, "support_dyn_tx", "slide", "1 0 0", damping=0.0, armature=0.001)
    _joint(support, "support_dyn_ty", "slide", "0 1 0", damping=0.0, armature=0.001)
    _joint(support, "support_dyn_tz", "slide", "0 0 1", damping=0.0, armature=0.001)
    _joint(support, "support_dyn_rx", "hinge", "1 0 0", damping=0.0, armature=0.001)
    _joint(support, "support_dyn_ry", "hinge", "0 1 0", damping=0.0, armature=0.001)
    _joint(support, "support_dyn_rz", "hinge", "0 0 1", damping=0.0, armature=0.001)
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
            "size": "0.035",
            "rgba": "0.0 0.45 1.0 0.35",
            "group": "4",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(
        support,
        "site",
        {
            "name": "trace_support_dynamic_anchor",
            "size": "0.025",
            "rgba": "0.0 0.45 1.0 1",
        },
    )
    worldbody.insert(object_idx, support)

    point_local = _point_local(row)
    ET.SubElement(
        equality,
        "weld",
        {
            "name": WELD_NAME,
            "body1": "object",
            "body2": SUPPORT_BODY,
            "relpose": (
                f"{point_local[0]:.8g} {point_local[1]:.8g} "
                f"{point_local[2]:.8g} 1 0 0 0"
            ),
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
        raise ValueError(f"{out} dims changed: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    support_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, SUPPORT_BODY)
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if support_id < 0 or object_id < 0:
        raise ValueError(f"{out} missing support/object body")
    if model.body_mocapid[support_id] >= 0:
        raise ValueError(f"{out} support body is mocap")
    if int(model.body_jntnum[support_id]) != 6:
        raise ValueError(f"{out} support joint count is {model.body_jntnum[support_id]}")
    support_jadr = int(model.body_jntadr[support_id])
    support_qadr = int(model.jnt_qposadr[support_jadr])
    support_dadr = int(model.jnt_dofadr[support_jadr])
    object_jadr = int(model.body_jntadr[object_id])
    object_qadr = int(model.jnt_qposadr[object_jadr])
    object_dadr = int(model.jnt_dofadr[object_jadr])
    if support_qadr != 36 or support_dadr != 35:
        raise ValueError(f"{out} support addresses got q={support_qadr}, d={support_dadr}")
    if object_qadr != model.nq - 7 or object_dadr != model.nv - 6:
        raise ValueError(f"{out} object is not last: q={object_qadr}, d={object_dadr}")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, WELD_NAME) < 0:
        raise ValueError(f"{out} missing equality {WELD_NAME}")
    return out


def _source_ref_dt(task_dir: Path) -> float:
    info = task_dir / "task_info.json"
    if not info.is_file():
        return 1.0 / 30.0
    data = json.loads(info.read_text(encoding="utf-8"))
    return float(data.get("ref_dt", 1.0 / 30.0))


def _support_qpos_from_object(qpos: np.ndarray, point_local: np.ndarray) -> np.ndarray:
    obj_pos = qpos[:, -7:-4].astype(np.float64)
    obj_quat = qpos[:, -4:].astype(np.float64)
    support_pos = obj_pos + _quat_apply(obj_quat, np.broadcast_to(point_local, obj_pos.shape))
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
    task_dir = BASE / row["source_task"]
    src = task_dir / "0/trajectory_kinematic.npz"
    if not src.is_file():
        raise FileNotFoundError(src)
    data = np.load(src, allow_pickle=True)
    qpos = data["qpos"].astype(np.float64)
    qvel = data["qvel"].astype(np.float64)
    if qpos.shape[1] != 43 or qvel.shape[1] != 41:
        raise ValueError(f"{src} expected 43/41 qpos/qvel, got {qpos.shape}/{qvel.shape}")

    point_local = _point_local(row)
    support_qpos = _support_qpos_from_object(qpos, point_local)
    support_qvel = _finite_velocity(support_qpos, _source_ref_dt(task_dir))

    qpos_aug = np.zeros((qpos.shape[0], 49), dtype=qpos.dtype)
    qvel_aug = np.zeros((qvel.shape[0], 47), dtype=qvel.dtype)
    qpos_aug[:, :36] = qpos[:, :36]
    qpos_aug[:, 36:42] = support_qpos
    qpos_aug[:, 42:] = qpos[:, -7:]
    qvel_aug[:, :35] = qvel[:, :35]
    qvel_aug[:, 35:41] = support_qvel
    qvel_aug[:, 41:] = qvel[:, -6:]

    out = REPO / row["data_relpath"]
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: data[key] for key in data.files if key not in {"qpos", "qvel"}}
    np.savez(out, qpos=qpos_aug, qvel=qvel_aug, **payload)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=Path, default=VARIANTS)
    args = parser.parse_args()

    for row in read_variants(args.variants):
        scene = generate_scene(row)
        data = generate_data(row)
        print(f"Wrote {scene.relative_to(REPO)}")
        print(f"Wrote {data.relative_to(REPO)}")


if __name__ == "__main__":
    main()
