#!/usr/bin/env python3
"""Generate E014 soft-weld support-anchor scene XMLs."""

from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
VARIANTS = REPO / "workspace/core4d_collab_retarget/scripts/E014/variants.tsv"

FIELDNAMES = [
    "variant",
    "source_task",
    "mask_slug",
    "person_idx",
    "queue",
    "role",
    "wave",
    "scene_name",
    "support_proxy_point_local_x",
    "support_proxy_point_local_y",
    "support_proxy_point_local_z",
    "weld_solref_timeconst",
    "weld_solimp_1",
    "weld_solimp_2",
    "weld_solimp_width",
    "support_proxy_gravity_scale",
    "hold_contact_rew_scale",
    "hold_contact_sigma",
    "hold_contact_start_eval_time",
    "hold_contact_end_eval_time",
    "hold_contact_require_ref_contact",
]

ANCHOR_BODY = "support_weld_anchor"
ANCHOR_GEOM = "support_weld_anchor_geom"
WELD_NAME = "e014_support_weld"


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
    q = q / np.clip(np.linalg.norm(q), 1e-8, None)
    qvec = q[1:]
    t = 2.0 * np.cross(qvec, v)
    return v + q[0] * t + np.cross(qvec, t)


def _first_support_pose(task_dir: Path, point_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(task_dir / "0/trajectory_kinematic.npz")
    qpos0 = data["qpos"][0]
    obj_pos = qpos0[-7:-4].astype(np.float64)
    obj_quat = qpos0[-4:].astype(np.float64)
    support_pos = obj_pos + _quat_apply(obj_quat, point_local)
    return support_pos, obj_quat / np.clip(np.linalg.norm(obj_quat), 1e-8, None)


def _remove_named_children(parent: ET.Element, tag: str, names: set[str]) -> None:
    for child in list(parent.findall(tag)):
        if child.get("name") in names:
            parent.remove(child)


def generate_scene(row: dict[str, str]) -> Path:
    task = row["source_task"]
    task_dir = BASE / task
    src = task_dir / "scene.xml"
    if not src.is_file():
        raise FileNotFoundError(src)

    point_local = np.array(
        [
            float(row["support_proxy_point_local_x"]),
            float(row["support_proxy_point_local_y"]),
            float(row["support_proxy_point_local_z"]),
        ],
        dtype=np.float64,
    )
    support_pos, support_quat = _first_support_pose(task_dir, point_local)

    tree = ET.parse(src)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{src} missing worldbody")

    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")

    _remove_named_children(worldbody, "body", {ANCHOR_BODY, "object_target"})
    _remove_named_children(equality, "weld", {WELD_NAME, "object_weld"})

    anchor = ET.Element(
        "body",
        {
            "name": ANCHOR_BODY,
            "mocap": "true",
            "pos": " ".join(f"{v:.8g}" for v in support_pos),
            "quat": " ".join(f"{v:.8g}" for v in support_quat),
        },
    )
    ET.SubElement(
        anchor,
        "geom",
        {
            "name": ANCHOR_GEOM,
            "type": "sphere",
            "size": "0.035",
            "rgba": "0.0 0.45 1.0 0.35",
            "group": "4",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(
        anchor,
        "site",
        {
            "name": "trace_support_weld_anchor",
            "size": "0.025",
            "rgba": "0.0 0.45 1.0 1",
        },
    )
    worldbody.append(anchor)

    ET.SubElement(
        equality,
        "weld",
        {
            "name": WELD_NAME,
            "body1": "object",
            "body2": ANCHOR_BODY,
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
    if model.nq != 43 or model.nu != 29:
        raise ValueError(f"{out} changed dimensions: nq={model.nq}, nu={model.nu}")
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ANCHOR_BODY)
    if body_id < 0 or model.body_mocapid[body_id] < 0:
        raise ValueError(f"{out} missing mocap body {ANCHOR_BODY}")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, WELD_NAME) < 0:
        raise ValueError(f"{out} missing weld equality {WELD_NAME}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=Path, default=VARIANTS)
    args = parser.parse_args()

    seen: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_variants(args.variants):
        key = (row["source_task"], row["scene_name"])
        prior = seen.get(key)
        if prior is not None:
            same = all(
                prior[name] == row[name]
                for name in [
                    "support_proxy_point_local_x",
                    "support_proxy_point_local_y",
                    "support_proxy_point_local_z",
                    "weld_solref_timeconst",
                    "weld_solimp_1",
                    "weld_solimp_2",
                    "weld_solimp_width",
                ]
            )
            if not same:
                raise ValueError(f"Conflicting scene parameters for {key}")
            continue
        seen[key] = row
        out = generate_scene(row)
        print(f"Wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
