#!/usr/bin/env python3
"""Generate E018 canonical support-proxy anchors and scenes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
E016_DIR = REPO / "workspace/core4d_collab_retarget/scripts/E016"
if str(E016_DIR) not in sys.path:
    sys.path.insert(0, str(E016_DIR))

import generate_e016_assets as e016_assets  # noqa: E402


E016_VARIANTS = REPO / "workspace/core4d_collab_retarget/scripts/E016/variants.tsv"
RESULTS = REPO / "workspace/core4d_collab_retarget/results/E018"
MANIFEST = RESULTS / "manifest.tsv"

ANCHOR_BODY = "support_weld_anchor"
ANCHOR_GEOM = "support_weld_anchor_geom"
WELD_NAME = "e018_support_weld"
CANONICAL_Z_FRAC = 0.62

E014_GT = {
    "box023_person2": {"face": "+x", "point": np.array([0.16, 0.0, 0.10], dtype=np.float64)},
    "box025_person2": {"face": "+y", "point": np.array([0.0, 0.38, 0.30], dtype=np.float64)},
}

FIELDNAMES = [
    "variant",
    "source_task",
    "derived_task",
    "mask_source_exp",
    "mask_slug",
    "person_idx",
    "queue",
    "role",
    "wave",
]

MANIFEST_FIELDS = FIELDNAMES + [
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
    "support_point_method",
    "source_variant",
    "anchor_policy",
    "anchor_face",
    "anchor_face_source",
    "canonical_z_frac",
    "object_half_x",
    "object_half_y",
    "object_half_z",
    "gt_anchor_available",
    "gt_anchor_x",
    "gt_anchor_y",
    "gt_anchor_z",
    "gt_anchor_face",
    "gt_anchor_dist_m",
]


def _read_tsv_rows(path: Path, fieldnames: list[str] | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [line for line in f if line.strip() and not line.startswith("#")]
    if fieldnames is None:
        return list(csv.DictReader(rows, delimiter="\t"))
    return list(csv.DictReader(rows, delimiter="\t", fieldnames=fieldnames))


def _quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return e016_assets._quat_apply(q, v)


def _remove_named_children(parent: ET.Element, tag: str, names: set[str]) -> None:
    for child in list(parent.findall(tag)):
        if child.get("name") in names:
            parent.remove(child)


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


def _face_axis_sign(face: str) -> tuple[int, float]:
    axis = 0 if face.endswith("x") else 1
    sign = 1.0 if face.startswith("+") else -1.0
    return axis, sign


def face_label(point: np.ndarray, half: np.ndarray) -> str:
    xy_norm = np.abs(point[:2]) / np.clip(half[:2], 1e-6, None)
    axis = int(np.argmax(xy_norm))
    sign = "+" if point[axis] >= 0.0 else "-"
    return f"{sign}{'xy'[axis]}"


def canonical_anchor(face: str, half: np.ndarray, z_frac: float = CANONICAL_Z_FRAC) -> np.ndarray:
    point = np.zeros(3, dtype=np.float64)
    axis, sign = _face_axis_sign(face)
    point[axis] = sign * float(half[axis])
    point[2] = float(z_frac * half[2])
    return point


def object_half(task: str) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(BASE / task / "scene.xml"))
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")
    if geom < 0:
        raise ValueError(f"{task} has no object_collision geom")
    return model.geom_size[geom, :3].astype(np.float64)


def _format_point(point: np.ndarray) -> tuple[str, str, str]:
    return (f"{point[0]:.8g}", f"{point[1]:.8g}", f"{point[2]:.8g}")


def _slug_from_e016_variant(name: str) -> str:
    return name[5:] if name.startswith("E016_") else name


def generate_support_scene(
    task_dir: Path,
    scene_name: str,
    point_local: np.ndarray,
    *,
    solref_timeconst: str,
    solimp_1: str,
    solimp_2: str,
    solimp_width: str,
) -> Path:
    src = task_dir / "scene.xml"
    qpos0 = np.load(task_dir / "0/trajectory_kinematic.npz")["qpos"][0]
    obj_pos = qpos0[-7:-4].astype(np.float64)
    obj_quat = qpos0[-4:].astype(np.float64)
    support_pos = obj_pos + _quat_apply(obj_quat, point_local)
    support_quat = obj_quat / np.clip(np.linalg.norm(obj_quat), 1e-8, None)

    tree = ET.parse(src)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{src} missing worldbody")
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")

    _remove_named_children(worldbody, "body", {ANCHOR_BODY, "object_target"})
    _remove_named_children(
        equality,
        "weld",
        {WELD_NAME, "e017_support_weld", "e016_support_weld", "e014_support_weld", "object_weld"},
    )

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
            "rgba": "0.9 0.78 0.05 0.45",
            "group": "4",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(anchor, "site", {"name": "trace_support_weld_anchor", "size": "0.025"})
    worldbody.append(anchor)
    ET.SubElement(
        equality,
        "weld",
        {
            "name": WELD_NAME,
            "body1": "object",
            "body2": ANCHOR_BODY,
            "relpose": f"{point_local[0]:.8g} {point_local[1]:.8g} {point_local[2]:.8g} 1 0 0 0",
            "solref": f"{solref_timeconst} 1",
            "solimp": f"{solimp_1} {solimp_2} {solimp_width}",
        },
    )
    _indent(root)
    out = task_dir / f"{scene_name}.xml"
    tree.write(out, encoding="unicode", xml_declaration=False)
    text = out.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        out.write_text(text + "\n", encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(str(out))
    if model.nq != 43 or model.nv != 41 or model.nu != 29:
        raise ValueError(f"{out} changed dims: nq/nv/nu={model.nq}/{model.nv}/{model.nu}")
    return out


def build_manifest_rows(base_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for base_row in base_rows:
        gt = E014_GT.get(base_row["source_task"])
        if gt is None:
            continue
        slug = _slug_from_e016_variant(base_row["variant"])
        half = object_half(base_row["source_task"])
        face = str(gt["face"])
        point = canonical_anchor(face, half)
        gt_point = np.asarray(gt["point"], dtype=np.float64)
        gt_dist = float(np.linalg.norm(point - gt_point))
        x, y, z = _format_point(point)
        gx, gy, gz = _format_point(gt_point)
        rows.append(
            {
                "variant": f"E018_{slug}_canonical_t02",
                "source_task": base_row["source_task"],
                "derived_task": f"{base_row['source_task']}_freejoint_legobj_e018",
                "mask_source_exp": base_row["mask_source_exp"],
                "mask_slug": base_row["mask_slug"],
                "person_idx": base_row["person_idx"],
                "queue": "local",
                "role": "main" if "box025" in base_row["source_task"] else "guard",
                "wave": "A",
                "scene_name": f"scene_e018_jointB_{slug}_canonical_t02",
                "support_proxy_point_local_x": x,
                "support_proxy_point_local_y": y,
                "support_proxy_point_local_z": z,
                "weld_solref_timeconst": "0.02",
                "weld_solimp_1": "0.9",
                "weld_solimp_2": "0.95",
                "weld_solimp_width": "0.001",
                "support_proxy_gravity_scale": "0.5",
                "hold_contact_rew_scale": "0.0",
                "hold_contact_sigma": "0.05",
                "hold_contact_start_eval_time": "0.64",
                "hold_contact_end_eval_time": "4.08",
                "hold_contact_require_ref_contact": "true",
                "support_point_method": "support_proxy_canonical",
                "source_variant": base_row["variant"],
                "anchor_policy": "canonical_face_center_upper",
                "anchor_face": face,
                "anchor_face_source": "e014_gt_template",
                "canonical_z_frac": f"{CANONICAL_Z_FRAC:.6g}",
                "object_half_x": f"{half[0]:.8g}",
                "object_half_y": f"{half[1]:.8g}",
                "object_half_z": f"{half[2]:.8g}",
                "gt_anchor_available": "true",
                "gt_anchor_x": gx,
                "gt_anchor_y": gy,
                "gt_anchor_z": gz,
                "gt_anchor_face": face_label(gt_point, half),
                "gt_anchor_dist_m": f"{gt_dist:.8g}",
            }
        )
    return rows


def write_manifest_and_scenes(rows: list[dict[str, str]], *, force: bool) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    prepared_tasks: set[str] = set()
    for row in rows:
        task = row["derived_task"]
        if task not in prepared_tasks:
            task_dir = e016_assets.copy_freejoint_case(row["source_task"], task, force=force)
            e016_assets.patch_leg_object_pairs(task_dir / "scene.xml", row["source_task"], task)
            prepared_tasks.add(task)
        else:
            task_dir = BASE / task
        point = np.array(
            [
                float(row["support_proxy_point_local_x"]),
                float(row["support_proxy_point_local_y"]),
                float(row["support_proxy_point_local_z"]),
            ],
            dtype=np.float64,
        )
        out = generate_support_scene(
            task_dir,
            row["scene_name"],
            point,
            solref_timeconst=row["weld_solref_timeconst"],
            solimp_1=row["weld_solimp_1"],
            solimp_2=row["weld_solimp_2"],
            solimp_width=row["weld_solimp_width"],
        )
        print(
            f"{row['variant']}: task={task} scene={out.relative_to(REPO)} "
            f"point={point.tolist()} gt_dist={row['gt_anchor_dist_m']}"
        )

    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS / "canonical_anchor_summary.json").write_text(
        json.dumps(
            {
                "num_variants": len(rows),
                "canonical_z_frac": CANONICAL_Z_FRAC,
                "variants": [
                    {
                        "variant": row["variant"],
                        "source_task": row["source_task"],
                        "point": [
                            float(row["support_proxy_point_local_x"]),
                            float(row["support_proxy_point_local_y"]),
                            float(row["support_proxy_point_local_z"]),
                        ],
                        "gt_dist_m": float(row["gt_anchor_dist_m"]),
                    }
                    for row in rows
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {MANIFEST.relative_to(REPO)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=Path, default=E016_VARIANTS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base_rows = _read_tsv_rows(args.variants, FIELDNAMES)
    rows = build_manifest_rows(base_rows)
    if not rows:
        raise SystemExit("No E018 canonical GT rows selected.")
    write_manifest_and_scenes(rows, force=args.force)


if __name__ == "__main__":
    main()
