#!/usr/bin/env python3
"""Generate E016 freejoint leg-object tasks and E014-style support-weld scenes."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
VARIANTS = REPO / "workspace/core4d_collab_retarget/scripts/E016/variants.tsv"
RESULTS = REPO / "workspace/core4d_collab_retarget/results/E016"
MANIFEST = RESULTS / "manifest.tsv"

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
    "support_point_method",
]

LEG_FOOT_GEOMS = [
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
]
ANCHOR_BODY = "support_weld_anchor"
ANCHOR_GEOM = "support_weld_anchor_geom"
WELD_NAME = "e016_support_weld"


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


def _quat_apply_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qi = q.astype(np.float64).copy()
    qi = qi / np.clip(np.linalg.norm(qi), 1e-8, None)
    qi[1:] *= -1.0
    return _quat_apply(qi, v)


def _remove_named_children(parent: ET.Element, tag: str, names: set[str]) -> None:
    for child in list(parent.findall(tag)):
        if child.get("name") in names:
            parent.remove(child)


def copy_freejoint_case(source_task: str, derived_task: str, *, force: bool) -> Path:
    src = BASE / source_task
    dst = BASE / derived_task
    if not src.is_dir():
        raise FileNotFoundError(src)
    if dst.exists():
        if not force:
            return dst
        shutil.rmtree(dst)
    (dst / "0").mkdir(parents=True, exist_ok=True)
    required = ["scene.xml", "0/trajectory_kinematic.npz"]
    optional = ["task_info.json", "scene_act_meta.json"]
    for rel in required:
        src_file = src / rel
        if not src_file.is_file():
            raise FileNotFoundError(src_file)
        dst_file = dst / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
    for rel in optional:
        src_file = src / rel
        if src_file.is_file():
            dst_file = dst / rel
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
    return dst


def patch_leg_object_pairs(scene_xml: Path, source_task: str, derived_task: str) -> None:
    text = scene_xml.read_text(encoding="utf-8")
    marker = "  </contact>"
    if marker not in text:
        raise ValueError(f"Missing {marker!r} in {scene_xml}")
    lines: list[str] = []
    added: list[str] = []
    for geom in LEG_FOOT_GEOMS:
        pair_name = f"{geom}_object"
        if f'name="{pair_name}"' in text:
            continue
        lines.append(
            f'    <pair name="{pair_name}" geom1="{geom}" geom2="object_collision" '
            'solref="0.008 1" friction="1 1" condim="3" />'
        )
        added.append(pair_name)
    if lines:
        scene_xml.write_text(
            text.replace(marker, "\n".join(lines) + "\n" + marker, 1),
            encoding="utf-8",
        )
    meta = {
        "source_task": source_task,
        "derived_task": derived_task,
        "patched_file": str(scene_xml.relative_to(REPO)),
        "scene_mode": "freejoint_scene_xml",
        "object_geom": "object_collision",
        "added_pairs": added,
        "note": "E016 derived freejoint scene with leg/foot-object contacts.",
    }
    (scene_xml.parent / "freejoint_leg_object_collision_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )


def _load_mask(row: dict[str, str], target_len: int) -> np.ndarray | None:
    mask_path = (
        REPO
        / "workspace/core4d/results"
        / row["mask_source_exp"]
        / "contact_masks"
        / row["mask_slug"]
        / "raw_contact_mask_3cm.npz"
    )
    if not mask_path.is_file():
        return None
    data = np.load(mask_path, allow_pickle=True)
    key = "eval_contact_mask_3cm" if "eval_contact_mask_3cm" in data else "spider_contact_mask_3cm"
    raw = data[key]
    person_idx = min(max(int(row["person_idx"]), 0), raw.shape[1] - 1)
    mask = raw[:, person_idx, :2].astype(bool)
    if len(mask) == target_len:
        return mask
    idx = np.round(np.linspace(0, len(mask) - 1, target_len)).astype(int)
    return mask[idx]


def infer_support_point(row: dict[str, str], task_dir: Path) -> tuple[np.ndarray, str]:
    model = mujoco.MjModel.from_xml_path(str(task_dir / "scene.xml"))
    data = mujoco.MjData(model)
    qpos = np.load(task_dir / "0/trajectory_kinematic.npz")["qpos"]
    qpos = qpos.reshape(-1, qpos.shape[-1])
    mask = _load_mask(row, len(qpos))

    left_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_palm")
    right_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    obj_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")
    half = model.geom_size[obj_geom, :3].astype(np.float64)

    points: list[np.ndarray] = []
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        obj_pos = data.xpos[obj_body].copy()
        obj_quat = q[-4:].astype(np.float64)
        active_left = mask is None or bool(mask[i, 0])
        active_right = mask is None or bool(mask[i, 1])
        if active_left:
            points.append(_quat_apply_inv(obj_quat, data.site_xpos[left_site] - obj_pos))
        if active_right:
            points.append(_quat_apply_inv(obj_quat, data.site_xpos[right_site] - obj_pos))
    if not points:
        return np.array([0.0, float(half[1]), max(0.08, float(0.6 * half[2]))]), "fallback_no_contact"

    centroid = np.mean(np.stack(points), axis=0)
    point = np.clip(centroid, -0.95 * half, 0.95 * half)
    xy_norm = np.abs(point[:2]) / np.clip(half[:2], 1e-6, None)
    axis = int(np.argmax(xy_norm))
    if xy_norm[axis] < 0.35:
        axis = int(np.argmax(half[:2]))
        sign = 1.0
    else:
        sign = 1.0 if point[axis] >= 0.0 else -1.0
    point[axis] = sign * half[axis]
    point[2] = float(np.clip(point[2], -0.25 * half[2], 0.85 * half[2]))
    if np.linalg.norm(point) < 0.05:
        point[1] = float(half[1])
        point[2] = max(0.08, float(0.6 * half[2]))
    return point.astype(np.float64), "mask_active_ref_palm_centroid_surface_clamp"


def generate_support_scene(task_dir: Path, scene_name: str, point_local: np.ndarray) -> Path:
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
    _remove_named_children(equality, "weld", {WELD_NAME, "e014_support_weld", "object_weld"})

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
            "solref": "0.02 1",
            "solimp": "0.9 0.95 0.001",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=Path, default=VARIANTS)
    parser.add_argument("--result-root", type=Path, default=RESULTS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.result_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in read_variants(args.variants):
        task_dir = copy_freejoint_case(row["source_task"], row["derived_task"], force=args.force)
        patch_leg_object_pairs(task_dir / "scene.xml", row["source_task"], row["derived_task"])
        point, method = infer_support_point(row, task_dir)
        scene_name = f"scene_e016_jointB_{row['variant'][5:]}"
        out = generate_support_scene(task_dir, scene_name, point)
        manifest_row = {
            **row,
            "scene_name": scene_name,
            "support_proxy_point_local_x": f"{point[0]:.8g}",
            "support_proxy_point_local_y": f"{point[1]:.8g}",
            "support_proxy_point_local_z": f"{point[2]:.8g}",
            "weld_solref_timeconst": "0.02",
            "weld_solimp_1": "0.9",
            "weld_solimp_2": "0.95",
            "weld_solimp_width": "0.001",
            "support_proxy_gravity_scale": "0.5",
            "hold_contact_rew_scale": "0.0",
            "support_point_method": method,
        }
        rows.append(manifest_row)
        print(
            f"{row['variant']}: task={row['derived_task']} scene={out.relative_to(REPO)} "
            f"point={point.tolist()}"
        )

    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {MANIFEST.relative_to(REPO)}")


if __name__ == "__main__":
    main()
