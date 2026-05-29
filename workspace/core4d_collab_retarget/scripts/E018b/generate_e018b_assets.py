#!/usr/bin/env python3
"""Generate E018b canonical support-proxy anchors and scenes."""

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
E017_AUDIT = REPO / "workspace/core4d_collab_retarget/results/E017/anchor_audit.csv"
RESULTS = REPO / "workspace/core4d_collab_retarget/results/E018b"
MANIFEST = RESULTS / "manifest.tsv"

ANCHOR_BODY = "support_weld_anchor"
ANCHOR_GEOM = "support_weld_anchor_geom"
WELD_NAME = "e018b_support_weld"
CANONICAL_Z_FRAC = 0.62
QUEUES = ("local", "remote_gpu0", "remote_gpu1")

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
    "anchor_audit_class",
    "anchor_current_face",
    "anchor_selected_face",
    "anchor_top_face",
    "anchor_top_face_frac",
    "anchor_partner_top_face",
    "anchor_partner_top_face_frac",
    "anchor_selected_partner_top_relation",
    "anchor_centroid_cancellation",
    "anchor_low_support",
    "reference_frames",
    "online_video_path",
]


def _read_tsv_rows(path: Path, fieldnames: list[str] | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [line for line in f if line.strip() and not line.startswith("#")]
    if fieldnames is None:
        return list(csv.DictReader(rows, delimiter="\t"))
    return list(csv.DictReader(rows, delimiter="\t", fieldnames=fieldnames))


def _read_csv_by_key(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row[key]: row for row in csv.DictReader(f)}


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
    """B1 修复 (exp_diagnostic_v2 §2)：支持全 6 面（含 ±z）。"""
    axis = "xyz".index(face[1])
    sign = 1.0 if face[0] == "+" else -1.0
    return axis, sign


def face_label(point: np.ndarray, half: np.ndarray) -> str:
    """B1 修复：全 3D argmax，可返回 ±z。"""
    norm = np.abs(point) / np.clip(half, 1e-6, None)
    axis = int(np.argmax(norm))
    sign = "+" if point[axis] >= 0.0 else "-"
    return f"{sign}{'xyz'[axis]}"


def canonical_anchor(face: str, half: np.ndarray, z_frac: float = CANONICAL_Z_FRAC) -> np.ndarray:
    """B1 修复：之前 z 永远固定到 z_frac·half[2]，与 ±z 面互斥；
    现在若 face 是 ±z，z 取 sign·half[2]；其它面 z 仍按 z_frac 历史行为。"""
    point = np.zeros(3, dtype=np.float64)
    axis, sign = _face_axis_sign(face)
    point[axis] = sign * float(half[axis])
    if axis != 2:  # ±x / ±y 面：z 保留历史的 z_frac·half[2] 习惯
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
        {
            WELD_NAME,
            "e018_support_weld",
            "e017_support_weld",
            "e016_support_weld",
            "e014_support_weld",
            "object_weld",
        },
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


def reference_frames(task: str) -> int:
    qpos = np.load(BASE / task / "0/trajectory_kinematic.npz")["qpos"]
    return int(qpos.reshape(-1, qpos.shape[-1]).shape[0])


def assign_queues(rows: list[dict[str, str]]) -> None:
    loads = {queue: 0 for queue in QUEUES}
    for row in sorted(rows, key=lambda r: int(r["reference_frames"]), reverse=True):
        queue = min(QUEUES, key=lambda q: (loads[q], QUEUES.index(q)))
        row["queue"] = queue
        loads[queue] += int(row["reference_frames"])


def _blank_gt() -> dict[str, str]:
    return {
        "gt_anchor_available": "false",
        "gt_anchor_x": "",
        "gt_anchor_y": "",
        "gt_anchor_z": "",
        "gt_anchor_face": "",
        "gt_anchor_dist_m": "nan",
    }


def build_manifest_rows(base_rows: list[dict[str, str]], audit_rows: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for base_row in base_rows:
        gt = E014_GT.get(base_row["source_task"])
        audit = audit_rows.get(base_row["variant"])
        if audit is None:
            raise ValueError(f"Missing E017 audit row for {base_row['variant']}")
        if gt is None:
            face = audit["selected_face"]
            face_source = "e017_audit_selected_face"
            gt_fields = _blank_gt()
            gt_dist = float("nan")
        else:
            face = str(gt["face"])
            face_source = "e014_gt_template"
        slug = _slug_from_e016_variant(base_row["variant"])
        half = object_half(base_row["source_task"])
        point = canonical_anchor(face, half)
        x, y, z = _format_point(point)
        if gt is not None:
            gt_point = np.asarray(gt["point"], dtype=np.float64)
            gt_dist = float(np.linalg.norm(point - gt_point))
            gx, gy, gz = _format_point(gt_point)
            gt_fields = {
                "gt_anchor_available": "true",
                "gt_anchor_x": gx,
                "gt_anchor_y": gy,
                "gt_anchor_z": gz,
                "gt_anchor_face": face_label(gt_point, half),
                "gt_anchor_dist_m": f"{gt_dist:.8g}",
            }
        rows.append(
            {
                "variant": f"E018b_{slug}_canonical_t02",
                "source_task": base_row["source_task"],
                "derived_task": f"{base_row['source_task']}_freejoint_legobj_e018b",
                "mask_source_exp": base_row["mask_source_exp"],
                "mask_slug": base_row["mask_slug"],
                "person_idx": base_row["person_idx"],
                "queue": "unassigned",
                "role": base_row["role"],
                "wave": "A",
                "scene_name": f"scene_e018b_jointB_{slug}_canonical_t02",
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
                "anchor_face_source": face_source,
                "canonical_z_frac": f"{CANONICAL_Z_FRAC:.6g}",
                "object_half_x": f"{half[0]:.8g}",
                "object_half_y": f"{half[1]:.8g}",
                "object_half_z": f"{half[2]:.8g}",
                **gt_fields,
                "anchor_audit_class": audit.get("audit_class", ""),
                "anchor_current_face": audit.get("current_face", ""),
                "anchor_selected_face": audit.get("selected_face", ""),
                "anchor_top_face": audit.get("top_face", ""),
                "anchor_top_face_frac": audit.get("top_face_frac", ""),
                "anchor_partner_top_face": audit.get("partner_top_face", ""),
                "anchor_partner_top_face_frac": audit.get("partner_top_face_frac", ""),
                "anchor_selected_partner_top_relation": audit.get("selected_partner_top_relation", ""),
                "anchor_centroid_cancellation": audit.get("centroid_cancellation", ""),
                "anchor_low_support": audit.get("low_support", ""),
                "reference_frames": str(reference_frames(base_row["source_task"])),
                "online_video_path": f"workspace/core4d_collab_retarget/results/E018b/online_video/E018b_{slug}_canonical_t02.mp4",
            }
        )
    assign_queues(rows)
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
            f"{row['variant']}: queue={row['queue']} task={task} scene={out.relative_to(REPO)} "
            f"face={row['anchor_face']} point={point.tolist()} gt_dist={row['gt_anchor_dist_m']}"
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
                        "queue": row["queue"],
                        "face": row["anchor_face"],
                        "face_source": row["anchor_face_source"],
                        "point": [
                            float(row["support_proxy_point_local_x"]),
                            float(row["support_proxy_point_local_y"]),
                            float(row["support_proxy_point_local_z"]),
                        ],
                        "gt_dist_m": float(row["gt_anchor_dist_m"]),
                    }
                    for row in rows
                ],
                "queue_loads_reference_frames": {
                    queue: sum(int(row["reference_frames"]) for row in rows if row["queue"] == queue)
                    for queue in QUEUES
                },
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
    parser.add_argument("--audit", type=Path, default=E017_AUDIT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base_rows = _read_tsv_rows(args.variants, FIELDNAMES)
    rows = build_manifest_rows(base_rows, _read_csv_by_key(args.audit, "source_variant"))
    if not rows:
        raise SystemExit("No E018b canonical rows selected.")
    write_manifest_and_scenes(rows, force=args.force)


if __name__ == "__main__":
    main()
