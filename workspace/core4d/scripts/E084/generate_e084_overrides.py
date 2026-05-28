#!/usr/bin/env python3
"""Generate E084 Hydra overrides for three Box021 constraint groups."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
CONVERT_DIR = REPO / "workspace/core4d/scripts/convert"
if str(CONVERT_DIR) not in sys.path:
    sys.path.insert(0, str(CONVERT_DIR))

from compute_palm_normal import compute_palm_normal_for_case  # noqa: E402


BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
VARIANTS = REPO / "workspace/core4d/scripts/E084/variants.tsv"
OUT_DIR = REPO / "examples/config/override"

UPPER_BODY_GEOMS = [
    "head_collision",
    "torso_collision",
    "pelvis_collision",
    "left_shoulder_yaw_collision",
    "right_shoulder_yaw_collision",
    "left_elbow_yaw_collision",
    "right_elbow_yaw_collision",
]

UPRIGHT_BODY_NAMES = [
    "pelvis",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]
UPRIGHT_BODY_WEIGHTS = [2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.5, 0.5]


def read_variants(path: Path) -> list[dict[str, str]]:
    fieldnames = [
        "variant",
        "source_task",
        "derived_task",
        "mask_source_dir",
        "mask_slug",
        "person_idx",
        "split",
        "role",
    ]
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(
            (line for line in f if line.strip() and not line.startswith("#")),
            delimiter="\t",
            fieldnames=fieldnames,
        )
        rows.extend(reader)
    return rows


def ref_npz_for_task(task: str) -> Path:
    path = BASE / task / "0" / "trajectory_kinematic.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def copy_mask(row: dict[str, str], result_root: Path) -> Path:
    mask_source_dir = Path(row["mask_source_dir"])
    if not mask_source_dir.is_absolute():
        mask_source_dir = REPO / mask_source_dir
    src = mask_source_dir / row["mask_slug"]
    if not src.is_dir():
        raise FileNotFoundError(src)
    dst = result_root / "contact_masks" / row["mask_slug"]
    dst.mkdir(parents=True, exist_ok=True)
    for name in [
        "raw_contact_mask_3cm.npz",
        "raw_contact_mask_3cm.csv",
        "audit_summary_3cm.json",
    ]:
        src_file = src / name
        if src_file.is_file():
            shutil.copy2(src_file, dst / name)
    return dst / "raw_contact_mask_3cm.npz"


def group_for_variant(variant: str) -> str:
    if variant.startswith("E084A_"):
        return "A"
    if variant.startswith("E084B_"):
        return "B"
    if variant.startswith("E084C_"):
        return "C"
    raise ValueError(f"Unknown E084 group for variant {variant}")


def yaml_list(values: list[object]) -> str:
    return json.dumps(values)


def group_block(group: str) -> str:
    upper_geoms = yaml_list(UPPER_BODY_GEOMS)
    if group == "A":
        return f"""
# Group A: explicit safety penalties for wrong physical contacts.
robot_object_penalty_scale: 5.0
robot_object_penalty_margin_m: 0.03
robot_object_penalty_deep_threshold_m: 0.0
robot_object_penalty_geom_names: {upper_geoms}
hand_floor_penalty_scale: 3.0
hand_floor_penalty_margin_m: 0.03
hand_floor_penalty_geom_names: ["lh", "rh"]
stability_penalty_scale: 1.0
stability_penalty_threshold: 0.60
"""
    if group == "B":
        return f"""
# Group B: stronger upright/body tracking and ctrl trust region.
ctrl_ref_guard_scale: 1.0
ctrl_ref_guard_robot_only: true
ctrl_ref_guard_sigma: 0.15
ctrl_ref_guard_start_eval_time: 0.6
ctrl_ref_guard_end_eval_time: 3.0
stability_penalty_scale: 1.0
stability_penalty_threshold: 0.60
task_body_rew_scale: 2.0
task_body_names: {yaml_list(UPRIGHT_BODY_NAMES)}
task_body_weights: {yaml_list(UPRIGHT_BODY_WEIGHTS)}
task_obj_use_exp: true
task_obj_pos_rew_scale: 0.5
task_obj_rot_rew_scale: 0.3
contact_hdmi_gain: 3.0
"""
    if group == "C":
        return f"""
# Group C: hand-contact semantics plus object lift/floor shaping.
robot_object_penalty_scale: 2.0
robot_object_penalty_margin_m: 0.02
robot_object_penalty_deep_threshold_m: 0.0
robot_object_penalty_geom_names: {upper_geoms}
hand_floor_penalty_scale: 2.0
hand_floor_penalty_margin_m: 0.02
hand_floor_penalty_geom_names: ["lh", "rh"]
object_lift_rew_scale: 2.0
object_lift_sigma: 0.05
object_floor_penalty_scale: 2.0
object_floor_margin_m: 0.02
task_obj_use_exp: true
task_obj_pos_rew_scale: 0.5
task_obj_rot_rew_scale: 0.3
contact_hdmi_gain: 5.0
"""
    raise ValueError(group)


def generate_override(row: dict[str, str], result_root: str) -> Path:
    variant = row["variant"]
    task = row["derived_task"]
    person_idx = int(row["person_idx"])
    result_root_path = REPO / result_root
    mask_path = copy_mask(row, result_root_path)

    scene_xml = BASE / task / "scene.xml"
    ref_npz = ref_npz_for_task(task)
    palm = compute_palm_normal_for_case(task, str(scene_xml), str(ref_npz))

    group = group_for_variant(variant)
    path = OUT_DIR / f"core4d_{variant}.yaml"
    content = f"""# @package _global_
# Auto-generated for E084 Box021 constraint group {group}.
defaults:
  - core4d_e074a_box023
  - _self_

task: {task}
contact_hdmi_mask_source: core4d_3cm
contact_hdmi_mask_path: {mask_path.relative_to(REPO)}
contact_hdmi_mask_person_idx: {person_idx}
contact_hdmi_mask_time_axis: auto
contact_hdmi_palm_normal_left: {palm['left']}
contact_hdmi_palm_normal_right: {palm['right']}
hold_contact_rew_scale: 0.0
hold_contact_start_eval_time: 0.0
hold_contact_end_eval_time: 0.0
{group_block(group)}"""
    path.write_text(content, encoding="utf-8")
    print(f"Wrote {path.relative_to(REPO)}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=Path, default=VARIANTS)
    parser.add_argument("--result-root", default="workspace/core4d/results/E084")
    args = parser.parse_args()

    for row in read_variants(args.variants):
        if row["role"] not in {"main", "guard"}:
            print(f"[SKIP] {row['variant']} role={row['role']}")
            continue
        generate_override(row, args.result_root)


if __name__ == "__main__":
    main()
