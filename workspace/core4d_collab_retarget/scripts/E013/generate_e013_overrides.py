#!/usr/bin/env python3
"""Generate E013 Hydra overrides for true-freejoint object oracle."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
CONVERT_DIR = REPO / "workspace/core4d/scripts/convert"
if str(CONVERT_DIR) not in sys.path:
    sys.path.insert(0, str(CONVERT_DIR))

from compute_palm_normal import compute_palm_normal_for_case  # noqa: E402


BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
VARIANTS = REPO / "workspace/core4d_collab_retarget/scripts/E013/variants.tsv"
OUT_DIR = REPO / "examples/config/override"
MASK_SOURCE_ROOT = REPO / "workspace/core4d_collab_retarget/results/E002/contact_masks"
MASK_FALLBACK_ROOT = REPO / "workspace/core4d/results/E081/contact_masks"

FIELDNAMES = [
    "variant",
    "source_task",
    "mask_slug",
    "person_idx",
    "queue",
    "role",
    "wave",
]


def read_variants(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(
            csv.DictReader(
                (line for line in f if line.strip() and not line.startswith("#")),
                delimiter="\t",
                fieldnames=FIELDNAMES,
            )
        )


def copy_mask(mask_slug: str, result_root: Path) -> Path:
    src = MASK_SOURCE_ROOT / mask_slug
    if not src.is_dir():
        fallback = MASK_FALLBACK_ROOT / mask_slug
        if not fallback.is_dir():
            raise FileNotFoundError(f"{src} (fallback also missing: {fallback})")
        src = fallback

    dst = result_root / "contact_masks" / mask_slug
    dst.mkdir(parents=True, exist_ok=True)
    for name in [
        "raw_contact_mask_3cm.npz",
        "raw_contact_mask_3cm.csv",
        "audit_summary_3cm.json",
    ]:
        src_file = src / name
        if src_file.is_file():
            shutil.copy2(src_file, dst / name)

    mask = dst / "raw_contact_mask_3cm.npz"
    if not mask.is_file():
        raise FileNotFoundError(mask)
    return mask


def generate_override(row: dict[str, str], result_root: str) -> Path:
    variant = row["variant"]
    task = row["source_task"]
    person_idx = int(row["person_idx"])
    result_root_path = REPO / result_root
    mask_path = copy_mask(row["mask_slug"], result_root_path)

    scene_xml = BASE / task / "scene.xml"
    ref_npz = BASE / task / "0/trajectory_kinematic.npz"
    if not scene_xml.is_file():
        raise FileNotFoundError(scene_xml)
    if not ref_npz.is_file():
        raise FileNotFoundError(ref_npz)
    palm = compute_palm_normal_for_case(task, str(scene_xml), str(ref_npz))

    override_name = f"core4d_collab_{variant}"
    path = OUT_DIR / f"{override_name}.yaml"
    content = f"""# @package _global_
# Auto-generated for E013 true-freejoint object oracle.
defaults:
  - core4d_e074a_box023
  - _self_

task: {task}
scene_name: scene
contact_guidance: false
object_pd_override: false
object_kinematic_override: true
object_kinematic_ref_dt: -1.0
object_kinematic_set_qvel: true
object_action_dims: 0
object_actuator_ids: []
object_actuator_names: []

partner_force_scale: 0.0
partner_force_spring_kp: 0.0
partner_force_spring_kd: -1.0
partner_force_spring_kp_rot: 0.0
partner_force_spring_kd_rot: -1.0
partner_force_ref_dt: -1.0
partner_force_point_local: []
partner_force_points_local: []
partner_force_force_clamp: 0.0
partner_force_torque_clamp: 0.0

support_proxy_enabled: false
support_proxy_mode: wrench
support_proxy_point_local: [0.0, 0.0, 0.0]
support_proxy_gravity_scale: 0.0
support_proxy_connector_kp: 0.0
support_proxy_connector_kd: 0.0
support_proxy_xy_velocity_scale: 1.0
support_proxy_max_xy_speed: 0.0
support_proxy_height_tau: 0.0
support_proxy_ref_dt: -1.0
support_proxy_force_clamp: 0.0
support_proxy_torque_clamp: 0.0

contact_hdmi_mask_source: core4d_3cm
contact_hdmi_mask_path: {mask_path.relative_to(REPO)}
contact_hdmi_mask_person_idx: {person_idx}
contact_hdmi_mask_time_axis: auto
contact_hdmi_palm_normal_left: {palm["left"]}
contact_hdmi_palm_normal_right: {palm["right"]}

hold_contact_rew_scale: 0.0
hold_contact_sigma: 0.05
hold_contact_start_eval_time: 0.64
hold_contact_end_eval_time: 4.08
hold_contact_require_ref_contact: true
"""
    path.write_text(content, encoding="utf-8")
    print(f"Wrote {path.relative_to(REPO)}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=Path, default=VARIANTS)
    parser.add_argument(
        "--result-root", default="workspace/core4d_collab_retarget/results/E013"
    )
    args = parser.parse_args()

    for row in read_variants(args.variants):
        generate_override(row, args.result_root)


if __name__ == "__main__":
    main()
