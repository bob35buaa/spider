#!/usr/bin/env python3
"""Generate Hydra overrides for E029 dynamic D6 support candidates."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e029_common as common  # noqa: E402

CONVERT_DIR = common.REPO / "workspace/core4d/scripts/convert"
if str(CONVERT_DIR) not in sys.path:
    sys.path.insert(0, str(CONVERT_DIR))
from compute_palm_normal import compute_palm_normal_for_case  # noqa: E402


MANIFEST = common.E029_RESULTS / "d6/manifest.tsv"
RESULT_ROOT = common.E029_RESULTS / "d6"
OUT_DIR = common.OVERRIDE_DIR


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _bool_yaml(value: str) -> str:
    return "true" if _parse_bool(value) else "false"


def _point_yaml(row: dict[str, str]) -> str:
    return (
        f"[{float(row['support_proxy_point_local_x']):.8g}, "
        f"{float(row['support_proxy_point_local_y']):.8g}, "
        f"{float(row['support_proxy_point_local_z']):.8g}]"
    )


def _copy_mask(row: dict[str, str], result_root: Path) -> Path:
    src_dir = Path(row["mask_path_source"]).parent
    if not src_dir.is_dir():
        raise FileNotFoundError(src_dir)
    dst_dir = result_root / "contact_masks" / row["mask_slug"]
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "raw_contact_mask_3cm.npz",
        "raw_contact_mask_3cm.csv",
        "audit_summary_3cm.json",
    ]:
        src = src_dir / name
        if src.is_file():
            shutil.copy2(src, dst_dir / name)
    mask = dst_dir / "raw_contact_mask_3cm.npz"
    if not mask.is_file():
        raise FileNotFoundError(mask)
    return mask


def generate_override(row: dict[str, str], result_root: Path) -> Path:
    variant = row["variant"]
    task = row["derived_task"]
    scene_xml = common.BASE / task / f"{row['scene_name']}.xml"
    ref_npz = common.REPO / row["data_relpath"]
    if not scene_xml.is_file():
        raise FileNotFoundError(scene_xml)
    if not ref_npz.is_file():
        raise FileNotFoundError(ref_npz)
    mask_path = _copy_mask(row, result_root)
    palm = compute_palm_normal_for_case(task, str(scene_xml), str(ref_npz))

    path = OUT_DIR / f"core4d_collab_{variant}.yaml"
    content = f"""# @package _global_
# Auto-generated for E029 COLA-style dynamic D6 support body.
defaults:
  - core4d_e074a_box023
  - _self_

task: {task}
scene_name: {row["scene_name"]}
data_path: {row["data_relpath"]}
contact_guidance: false
object_pd_override: false
object_kinematic_override: false
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

support_proxy_enabled: true
support_proxy_mode: dynamic_weld
support_proxy_mocap_body_name: ""
support_proxy_mocap_quat_mode: object_ref
support_proxy_point_local: {_point_yaml(row)}
support_proxy_gravity_scale: 0.0
support_proxy_connector_kp: 0.0
support_proxy_connector_kd: 0.0
support_proxy_xy_velocity_scale: 1.0
support_proxy_max_xy_speed: 0.0
support_proxy_height_tau: 0.0
support_proxy_ref_dt: -1.0
support_proxy_force_clamp: 0.0
support_proxy_torque_clamp: 0.0

support_dynamic_body_name: support_dynamic_anchor
support_dynamic_mass: {row["support_dynamic_mass"]}
support_dynamic_pos_kp: {row["support_dynamic_pos_kp"]}
support_dynamic_pos_kd: {row["support_dynamic_pos_kd"]}
support_dynamic_rot_kp: {row["support_dynamic_rot_kp"]}
support_dynamic_rot_kd: {row["support_dynamic_rot_kd"]}
support_dynamic_force_clamp: {row["support_dynamic_force_clamp"]}
support_dynamic_torque_clamp: {row["support_dynamic_torque_clamp"]}

contact_hdmi_mask_source: core4d_3cm
contact_hdmi_mask_path: {mask_path.resolve().relative_to(common.REPO)}
contact_hdmi_mask_person_idx: {int(row["person_idx"])}
contact_hdmi_mask_time_axis: auto
contact_hdmi_palm_normal_left: {palm["left"]}
contact_hdmi_palm_normal_right: {palm["right"]}

hold_contact_rew_scale: {row["hold_contact_rew_scale"]}
hold_contact_sigma: {row["hold_contact_sigma"]}
hold_contact_start_eval_time: {row["hold_contact_start_eval_time"]}
hold_contact_end_eval_time: {row["hold_contact_end_eval_time"]}
hold_contact_require_ref_contact: {_bool_yaml(row["hold_contact_require_ref_contact"])}
"""
    path.write_text(content, encoding="utf-8")
    print(f"Wrote {common.rel_or_abs(path)}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    args = parser.parse_args()
    args.result_root.mkdir(parents=True, exist_ok=True)
    for row in read_manifest(args.manifest):
        generate_override(row, args.result_root)


if __name__ == "__main__":
    main()

