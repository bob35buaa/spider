#!/usr/bin/env python3
"""Generate E025 manifest and Hydra overrides for contact/collision repair."""

from __future__ import annotations

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
WS = REPO / "workspace/core4d_collab_retarget"
VARIANTS = WS / "scripts/E025/variants.tsv"
E018B_RESULTS = WS / "results/E018b"
E018B_MANIFEST = E018B_RESULTS / "manifest.tsv"
RESULTS = WS / "results/E025"
MANIFEST = RESULTS / "manifest.tsv"
OUT_DIR = REPO / "examples/config/override"

FIELDNAMES = [
    "variant",
    "source_variant",
    "case_slug",
    "queue",
    "role",
    "wave",
    "hold_contact_rew_scale",
    "contact_hdmi_gain",
    "contact_hdmi_sigma",
    "contact_hdmi_ori_weight",
    "contact_hdmi_ori_mode",
    "robot_object_penalty_scale",
    "robot_object_penalty_margin_m",
    "robot_object_penalty_deep_threshold_m",
    "leg_object_penalty_scale",
    "leg_object_penalty_margin_m",
    "leg_object_penalty_geom_names",
    "notes",
]

EXTRA_FIELDS = [
    "E025_source_variant",
    "E025_case_slug",
    "E025_role",
    "E025_wave",
    "E025_hold_contact_rew_scale",
    "E025_contact_hdmi_gain",
    "E025_contact_hdmi_sigma",
    "E025_contact_hdmi_ori_weight",
    "E025_contact_hdmi_ori_mode",
    "E025_robot_object_penalty_scale",
    "E025_robot_object_penalty_margin_m",
    "E025_robot_object_penalty_deep_threshold_m",
    "E025_leg_object_penalty_scale",
    "E025_leg_object_penalty_margin_m",
    "E025_leg_object_penalty_geom_names",
    "E025_notes",
    "contact_hdmi_mask_path",
    "contact_hdmi_mask_time_axis",
]


def _read_tsv(path: Path, fieldnames: list[str] | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        lines = [line for line in f if line.strip() and not line.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t", fieldnames=fieldnames))


def _e018b_manifest() -> dict[str, dict[str, str]]:
    return {row["variant"]: row for row in _read_tsv(E018B_MANIFEST)}


def _point_yaml(row: dict[str, str]) -> str:
    return (
        f"[{float(row['support_proxy_point_local_x']):.8g}, "
        f"{float(row['support_proxy_point_local_y']):.8g}, "
        f"{float(row['support_proxy_point_local_z']):.8g}]"
    )


def _bool_yaml(value: str) -> str:
    return "true" if value.strip().lower() in {"1", "true", "yes", "y"} else "false"


def _list_yaml(value: str) -> str:
    value = value.strip()
    if not value:
        return "[]"
    if value == "default":
        names = [
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
    else:
        names = [item.strip() for item in value.split(",") if item.strip()]
    return "[" + ", ".join(f'"{name}"' for name in names) + "]"


def _copy_mask(row: dict[str, str], variant: str) -> str:
    src_dir = E018B_RESULTS / "contact_masks" / row["mask_slug"]
    if not src_dir.is_dir():
        raise FileNotFoundError(src_dir)
    dst_dir = RESULTS / "contact_masks" / variant
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ("raw_contact_mask_3cm.npz", "raw_contact_mask_3cm.csv", "audit_summary_3cm.json"):
        src = src_dir / name
        if src.is_file():
            shutil.copy2(src, dst_dir / name)
    mask = dst_dir / "raw_contact_mask_3cm.npz"
    if not mask.is_file():
        raise FileNotFoundError(mask)
    return str(mask.relative_to(REPO))


def _build_manifest() -> list[dict[str, str]]:
    e018b_rows = _e018b_manifest()
    rows: list[dict[str, str]] = []
    RESULTS.mkdir(parents=True, exist_ok=True)
    for variant in _read_tsv(VARIANTS, FIELDNAMES):
        source = e018b_rows[variant["source_variant"]]
        mask_path = _copy_mask(source, variant["variant"])
        meta = dict(source)
        meta.update(
            {
                "variant": variant["variant"],
                "queue": variant["queue"],
                "role": variant["role"],
                "wave": variant["wave"],
                "source_variant": variant["source_variant"],
                "online_video_path": (
                    f"workspace/core4d_collab_retarget/results/E025/online_video/{variant['variant']}.mp4"
                ),
                "E025_source_variant": variant["source_variant"],
                "E025_case_slug": variant["case_slug"],
                "E025_role": variant["role"],
                "E025_wave": variant["wave"],
                "E025_hold_contact_rew_scale": variant["hold_contact_rew_scale"],
                "E025_contact_hdmi_gain": variant["contact_hdmi_gain"],
                "E025_contact_hdmi_sigma": variant["contact_hdmi_sigma"],
                "E025_contact_hdmi_ori_weight": variant["contact_hdmi_ori_weight"],
                "E025_contact_hdmi_ori_mode": variant["contact_hdmi_ori_mode"],
                "E025_robot_object_penalty_scale": variant["robot_object_penalty_scale"],
                "E025_robot_object_penalty_margin_m": variant["robot_object_penalty_margin_m"],
                "E025_robot_object_penalty_deep_threshold_m": variant["robot_object_penalty_deep_threshold_m"],
                "E025_leg_object_penalty_scale": variant["leg_object_penalty_scale"],
                "E025_leg_object_penalty_margin_m": variant["leg_object_penalty_margin_m"],
                "E025_leg_object_penalty_geom_names": variant["leg_object_penalty_geom_names"],
                "E025_notes": variant["notes"],
                "contact_hdmi_mask_path": mask_path,
                "contact_hdmi_mask_time_axis": "auto",
            }
        )
        rows.append(meta)
    if not rows:
        raise SystemExit("No E025 variants selected.")
    source_fields = list(e018b_rows[next(iter(e018b_rows))].keys())
    fieldnames = source_fields + [field for field in EXTRA_FIELDS if field not in source_fields]
    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {MANIFEST.relative_to(REPO)} ({len(rows)} variants)")
    return rows


def _write_override(row: dict[str, str]) -> Path:
    variant = row["variant"]
    task = row["derived_task"]
    scene_xml = BASE / task / f"{row['scene_name']}.xml"
    ref_npz = BASE / task / "0/trajectory_kinematic.npz"
    mask_path = REPO / row["contact_hdmi_mask_path"]
    if not scene_xml.is_file():
        raise FileNotFoundError(scene_xml)
    if not ref_npz.is_file():
        raise FileNotFoundError(ref_npz)
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    palm = compute_palm_normal_for_case(task, str(scene_xml), str(ref_npz))

    path = OUT_DIR / f"core4d_collab_{variant}.yaml"
    content = f"""# @package _global_
# Auto-generated for E025 robot-side contact/collision repair.
defaults:
  - core4d_e074a_box023
  - _self_

task: {task}
scene_name: {row["scene_name"]}
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
support_proxy_mode: mocap_pad
support_proxy_mocap_body_name: support_weld_anchor
support_proxy_mocap_quat_mode: object_ref
support_proxy_point_local: {_point_yaml(row)}
support_proxy_gravity_scale: {row["support_proxy_gravity_scale"]}
support_proxy_connector_kp: 0.0
support_proxy_connector_kd: 0.0
support_proxy_xy_velocity_scale: 1.0
support_proxy_max_xy_speed: 0.0
support_proxy_height_tau: 0.0
support_proxy_ref_dt: -1.0
support_proxy_force_clamp: 0.0
support_proxy_torque_clamp: 0.0

contact_hdmi_mask_source: core4d_3cm
contact_hdmi_mask_path: {row["contact_hdmi_mask_path"]}
contact_hdmi_mask_person_idx: {int(row["person_idx"])}
contact_hdmi_mask_time_axis: {row["contact_hdmi_mask_time_axis"]}
contact_hdmi_gain: {row["E025_contact_hdmi_gain"]}
contact_hdmi_sigma: {row["E025_contact_hdmi_sigma"]}
contact_hdmi_ori_weight: {row["E025_contact_hdmi_ori_weight"]}
contact_hdmi_ori_mode: {row["E025_contact_hdmi_ori_mode"]}
contact_hdmi_palm_normal_left: {palm["left"]}
contact_hdmi_palm_normal_right: {palm["right"]}

hold_contact_rew_scale: {row["E025_hold_contact_rew_scale"]}
hold_contact_sigma: {row["hold_contact_sigma"]}
hold_contact_start_eval_time: {row["hold_contact_start_eval_time"]}
hold_contact_end_eval_time: {row["hold_contact_end_eval_time"]}
hold_contact_require_ref_contact: {_bool_yaml(row["hold_contact_require_ref_contact"])}

robot_object_penalty_scale: {row["E025_robot_object_penalty_scale"]}
robot_object_penalty_margin_m: {row["E025_robot_object_penalty_margin_m"]}
robot_object_penalty_deep_threshold_m: {row["E025_robot_object_penalty_deep_threshold_m"]}
leg_object_penalty_scale: {row["E025_leg_object_penalty_scale"]}
leg_object_penalty_margin_m: {row["E025_leg_object_penalty_margin_m"]}
leg_object_penalty_geom_names: {_list_yaml(row["E025_leg_object_penalty_geom_names"])}
"""
    path.write_text(content, encoding="utf-8")
    print(f"Wrote {path.relative_to(REPO)}")
    return path


def main() -> None:
    rows = _build_manifest()
    for row in rows:
        _write_override(row)
    print("E025 overrides done.")


if __name__ == "__main__":
    main()
