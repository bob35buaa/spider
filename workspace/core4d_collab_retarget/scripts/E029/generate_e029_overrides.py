#!/usr/bin/env python3
"""Generate E029 manifest and Hydra overrides for bucket001 stability controls."""

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
VARIANTS = WS / "scripts/E029/variants.tsv"
RESULTS = WS / "results/E029"
MANIFEST = RESULTS / "manifest.tsv"
OUT_DIR = REPO / "examples/config/override"
SOURCE_MANIFESTS = {
    "E018b": WS / "results/E018b/manifest.tsv",
    "E024": WS / "results/E024/manifest.tsv",
}
E018B_RESULTS = WS / "results/E018b"

FIELDNAMES = [
    "variant",
    "source_manifest",
    "source_variant",
    "case_slug",
    "queue",
    "role",
    "wave",
    "stability_penalty_scale",
    "stability_penalty_threshold",
    "local_frame_root_sigma",
    "contact_hdmi_gain",
    "contact_hdmi_sigma",
    "hold_contact_rew_scale",
    "robot_object_penalty_scale",
    "robot_object_penalty_margin_m",
    "robot_object_penalty_deep_threshold_m",
    "upright_barrier_scale",
    "upright_barrier_threshold_m",
    "upright_barrier_margin_m",
    "upright_barrier_power",
    "upright_score_cap_scale",
    "upright_score_cap_threshold_m",
    "root_tilt_penalty_scale",
    "root_tilt_penalty_max_deg",
    "root_tilt_penalty_power",
    "foot_support_penalty_scale",
    "foot_support_max_z_m",
    "foot_support_low_pelvis_threshold_m",
    "posture_contact_gate_enabled",
    "posture_contact_gate_pelvis_z_m",
    "posture_contact_gate_max_root_tilt_deg",
    "posture_contact_gate_require_foot_support",
    "posture_contact_gate_hold_contact",
    "notes",
]

EXTRA_FIELDS = [
    "E029_source_manifest",
    "E029_source_variant",
    "E029_case_slug",
    "E029_role",
    "E029_wave",
    "E029_stability_penalty_scale",
    "E029_stability_penalty_threshold",
    "E029_local_frame_root_sigma",
    "E029_contact_hdmi_gain",
    "E029_contact_hdmi_sigma",
    "E029_hold_contact_rew_scale",
    "E029_robot_object_penalty_scale",
    "E029_robot_object_penalty_margin_m",
    "E029_robot_object_penalty_deep_threshold_m",
    "E029_upright_barrier_scale",
    "E029_upright_barrier_threshold_m",
    "E029_upright_barrier_margin_m",
    "E029_upright_barrier_power",
    "E029_upright_score_cap_scale",
    "E029_upright_score_cap_threshold_m",
    "E029_root_tilt_penalty_scale",
    "E029_root_tilt_penalty_max_deg",
    "E029_root_tilt_penalty_power",
    "E029_foot_support_penalty_scale",
    "E029_foot_support_max_z_m",
    "E029_foot_support_low_pelvis_threshold_m",
    "E029_posture_contact_gate_enabled",
    "E029_posture_contact_gate_pelvis_z_m",
    "E029_posture_contact_gate_max_root_tilt_deg",
    "E029_posture_contact_gate_require_foot_support",
    "E029_posture_contact_gate_hold_contact",
    "E029_notes",
    "contact_hdmi_mask_path",
    "contact_hdmi_mask_time_axis",
]


def _read_tsv(path: Path, fieldnames: list[str] | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        lines = [line for line in f if line.strip() and not line.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t", fieldnames=fieldnames))


def _source_rows() -> dict[str, dict[str, dict[str, str]]]:
    out: dict[str, dict[str, dict[str, str]]] = {}
    for name, path in SOURCE_MANIFESTS.items():
        out[name] = {row["variant"]: row for row in _read_tsv(path)}
    return out


def _point_yaml(row: dict[str, str]) -> str:
    return (
        f"[{float(row['support_proxy_point_local_x']):.8g}, "
        f"{float(row['support_proxy_point_local_y']):.8g}, "
        f"{float(row['support_proxy_point_local_z']):.8g}]"
    )


def _bool_yaml(value: str) -> str:
    return "true" if value.strip().lower() in {"1", "true", "yes", "y"} else "false"


def _copy_mask(source: dict[str, str], variant: str) -> str:
    if source.get("contact_hdmi_mask_path"):
        src_mask = REPO / source["contact_hdmi_mask_path"]
        if not src_mask.is_file():
            raise FileNotFoundError(src_mask)
        src_dir = src_mask.parent
    else:
        src_dir = E018B_RESULTS / "contact_masks" / source["mask_slug"]
        src_mask = src_dir / "raw_contact_mask_3cm.npz"
        if not src_mask.is_file():
            raise FileNotFoundError(src_mask)
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
    sources = _source_rows()
    rows: list[dict[str, str]] = []
    RESULTS.mkdir(parents=True, exist_ok=True)
    for variant in _read_tsv(VARIANTS, FIELDNAMES):
        source_name = variant["source_manifest"]
        source_variant = variant["source_variant"]
        source = sources[source_name][source_variant]
        mask_path = _copy_mask(source, variant["variant"])
        meta = dict(source)
        meta.update(
            {
                "variant": variant["variant"],
                "queue": variant["queue"],
                "role": variant["role"],
                "wave": variant["wave"],
                "source_variant": source_variant,
                "online_video_path": (
                    f"workspace/core4d_collab_retarget/results/E029/online_video/{variant['variant']}.mp4"
                ),
                "E029_source_manifest": source_name,
                "E029_source_variant": source_variant,
                "E029_case_slug": variant["case_slug"],
                "E029_role": variant["role"],
                "E029_wave": variant["wave"],
                "E029_stability_penalty_scale": variant["stability_penalty_scale"],
                "E029_stability_penalty_threshold": variant["stability_penalty_threshold"],
                "E029_local_frame_root_sigma": variant["local_frame_root_sigma"],
                "E029_contact_hdmi_gain": variant["contact_hdmi_gain"],
                "E029_contact_hdmi_sigma": variant["contact_hdmi_sigma"],
                "E029_hold_contact_rew_scale": variant["hold_contact_rew_scale"],
                "E029_robot_object_penalty_scale": variant["robot_object_penalty_scale"],
                "E029_robot_object_penalty_margin_m": variant["robot_object_penalty_margin_m"],
                "E029_robot_object_penalty_deep_threshold_m": variant[
                    "robot_object_penalty_deep_threshold_m"
                ],
                "E029_upright_barrier_scale": variant["upright_barrier_scale"],
                "E029_upright_barrier_threshold_m": variant["upright_barrier_threshold_m"],
                "E029_upright_barrier_margin_m": variant["upright_barrier_margin_m"],
                "E029_upright_barrier_power": variant["upright_barrier_power"],
                "E029_upright_score_cap_scale": variant["upright_score_cap_scale"],
                "E029_upright_score_cap_threshold_m": variant[
                    "upright_score_cap_threshold_m"
                ],
                "E029_root_tilt_penalty_scale": variant["root_tilt_penalty_scale"],
                "E029_root_tilt_penalty_max_deg": variant["root_tilt_penalty_max_deg"],
                "E029_root_tilt_penalty_power": variant["root_tilt_penalty_power"],
                "E029_foot_support_penalty_scale": variant["foot_support_penalty_scale"],
                "E029_foot_support_max_z_m": variant["foot_support_max_z_m"],
                "E029_foot_support_low_pelvis_threshold_m": variant[
                    "foot_support_low_pelvis_threshold_m"
                ],
                "E029_posture_contact_gate_enabled": variant[
                    "posture_contact_gate_enabled"
                ],
                "E029_posture_contact_gate_pelvis_z_m": variant[
                    "posture_contact_gate_pelvis_z_m"
                ],
                "E029_posture_contact_gate_max_root_tilt_deg": variant[
                    "posture_contact_gate_max_root_tilt_deg"
                ],
                "E029_posture_contact_gate_require_foot_support": variant[
                    "posture_contact_gate_require_foot_support"
                ],
                "E029_posture_contact_gate_hold_contact": variant[
                    "posture_contact_gate_hold_contact"
                ],
                "E029_notes": variant["notes"],
                "contact_hdmi_mask_path": mask_path,
                "contact_hdmi_mask_time_axis": "auto",
            }
        )
        rows.append(meta)
    if not rows:
        raise SystemExit("No E029 variants selected.")
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    for key in EXTRA_FIELDS:
        if key not in fieldnames:
            fieldnames.append(key)
    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
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
# Auto-generated for E029 bucket001 stability / posture-valid contact.
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
contact_hdmi_gain: {row["E029_contact_hdmi_gain"]}
contact_hdmi_sigma: {row["E029_contact_hdmi_sigma"]}
contact_hdmi_ori_weight: 0.0
contact_hdmi_ori_mode: multiply
contact_hdmi_palm_normal_left: {palm["left"]}
contact_hdmi_palm_normal_right: {palm["right"]}

hold_contact_rew_scale: {row["E029_hold_contact_rew_scale"]}
hold_contact_sigma: {row["hold_contact_sigma"]}
hold_contact_start_eval_time: {row["hold_contact_start_eval_time"]}
hold_contact_end_eval_time: {row["hold_contact_end_eval_time"]}
hold_contact_require_ref_contact: {_bool_yaml(row["hold_contact_require_ref_contact"])}

stability_penalty_scale: {row["E029_stability_penalty_scale"]}
stability_penalty_threshold: {row["E029_stability_penalty_threshold"]}
local_frame_root_sigma: {row["E029_local_frame_root_sigma"]}

robot_object_penalty_geom_names: ["lh", "rh"]
robot_object_penalty_scale: {row["E029_robot_object_penalty_scale"]}
robot_object_penalty_margin_m: {row["E029_robot_object_penalty_margin_m"]}
robot_object_penalty_deep_threshold_m: {row["E029_robot_object_penalty_deep_threshold_m"]}

upright_barrier_scale: {row["E029_upright_barrier_scale"]}
upright_barrier_threshold_m: {row["E029_upright_barrier_threshold_m"]}
upright_barrier_margin_m: {row["E029_upright_barrier_margin_m"]}
upright_barrier_power: {row["E029_upright_barrier_power"]}
upright_score_cap_scale: {row["E029_upright_score_cap_scale"]}
upright_score_cap_threshold_m: {row["E029_upright_score_cap_threshold_m"]}
root_tilt_penalty_scale: {row["E029_root_tilt_penalty_scale"]}
root_tilt_penalty_max_deg: {row["E029_root_tilt_penalty_max_deg"]}
root_tilt_penalty_power: {row["E029_root_tilt_penalty_power"]}
foot_support_penalty_scale: {row["E029_foot_support_penalty_scale"]}
foot_support_max_z_m: {row["E029_foot_support_max_z_m"]}
foot_support_low_pelvis_threshold_m: {row["E029_foot_support_low_pelvis_threshold_m"]}
posture_contact_gate_enabled: {_bool_yaml(row["E029_posture_contact_gate_enabled"])}
posture_contact_gate_pelvis_z_m: {row["E029_posture_contact_gate_pelvis_z_m"]}
posture_contact_gate_max_root_tilt_deg: {row["E029_posture_contact_gate_max_root_tilt_deg"]}
posture_contact_gate_require_foot_support: {_bool_yaml(row["E029_posture_contact_gate_require_foot_support"])}
posture_contact_gate_hold_contact: {_bool_yaml(row["E029_posture_contact_gate_hold_contact"])}
"""
    path.write_text(content, encoding="utf-8")
    print(f"Wrote {path.relative_to(REPO)}")
    return path


def main() -> None:
    rows = _build_manifest()
    for row in rows:
        _write_override(row)
    print("E029 overrides done.")


if __name__ == "__main__":
    main()
