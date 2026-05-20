#!/usr/bin/env python3
"""Generate E030 derived tasks, manifest, case-scope audit, and overrides."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


REPO = Path(__file__).resolve().parents[4]
CONVERT_DIR = REPO / "workspace/core4d/scripts/convert"
if str(CONVERT_DIR) not in sys.path:
    sys.path.insert(0, str(CONVERT_DIR))

from compute_palm_normal import compute_palm_normal_for_case  # noqa: E402


BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
WS = REPO / "workspace/core4d_collab_retarget"
VARIANTS = WS / "scripts/E030/variants.tsv"
RESULTS = WS / "results/E030"
MANIFEST = RESULTS / "manifest.tsv"
OUT_DIR = REPO / "examples/config/override"
QUALITY_AUDIT = WS / "results/E027/data_quality/case_quality_audit.csv"
SOURCE_MANIFESTS = {
    "E018b": WS / "results/E018b/manifest.tsv",
    "E022": WS / "results/E022/manifest.tsv",
    "E023": WS / "results/E023/manifest.tsv",
    "E024": WS / "results/E024/manifest.tsv",
    "E025": WS / "results/E025/manifest.tsv",
    "E029": WS / "results/E029/manifest.tsv",
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
    "patch_mode",
    "lowerbody_radius",
    "foot_radius",
    "hold_contact_rew_scale",
    "contact_hdmi_gain",
    "contact_hdmi_sigma",
    "contact_hdmi_ori_weight",
    "contact_hdmi_ori_mode",
    "robot_object_barrier_scale",
    "robot_object_barrier_margin_m",
    "robot_object_barrier_power",
    "robot_object_score_cap_scale",
    "robot_object_score_cap_threshold_m",
    "contact_penetration_gate_enabled",
    "contact_penetration_gate_margin_m",
    "contact_penetration_gate_hold_contact",
    "leg_object_penalty_scale",
    "leg_object_penalty_margin_m",
    "leg_object_penalty_geom_names",
    "stability_penalty_scale",
    "stability_penalty_threshold",
    "local_frame_root_sigma",
    "posture_contact_gate_enabled",
    "posture_contact_gate_pelvis_z_m",
    "posture_contact_gate_max_root_tilt_deg",
    "posture_contact_gate_require_foot_support",
    "posture_contact_gate_hold_contact",
    "notes",
]

EXTRA_FIELDS = [
    "E030_source_manifest",
    "E030_source_variant",
    "E030_source_derived_task",
    "E030_case_slug",
    "E030_role",
    "E030_wave",
    "E030_patch_mode",
    "E030_disabled_leg_object_pairs",
    "E030_shrunk_leg_geoms",
    "E030_lowerbody_radius",
    "E030_foot_radius",
    "E030_hold_contact_rew_scale",
    "E030_contact_hdmi_gain",
    "E030_contact_hdmi_sigma",
    "E030_contact_hdmi_ori_weight",
    "E030_contact_hdmi_ori_mode",
    "E030_robot_object_barrier_scale",
    "E030_robot_object_barrier_margin_m",
    "E030_robot_object_barrier_power",
    "E030_robot_object_score_cap_scale",
    "E030_robot_object_score_cap_threshold_m",
    "E030_contact_penetration_gate_enabled",
    "E030_contact_penetration_gate_margin_m",
    "E030_contact_penetration_gate_hold_contact",
    "E030_leg_object_penalty_scale",
    "E030_leg_object_penalty_margin_m",
    "E030_leg_object_penalty_geom_names",
    "E030_stability_penalty_scale",
    "E030_stability_penalty_threshold",
    "E030_local_frame_root_sigma",
    "E030_posture_contact_gate_enabled",
    "E030_posture_contact_gate_pelvis_z_m",
    "E030_posture_contact_gate_max_root_tilt_deg",
    "E030_posture_contact_gate_require_foot_support",
    "E030_posture_contact_gate_hold_contact",
    "E030_notes",
    "contact_hdmi_mask_path",
    "contact_hdmi_mask_time_axis",
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
FOOT_GEOMS = {"lf0", "lf1", "lf2", "lf3", "rf0", "rf1", "rf2", "rf3"}


def _read_tsv(path: Path, fieldnames: list[str] | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        lines = [line for line in f if line.strip() and not line.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t", fieldnames=fieldnames))


def _read_csv(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["case"]: row for row in csv.DictReader(f)}


def _source_rows() -> dict[str, dict[str, dict[str, str]]]:
    out: dict[str, dict[str, dict[str, str]]] = {}
    for name, path in SOURCE_MANIFESTS.items():
        if path.is_file():
            out[name] = {row["variant"]: row for row in _read_tsv(path)}
    return out


def _copy_task(src_task: str, dst_task: str, *, force: bool) -> None:
    src = BASE / src_task
    dst = BASE / dst_task
    if not src.is_dir():
        raise FileNotFoundError(src)
    if force and dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, dirs_exist_ok=True)


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


def _write_tree(tree: ET.ElementTree, path: Path) -> None:
    root = tree.getroot()
    _indent(root)
    tree.write(path, encoding="unicode", xml_declaration=False)
    text = path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        path.write_text(text + "\n", encoding="utf-8")


def _validate_scene(path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(path))
    if model.nq != 43 or model.nv != 41 or model.nu != 29:
        raise ValueError(f"{path} changed dims: nq/nv/nu={model.nq}/{model.nv}/{model.nu}")


def _shrink_leg_geoms(root: ET.Element, *, lowerbody_radius: float, foot_radius: float) -> list[str]:
    changed: list[str] = []
    for geom in root.iter("geom"):
        name = geom.get("name", "")
        if name not in LEG_FOOT_GEOMS:
            continue
        target = foot_radius if name in FOOT_GEOMS else lowerbody_radius
        size = geom.get("size", "").split()
        if not size:
            size = [f"{target:.8g}"]
        else:
            size[0] = f"{target:.8g}"
        geom.set("size", " ".join(size))
        changed.append(name)
    return sorted(set(changed))


def _patch_task_xmls(
    task: str,
    patch_mode: str,
    *,
    lowerbody_radius: float,
    foot_radius: float,
) -> tuple[list[str], list[str]]:
    disabled: list[str] = []
    shrunk: set[str] = set()
    for xml_path in sorted((BASE / task).glob("scene*.xml")):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        if patch_mode == "none":
            pass
        elif patch_mode == "lowerbody_proxy_tiny":
            shrunk.update(
                _shrink_leg_geoms(
                    root,
                    lowerbody_radius=lowerbody_radius,
                    foot_radius=foot_radius,
                )
            )
        elif patch_mode == "legpair_off":
            raise ValueError("E030 intentionally forbids legpair_off as a success path")
        else:
            raise ValueError(f"Unsupported E030 patch_mode={patch_mode}")
        _write_tree(tree, xml_path)
        _validate_scene(xml_path)
    return disabled, sorted(shrunk)


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


def _dst_task(source_task: str, variant: str) -> str:
    slug = variant.removeprefix("E030_")
    return f"{source_task}_freejoint_legobj_e030_{slug}"


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
        names = LEG_FOOT_GEOMS
    else:
        names = [item.strip() for item in value.split(",") if item.strip()]
    return "[" + ", ".join(f'"{name}"' for name in names) + "]"


def _build_manifest(force: bool) -> list[dict[str, str]]:
    sources = _source_rows()
    rows: list[dict[str, str]] = []
    RESULTS.mkdir(parents=True, exist_ok=True)
    for variant in _read_tsv(VARIANTS, FIELDNAMES):
        source_name = variant["source_manifest"]
        source_variant = variant["source_variant"]
        if source_name not in sources:
            raise KeyError(f"unknown source_manifest {source_name}")
        if source_variant not in sources[source_name]:
            raise KeyError(f"unknown source_variant {source_name}:{source_variant}")
        source = sources[source_name][source_variant]
        dst_task = _dst_task(source["source_task"], variant["variant"])
        _copy_task(source["derived_task"], dst_task, force=force)
        disabled, shrunk = _patch_task_xmls(
            dst_task,
            variant["patch_mode"],
            lowerbody_radius=float(variant["lowerbody_radius"] or 0.0),
            foot_radius=float(variant["foot_radius"] or 0.0),
        )
        mask_path = _copy_mask(source, variant["variant"])
        meta = dict(source)
        meta.update(
            {
                "variant": variant["variant"],
                "derived_task": dst_task,
                "queue": variant["queue"],
                "role": variant["role"],
                "wave": variant["wave"],
                "source_variant": source_variant,
                "online_video_path": (
                    f"workspace/core4d_collab_retarget/results/E030/online_video/{variant['variant']}.mp4"
                ),
                "E030_source_manifest": source_name,
                "E030_source_variant": source_variant,
                "E030_source_derived_task": source["derived_task"],
                "E030_case_slug": variant["case_slug"],
                "E030_role": variant["role"],
                "E030_wave": variant["wave"],
                "E030_patch_mode": variant["patch_mode"],
                "E030_disabled_leg_object_pairs": ",".join(disabled),
                "E030_shrunk_leg_geoms": ",".join(shrunk),
                "E030_lowerbody_radius": variant["lowerbody_radius"],
                "E030_foot_radius": variant["foot_radius"],
                "E030_hold_contact_rew_scale": variant["hold_contact_rew_scale"],
                "E030_contact_hdmi_gain": variant["contact_hdmi_gain"],
                "E030_contact_hdmi_sigma": variant["contact_hdmi_sigma"],
                "E030_contact_hdmi_ori_weight": variant["contact_hdmi_ori_weight"],
                "E030_contact_hdmi_ori_mode": variant["contact_hdmi_ori_mode"],
                "E030_robot_object_barrier_scale": variant["robot_object_barrier_scale"],
                "E030_robot_object_barrier_margin_m": variant["robot_object_barrier_margin_m"],
                "E030_robot_object_barrier_power": variant["robot_object_barrier_power"],
                "E030_robot_object_score_cap_scale": variant["robot_object_score_cap_scale"],
                "E030_robot_object_score_cap_threshold_m": variant[
                    "robot_object_score_cap_threshold_m"
                ],
                "E030_contact_penetration_gate_enabled": variant[
                    "contact_penetration_gate_enabled"
                ],
                "E030_contact_penetration_gate_margin_m": variant[
                    "contact_penetration_gate_margin_m"
                ],
                "E030_contact_penetration_gate_hold_contact": variant[
                    "contact_penetration_gate_hold_contact"
                ],
                "E030_leg_object_penalty_scale": variant["leg_object_penalty_scale"],
                "E030_leg_object_penalty_margin_m": variant["leg_object_penalty_margin_m"],
                "E030_leg_object_penalty_geom_names": variant[
                    "leg_object_penalty_geom_names"
                ],
                "E030_stability_penalty_scale": variant["stability_penalty_scale"],
                "E030_stability_penalty_threshold": variant["stability_penalty_threshold"],
                "E030_local_frame_root_sigma": variant["local_frame_root_sigma"],
                "E030_posture_contact_gate_enabled": variant["posture_contact_gate_enabled"],
                "E030_posture_contact_gate_pelvis_z_m": variant[
                    "posture_contact_gate_pelvis_z_m"
                ],
                "E030_posture_contact_gate_max_root_tilt_deg": variant[
                    "posture_contact_gate_max_root_tilt_deg"
                ],
                "E030_posture_contact_gate_require_foot_support": variant[
                    "posture_contact_gate_require_foot_support"
                ],
                "E030_posture_contact_gate_hold_contact": variant[
                    "posture_contact_gate_hold_contact"
                ],
                "E030_notes": variant["notes"],
                "contact_hdmi_mask_path": mask_path,
                "contact_hdmi_mask_time_axis": "auto",
            }
        )
        (BASE / dst_task / "e030_geometry_patch_meta.json").write_text(
            json.dumps(
                {
                    "variant": variant["variant"],
                    "source_manifest": source_name,
                    "source_variant": source_variant,
                    "source_derived_task": source["derived_task"],
                    "patch_mode": variant["patch_mode"],
                    "disabled_leg_object_pairs": disabled,
                    "shrunk_leg_geoms": shrunk,
                    "lowerbody_radius": variant["lowerbody_radius"],
                    "foot_radius": variant["foot_radius"],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        rows.append(meta)
        print(
            f"{variant['variant']}: task={dst_task} patch={variant['patch_mode']} "
            f"disabled={len(disabled)} shrunk={len(shrunk)}"
        )
    if not rows:
        raise SystemExit("No E030 variants selected.")
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
# Auto-generated for E030 lower-body geometry / surface-control repair.
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
contact_hdmi_dynamic_target: true
contact_hdmi_target_uses_eef_offset: true
contact_hdmi_target_left: []
contact_hdmi_target_right: []
contact_hdmi_gain: {row["E030_contact_hdmi_gain"]}
contact_hdmi_sigma: {row["E030_contact_hdmi_sigma"]}
contact_hdmi_ori_weight: {row["E030_contact_hdmi_ori_weight"]}
contact_hdmi_ori_mode: {row["E030_contact_hdmi_ori_mode"]}
contact_hdmi_palm_normal_left: {palm["left"]}
contact_hdmi_palm_normal_right: {palm["right"]}

hold_contact_rew_scale: {row["E030_hold_contact_rew_scale"]}
hold_contact_sigma: {row["hold_contact_sigma"]}
hold_contact_start_eval_time: {row["hold_contact_start_eval_time"]}
hold_contact_end_eval_time: {row["hold_contact_end_eval_time"]}
hold_contact_require_ref_contact: {_bool_yaml(row["hold_contact_require_ref_contact"])}

robot_object_penalty_geom_names: ["lh", "rh"]
robot_object_penalty_scale: 0.0
robot_object_penalty_margin_m: 0.0
robot_object_penalty_deep_threshold_m: 0.02
robot_object_barrier_scale: {row["E030_robot_object_barrier_scale"]}
robot_object_barrier_margin_m: {row["E030_robot_object_barrier_margin_m"]}
robot_object_barrier_power: {row["E030_robot_object_barrier_power"]}
robot_object_barrier_normalize_by_margin: true
robot_object_score_cap_scale: {row["E030_robot_object_score_cap_scale"]}
robot_object_score_cap_threshold_m: {row["E030_robot_object_score_cap_threshold_m"]}
contact_penetration_gate_enabled: {_bool_yaml(row["E030_contact_penetration_gate_enabled"])}
contact_penetration_gate_margin_m: {row["E030_contact_penetration_gate_margin_m"]}
contact_penetration_gate_hold_contact: {_bool_yaml(row["E030_contact_penetration_gate_hold_contact"])}

leg_object_penalty_scale: {row["E030_leg_object_penalty_scale"]}
leg_object_penalty_margin_m: {row["E030_leg_object_penalty_margin_m"]}
leg_object_penalty_geom_names: {_list_yaml(row["E030_leg_object_penalty_geom_names"])}

stability_penalty_scale: {row["E030_stability_penalty_scale"]}
stability_penalty_threshold: {row["E030_stability_penalty_threshold"]}
local_frame_root_sigma: {row["E030_local_frame_root_sigma"]}

posture_contact_gate_enabled: {_bool_yaml(row["E030_posture_contact_gate_enabled"])}
posture_contact_gate_pelvis_z_m: {row["E030_posture_contact_gate_pelvis_z_m"]}
posture_contact_gate_max_root_tilt_deg: {row["E030_posture_contact_gate_max_root_tilt_deg"]}
posture_contact_gate_require_foot_support: {_bool_yaml(row["E030_posture_contact_gate_require_foot_support"])}
posture_contact_gate_hold_contact: {_bool_yaml(row["E030_posture_contact_gate_hold_contact"])}
"""
    path.write_text(content, encoding="utf-8")
    print(f"Wrote {path.relative_to(REPO)}")
    return path


def _write_case_scope(rows: list[dict[str, str]]) -> None:
    audit = _read_csv(QUALITY_AUDIT)
    lines = [
        "# E030 Case Scope",
        "",
        "E030 follows the E027 multi-evidence data protocol: algorithm failures are not data-discard evidence.",
        "",
        "| Variant | Case | Role | Source | Quality Label | Baseline Contact | Baseline Deep Pen | Rationale |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        case = row["E030_case_slug"]
        qa = audit.get(case, {})
        lines.append(
            "| {variant} | `{case}` | `{role}` | `{source}` | `{label}` | {contact} | {deep} | {notes} |".format(
                variant=row["variant"],
                case=case,
                role=row["E030_role"],
                source=f"{row['E030_source_manifest']}:{row['E030_source_variant']}",
                label=qa.get("quality_label", ""),
                contact=qa.get("best_contact5_pct", ""),
                deep=qa.get("best_deep_pen_pct", ""),
                notes=row["E030_notes"],
            )
        )
    lines.extend(
        [
            "",
            "Excluded from E030 full variants:",
            "",
            "- `desk021_p1`: only case with `discard_from_success_denominator=True`; retained in P1 caveat tables.",
            "- `bucket001_p1`: E029 fixed fall but contact stayed `0%`; next step is reachability/support timing audit, not another stability sweep.",
            "- `bucket001_p2`: retained as usable-with-caveat bucket penetration evidence; E030 uses `bucket005_s2_p1` as the shortcut guard to keep the batch bounded.",
        ]
    )
    (RESULTS / "case_scope.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    rows = _build_manifest(force=args.force)
    _write_case_scope(rows)
    for row in rows:
        _write_override(row)
    print("E030 assets and overrides done.")


if __name__ == "__main__":
    main()
