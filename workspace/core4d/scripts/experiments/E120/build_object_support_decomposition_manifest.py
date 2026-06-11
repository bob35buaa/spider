#!/usr/bin/env python3
"""Build E120 object-support decomposition diagnostic manifest."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
OVERRIDE_ROOT = REPO / "examples/config/override"
SCRIPTS_ROOT = REPO / "workspace/core4d/scripts/E120"
RESULTS_ROOT = REPO / "workspace/core4d/results/E120"
PREFLIGHT_ROOT = RESULTS_ROOT / "preflight"
E113_VARIANTS = REPO / "workspace/core4d/scripts/E113/variants.tsv"
E113_PARETO = REPO / "workspace/core4d/results/E113/cem/full/pareto_decisions.tsv"
E114_DIAGNOSTICS = REPO / "workspace/core4d/results/E114/rl_handoff_gate/diagnostic_queues.tsv"
E100_TARGET_ROOT = REPO / "workspace/core4d/results/E100/fingertip_targets"
VARIANTS_TSV = SCRIPTS_ROOT / "variants.tsv"
PREFLIGHT_TSV = PREFLIGHT_ROOT / "phaseA_preflight.tsv"
SUMMARY_JSON = PREFLIGHT_ROOT / "phaseA_manifest_summary.json"
SUMMARY_MD = PREFLIGHT_ROOT / "phaseA_manifest_summary.md"

LEG_GEOMS = [
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

ROBOT_OBJECT_GEOMS = [
    "head_collision",
    "torso_collision",
    "pelvis_collision",
    "left_shoulder_yaw_collision",
    "right_shoulder_yaw_collision",
    "left_elbow_yaw_collision",
    "right_elbow_yaw_collision",
]

NONHAND_SUPPORT_GEOMS = ROBOT_OBJECT_GEOMS + LEG_GEOMS

FIELDS = [
    "ordinal",
    "variant",
    "source_task",
    "derived_task",
    "person_idx",
    "split",
    "case_id",
    "e109_case_id",
    "object_key",
    "ablation",
    "mask_path",
    "source_override",
    "baseline_npz_path",
    "e113_hold_npz_path",
    "e113_hold_video_path",
    "phase_scope",
    "diagnostic_queue",
    "holdout_reason",
]

E113_FIELDS = [
    "ordinal",
    "variant",
    "source_task",
    "derived_task",
    "person_idx",
    "split",
    "case_id",
    "e109_case_id",
    "object_key",
    "ablation",
    "mask_path",
    "source_override",
    "baseline_npz_path",
    "phase_scope",
    "holdout_reason",
]


@dataclass(frozen=True)
class VariantSpec:
    ablation: str
    target_source: str
    contact_gain: float
    hand_support_scale: float = 3.0
    nonhand_support_scale: float = 1.5
    nonhand_start_eval_time: float = 0.6
    carry_corridor_scale: float = 2.0
    target_uses_eef_offset: bool = True


RUNNABLE_VARIANTS = [
    VariantSpec(
        "support_ref_direct",
        target_source="ref_fk",
        contact_gain=5.0,
        target_uses_eef_offset=True,
    ),
    VariantSpec(
        "support_ref_staged",
        target_source="ref_fk",
        contact_gain=5.0,
        nonhand_support_scale=0.8,
        nonhand_start_eval_time=1.1,
        target_uses_eef_offset=True,
    ),
    VariantSpec(
        "support_surface_direct",
        target_source="external",
        contact_gain=3.0,
        target_uses_eef_offset=False,
    ),
]

CASE_SPLITS = {
    "box021_029_p2": ("local-gpu0", "main_upright_carry"),
    "box021_035_p2": ("remote-gpu0", "companion_lowerbody_repair"),
    "box021_035_p1": ("remote-gpu1", "strict_guard"),
    "box004_083_p2": ("local-gpu0", "box004_contact_guard"),
}

TARGET_KEYS = {
    "box021_029_p2": "d003_box021_20231018_029_p2",
    "box021_035_p2": "d003_box021_20231011_035_p2",
    "box021_035_p1": "box021_person1",
    "box004_082_p1": "e091_box004_20231003_2_082_p1",
    "box004_083_p2": "e091_box004_20231003_2_083_p2",
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def repo_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def read_e113_variants() -> dict[str, dict[str, str]]:
    with E113_VARIANTS.open("r", encoding="utf-8", newline="") as f:
        lines = (line for line in f if line.strip() and not line.startswith("#"))
        return {
            row["case_id"]: row
            for row in csv.DictReader(lines, fieldnames=E113_FIELDS, delimiter="\t")
        }


def read_tsv(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row[key]: row for row in csv.DictReader(f, delimiter="\t")}


def target_path_for(case_id: str) -> Path:
    return E100_TARGET_ROOT / TARGET_KEYS[case_id] / "spider_contact_target_object_local.npz"


def validate_mask(path: Path, person_idx: int) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    key = "eval_contact_mask_3cm" if "eval_contact_mask_3cm" in data.files else "spider_contact_mask_3cm"
    if key not in data.files:
        raise KeyError(f"{path} missing eval_contact_mask_3cm or spider_contact_mask_3cm")
    mask = np.asarray(data[key])
    if mask.ndim != 3 or mask.shape[2] != 2:
        raise ValueError(f"{path}:{key} expected (T, persons, 2), got {mask.shape}")
    if person_idx >= mask.shape[1]:
        raise ValueError(f"{path}:{key} person_idx {person_idx} outside shape {mask.shape}")
    person_mask = mask[:, person_idx, :].astype(bool)
    return {
        "mask_key": key,
        "mask_frames": int(mask.shape[0]),
        "mask_active_left_frac": float(person_mask[:, 0].mean()),
        "mask_active_right_frac": float(person_mask[:, 1].mean()),
        "mask_active_either_frac": float(person_mask.any(axis=1).mean()),
    }


def validate_target(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    key = "spider_contact_target_object_local"
    if key not in data.files:
        raise KeyError(f"{path} missing {key}")
    target = np.asarray(data[key])
    if target.ndim != 3 or target.shape[1:] != (2, 3):
        raise ValueError(f"{path}:{key} expected (T,2,3), got {target.shape}")
    return {"target_key": key, "target_frames": int(target.shape[0])}


def scene_loads(task: str) -> bool:
    scene = TASK_ROOT / task / "scene_act.xml"
    mujoco.MjModel.from_xml_path(str(scene))
    return True


def build_rows() -> list[dict[str, Any]]:
    e113_by_case = read_e113_variants()
    pareto_by_variant = read_tsv(E113_PARETO, "variant")
    diag_by_variant = read_tsv(E114_DIAGNOSTICS, "variant")
    rows: list[dict[str, Any]] = []
    ordinal = 1
    for case_id, (split, phase_scope) in CASE_SPLITS.items():
        base = e113_by_case[case_id]
        e113_variant = base["variant"]
        pareto = pareto_by_variant[e113_variant]
        diag = diag_by_variant[e113_variant]
        for spec in RUNNABLE_VARIANTS:
            rows.append(
                {
                    "ordinal": ordinal,
                    "variant": f"E120_{case_id}_{spec.ablation}",
                    "source_task": base["source_task"],
                    "derived_task": base["derived_task"],
                    "person_idx": base["person_idx"],
                    "split": split,
                    "case_id": case_id,
                    "e109_case_id": base["e109_case_id"],
                    "object_key": base["object_key"],
                    "ablation": spec.ablation,
                    "mask_path": base["mask_path"],
                    "source_override": f"core4d_{e113_variant}",
                    "baseline_npz_path": base["baseline_npz_path"],
                    "e113_hold_npz_path": pareto["hold_npz_path"],
                    "e113_hold_video_path": pareto["hold_video_path"],
                    "phase_scope": phase_scope,
                    "diagnostic_queue": diag["diagnostic_queue"],
                    "holdout_reason": "",
                    "_spec": spec,
                    "_target_path": rel(target_path_for(case_id)),
                }
            )
            ordinal += 1
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str], comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(f"# {comment}\n")
        f.write("# " + "\t".join(fields) + "\n")
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def override_text(row: dict[str, Any]) -> str:
    spec: VariantSpec = row["_spec"]
    leg_names = ", ".join(f'"{name}"' for name in LEG_GEOMS)
    robot_object_names = ", ".join(f'"{name}"' for name in ROBOT_OBJECT_GEOMS)
    nonhand_support_names = ", ".join(f'"{name}"' for name in NONHAND_SUPPORT_GEOMS)
    dynamic = "true"
    target_path = row["_target_path"] if spec.target_source == "external" else ""
    uses_offset = "true" if spec.target_uses_eef_offset else "false"
    return f"""# @package _global_
# Auto-generated by workspace/core4d/scripts/E120/build_object_support_decomposition_manifest.py.
# E120 object-support decomposition; case={row['case_id']} ablation={row['ablation']}.
defaults:
  - {row['source_override']}
  - _self_

task: {row['derived_task']}

contact_hdmi_gain: {spec.contact_gain:.1f}
contact_hdmi_dynamic_target: {dynamic}
contact_hdmi_target_source: {spec.target_source}
contact_hdmi_target_path: {target_path}
contact_hdmi_target_time_axis: auto
contact_hdmi_target_uses_eef_offset: {uses_offset}

hand_support_rew_scale: {spec.hand_support_scale:.1f}
hand_support_sigma: 0.015
hand_support_margin_m: 0.01
hand_support_gate_source: contact_mask_time_window
hand_support_start_eval_time: 0.6
hand_support_end_eval_time: 3.0
hand_support_geom_names: ["lh", "rh"]
hand_support_geom_ids: []
nonhand_support_penalty_scale: {spec.nonhand_support_scale:.1f}
nonhand_support_penalty_margin_m: 0.02
nonhand_support_penalty_gate_source: contact_mask_time_window
nonhand_support_penalty_start_eval_time: {spec.nonhand_start_eval_time:.1f}
nonhand_support_penalty_end_eval_time: 3.0
nonhand_support_penalty_geom_names: [{nonhand_support_names}]
nonhand_support_penalty_geom_ids: []

carry_corridor_rew_scale: {spec.carry_corridor_scale:.1f}
carry_corridor_gate_source: contact_mask_time_window
carry_corridor_start_eval_time: 0.6
carry_corridor_end_eval_time: 3.0
carry_corridor_hand_target_threshold_m: 0.02
carry_corridor_hand_sigma: 0.06
carry_corridor_clearance_min_m: 0.04
carry_corridor_clearance_max_m: 0.22
carry_corridor_clearance_sigma: 0.06
carry_corridor_pelvis_min_m: 0.60
carry_corridor_pelvis_sigma: 0.06
carry_corridor_rot_sigma: 0.30
carry_corridor_leg_margin_m: 0.02
carry_corridor_leg_sigma: 0.04
carry_corridor_leg_geom_names: [{leg_names}]
carry_corridor_leg_geom_ids: []

leg_object_penalty_scale: 0.0
leg_object_penalty_margin_m: 0.02
leg_object_penalty_geom_names: [{leg_names}]
leg_object_penalty_geom_ids: []
leg_object_penalty_gate_source: contact_mask_time_window
leg_object_penalty_start_eval_time: 0.6
leg_object_penalty_end_eval_time: 3.0
leg_object_penalty_hand_target_threshold_m: 0.08

# E120 keeps E119 posture guard, but the primary variable is support decomposition.
robot_object_penalty_scale: 0.5
robot_object_penalty_margin_m: 0.0
robot_object_penalty_deep_threshold_m: 0.0
robot_object_penalty_geom_names: [{robot_object_names}]
robot_object_penalty_geom_ids: []
hand_object_deep_penalty_scale: 5.0
hand_object_deep_penalty_threshold_m: 0.01
hand_object_deep_penalty_geom_names: ["lh", "rh"]
hand_object_deep_penalty_geom_ids: []
hand_floor_penalty_scale: 1.0
hand_floor_penalty_margin_m: 0.03
hand_floor_penalty_geom_names: ["lh", "rh"]
hand_floor_penalty_geom_ids: []
cem_safety_gate_enabled: true
cem_safety_gate_mode: elite_filter
cem_safety_gate_geom_names: [{robot_object_names}]
cem_safety_gate_geom_ids: []
cem_safety_gate_min_sdf_m: -0.005
cem_safety_gate_max_violation_pct: 0.0
cem_safety_gate_min_valid_frac: 0.02
cem_safety_gate_fallback: least_violation
ctrl_ref_guard_scale: 0.8
ctrl_ref_guard_robot_only: true
ctrl_ref_guard_sigma: 0.15
ctrl_ref_guard_start_eval_time: 0.6
ctrl_ref_guard_end_eval_time: 3.0
hold_contact_rew_scale: 0.0
hold_contact_sigma: 0.05
hold_contact_start_eval_time: 0.6
hold_contact_end_eval_time: 3.0
hold_contact_require_ref_contact: true
stability_penalty_scale: 1.0
stability_penalty_threshold: 0.60
task_body_rew_scale: 1.5
task_body_names: ["pelvis", "waist_yaw_link", "waist_roll_link", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
task_body_weights: [2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.5, 0.5]
task_obj_use_exp: true
task_obj_pos_rew_scale: 0.5
task_obj_rot_rew_scale: 0.6
object_clearance_rew_scale: 0.0
object_clearance_penalty_scale: 0.0

video_camera: auto
"""


def write_overrides(rows: list[dict[str, Any]]) -> list[str]:
    paths = []
    OVERRIDE_ROOT.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = OVERRIDE_ROOT / f"core4d_{row['variant']}.yaml"
        path.write_text(override_text(row), encoding="utf-8")
        paths.append(rel(path))
    return paths


def write_preflight(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    preflight_rows: list[dict[str, Any]] = []
    for row in rows:
        spec: VariantSpec = row["_spec"]
        mask_info = validate_mask(repo_path(row["mask_path"]), int(row["person_idx"]))
        target_required = spec.target_source == "external"
        target_info: dict[str, Any] = {"target_key": "", "target_frames": ""}
        target_exists = True
        if target_required:
            target_path = repo_path(row["_target_path"])
            target_exists = target_path.is_file()
            target_info = validate_target(target_path)
        out = {
            **{field: row[field] for field in FIELDS},
            "scene_act_exists": (TASK_ROOT / row["derived_task"] / "scene_act.xml").is_file(),
            "scene_act_loads": scene_loads(row["derived_task"]),
            "trajectory_exists": (TASK_ROOT / row["derived_task"] / "0/trajectory_kinematic.npz").is_file(),
            "mask_exists": repo_path(row["mask_path"]).is_file(),
            "baseline_exists": repo_path(row["baseline_npz_path"]).is_file(),
            "e113_hold_exists": repo_path(row["e113_hold_npz_path"]).is_file(),
            "source_override_exists": (OVERRIDE_ROOT / f"{row['source_override']}.yaml").is_file(),
            "override_exists": (OVERRIDE_ROOT / f"core4d_{row['variant']}.yaml").is_file(),
            "target_required": target_required,
            "target_exists": target_exists,
            "target_source": spec.target_source,
            "hand_support_scale": spec.hand_support_scale,
            "nonhand_support_scale": spec.nonhand_support_scale,
            "nonhand_start_eval_time": spec.nonhand_start_eval_time,
            "carry_corridor_scale": spec.carry_corridor_scale,
            "cem_safety_gate": True,
            "leg_geom_count": len(LEG_GEOMS),
            "robot_object_geom_count": len(ROBOT_OBJECT_GEOMS),
            "nonhand_support_geom_count": len(NONHAND_SUPPORT_GEOMS),
            **mask_info,
            **target_info,
        }
        preflight_rows.append(out)
    preflight_fields = FIELDS + [
        "scene_act_exists",
        "scene_act_loads",
        "trajectory_exists",
        "mask_exists",
        "baseline_exists",
        "e113_hold_exists",
        "source_override_exists",
        "override_exists",
        "target_required",
        "target_exists",
        "target_source",
        "hand_support_scale",
        "nonhand_support_scale",
        "nonhand_start_eval_time",
        "carry_corridor_scale",
        "cem_safety_gate",
        "leg_geom_count",
        "robot_object_geom_count",
        "nonhand_support_geom_count",
        "mask_key",
        "mask_frames",
        "mask_active_left_frac",
        "mask_active_right_frac",
        "mask_active_either_frac",
        "target_key",
        "target_frames",
    ]
    write_tsv(PREFLIGHT_TSV, preflight_rows, preflight_fields, "E120 Phase A preflight")
    required = [
        "scene_act_exists",
        "scene_act_loads",
        "trajectory_exists",
        "mask_exists",
        "baseline_exists",
        "e113_hold_exists",
        "source_override_exists",
        "override_exists",
    ]
    all_ok = all(all(row[key] is True for key in required) and (not row["target_required"] or row["target_exists"] is True) for row in preflight_rows)
    summary = {
        "variants": len(rows),
        "cases": len(CASE_SPLITS),
        "ablations": [spec.ablation for spec in RUNNABLE_VARIANTS],
        "all_preflight_ok": all_ok,
        "preflight_tsv": rel(PREFLIGHT_TSV),
        "variants_tsv": rel(VARIANTS_TSV),
        "override_paths": [rel(OVERRIDE_ROOT / f"core4d_{row['variant']}.yaml") for row in rows],
    }
    return preflight_rows, summary


def write_summary_md(rows: list[dict[str, Any]], preflight_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    split_counts: dict[str, int] = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    lines = [
        "# E120 Object Support Decomposition Manifest Summary",
        "",
        f"- variants: `{summary['variants']}`",
        f"- cases: `{summary['cases']}`",
        f"- ablations: `{len(summary['ablations'])}`",
        f"- all_preflight_ok: `{summary['all_preflight_ok']}`",
        "",
        "| split | variants |",
        "|---|---:|",
    ]
    for split, count in sorted(split_counts.items()):
        lines.append(f"| `{split}` | {count} |")
    lines.extend(
        [
            "",
            "| case | ablation | split | target | hand support | non-hand penalty | mask either | phase | queue |",
            "|---|---|---|---|---:|---:|---:|---|---|",
        ]
    )
    by_variant = {row["variant"]: row for row in preflight_rows}
    for row in rows:
        pf = by_variant[row["variant"]]
        lines.append(
            f"| `{row['case_id']}` | `{row['ablation']}` | `{row['split']}` | "
            f"`{pf['target_source']}` | {float(pf['hand_support_scale']):.1f} | "
            f"{float(pf['nonhand_support_scale']):.1f} | {float(pf['mask_active_either_frac']) * 100:.1f}% | "
            f"`{row['phase_scope']}` | `{row['diagnostic_queue']}` |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = build_rows()
    write_overrides(rows)
    write_tsv(VARIANTS_TSV, rows, FIELDS, "E120 object-support decomposition variants")
    preflight_rows, summary = write_preflight(rows)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary_md(rows, preflight_rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
