#!/usr/bin/env python3
"""Build E116 surface-target + upright-guard diagnostic manifest."""

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
SCRIPTS_ROOT = REPO / "workspace/core4d/scripts/E116"
RESULTS_ROOT = REPO / "workspace/core4d/results/E116"
PREFLIGHT_ROOT = RESULTS_ROOT / "preflight"
E113_VARIANTS = REPO / "workspace/core4d/scripts/E113/variants.tsv"
E113_PARETO = REPO / "workspace/core4d/results/E113/cem/full/pareto_decisions.tsv"
E114_DIAGNOSTICS = REPO / "workspace/core4d/results/E114/rl_handoff_gate/diagnostic_queues.tsv"
E100_TARGET_ROOT = REPO / "workspace/core4d/results/E100/fingertip_targets"
VARIANTS_TSV = SCRIPTS_ROOT / "variants.tsv"
PREFLIGHT_TSV = PREFLIGHT_ROOT / "phaseA_preflight.tsv"
SUMMARY_JSON = PREFLIGHT_ROOT / "phaseA_manifest_summary.json"
SUMMARY_MD = PREFLIGHT_ROOT / "phaseA_manifest_summary.md"

UPPERBODY_GEOMS = [
    "head_collision",
    "torso_collision",
    "pelvis_collision",
    "left_shoulder_yaw_collision",
    "right_shoulder_yaw_collision",
    "left_elbow_yaw_collision",
    "right_elbow_yaw_collision",
]

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
    upright_guard: bool
    safety: bool
    contact_gain: float = 3.0


RUNNABLE_VARIANTS = [
    VariantSpec("surface_light", upright_guard=False, safety=False),
    VariantSpec("surface_light_safety", upright_guard=False, safety=True),
    VariantSpec("surface_upright_safety", upright_guard=True, safety=True),
]

CASE_SPLITS = {
    "box021_029_p2": ("remote-gpu0", "main_surface_target"),
    "box021_035_p2": ("remote-gpu1", "companion_lowerbody_repair"),
    "box021_035_p1": ("remote-gpu1", "strict_guard"),
    "box004_082_p1": ("local-gpu0", "strict_guard"),
    "box004_083_p2": ("local-gpu0", "strict_guard"),
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
        return {row["case_id"]: row for row in csv.DictReader(lines, fieldnames=E113_FIELDS, delimiter="\t")}


def read_tsv(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row[key]: row for row in csv.DictReader(f, delimiter="\t")}


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
    if "spider_contact_target_object_local" not in data.files:
        raise KeyError(f"{path} missing spider_contact_target_object_local")
    target = np.asarray(data["spider_contact_target_object_local"])
    if target.ndim != 3 or target.shape[1:] != (2, 3):
        raise ValueError(f"{path}: target expected (T,2,3), got {target.shape}")
    active_frac = ""
    if "active" in data.files:
        active = np.asarray(data["active"]).astype(bool)
        active_frac = float(active.mean())
    return {
        "target_exists": path.is_file(),
        "target_key": "spider_contact_target_object_local",
        "target_frames": int(target.shape[0]),
        "target_active_frac": active_frac,
    }


def scene_loads(task: str) -> bool:
    scene = TASK_ROOT / task / "scene_act.xml"
    mujoco.MjModel.from_xml_path(str(scene))
    return True


def target_path_for(case_id: str) -> Path:
    return E100_TARGET_ROOT / TARGET_KEYS[case_id] / "spider_contact_target_object_local.npz"


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
                    "variant": f"E116_{case_id}_{spec.ablation}",
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
                    "_contact_gain": spec.contact_gain,
                    "_upright_guard": spec.upright_guard,
                    "_safety": spec.safety,
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
    upperbody_names = ", ".join(f'"{name}"' for name in UPPERBODY_GEOMS)
    target_path = row["_target_path"]
    safety_enabled = bool(row["_safety"])
    upright_enabled = bool(row["_upright_guard"])
    robot_object_scale = 2.0 if safety_enabled else 0.0
    robot_object_names = f"[{upperbody_names}]" if safety_enabled else "[]"
    hand_deep_scale = 5.0 if safety_enabled else 0.0
    hand_deep_names = '["lh", "rh"]' if safety_enabled else "[]"
    hand_floor_scale = 1.0 if safety_enabled else 0.0
    hand_floor_names = '["lh", "rh"]' if safety_enabled else "[]"
    ctrl_guard_scale = 0.8 if upright_enabled else 0.0
    stability_scale = 1.0 if upright_enabled else 0.0
    task_body_scale = 1.5 if upright_enabled else 0.0
    task_obj_exp = "true" if upright_enabled else "false"
    lines = [
        "# @package _global_",
        "# Auto-generated by workspace/core4d/scripts/E116/build_surface_target_upright_manifest.py.",
        f"# E116 surface-target/upright diagnostic; case={row['case_id']} ablation={row['ablation']}.",
        "defaults:",
        f"  - {row['source_override']}",
        "  - _self_",
        "",
        f"task: {row['derived_task']}",
        "",
        "contact_hdmi_dynamic_target: true",
        "contact_hdmi_target_source: external",
        f"contact_hdmi_target_path: {target_path}",
        "contact_hdmi_target_time_axis: auto",
        "contact_hdmi_target_uses_eef_offset: false",
        f"contact_hdmi_gain: {float(row['_contact_gain']):.1f}",
        "",
        "# Set every inherited safety/upright knob exactly once so E116 ablations stay interpretable.",
        f"robot_object_penalty_scale: {robot_object_scale:.1f}",
        "robot_object_penalty_margin_m: 0.02",
        "robot_object_penalty_deep_threshold_m: 0.0",
        f"robot_object_penalty_geom_names: {robot_object_names}",
        "robot_object_penalty_geom_ids: []",
        f"hand_object_deep_penalty_scale: {hand_deep_scale:.1f}",
        "hand_object_deep_penalty_threshold_m: 0.01",
        f"hand_object_deep_penalty_geom_names: {hand_deep_names}",
        "hand_object_deep_penalty_geom_ids: []",
        f"hand_floor_penalty_scale: {hand_floor_scale:.1f}",
        "hand_floor_penalty_margin_m: 0.03",
        f"hand_floor_penalty_geom_names: {hand_floor_names}",
        "hand_floor_penalty_geom_ids: []",
        "cem_safety_gate_enabled: false",
        "cem_safety_gate_geom_names: []",
        "cem_safety_gate_geom_ids: []",
        f"ctrl_ref_guard_scale: {ctrl_guard_scale:.1f}",
        "ctrl_ref_guard_robot_only: true",
        "ctrl_ref_guard_sigma: 0.15",
        "ctrl_ref_guard_start_eval_time: 0.6",
        "ctrl_ref_guard_end_eval_time: 3.0",
        f"stability_penalty_scale: {stability_scale:.1f}",
        "stability_penalty_threshold: 0.60",
        f"task_body_rew_scale: {task_body_scale:.1f}",
        'task_body_names: ["pelvis", "waist_yaw_link", "waist_roll_link", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]',
        "task_body_weights: [2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.5, 0.5]",
        f"task_obj_use_exp: {task_obj_exp}",
        "task_obj_pos_rew_scale: 0.5",
        "task_obj_rot_rew_scale: 0.3",
    ]
    lines.extend(["", "video_camera: auto", ""])
    return "\n".join(lines)


def write_overrides(rows: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    for row in rows:
        path = OVERRIDE_ROOT / f"core4d_{row['variant']}.yaml"
        path.write_text(override_text(row), encoding="utf-8")
        paths.append(rel(path))
    return paths


def preflight(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        person_idx = int(row["person_idx"])
        task = row["derived_task"]
        scene = TASK_ROOT / task / "scene_act.xml"
        traj = TASK_ROOT / task / "0/trajectory_kinematic.npz"
        mask_path = repo_path(str(row["mask_path"]))
        target_path = repo_path(str(row["_target_path"]))
        baseline = repo_path(str(row["baseline_npz_path"]))
        e113_hold = repo_path(str(row["e113_hold_npz_path"]))
        source_override = OVERRIDE_ROOT / f"{row['source_override']}.yaml"
        override = OVERRIDE_ROOT / f"core4d_{row['variant']}.yaml"
        mask_info = validate_mask(mask_path, person_idx)
        target_info = validate_target(target_path)
        out.append(
            {
                **{field: row[field] for field in FIELDS},
                "surface_target_path": row["_target_path"],
                "scene_act_exists": scene.is_file(),
                "scene_act_loads": scene_loads(task),
                "trajectory_exists": traj.is_file(),
                "mask_exists": mask_path.is_file(),
                "baseline_exists": baseline.is_file(),
                "e113_hold_exists": e113_hold.is_file(),
                "source_override_exists": source_override.is_file(),
                "override_exists": override.is_file(),
                "upright_guard": "enabled" if row["_upright_guard"] else "disabled",
                "light_safety": "enabled" if row["_safety"] else "disabled",
                **mask_info,
                **target_info,
            }
        )
    return out


def write_summary(rows: list[dict[str, Any]], pf_rows: list[dict[str, Any]], override_paths: list[str]) -> None:
    summary = {
        "stage": "E116_phaseA_manifest",
        "variant_rows": len(rows),
        "case_count": len(CASE_SPLITS),
        "ablation_count": len(RUNNABLE_VARIANTS),
        "splits": {split: sum(1 for row in rows if row["split"] == split) for split, _ in set(CASE_SPLITS.values())},
        "all_preflight_ok": all(
            bool(row["scene_act_exists"])
            and bool(row["scene_act_loads"])
            and bool(row["trajectory_exists"])
            and bool(row["mask_exists"])
            and bool(row["target_exists"])
            and bool(row["baseline_exists"])
            and bool(row["e113_hold_exists"])
            and bool(row["source_override_exists"])
            and bool(row["override_exists"])
            for row in pf_rows
        ),
        "override_paths": override_paths,
    }
    PREFLIGHT_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# E116 Surface Target + Upright Guard Manifest Summary",
        "",
        f"- variants: `{summary['variant_rows']}`",
        f"- cases: `{summary['case_count']}`",
        f"- ablations: `{summary['ablation_count']}`",
        f"- all_preflight_ok: `{summary['all_preflight_ok']}`",
        "",
        "| split | variants |",
        "|---|---:|",
    ]
    for split, count in sorted(summary["splits"].items()):
        lines.append(f"| `{split}` | {count} |")
    lines.extend(["", "| case | ablation | split | target frames | mask either | phase | queue |", "|---|---|---|---:|---:|---|---|"])
    for row in pf_rows:
        lines.append(
            f"| `{row['case_id']}` | `{row['ablation']}` | `{row['split']}` | "
            f"{int(row['target_frames'])} | {float(row['mask_active_either_frac']) * 100:.1f}% | "
            f"`{row['phase_scope']}` | `{row['diagnostic_queue']}` |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = build_rows()
    override_paths = write_overrides(rows)
    public_rows = [{field: row[field] for field in FIELDS} for row in rows]
    write_tsv(VARIANTS_TSV, public_rows, FIELDS, "E116 surface-target + upright-guard diagnostic variants")
    pf_rows = preflight(rows)
    write_tsv(PREFLIGHT_TSV, pf_rows, list(pf_rows[0]), "E116 Phase A preflight")
    write_summary(rows, pf_rows, override_paths)
    print(f"wrote {rel(VARIANTS_TSV)}")
    print(f"wrote {rel(PREFLIGHT_TSV)}")
    print(f"wrote {rel(SUMMARY_MD)}")


if __name__ == "__main__":
    main()
