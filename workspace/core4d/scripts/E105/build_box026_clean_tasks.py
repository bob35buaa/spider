#!/usr/bin/env python3
"""Build clean E105 Box026 derived tasks, variants, overrides, and preflight."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parent / "E083"))
sys.path.insert(0, str(THIS.parent / "E103"))

import create_upperobj_cases as upperobj  # type: ignore  # noqa: E402
from audit_scene_inertials import read_scene as audit_scene  # type: ignore  # noqa: E402
from e105_common import OVERRIDE_ROOT, REPO, RESULTS_ROOT, SOURCE_CASES, TASK_ROOT, rel, rows, write_variants  # noqa: E402


PREFLIGHT_TSV = RESULTS_ROOT / "clean_scene_preflight.tsv"
BUILD_META = RESULTS_ROOT / "task_build_meta.json"


def copy_clean_task(source_task: str, derived_task: str, force: bool) -> Path:
    src = TASK_ROOT / source_task
    dst = TASK_ROOT / derived_task
    if not src.is_dir():
        raise FileNotFoundError(src)
    if "_e092_" in derived_task:
        raise ValueError(f"E105 derived task must not reuse old E092 name: {derived_task}")
    if dst.exists():
        if force:
            shutil.rmtree(dst)
        else:
            print(f"[SKIP] derived task exists: {rel(dst)}")
            return dst
    (dst / "0").mkdir(parents=True, exist_ok=True)
    for item in [
        "scene.xml",
        "scene_act.xml",
        "scene_act_meta.json",
        "task_info.json",
        "0/trajectory_kinematic.npz",
    ]:
        src_file = src / item
        if not src_file.is_file():
            raise FileNotFoundError(src_file)
        dst_file = dst / item
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
    return dst


def object_mass(scene: Path) -> float:
    root = ET.parse(scene).getroot()
    obj = next((body for body in root.iter("body") if body.get("name") == "object"), None)
    inertial = obj.find("inertial") if obj is not None else None
    if inertial is None:
        raise ValueError(f"object inertial missing in {scene}")
    return float(inertial.get("mass", "nan"))


def validate_task(task: str, source_task: str) -> dict[str, Any]:
    task_dir = TASK_ROOT / task
    scene = task_dir / "scene.xml"
    scene_act = task_dir / "scene_act.xml"
    model_scene = mujoco.MjModel.from_xml_path(str(scene))
    model_act = mujoco.MjModel.from_xml_path(str(scene_act))
    qpos = np.load(task_dir / "0/trajectory_kinematic.npz", allow_pickle=True)["qpos"]
    source_qpos = np.load(TASK_ROOT / source_task / "0/trajectory_kinematic.npz", allow_pickle=True)["qpos"]
    root = ET.parse(scene_act).getroot()
    pair_names = {pair.get("name") for pair in root.iter("pair")}
    missing_leg = sorted({f"{geom}_object" for geom in upperobj.LEG_FOOT_GEOMS} - pair_names)
    missing_upper = sorted({f"{geom}_object" for geom in upperobj.UPPER_BODY_GEOMS} - pair_names)
    audit = audit_scene(scene)
    audit_act = audit_scene(scene_act)
    clean_robot = (
        audit.get("robot_polluted_mass_29_632") == "False"
        and int(audit.get("robot_inertial_unique_pairs") or 0) > 5
        and audit_act.get("robot_polluted_mass_29_632") == "False"
        and int(audit_act.get("robot_inertial_unique_pairs") or 0) > 5
    )
    ok = (
        model_scene.nq == 43
        and model_scene.nv == 41
        and model_scene.nu == 29
        and model_act.nq == 42
        and model_act.nv == 41
        and model_act.nu == 35
        and qpos.shape == source_qpos.shape
        and bool(np.allclose(qpos, source_qpos))
        and not missing_leg
        and not missing_upper
        and clean_robot
    )
    report: dict[str, Any] = {
        "task": task,
        "source_task": source_task,
        "scene_nq": int(model_scene.nq),
        "scene_nv": int(model_scene.nv),
        "scene_nu": int(model_scene.nu),
        "scene_act_nq": int(model_act.nq),
        "scene_act_nv": int(model_act.nv),
        "scene_act_nu": int(model_act.nu),
        "qpos_shape": list(qpos.shape),
        "qpos_matches_source": bool(np.allclose(qpos, source_qpos)),
        "object_mass_kg": object_mass(scene_act),
        "scene_robot_polluted_mass_29_632": audit.get("robot_polluted_mass_29_632", ""),
        "scene_act_robot_polluted_mass_29_632": audit_act.get("robot_polluted_mass_29_632", ""),
        "scene_robot_inertial_unique_pairs": audit.get("robot_inertial_unique_pairs", ""),
        "scene_act_robot_inertial_unique_pairs": audit_act.get("robot_inertial_unique_pairs", ""),
        "missing_leg_pairs": missing_leg,
        "missing_upper_pairs": missing_upper,
        "validation_ok": ok,
    }
    if not ok:
        raise RuntimeError(f"E105 task validation failed for {task}: {report}")
    return report


def patch_task_info(task: str, source_task: str, validation: dict[str, Any]) -> None:
    path = TASK_ROOT / task / "task_info.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    data.update(
        {
            "task": task,
            "source_task_e105": source_task,
            "e105_clean_scene": True,
            "e105_validation": validation,
            "e105_note": "Clean derived task for E105; old _e092_* task directories are not reused.",
        }
    )
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_override(row: dict[str, str]) -> Path:
    path = OVERRIDE_ROOT / f"core4d_{row['variant']}.yaml"
    if row["route"] == "ref_fk_clean":
        content = f"""# @package _global_
# Auto-generated by workspace/core4d/scripts/E105/build_box026_clean_tasks.py.
# E105 clean-scene ref_fk rerun aligned to {row['old_variant']}.
defaults:
  - core4d_E089A_box021_person1_upperobj
  - _self_

task: {row['derived_task']}

contact_hdmi_target_source: ref_fk
contact_hdmi_target_path: ""
contact_hdmi_target_uses_eef_offset: true
contact_hdmi_mask_source: ""
contact_hdmi_mask_path: ""
contact_hdmi_mask_person_idx: {row['person_idx']}
contact_hdmi_mask_time_axis: auto
hold_contact_rew_scale: 0.0
hold_contact_start_eval_time: 0.0
hold_contact_end_eval_time: 0.0
video_camera: auto
"""
    elif row["route"] == "adaptive_clean":
        content = f"""# @package _global_
# Auto-generated by workspace/core4d/scripts/E105/build_box026_clean_tasks.py.
# E105 clean-scene adaptive-support rerun aligned to {row['old_variant']}.
defaults:
  - core4d_{row['source_variant']}
  - _self_

task: {row['derived_task']}

contact_hdmi_dynamic_target: true
contact_hdmi_target_source: external
contact_hdmi_target_path: {row['target_npz']}
contact_hdmi_target_time_axis: auto
contact_hdmi_target_uses_eef_offset: false
contact_hdmi_eef_offset: [0.05, 0.0, 0.0]
contact_hdmi_mask_source: ""
contact_hdmi_mask_path: ""
contact_hdmi_mask_person_idx: {row['person_idx']}
contact_hdmi_mask_time_axis: auto
video_camera: auto
"""
    elif row["route"] == "fingertip_clean":
        content = f"""# @package _global_
# Auto-generated by workspace/core4d/scripts/E105/build_box026_clean_tasks.py.
# E105 clean-scene E100/E101-style fingertip target ablation.
defaults:
  - core4d_{row['source_variant']}
  - _self_

task: {row['derived_task']}

contact_hdmi_dynamic_target: true
contact_hdmi_target_source: external
contact_hdmi_target_path: {row['target_npz']}
contact_hdmi_target_time_axis: auto
contact_hdmi_target_uses_eef_offset: false
video_camera: auto
"""
    else:
        raise ValueError(row["route"])
    path.write_text(content, encoding="utf-8")
    return path


def write_preflight(built: list[dict[str, Any]]) -> None:
    tasks = [
        "box026_person1",
        "box026_person2",
        SOURCE_CASES["039"]["source_task"],
        SOURCE_CASES["135"]["source_task"],
        f"{SOURCE_CASES['039']['source_task']}_e092_dyn",
        f"{SOURCE_CASES['135']['source_task']}_e092_dyn",
        f"{SOURCE_CASES['039']['source_task']}_e092_omni",
        f"{SOURCE_CASES['135']['source_task']}_e092_omni",
        SOURCE_CASES["039"]["derived_task"],
        SOURCE_CASES["135"]["derived_task"],
    ]
    rows_out = []
    for task in tasks:
        scene = TASK_ROOT / task / "scene.xml"
        scene_act = TASK_ROOT / task / "scene_act.xml"
        if not scene.is_file():
            rows_out.append({"task": task, "scene_xml": rel(scene), "exists": "False", "gate": "MISSING"})
            continue
        audit = audit_scene(scene)
        act_audit = audit_scene(scene_act) if scene_act.is_file() else {}
        polluted = audit.get("robot_polluted_mass_29_632") == "True" or act_audit.get("robot_polluted_mass_29_632") == "True"
        is_old = "_e092_" in task
        gate = "INVALIDATED_OLD_DERIVED" if is_old and polluted else ("PASS" if not polluted else "FAIL")
        rows_out.append(
            {
                "task": task,
                "scene_xml": rel(scene),
                "scene_act_xml": rel(scene_act) if scene_act.is_file() else "",
                "exists": "True",
                "scene_status": audit.get("status", ""),
                "scene_act_status": act_audit.get("status", ""),
                "robot_polluted_mass_29_632": str(polluted),
                "robot_inertial_unique_pairs": audit.get("robot_inertial_unique_pairs", ""),
                "scene_act_robot_inertial_unique_pairs": act_audit.get("robot_inertial_unique_pairs", ""),
                "gate": gate,
            }
        )
    fields = sorted({key for row in rows_out for key in row})
    PREFLIGHT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with PREFLIGHT_TSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    built: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows():
        source = row["source_task"]
        derived = row["derived_task"]
        if derived in seen:
            continue
        seen.add(derived)
        dst = copy_clean_task(source, derived, args.force)
        added_leg, added_upper = upperobj.patch_scene_act(dst / "scene_act.xml", source, derived)
        validation = validate_task(derived, source)
        patch_task_info(derived, source, validation)
        built.append(
            {
                "source_task": source,
                "derived_task": derived,
                "task_dir": rel(dst),
                "added_leg_pairs": added_leg,
                "added_upper_pairs": added_upper,
                "validation": validation,
            }
        )
        print(f"[E105-task] {source} -> {derived} validation=ok qpos={validation['qpos_shape']}")

    write_variants()
    overrides = [rel(write_override(row)) for row in rows()]
    write_preflight(built)
    BUILD_META.parent.mkdir(parents=True, exist_ok=True)
    BUILD_META.write_text(
        json.dumps({"tasks": built, "overrides": overrides}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {rel(PREFLIGHT_TSV)}")
    print(f"wrote {rel(BUILD_META)}")


if __name__ == "__main__":
    main()
