#!/usr/bin/env python3
"""Build E106 clean derived tasks, ref-FK overrides, and variants."""

from __future__ import annotations

import argparse
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
from e106_common import (  # noqa: E402
    OVERRIDE_ROOT,
    RESULTS_ROOT,
    TASK_ROOT,
    VARIANT_FIELDS,
    VARIANTS_TSV,
    read_candidates,
    rel,
    write_tsv,
)


BUILD_META = RESULTS_ROOT / "clean_task_build_meta.json"
PREFLIGHT_TSV = RESULTS_ROOT / "clean_task_preflight.tsv"
FAILURES_TSV = RESULTS_ROOT / "preprocess_failures.tsv"


def object_mass(scene: Path) -> float:
    root = ET.parse(scene).getroot()
    obj = next((body for body in root.iter("body") if body.get("name") == "object"), None)
    inertial = obj.find("inertial") if obj is not None else None
    if inertial is None:
        raise ValueError(f"object inertial missing in {scene}")
    return float(inertial.get("mass", "nan"))


def copy_task(source_task: str, derived_task: str, force: bool) -> Path:
    src = TASK_ROOT / source_task
    dst = TASK_ROOT / derived_task
    if "_e092_" in source_task or "_e092_" in derived_task:
        raise ValueError(f"E106 refuses legacy polluted _e092_ path: {source_task} -> {derived_task}")
    if not (src / "0/trajectory_kinematic.npz").is_file():
        raise FileNotFoundError(src / "0/trajectory_kinematic.npz")
    if dst.exists():
        if force:
            shutil.rmtree(dst)
        else:
            print(f"[SKIP] derived task exists: {rel(dst)}")
            return dst
    shutil.copytree(src, dst)
    return dst


def validate_task(row: dict[str, str]) -> dict[str, Any]:
    src = TASK_ROOT / row["source_task"]
    dst = TASK_ROOT / row["derived_task"]
    scene = dst / "scene.xml"
    scene_act = dst / "scene_act.xml"
    model_scene = mujoco.MjModel.from_xml_path(str(scene))
    model_act = mujoco.MjModel.from_xml_path(str(scene_act))
    qpos = np.load(dst / "0/trajectory_kinematic.npz", allow_pickle=True)["qpos"]
    source_qpos = np.load(src / "0/trajectory_kinematic.npz", allow_pickle=True)["qpos"]
    root = ET.parse(scene_act).getroot()
    pair_names = {pair.get("name") for pair in root.iter("pair")}
    missing_leg = sorted({f"{geom}_object" for geom in upperobj.LEG_FOOT_GEOMS} - pair_names)
    missing_upper = sorted({f"{geom}_object" for geom in upperobj.UPPER_BODY_GEOMS} - pair_names)
    audit = audit_scene(scene)
    audit_act = audit_scene(scene_act)
    clean_robot = (
        audit.get("robot_polluted_mass_29_632") == "False"
        and audit_act.get("robot_polluted_mass_29_632") == "False"
        and int(audit.get("robot_inertial_unique_pairs") or 0) > 5
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
        "variant": row["variant"],
        "source_task": row["source_task"],
        "derived_task": row["derived_task"],
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
        "missing_leg_pairs": ",".join(missing_leg),
        "missing_upper_pairs": ",".join(missing_upper),
        "validation_ok": str(ok),
    }
    if not ok:
        raise RuntimeError(f"E106 task validation failed for {row['derived_task']}: {report}")
    return report


def patch_task_info(row: dict[str, str], validation: dict[str, Any]) -> None:
    path = TASK_ROOT / row["derived_task"] / "task_info.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    data.update(
        {
            "task": row["derived_task"],
            "source_task_e106": row["source_task"],
            "e106_variant": row["variant"],
            "e106_clean_scene": True,
            "e106_route": "ref_fk_clean",
            "e106_validation": validation,
            "e106_note": "Clean derived task for E106 30-candidate Box026 batch; old _e092_* paths are never reused.",
        }
    )
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_override(row: dict[str, str]) -> Path:
    path = OVERRIDE_ROOT / f"core4d_{row['variant']}.yaml"
    content = f"""# @package _global_
# Auto-generated by workspace/core4d/scripts/E106/build_e106_clean_tasks.py.
# E106 Box026 30-candidate clean ref-FK batch.
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
    path.write_text(content, encoding="utf-8")
    return path


def preprocess_failures() -> set[str]:
    if not FAILURES_TSV.is_file():
        return set()
    failed: set[str] = set()
    for line in FAILURES_TSV.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        failed.add(line.split("\t", 1)[0])
    return failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--require-all-ready", action="store_true")
    args = parser.parse_args()

    failed_sources = preprocess_failures()
    built: list[dict[str, Any]] = []
    variant_rows: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for row in read_candidates():
        if row["source_task"] in failed_sources:
            skipped.append({"variant": row["variant"], "source_task": row["source_task"], "reason": "preprocess_failed"})
            continue
        source_traj = TASK_ROOT / row["source_task"] / "0/trajectory_kinematic.npz"
        if not source_traj.is_file():
            skipped.append({"variant": row["variant"], "source_task": row["source_task"], "reason": rel(source_traj)})
            continue
        dst = copy_task(row["source_task"], row["derived_task"], args.force)
        added_leg, added_upper = upperobj.patch_scene_act(dst / "scene_act.xml", row["source_task"], row["derived_task"])
        validation = validate_task(row)
        patch_task_info(row, validation)
        override = write_override(row)
        variant_rows.append({field: row[field] for field in VARIANT_FIELDS})
        built.append(
            {
                "variant": row["variant"],
                "source_task": row["source_task"],
                "derived_task": row["derived_task"],
                "task_dir": rel(dst),
                "override": rel(override),
                "added_leg_pairs": added_leg,
                "added_upper_pairs": added_upper,
                "validation": validation,
            }
        )
        print(f"[E106-task] {row['source_task']} -> {row['derived_task']} qpos={validation['qpos_shape']}")

    write_tsv(VARIANTS_TSV, variant_rows, VARIANT_FIELDS, "E106 runnable variants generated from candidates.tsv")
    PREFLIGHT_TSV.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for item in built for key in item.get("validation", {})} | {"variant", "source_task", "derived_task"})
    write_tsv(PREFLIGHT_TSV, [{**item["validation"]} for item in built], fields)
    BUILD_META.write_text(
        json.dumps({"built": built, "skipped": skipped}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if skipped:
        print(f"[E106-task] skipped missing source tasks: {len(skipped)}")
        for item in skipped:
            print(f"  - {item['variant']}: {item['reason']}")
    expected = 30 - len(failed_sources)
    unexpected_skipped = [item for item in skipped if item["reason"] != "preprocess_failed"]
    if args.require_all_ready and (len(built) != expected or unexpected_skipped):
        raise SystemExit(
            "E106 requires all non-failed tasks before CEM; "
            f"built={len(built)} expected={expected} preprocess_failures={len(failed_sources)} "
            f"unexpected_skipped={len(unexpected_skipped)}"
        )
    print(f"wrote {rel(VARIANTS_TSV)} rows={len(variant_rows)}")
    print(f"wrote {rel(PREFLIGHT_TSV)}")
    print(f"wrote {rel(BUILD_META)}")


if __name__ == "__main__":
    main()
