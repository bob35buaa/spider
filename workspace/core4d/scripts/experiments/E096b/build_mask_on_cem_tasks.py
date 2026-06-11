#!/usr/bin/env python3
"""Build E096b mask-on full-CEM tasks from E096 ready case manifest."""

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


REPO = Path(__file__).resolve().parents[4]
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
OVERRIDE_ROOT = REPO / "examples/config/override"
DEFAULT_MANIFEST = REPO / "workspace/core4d/results/E096/contact_semantics/case_manifest.tsv"
RESULT_ROOT = REPO / "workspace/core4d/results/E096b"
MASK_ROOT = RESULT_ROOT / "contact_masks"
VARIANTS_OUT = REPO / "workspace/core4d/scripts/E096b/variants.tsv"
META_OUT = RESULT_ROOT / "cem/task_build_meta.json"
E083_DIR = REPO / "workspace/core4d/scripts/E083"

if str(E083_DIR) not in sys.path:
    sys.path.insert(0, str(E083_DIR))

import create_upperobj_cases as upperobj  # type: ignore  # noqa: E402


FIELDS = [
    "route",
    "case_id",
    "variant",
    "source_task",
    "derived_task",
    "person_idx",
    "split",
    "object",
    "role",
    "note",
    "ready",
    "blocker",
]


CASE_NUM = {
    "box004_083_p1": "P1",
    "box004_082_p1": "P2",
}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def object_mass(scene: Path) -> float:
    root = ET.parse(scene).getroot()
    obj = next((body for body in root.iter("body") if body.get("name") == "object"), None)
    inertial = obj.find("inertial") if obj is not None else None
    if inertial is None:
        raise ValueError(f"object inertial not found in {scene}")
    return float(inertial.get("mass", "nan"))


def validate_task(task_dir: Path) -> dict[str, Any]:
    scene_act = task_dir / "scene_act.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_act))
    qpos = np.load(task_dir / "0/trajectory_kinematic.npz", allow_pickle=True)["qpos"]
    root = ET.parse(scene_act).getroot()
    pair_names = {pair.get("name") for pair in root.iter("pair")}
    required_leg = {f"{geom}_object" for geom in upperobj.LEG_FOOT_GEOMS}
    required_upper = {f"{geom}_object" for geom in upperobj.UPPER_BODY_GEOMS}
    missing_leg = sorted(required_leg - pair_names)
    missing_upper = sorted(required_upper - pair_names)
    mass = object_mass(scene_act)
    ok = (
        model.nq == 42
        and model.nv == 41
        and model.nu == 35
        and model.npair >= 49
        and qpos.ndim == 2
        and qpos.shape[1] == 43
        and not missing_leg
        and not missing_upper
        and mass > 0
    )
    report = {
        "model_nq": int(model.nq),
        "model_nv": int(model.nv),
        "model_nu": int(model.nu),
        "model_npair": int(model.npair),
        "trajectory_qpos_shape": list(qpos.shape),
        "object_mass_kg": mass,
        "missing_leg_pairs": missing_leg,
        "missing_upper_pairs": missing_upper,
        "validation_ok": ok,
    }
    if not ok:
        raise RuntimeError(f"E096b task validation failed for {task_dir}: {report}")
    return report


def copy_mask(row: dict[str, str]) -> dict[str, str]:
    mask_src = Path(row["mask_path"])
    audit_src = Path(row["audit_summary"])
    if not mask_src.is_file():
        raise FileNotFoundError(mask_src)
    if not audit_src.is_file():
        raise FileNotFoundError(audit_src)
    out_dir = MASK_ROOT / row["task"]
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dst = out_dir / "raw_contact_mask_3cm.npz"
    audit_dst = out_dir / "audit_summary_3cm.json"
    shutil.copy2(mask_src, mask_dst)
    shutil.copy2(audit_src, audit_dst)
    return {"mask_path_repo": rel(mask_dst), "audit_summary_repo": rel(audit_dst)}


def variant_row(case: dict[str, str], mask_meta: dict[str, str]) -> dict[str, str]:
    case_no = CASE_NUM[case["case_id"]]
    variant = f"E096b{case_no}_{case['case_id']}_mask_cem"
    split = "local" if case_no == "P1" else "remote-gpu0"
    return {
        "route": "cem_ref_fk",
        "case_id": case["case_id"],
        "variant": variant,
        "source_task": case["task"],
        "derived_task": f"{case['task']}_e096b_mask_cem",
        "person_idx": case["person_idx"],
        "split": split,
        "object": case["object_group"],
        "role": f"E096b_{case_no}_mask_on",
        "note": f"E096b mask-on full CEM; mask={mask_meta['mask_path_repo']}",
        "ready": "True",
        "blocker": "none",
    }


def patch_task_info(dst: Path, row: dict[str, str], validation: dict[str, Any], mask_meta: dict[str, str]) -> None:
    path = dst / "task_info.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    data.update(
        {
            "task": row["derived_task"],
            "source_task_e096b": row["source_task"],
            "e096b_route": row["route"],
            "e096b_variant": row["variant"],
            "e096b_case_id": row["case_id"],
            "e096b_note": row["note"],
            "e096b_mask": mask_meta,
            "e096b_validation": validation,
        }
    )
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_override(row: dict[str, str], mask_meta: dict[str, str]) -> Path:
    path = OVERRIDE_ROOT / f"core4d_{row['variant']}.yaml"
    text = f"""# @package _global_
# Auto-generated by workspace/core4d/scripts/E096b/build_mask_on_cem_tasks.py.
# E096b full CEM; case={row['case_id']} source={row['source_task']}.
defaults:
  - core4d_E089A_box021_person1_upperobj
  - _self_

task: {row['derived_task']}

# E096b is the mask-on rerun of E096: same ref-FK wrist5cm target,
# but with the correct CORE4D 3cm per-EEF raw contact mask.
contact_hdmi_target_source: ref_fk
contact_hdmi_target_path: ""
contact_hdmi_target_uses_eef_offset: true
contact_hdmi_mask_source: core4d_3cm
contact_hdmi_mask_path: {mask_meta['mask_path_repo']}
contact_hdmi_mask_person_idx: {row['person_idx']}
contact_hdmi_mask_time_axis: auto
hold_contact_rew_scale: 0.0
hold_contact_start_eval_time: 0.0
hold_contact_end_eval_time: 0.0
"""
    path.write_text(text, encoding="utf-8")
    return path


def build_one(case: dict[str, str], *, force: bool) -> dict[str, Any]:
    mask_meta = copy_mask(case)
    row = variant_row(case, mask_meta)
    src = TASK_ROOT / row["source_task"]
    dst = TASK_ROOT / row["derived_task"]
    if not src.is_dir():
        raise FileNotFoundError(src)
    if not (src / "0/trajectory_kinematic.npz").is_file():
        raise FileNotFoundError(src / "0/trajectory_kinematic.npz")
    if dst.exists():
        if force:
            shutil.rmtree(dst)
        else:
            print(f"[SKIP] derived task exists: {rel(dst)}")
    if not dst.exists():
        shutil.copytree(src, dst)

    added_leg, added_upper = upperobj.patch_scene_act(
        dst / "scene_act.xml", row["source_task"], row["derived_task"]
    )
    validation = validate_task(dst)
    patch_task_info(dst, row, validation, mask_meta)
    override = write_override(row, mask_meta)
    meta = {
        **row,
        "task_dir": rel(dst),
        "override": rel(override),
        "mask": mask_meta,
        "added_leg_pairs": added_leg,
        "added_upper_pairs": added_upper,
        "validation": validation,
    }
    print(
        f"{row['case_id']} {row['source_task']} -> {row['derived_task']}: "
        f"mask={mask_meta['mask_path_repo']} npair={validation['model_npair']} validation=ok"
    )
    return meta


def write_variants(rows: list[dict[str, str]]) -> None:
    VARIANTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with VARIANTS_OUT.open("w", encoding="utf-8", newline="") as f:
        f.write("# E096b variants generated by build_mask_on_cem_tasks.py\n")
        f.write("# " + "\t".join(FIELDS) + "\n")
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for row in rows:
            writer.writerow([row[field] for field in FIELDS])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    meta: list[dict[str, Any]] = []
    for case in read_manifest(args.manifest):
        if case["case_id"] not in CASE_NUM:
            continue
        if str(case.get("ready", "")).lower() != "true":
            continue
        item = build_one(case, force=args.force)
        rows.append({field: item[field] for field in FIELDS})
        meta.append(item)

    if not rows:
        raise SystemExit("No ready E096b mask-on cases found")
    write_variants(rows)
    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    META_OUT.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {rel(VARIANTS_OUT)}")
    print(f"wrote {rel(META_OUT)}")


if __name__ == "__main__":
    main()
