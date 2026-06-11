#!/usr/bin/env python3
"""Build E092 three-case tasks and route manifest.

E092 uses the three E091 Stage2b tasks that already have OmniRetarget
visualizations. For each case this script creates two derived tasks:

- spider_dyn: used by Stage A SPIDER/MJWP dynamic retargeting.
- rl_omni: used by Stage C direct OmniRetarget training control.

Both derived tasks keep the original E091 kinematic reference, but add the
leg/foot and upper-body object collision pairs used by the E083+ safety stack.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
OVERRIDE_ROOT = REPO / "examples/config/override"
VARIANTS_OUT = REPO / "workspace/core4d/scripts/E092/variants.tsv"
META_OUT = REPO / "workspace/core4d/results/E092/task_build_meta.json"
E083_DIR = REPO / "workspace/core4d/scripts/E083"

if str(E083_DIR) not in sys.path:
    sys.path.insert(0, str(E083_DIR))

import create_upperobj_cases as upperobj  # type: ignore  # noqa: E402


CASES = [
    {
        "case_id": "C1",
        "object": "box004",
        "source_task": "e091_box004_20231003_2_083_p2",
        "person_idx": "1",
        "split": "local",
        "role": "positive_seed",
        "note": "E091 D005b PASS; smoke pelvis collapse follow-up.",
    },
    {
        "case_id": "C2",
        "object": "Box026",
        "source_task": "e091_box026_20231018_039_p2",
        "person_idx": "1",
        "split": "remote-gpu0",
        "role": "low_support_near_pass",
        "note": "E091 D005b reject: support_face_either=19.5%.",
    },
    {
        "case_id": "C3",
        "object": "Box026",
        "source_task": "e091_box026_20231020_135_p2",
        "person_idx": "1",
        "split": "remote-gpu1",
        "role": "inside_risk_near_pass",
        "note": "E091 D005b reject: right wrist inside=12.2%.",
    },
]


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _object_mass(scene: Path) -> float:
    root = ET.parse(scene).getroot()
    obj = next((body for body in root.iter("body") if body.get("name") == "object"), None)
    inertial = obj.find("inertial") if obj is not None else None
    if inertial is None:
        raise ValueError(f"object inertial not found in {scene}")
    return float(inertial.get("mass", "nan"))


def _route_variant(case: dict[str, str], route: str) -> dict[str, str]:
    short = {
        "e091_box004_20231003_2_083_p2": "box004_083_p2",
        "e091_box026_20231018_039_p2": "box026_039_p2",
        "e091_box026_20231020_135_p2": "box026_135_p2",
    }[case["source_task"]]
    prefix = {
        "spider_dyn": "D",
        "rl_omni": "O",
    }[route]
    route_suffix = {
        "spider_dyn": "dyn",
        "rl_omni": "omni",
    }[route]
    derived_suffix = {
        "spider_dyn": "e092_dyn",
        "rl_omni": "e092_omni",
    }[route]
    variant = f"E092{prefix}{case['case_id'][1]}_{short}_{route_suffix}"
    return {
        "route": route,
        "case_id": case["case_id"],
        "variant": variant,
        "source_task": case["source_task"],
        "derived_task": f"{case['source_task']}_{derived_suffix}",
        "person_idx": case["person_idx"],
        "split": case["split"],
        "object": case["object"],
        "role": case["role"],
        "note": case["note"],
    }


def _validate_task(task_dir: Path) -> dict[str, object]:
    scene_act = task_dir / "scene_act.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_act))
    qpos = np.load(task_dir / "0/trajectory_kinematic.npz", allow_pickle=True)["qpos"]
    root = ET.parse(scene_act).getroot()
    pair_names = {pair.get("name") for pair in root.iter("pair")}
    required_leg = {f"{geom}_object" for geom in upperobj.LEG_FOOT_GEOMS}
    required_upper = {f"{geom}_object" for geom in upperobj.UPPER_BODY_GEOMS}
    missing_leg = sorted(required_leg - pair_names)
    missing_upper = sorted(required_upper - pair_names)
    mass = _object_mass(scene_act)
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
        raise RuntimeError(f"E092 task validation failed for {task_dir}: {report}")
    return report


def _patch_task_info(dst: Path, row: dict[str, str], validation: dict[str, object]) -> None:
    path = dst / "task_info.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    data.update(
        {
            "task": row["derived_task"],
            "source_task_e092": row["source_task"],
            "e092_route": row["route"],
            "e092_variant": row["variant"],
            "e092_case_id": row["case_id"],
            "e092_note": row["note"],
            "e092_validation": validation,
        }
    )
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _write_override(row: dict[str, str]) -> Path:
    path = OVERRIDE_ROOT / f"core4d_{row['variant']}.yaml"
    route_desc = {
        "spider_dyn": "Stage A SPIDER dynamic retargeting",
        "rl_omni": "Stage C direct OmniRetarget training control",
    }[row["route"]]
    text = f"""# @package _global_
# Auto-generated by workspace/core4d/scripts/E092/build_three_case_tasks.py.
# E092 {route_desc}; route={row['route']} case={row['case_id']}.
defaults:
  - core4d_E089A_box021_person1_upperobj
  - _self_

task: {row['derived_task']}

# Current repository training entry for E092 uses the established MJWP stack.
# Reference source is the task's own FK trajectory; external CORE4D masks are
# disabled because E091 Stage2b cases do not have aligned 3cm masks.
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
"""
    path.write_text(text, encoding="utf-8")
    return path


def _build_one(row: dict[str, str], *, force: bool) -> dict[str, object]:
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
            print(f"[SKIP] derived task exists: {_relative(dst)}")
    if not dst.exists():
        shutil.copytree(src, dst)

    added_leg, added_upper = upperobj.patch_scene_act(
        dst / "scene_act.xml", row["source_task"], row["derived_task"]
    )
    validation = _validate_task(dst)
    _patch_task_info(dst, row, validation)
    override = _write_override(row)
    meta = {
        **row,
        "task_dir": _relative(dst),
        "override": _relative(override),
        "added_leg_pairs": added_leg,
        "added_upper_pairs": added_upper,
        "validation": validation,
    }
    print(
        f"{row['route']} {row['source_task']} -> {row['derived_task']}: "
        f"npair={validation['model_npair']} mass={validation['object_mass_kg']} validation=ok"
    )
    return meta


def _write_variants(rows: list[dict[str, str]]) -> None:
    VARIANTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with VARIANTS_OUT.open("w", encoding="utf-8", newline="") as f:
        f.write("# E092 variants generated by build_three_case_tasks.py\n")
        f.write(
            "# route\tcase_id\tvariant\tsource_task\tderived_task\tperson_idx\t"
            "split\tobject\trole\tnote\n"
        )
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for row in rows:
            writer.writerow(
                [
                    row["route"],
                    row["case_id"],
                    row["variant"],
                    row["source_task"],
                    row["derived_task"],
                    row["person_idx"],
                    row["split"],
                    row["object"],
                    row["role"],
                    row["note"],
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--routes",
        default="spider_dyn,rl_omni",
        help="comma-separated routes to build: spider_dyn,rl_omni",
    )
    args = parser.parse_args()

    routes = [route.strip() for route in args.routes.split(",") if route.strip()]
    rows: list[dict[str, str]] = []
    for case in CASES:
        for route in routes:
            rows.append(_route_variant(case, route))

    built = [_build_one(row, force=args.force) for row in rows]
    _write_variants(rows)
    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    META_OUT.write_text(json.dumps(built, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {_relative(VARIANTS_OUT)}")
    print(f"wrote {_relative(META_OUT)}")


if __name__ == "__main__":
    main()
