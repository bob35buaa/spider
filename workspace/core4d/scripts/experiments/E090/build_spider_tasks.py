#!/usr/bin/env python3
"""Build E090 SPIDER smoke tasks from topface-preIK retarget outputs.

The retargeted E090 tasks are geometrically repaired, but their scenes are still
plain retarget scenes. SPIDER smoke should use the same safety setup as the
E083/E088 line: leg/upper-body object collision pairs and Box021 mass fixed to
10kg. This script creates deterministic derived tasks and matching Hydra
overrides for the two Phase-3 smoke candidates.
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
VARIANTS_OUT = REPO / "workspace/core4d/scripts/E090/variants_smoke.tsv"
SMOKE_META_OUT = REPO / "workspace/core4d/results/E090/smoke_task_meta.json"
E083_DIR = REPO / "workspace/core4d/scripts/E083"

if str(E083_DIR) not in sys.path:
    sys.path.insert(0, str(E083_DIR))

import create_upperobj_cases as upperobj  # type: ignore  # noqa: E402


SMOKE_CASES = [
    {
        "variant": "E090S1_box021_20231011_035_p2_btop",
        "source_task": "d003_box021_20231011_035_p2_btop_preik_e090",
        "derived_task": "d003_box021_20231011_035_p2_btop_preik_e090_upperobj_m10_smoke",
        "person_idx": "1",
        "split": "local",
        "role": "main",
        "note": "topface-preIK canonical gate pass; T=135",
    },
    {
        "variant": "E090S2_box021_20231020_019_p1_btop",
        "source_task": "d003_box021_20231020_019_p1_btop_preik_e090",
        "derived_task": "d003_box021_20231020_019_p1_btop_preik_e090_upperobj_m10_smoke",
        "person_idx": "0",
        "split": "local",
        "role": "main",
        "note": "topface-preIK canonical gate pass; T=101",
    },
]


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _patch_object_mass(scene: Path, new_mass: float) -> dict[str, object]:
    tree = ET.parse(scene)
    root = tree.getroot()
    obj = next((body for body in root.iter("body") if body.get("name") == "object"), None)
    if obj is None:
        raise ValueError(f"object body not found in {scene}")
    inertial = obj.find("inertial")
    if inertial is None:
        raise ValueError(f"object inertial not found in {scene}")
    old_mass = float(inertial.get("mass", "0"))
    if old_mass <= 0:
        raise ValueError(f"invalid object mass {old_mass} in {scene}")
    old_inertia = [float(x) for x in inertial.get("diaginertia", "").split()]
    if len(old_inertia) != 3:
        raise ValueError(f"invalid object diaginertia in {scene}")
    scale = new_mass / old_mass
    new_inertia = [x * scale for x in old_inertia]
    inertial.set("mass", f"{new_mass:.6g}")
    inertial.set("diaginertia", " ".join(f"{x:.6g}" for x in new_inertia))
    tree.write(scene, encoding="utf-8", xml_declaration=False)
    return {
        "scene": _relative(scene),
        "old_mass_kg": old_mass,
        "new_mass_kg": new_mass,
        "old_diaginertia": old_inertia,
        "new_diaginertia": new_inertia,
        "scale": scale,
    }


def _patch_task_info(dst: Path, row: dict[str, str], mass_meta: list[dict[str, object]]) -> None:
    info_path = dst / "task_info.json"
    data = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    data.update(
        {
            "task": row["derived_task"],
            "source_task_e090_smoke": row["source_task"],
            "e090_variant": row["variant"],
            "e090_note": row["note"],
            "e090_smoke_mass_kg": 10.0,
            "e090_mass_patch": mass_meta,
        }
    )
    info_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _validate_task(dst: Path) -> dict[str, object]:
    scene_act = dst / "scene_act.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_act))
    qpos = np.load(dst / "0/trajectory_kinematic.npz", allow_pickle=True)["qpos"]

    tree = ET.parse(scene_act)
    root = tree.getroot()
    pair_names = {pair.get("name") for pair in root.iter("pair")}
    required_leg = {f"{geom}_object" for geom in upperobj.LEG_FOOT_GEOMS}
    required_upper = {f"{geom}_object" for geom in upperobj.UPPER_BODY_GEOMS}
    missing_leg = sorted(required_leg - pair_names)
    missing_upper = sorted(required_upper - pair_names)

    obj = next((body for body in root.iter("body") if body.get("name") == "object"), None)
    inertial = obj.find("inertial") if obj is not None else None
    mass = float(inertial.get("mass", "nan")) if inertial is not None else float("nan")
    ok = (
        model.nq == 42
        and model.nv == 41
        and model.nu == 35
        and model.npair == 49
        and qpos.ndim == 2
        and qpos.shape[1] == 43
        and not missing_leg
        and not missing_upper
        and abs(mass - 10.0) < 1e-6
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
        raise RuntimeError(f"E090 smoke task validation failed for {dst}: {report}")
    return report


def _write_override(row: dict[str, str]) -> Path:
    path = OVERRIDE_ROOT / f"core4d_{row['variant']}.yaml"
    text = f"""# @package _global_
# Auto-generated by workspace/core4d/scripts/E090/build_spider_tasks.py.
# E090 smoke: topface-preIK Box021 retarget under E088/E089 safety reward stack.
defaults:
  - core4d_E089A_box021_person1_upperobj
  - _self_

task: {row['derived_task']}

# No raw external target/mask exists for the repaired E090 task. Keep the E089B
# isolation choice: ref_fk target, no old CORE4D contact mask.
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
    mass_meta = [
        _patch_object_mass(dst / scene_name, 10.0)
        for scene_name in ("scene.xml", "scene_act.xml")
    ]
    _patch_task_info(dst, row, mass_meta)
    override = _write_override(row)

    meta = {
        **row,
        "task_dir": _relative(dst),
        "override": _relative(override),
        "added_leg_pairs": added_leg,
        "added_upper_pairs": added_upper,
        "mass_patch": mass_meta,
        "validation": _validate_task(dst),
    }
    print(
        f"{row['source_task']} -> {row['derived_task']}: "
        f"leg_pairs_added={len(added_leg)} upper_pairs_added={len(added_upper)} "
        "mass=10kg validation=ok"
    )
    return meta


def _write_variants(rows: list[dict[str, str]]) -> None:
    VARIANTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with VARIANTS_OUT.open("w", encoding="utf-8", newline="") as f:
        f.write("# E090 SPIDER smoke variants generated by build_spider_tasks.py\n")
        f.write(
            "# variant\tsource_task\tderived_task\tmask_source_dir\tmask_slug\t"
            "person_idx\tsplit\trole\n"
        )
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for row in rows:
            writer.writerow(
                [
                    row["variant"],
                    row["source_task"],
                    row["derived_task"],
                    "-",
                    "-",
                    row["person_idx"],
                    row["split"],
                    row["role"],
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke"], default="smoke")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = [dict(row) for row in SMOKE_CASES]
    built = [_build_one(row, force=args.force) for row in rows]
    _write_variants(rows)
    SMOKE_META_OUT.parent.mkdir(parents=True, exist_ok=True)
    SMOKE_META_OUT.write_text(json.dumps(built, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {_relative(VARIANTS_OUT)}")
    print(f"wrote {_relative(SMOKE_META_OUT)}")


if __name__ == "__main__":
    main()
