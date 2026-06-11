#!/usr/bin/env python3
"""Build E107 selected-4 Box021 CEM manifest, overrides, and preflight."""

from __future__ import annotations

import argparse
import json
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
from e107_common import (  # noqa: E402
    CANDIDATE_FIELDS,
    CANDIDATES_TSV,
    GATE_SUMMARY_TSV,
    OVERRIDE_ROOT,
    PERSON_IDX,
    RESULTS_ROOT,
    SELECTED_JSON,
    TASK_ROOT,
    VARIANT_FIELDS,
    VARIANTS_TSV,
    rel,
    selected_id_to_clean_task,
    selected_id_to_source_task,
    split_for_ordinal,
    write_tsv,
)


PREFLIGHT_TSV = RESULTS_ROOT / "selected4_clean_task_preflight.tsv"
SUMMARY_MD = RESULTS_ROOT / "selected4_manifest_summary.md"
SUMMARY_JSON = RESULTS_ROOT / "selected4_manifest_summary.json"


def read_gate() -> dict[str, dict[str, str]]:
    rows = {}
    import csv

    with GATE_SUMMARY_TSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows[row["derived_task"]] = row
    return rows


def split_task_parts(source_task: str) -> tuple[str, str, str]:
    # d003_box021_20231011_034_p1
    parts = source_task.split("_")
    if len(parts) < 5 or parts[0] != "d003" or parts[1] != "box021":
        raise ValueError(f"unexpected source task: {source_task}")
    return parts[2], parts[3], parts[4]


def variant_name(ordinal: int, source_task: str) -> str:
    short = source_task.removeprefix("d003_")
    return f"E107C{ordinal:02d}_{short}_ref_fk_clean"


def selected_rows(path: Path) -> list[dict[str, str]]:
    selected = json.loads(path.read_text(encoding="utf-8"))
    gate = read_gate()
    out: list[dict[str, str]] = []
    for ordinal, selected_id in enumerate(selected, start=1):
        source_task = selected_id_to_source_task(selected_id)
        derived_task = selected_id_to_clean_task(selected_id)
        if derived_task not in gate:
            raise KeyError(f"selected task not in E107 gate summary: {derived_task}")
        grow = gate[derived_task]
        if grow.get("cem_gate") != "True":
            raise RuntimeError(f"selected task is not cem_ready: {derived_task} gate={grow.get('failure_mode')}")
        date, seq, pshort = split_task_parts(source_task)
        person = "person1" if pshort == "p1" else "person2"
        out.append(
            {
                "ordinal": str(ordinal),
                "selected_id": selected_id,
                "variant": variant_name(ordinal, source_task),
                "source_task": source_task,
                "derived_task": derived_task,
                "split": split_for_ordinal(ordinal),
                "date": date,
                "seq": seq,
                "person": person,
                "person_idx": PERSON_IDX[person],
                "object_key": "box021",
                "source_scene_task": grow.get("source_scene_task", f"box021_{person}"),
                "qpos_shape": grow.get("qpos_shape", ""),
                "target_both_active_frac_3cm": grow.get("target_both_active_frac_3cm", ""),
                "target_both_active_frac_5cm": grow.get("target_both_active_frac_5cm", ""),
                "raw_contact_proxy_path": grow.get("raw_contact_proxy_path", ""),
            }
        )
    return out


def object_mass(scene: Path) -> float:
    root = ET.parse(scene).getroot()
    obj = next((body for body in root.iter("body") if body.get("name") == "object"), None)
    inertial = obj.find("inertial") if obj is not None else None
    if inertial is None:
        raise ValueError(f"object inertial missing in {scene}")
    return float(inertial.get("mass", "nan"))


def validate_task(row: dict[str, str]) -> dict[str, Any]:
    task_dir = TASK_ROOT / row["derived_task"]
    scene = task_dir / "scene.xml"
    scene_act = task_dir / "scene_act.xml"
    model_scene = mujoco.MjModel.from_xml_path(str(scene))
    model_act = mujoco.MjModel.from_xml_path(str(scene_act))
    qpos = np.load(task_dir / "0/trajectory_kinematic.npz", allow_pickle=True)["qpos"]
    root = ET.parse(scene_act).getroot()
    pair_names = {pair.get("name") for pair in root.iter("pair")}
    missing_leg = sorted({f"{geom}_object" for geom in upperobj.LEG_FOOT_GEOMS} - pair_names)
    missing_upper = sorted({f"{geom}_object" for geom in upperobj.UPPER_BODY_GEOMS} - pair_names)
    audit = audit_scene(scene)
    audit_act = audit_scene(scene_act)
    ok = (
        model_scene.nq == 43
        and model_scene.nv == 41
        and model_scene.nu == 29
        and model_act.nq == 42
        and model_act.nv == 41
        and model_act.nu == 35
        and qpos.ndim == 2
        and qpos.shape[1] == 43
        and audit.get("robot_polluted_mass_29_632") == "False"
        and audit_act.get("robot_polluted_mass_29_632") == "False"
        and not missing_leg
        and not missing_upper
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
        raise RuntimeError(f"E107 selected task validation failed: {report}")
    return report


def write_override(row: dict[str, str]) -> Path:
    path = OVERRIDE_ROOT / f"core4d_{row['variant']}.yaml"
    content = f"""# @package _global_
# Auto-generated by workspace/core4d/scripts/E107/build_selected4_cem_manifest.py.
# E107 Box021 selected-4 clean ref-FK full CEM.
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


def write_summary(rows: list[dict[str, str]], validations: list[dict[str, Any]], overrides: list[Path]) -> None:
    data = {
        "selected_count": len(rows),
        "splits": {row["variant"]: row["split"] for row in rows},
        "variants_tsv": rel(VARIANTS_TSV),
        "candidates_tsv": rel(CANDIDATES_TSV),
        "preflight_tsv": rel(PREFLIGHT_TSV),
        "overrides": [rel(path) for path in overrides],
    }
    SUMMARY_JSON.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# E107 Box021 Selected-4 Manifest Summary",
        "",
        f"- selected variants: `{len(rows)}`",
        f"- candidates: `{rel(CANDIDATES_TSV)}`",
        f"- variants: `{rel(VARIANTS_TSV)}`",
        f"- preflight: `{rel(PREFLIGHT_TSV)}`",
        "",
        "| ordinal | variant | task | split | qpos | validation |",
        "|---:|---|---|---|---|---|",
    ]
    by_variant = {item["variant"]: item for item in validations}
    for row in rows:
        val = by_variant[row["variant"]]
        lines.append(
            f"| {row['ordinal']} | `{row['variant']}` | `{row['derived_task']}` | `{row['split']}` | "
            f"`{val['qpos_shape']}` | `{val['validation_ok']}` |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", type=Path, default=SELECTED_JSON)
    args = parser.parse_args()

    rows = selected_rows(args.selected)
    validations = [validate_task(row) for row in rows]
    overrides = [write_override(row) for row in rows]
    write_tsv(CANDIDATES_TSV, rows, CANDIDATE_FIELDS, "E107 Box021 selected-4 clean CEM candidates")
    write_tsv(VARIANTS_TSV, [{field: row[field] for field in VARIANT_FIELDS} for row in rows], VARIANT_FIELDS, "E107 runnable selected-4 variants")
    fields = sorted({key for item in validations for key in item})
    write_tsv(PREFLIGHT_TSV, validations, fields)
    write_summary(rows, validations, overrides)
    print(f"wrote {rel(CANDIDATES_TSV)} rows={len(rows)}")
    print(f"wrote {rel(VARIANTS_TSV)} rows={len(rows)}")
    print(f"wrote {rel(PREFLIGHT_TSV)}")
    print(f"wrote {rel(SUMMARY_MD)}")


if __name__ == "__main__":
    main()
