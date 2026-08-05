#!/usr/bin/env python3
"""Derive the 15 E188 scenes by scaling only object mass and diagonal inertia."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from common import (
    E187_EVAL,
    REPO,
    S0,
    e187_eval_rows,
    queue_case_set,
    relative,
    repo_path,
    sha256,
    verify_upstream,
    write_tsv,
)

SCENE_MANIFEST = S0 / "scene_manifest.tsv"
MASS_AUDIT = S0 / "mass_audit.tsv"
INVARIANT_DIFF = S0 / "invariant_diff.tsv"
TARGET_NAME = "scene_act_E188_mass5kg.xml"
NEW_MASS = 5.0


def object_inertial(root: ET.Element) -> ET.Element:
    bodies = root.findall(".//body[@name='object']")
    if len(bodies) != 1:
        raise RuntimeError(f"expected one object body, found {len(bodies)}")
    inertials = bodies[0].findall("inertial")
    if len(inertials) != 1:
        raise RuntimeError(f"expected one object inertial, found {len(inertials)}")
    return inertials[0]


def floats(raw: str) -> tuple[float, ...]:
    return tuple(float(value) for value in raw.split())


def canonical(element: ET.Element) -> tuple[Any, ...]:
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        (element.text or "").strip(),
        tuple(canonical(child) for child in element),
    )


def build_tree(source: Path) -> tuple[ET.ElementTree, dict[str, Any]]:
    tree = ET.parse(source)
    source_root = tree.getroot()
    inertial = object_inertial(source_root)
    old_mass = float(inertial.attrib["mass"])
    old_inertia = floats(inertial.attrib["diaginertia"])
    if abs(old_mass - 2.0) > 1e-9 or len(old_inertia) != 3:
        raise RuntimeError(f"source is not an eligible 2kg scene: {source}")
    scale = NEW_MASS / old_mass
    new_inertia = tuple(value * scale for value in old_inertia)
    inertial.set("mass", f"{NEW_MASS:.9g}")
    inertial.set("diaginertia", " ".join(f"{value:.12g}" for value in new_inertia))

    reverted = copy.deepcopy(source_root)
    reverted_inertial = object_inertial(reverted)
    reverted_inertial.set("mass", f"{old_mass:.3f}")
    reverted_inertial.set("diaginertia", " ".join(f"{value:.8f}" for value in old_inertia))
    if canonical(reverted) != canonical(ET.parse(source).getroot()):
        raise RuntimeError(f"semantic XML diff exceeds inertial whitelist: {source}")
    return tree, {
        "old_mass_kg": old_mass,
        "new_mass_kg": NEW_MASS,
        "inertia_scale": scale,
        "old_diaginertia": " ".join(f"{value:.8f}" for value in old_inertia),
        "new_diaginertia": " ".join(f"{value:.12g}" for value in new_inertia),
        "changed_xpath_count": 2,
        "allowed_xpaths": (
            "/mujoco/worldbody/body[@name='object']/inertial/@mass;"
            "/mujoco/worldbody/body[@name='object']/inertial/@diaginertia"
        ),
    }


def encoded_tree(tree: ET.ElementTree) -> bytes:
    ET.indent(tree, space="  ")
    return ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True) + b"\n"


def compile_scene(path: Path) -> tuple[int, int]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(path))
    return int(model.nq), int(model.nbody)


def build(*, freeze: bool) -> dict[str, Any]:
    verify_upstream()
    expected = queue_case_set()
    selected: list[dict[str, Any]] = []
    skipped: list[str] = []
    for row in e187_eval_rows():
        source = repo_path(row["scene_act"]).resolve()
        mass = float(object_inertial(ET.parse(source).getroot()).attrib["mass"])
        if abs(mass - 2.0) > 1e-9:
            skipped.append(row["case_id"])
            continue
        if row["case_id"] not in expected:
            raise RuntimeError(f"2kg row missing from plan queue: {row['case_id']}")
        target = source.parent / TARGET_NAME
        tree, audit = build_tree(source)
        payload = encoded_tree(tree)
        if freeze:
            if target.is_file() and target.read_bytes() != payload:
                raise RuntimeError(f"refusing to replace non-identical E188 scene: {target}")
            if not target.exists():
                temporary = target.with_suffix(".xml.tmp")
                temporary.write_bytes(payload)
                os.replace(temporary, target)
            compile_path = target
        else:
            descriptor, raw = tempfile.mkstemp(prefix=".e188_preflight_", suffix=".xml", dir=source.parent)
            os.close(descriptor)
            compile_path = Path(raw)
            compile_path.write_bytes(payload)
        try:
            nq, nbody = compile_scene(compile_path)
        finally:
            if not freeze:
                compile_path.unlink(missing_ok=True)
        selected.append(
            {
                "ordinal": len(selected) + 1,
                "e187_ordinal": int(row["ordinal"]),
                "case_id": row["case_id"],
                "object_key": row["object_key"],
                "target_task": Path(row["trajectory"]).parents[1].name,
                "source_scene": relative(source),
                "source_scene_sha256": sha256(source),
                "scene_act": relative(target),
                "scene_sha256": sha256(target) if freeze else __import__("hashlib").sha256(payload).hexdigest(),
                "trajectory": row["trajectory"],
                "contact_mask": row["contact_mask"],
                "nq": nq,
                "nbody": nbody,
                **audit,
            }
        )
    if {row["case_id"] for row in selected} != expected or len(skipped) != 7:
        raise RuntimeError("E188 mass selection is not exact 15 treatment / 7 skip")
    if freeze:
        write_tsv(SCENE_MANIFEST, selected)
        write_tsv(
            MASS_AUDIT,
            [
                {key: row[key] for key in (
                    "ordinal", "case_id", "object_key", "old_mass_kg", "new_mass_kg",
                    "inertia_scale", "old_diaginertia", "new_diaginertia", "scene_act", "scene_sha256"
                )}
                for row in selected
            ],
        )
        write_tsv(
            INVARIANT_DIFF,
            [
                {
                    "ordinal": row["ordinal"],
                    "case_id": row["case_id"],
                    "status": "PASS",
                    "changed_xpath_count": row["changed_xpath_count"],
                    "allowed_xpaths": row["allowed_xpaths"],
                    "robot_inertial_changes": 0,
                    "source_scene_sha256": row["source_scene_sha256"],
                    "target_scene_sha256": row["scene_sha256"],
                }
                for row in selected
            ],
        )
    return {
        "status": "PASS",
        "mode": "freeze" if freeze else "preflight",
        "selected_rows": len(selected),
        "skipped_5_to_5_rows": len(skipped),
        "objects": {key: sum(row["object_key"] == key for row in selected) for key in ("bucket003", "bucket004", "bucket007")},
        "mujoco_load_pass": len(selected),
        "source_authority": relative(E187_EVAL),
        "written": freeze,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "freeze"))
    args = parser.parse_args()
    print(json.dumps(build(freeze=args.mode == "freeze"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
