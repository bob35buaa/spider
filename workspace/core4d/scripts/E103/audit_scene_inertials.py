#!/usr/bin/env python3
"""Audit CORE4D scene inertials for E103.

This script is intentionally read-only. It checks whether robot link inertials
look like a clean G1 scene or whether object inertial values were accidentally
copied into robot links.
"""

from __future__ import annotations

import argparse
import csv
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


DEFAULT_ROOT = Path("example_datasets/processed/core4d/unitree_g1/humanoid_object")
OBJECT_BODY = "object"
POLLUTED_MASS = 29.632


def fmt_float(value: float | None, digits: int = 6) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def fnum(value: str | None) -> float:
    try:
        return float(value or "nan")
    except ValueError:
        return float("nan")


def read_scene(scene: Path) -> dict[str, str]:
    task = scene.parent.name
    row: dict[str, str] = {
        "task": task,
        "scene_xml": str(scene),
        "mujoco_load_ok": "False",
        "mujoco_error": "",
        "nq": "",
        "nv": "",
        "nu": "",
        "status": "unknown",
    }

    try:
        model = mujoco.MjModel.from_xml_path(str(scene))
        row.update(
            {
                "mujoco_load_ok": "True",
                "nq": str(int(model.nq)),
                "nv": str(int(model.nv)),
                "nu": str(int(model.nu)),
            }
        )
    except Exception as exc:  # noqa: BLE001 - audit must report all XML failures.
        row["mujoco_error"] = f"{type(exc).__name__}: {exc}"

    try:
        root = ET.parse(scene).getroot()
    except Exception as exc:  # noqa: BLE001
        row["status"] = "xml_parse_error"
        row["xml_error"] = f"{type(exc).__name__}: {exc}"
        return row

    robot_inertials: list[tuple[str, str, str]] = []
    object_inertial: tuple[str, str, str] | None = None
    samples: dict[str, tuple[str, str, str]] = {}

    for body in root.iter("body"):
        body_name = body.get("name", "")
        inertial = None
        for child in body:
            if child.tag == "inertial":
                inertial = child
                break
        if inertial is None:
            continue
        item = (body_name, inertial.get("mass", ""), inertial.get("diaginertia", ""))
        if body_name == OBJECT_BODY:
            object_inertial = item
        else:
            robot_inertials.append(item)
            if body_name in {"pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "torso_link"}:
                samples[body_name] = item

    unique_robot_pairs = sorted({(mass, inertia) for _, mass, inertia in robot_inertials})
    robot_masses = [fnum(mass) for _, mass, _ in robot_inertials]
    object_mass = fnum(object_inertial[1]) if object_inertial else float("nan")

    robot_all_same = bool(robot_inertials) and len(unique_robot_pairs) == 1
    robot_polluted_mass = any(abs(mass - POLLUTED_MASS) < 1e-3 for mass in robot_masses if not math.isnan(mass))
    pelvis_mass = fnum(samples.get("pelvis", ("", "", ""))[1])
    hip_mass = fnum(samples.get("left_hip_pitch_link", ("", "", ""))[1])

    status_parts: list[str] = []
    if row["mujoco_load_ok"] != "True":
        status_parts.append("mujoco_load_error")
    if not robot_inertials:
        status_parts.append("missing_robot_inertials")
    if object_inertial is None:
        status_parts.append("missing_object_inertial")
    if robot_all_same and robot_polluted_mass:
        status_parts.append("polluted_robot_inertial")
    elif len(unique_robot_pairs) <= 5:
        status_parts.append("robot_inertial_low_diversity")
    if not math.isnan(pelvis_mass) and abs(pelvis_mass - POLLUTED_MASS) < 1e-3:
        if "polluted_robot_inertial" not in status_parts:
            status_parts.append("polluted_robot_inertial")
    if not math.isnan(object_mass) and object_mass > 20:
        status_parts.append("object_mass_policy_review")
    if row["mujoco_load_ok"] == "True" and not status_parts:
        status_parts.append("clean")

    row.update(
        {
            "robot_inertial_count": str(len(robot_inertials)),
            "robot_inertial_unique_pairs": str(len(unique_robot_pairs)),
            "robot_inertial_all_same": str(robot_all_same),
            "robot_polluted_mass_29_632": str(robot_polluted_mass),
            "pelvis_mass": fmt_float(pelvis_mass),
            "left_hip_pitch_mass": fmt_float(hip_mass),
            "torso_mass": fmt_float(fnum(samples.get("torso_link", ("", "", ""))[1])),
            "object_mass": fmt_float(object_mass),
            "object_diaginertia": object_inertial[2] if object_inertial else "",
            "robot_inertial_sample": "; ".join(
                f"{name}:{mass}:{inertia}" for name, mass, inertia in robot_inertials[:5]
            ),
            "status": ";".join(status_parts),
        }
    )
    return row


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    scenes = sorted(args.root.glob("*/scene.xml"))
    rows = [read_scene(scene) for scene in scenes]
    fields = [
        "task",
        "scene_xml",
        "status",
        "mujoco_load_ok",
        "mujoco_error",
        "nq",
        "nv",
        "nu",
        "robot_inertial_count",
        "robot_inertial_unique_pairs",
        "robot_inertial_all_same",
        "robot_polluted_mass_29_632",
        "pelvis_mass",
        "left_hip_pitch_mass",
        "torso_mass",
        "object_mass",
        "object_diaginertia",
        "robot_inertial_sample",
    ]
    write_tsv(args.out, rows, fields)

    polluted = sum("polluted_robot_inertial" in row["status"] for row in rows)
    clean = sum(row["status"] == "clean" for row in rows)
    print(f"audited scenes={len(rows)} clean={clean} polluted_robot_inertial={polluted}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
