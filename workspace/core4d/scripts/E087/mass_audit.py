#!/usr/bin/env python3
"""E087 mass/inertia audit for CORE4D object scenes."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
OUT = REPO / "workspace/core4d/results/E087/mass_audit"


FOCUS_TASKS = [
    "d003_box021_20231018_029_p2_upperobj_e083",
    "d003_box021_20231011_035_p2_upperobj_e083",
    "d003_box021_20231020_019_p1_upperobj_e083",
    "box023_person2_upperobj_e083",
    "box025_person2_legobj",
    "box025_person2",
]


def _safe_model(path: Path) -> mujoco.MjModel | None:
    try:
        return mujoco.MjModel.from_xml_path(str(path))
    except Exception:
        return None


def _row_for_scene(scene: Path) -> dict[str, object] | None:
    model = _safe_model(scene)
    if model is None:
        return None
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")
    if body_id < 0 or geom_id < 0:
        return None
    half = model.geom_size[geom_id, :3].astype(float)
    volume = float(np.prod(half * 2.0))
    mass = float(model.body_mass[body_id])
    density = mass / volume if volume > 0 else float("nan")
    return {
        "task": scene.parent.name,
        "scene": str(scene.relative_to(REPO)),
        "mass_kg": mass,
        "inertia_x": float(model.body_inertia[body_id, 0]),
        "inertia_y": float(model.body_inertia[body_id, 1]),
        "inertia_z": float(model.body_inertia[body_id, 2]),
        "half_x_m": float(half[0]),
        "half_y_m": float(half[1]),
        "half_z_m": float(half[2]),
        "collision_volume_m3": volume,
        "density_kg_m3": density,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for scene in sorted(TASK_ROOT.glob("*/scene_act.xml")):
        row = _row_for_scene(scene)
        if row is not None:
            rows.append(row)

    fieldnames = [
        "task",
        "scene",
        "mass_kg",
        "inertia_x",
        "inertia_y",
        "inertia_z",
        "half_x_m",
        "half_y_m",
        "half_z_m",
        "collision_volume_m3",
        "density_kg_m3",
    ]
    csv_path = OUT / "object_mass_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    focus = [r for r in rows if r["task"] in FOCUS_TASKS or "box021" in str(r["task"])]
    by_mass: dict[str, int] = {}
    for row in rows:
        key = f"{float(row['mass_kg']):.3f}"
        by_mass[key] = by_mass.get(key, 0) + 1
    summary = {
        "num_scenes": len(rows),
        "mass_histogram": by_mass,
        "focus": focus,
        "box021_mass_kg": sorted({float(r["mass_kg"]) for r in rows if "box021" in str(r["task"])}),
        "box023_mass_kg": sorted({float(r["mass_kg"]) for r in rows if "box023" in str(r["task"])}),
        "box025_mass_kg": sorted({float(r["mass_kg"]) for r in rows if "box025" in str(r["task"])}),
    }
    summary_path = OUT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

