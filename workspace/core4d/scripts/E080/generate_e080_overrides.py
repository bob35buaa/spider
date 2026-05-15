#!/usr/bin/env python3
"""Generate E080 box025 boundary-control Hydra overrides."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
CONVERT_DIR = REPO / "workspace/core4d/scripts/convert"
if str(CONVERT_DIR) not in sys.path:
    sys.path.insert(0, str(CONVERT_DIR))

from compute_palm_normal import compute_palm_normal_for_case  # noqa: E402


BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
VARIANTS = REPO / "workspace/core4d/scripts/E080/variants.tsv"
OUT_DIR = REPO / "examples/config/override"


def read_variants(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    fieldnames = ["variant", "task", "mask_slug", "person_idx", "split", "role"]
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(
            (line for line in f if line.strip() and not line.startswith("#")),
            delimiter="\t",
            fieldnames=fieldnames,
        )
        for row in reader:
            rows.append(row)
    return rows


def ref_npz_for_task(task: str) -> Path:
    case_dir = BASE / task / "0"
    for name in ("trajectory_kinematic_dual.npz", "trajectory_kinematic.npz"):
        path = case_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No trajectory npz found under {case_dir}")


def generate_override(row: dict[str, str], result_root: str) -> Path:
    variant = row["variant"]
    task = row["task"]
    mask_slug = row["mask_slug"]
    person_idx = int(row["person_idx"])

    scene_xml = BASE / task / "scene.xml"
    ref_npz = ref_npz_for_task(task)
    palm = compute_palm_normal_for_case(task, str(scene_xml), str(ref_npz))

    path = OUT_DIR / f"core4d_{variant}.yaml"
    content = f"""# @package _global_
# Auto-generated for E080 box025 boundary-control validation.
defaults:
  - core4d_e074a_box023
  - _self_

task: {task}
contact_hdmi_mask_source: core4d_3cm
contact_hdmi_mask_path: {result_root}/contact_masks/{mask_slug}/raw_contact_mask_3cm.npz
contact_hdmi_mask_person_idx: {person_idx}
contact_hdmi_mask_time_axis: auto
contact_hdmi_palm_normal_left: {palm['left']}
contact_hdmi_palm_normal_right: {palm['right']}
hold_contact_rew_scale: 0.0
# E080 keeps the E079 main validation rule: no box023-specific hold window.
hold_contact_start_eval_time: 0.0
hold_contact_end_eval_time: 0.0
"""
    path.write_text(content, encoding="utf-8")
    print(f"Wrote {path.relative_to(REPO)}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=Path, default=VARIANTS)
    parser.add_argument(
        "--result-root",
        default="workspace/core4d/results/E080",
        help="Project-relative result root containing contact_masks.",
    )
    args = parser.parse_args()

    for row in read_variants(args.variants):
        if row["role"] not in {"main", "guard"}:
            print(f"[SKIP] {row['variant']} role={row['role']}")
            continue
        generate_override(row, args.result_root)


if __name__ == "__main__":
    main()
