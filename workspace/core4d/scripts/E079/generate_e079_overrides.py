#!/usr/bin/env python3
"""Generate E079 per-case Hydra overrides."""

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
VARIANTS = REPO / "workspace/core4d/scripts/E079/variants.tsv"
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
        p = case_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"No trajectory npz found under {case_dir}")


def generate_override(row: dict[str, str], result_root: str) -> Path:
    variant = row["variant"]
    task = row["task"]
    mask_slug = row["mask_slug"]
    person_idx = int(row["person_idx"])
    role = row["role"]

    scene_xml = BASE / task / "scene.xml"
    ref_npz = ref_npz_for_task(task)
    palm = compute_palm_normal_for_case(task, str(scene_xml), str(ref_npz))

    hold_scale = 1.0 if role == "calib" else 0.0
    defaults = "core4d_e075b_box023" if role == "calib" else "core4d_e074a_box023"
    path = OUT_DIR / f"core4d_{variant}.yaml"
    content = f"""# @package _global_
# Auto-generated for E079 CORE4D generalization.
defaults:
  - {defaults}
  - _self_

task: {task}
contact_hdmi_mask_source: core4d_3cm
contact_hdmi_mask_path: {result_root}/contact_masks/{mask_slug}/raw_contact_mask_3cm.npz
contact_hdmi_mask_person_idx: {person_idx}
contact_hdmi_mask_time_axis: auto
contact_hdmi_palm_normal_left: {palm['left']}
contact_hdmi_palm_normal_right: {palm['right']}
hold_contact_rew_scale: {hold_scale}
"""
    if role != "calib":
        content += """# Main E079 validation disables the box023-specific hold window.
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
        default="workspace/core4d/results/E079",
        help="Project-relative result root containing contact_masks.",
    )
    parser.add_argument(
        "--include-calib",
        action="store_true",
        help="Also generate calibration variants.",
    )
    args = parser.parse_args()

    rows = read_variants(args.variants)
    for row in rows:
        if row["role"] == "calib" and not args.include_calib:
            continue
        if row["role"] not in {"main", "guard", "calib"}:
            print(f"[SKIP] {row['variant']} role={row['role']}")
            continue
        generate_override(row, args.result_root)


if __name__ == "__main__":
    main()
