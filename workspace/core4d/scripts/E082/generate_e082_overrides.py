#!/usr/bin/env python3
"""Generate E082 Hydra overrides for D003 Box021 leg-object variants."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
CONVERT_DIR = REPO / "workspace/core4d/scripts/convert"
if str(CONVERT_DIR) not in sys.path:
    sys.path.insert(0, str(CONVERT_DIR))

from compute_palm_normal import compute_palm_normal_for_case  # noqa: E402


BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
VARIANTS = REPO / "workspace/core4d/scripts/E082/variants.tsv"
OUT_DIR = REPO / "examples/config/override"


def read_variants(path: Path) -> list[dict[str, str]]:
    fieldnames = [
        "variant",
        "source_task",
        "derived_task",
        "mask_source_dir",
        "mask_slug",
        "person_idx",
        "split",
        "role",
    ]
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(
            (line for line in f if line.strip() and not line.startswith("#")),
            delimiter="\t",
            fieldnames=fieldnames,
        )
        rows.extend(reader)
    return rows


def ref_npz_for_task(task: str) -> Path:
    path = BASE / task / "0" / "trajectory_kinematic.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def copy_mask(row: dict[str, str], result_root: Path) -> Path:
    mask_source_dir = Path(row["mask_source_dir"])
    if not mask_source_dir.is_absolute():
        mask_source_dir = REPO / mask_source_dir
    src = mask_source_dir / row["mask_slug"]
    if not src.is_dir():
        raise FileNotFoundError(src)
    dst = result_root / "contact_masks" / row["mask_slug"]
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["raw_contact_mask_3cm.npz", "raw_contact_mask_3cm.csv", "audit_summary_3cm.json"]:
        src_file = src / name
        if src_file.is_file():
            shutil.copy2(src_file, dst / name)
    return dst / "raw_contact_mask_3cm.npz"


def generate_override(row: dict[str, str], result_root: str) -> Path:
    variant = row["variant"]
    task = row["derived_task"]
    person_idx = int(row["person_idx"])
    result_root_path = REPO / result_root
    mask_path = copy_mask(row, result_root_path)

    scene_xml = BASE / task / "scene.xml"
    ref_npz = ref_npz_for_task(task)
    palm = compute_palm_normal_for_case(task, str(scene_xml), str(ref_npz))

    path = OUT_DIR / f"core4d_{variant}.yaml"
    content = f"""# @package _global_
# Auto-generated for E082 D003 Box021 E081-style leg-object validation.
defaults:
  - core4d_e074a_box023
  - _self_

task: {task}
contact_hdmi_mask_source: core4d_3cm
contact_hdmi_mask_path: {mask_path.relative_to(REPO)}
contact_hdmi_mask_person_idx: {person_idx}
contact_hdmi_mask_time_axis: auto
contact_hdmi_palm_normal_left: {palm['left']}
contact_hdmi_palm_normal_right: {palm['right']}
hold_contact_rew_scale: 0.0
# E082 keeps the E079/E081 main validation rule: no box023-specific hold window.
hold_contact_start_eval_time: 0.0
hold_contact_end_eval_time: 0.0
"""
    path.write_text(content, encoding="utf-8")
    print(f"Wrote {path.relative_to(REPO)}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=Path, default=VARIANTS)
    parser.add_argument("--result-root", default="workspace/core4d/results/E082")
    args = parser.parse_args()

    for row in read_variants(args.variants):
        if row["role"] not in {"main", "guard"}:
            print(f"[SKIP] {row['variant']} role={row['role']}")
            continue
        generate_override(row, args.result_root)


if __name__ == "__main__":
    main()
