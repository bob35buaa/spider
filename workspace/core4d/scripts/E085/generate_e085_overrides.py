#!/usr/bin/env python3
"""Generate E085 overrides that replace the ref-FK contact target with raw targets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
CASES = REPO / "workspace/core4d/scripts/E085/target_cases.tsv"
OUT_DIR = REPO / "examples/config/override"
RAW_TARGET_ROOT = REPO / "workspace/core4d/results/E085/raw_targets"

BASE_BY_VARIANT = {
    "E085A_rawtarget_main": "core4d_E084C_d003_box021_20231018_029_p2_semantic",
    "E085A_rawtarget_guard": "core4d_E084C_box023_p2_semantic_guard",
}


def read_cases(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [
            row
            for row in csv.DictReader(f, delimiter="\t")
            if row and not row["variant"].startswith("#")
        ]


def rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def generate(row: dict[str, str]) -> Path:
    variant = row["variant"]
    base = BASE_BY_VARIANT.get(variant)
    if not base:
        raise KeyError(f"No base override registered for {variant}")
    target_path = RAW_TARGET_ROOT / variant / "raw_contact_targets.npz"
    if not target_path.is_file():
        raise FileNotFoundError(target_path)

    out = OUT_DIR / f"core4d_{variant}.yaml"
    content = f"""# @package _global_
# Auto-generated for E085 raw-contact target repair.
defaults:
  - {base}
  - _self_

task: {row["task"]}
contact_hdmi_dynamic_target: true
contact_hdmi_target_source: external
contact_hdmi_target_path: {rel(target_path)}
contact_hdmi_target_time_axis: auto
contact_hdmi_target_uses_eef_offset: false
"""
    out.write_text(content, encoding="utf-8")
    print(f"Wrote {rel(out)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=CASES)
    args = parser.parse_args()
    for row in read_cases(args.cases):
        generate(row)


if __name__ == "__main__":
    main()
