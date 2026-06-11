#!/usr/bin/env python3
"""Generate E087 override YAMLs."""

from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
OVERRIDE_DIR = REPO / "examples/config/override"


OVERRIDES = {
    "core4d_E087A_m5_rawtarget_main.yaml": """# @package _global_
# E087A: E085 raw target with Box021 mass scaled to 5kg.
defaults:
  - core4d_E085A_rawtarget_main
  - _self_

task: d003_box021_20231018_029_p2_upperobj_e083_m5_e087
""",
    "core4d_E087B_m10_rawtarget_main.yaml": """# @package _global_
# E087B: E085 raw target with Box021 mass scaled to 10kg.
defaults:
  - core4d_E085A_rawtarget_main
  - _self_

task: d003_box021_20231018_029_p2_upperobj_e083_m10_e087
""",
    "core4d_E087C_m5_safe_main.yaml": """# @package _global_
# E087C: 5kg Box021 plus safety-dominant reward tuning.
defaults:
  - core4d_E087A_m5_rawtarget_main
  - _self_

task: d003_box021_20231018_029_p2_upperobj_e083_m5_e087
contact_hdmi_gain: 2.0
task_obj_use_exp: true
task_obj_pos_rew_scale: 0.25
task_obj_rot_rew_scale: 0.15
robot_object_penalty_scale: 12.0
robot_object_penalty_margin_m: 0.04
robot_object_penalty_deep_threshold_m: 0.0
hand_object_deep_penalty_scale: 10.0
hand_object_deep_penalty_threshold_m: 0.01
hand_object_deep_penalty_geom_names: ["lh", "rh"]
hand_floor_penalty_scale: 4.0
hand_floor_penalty_margin_m: 0.03
stability_penalty_scale: 2.0
stability_penalty_threshold: 0.65
object_lift_rew_scale: 1.0
object_floor_penalty_scale: 4.0
object_floor_margin_m: 0.02
""",
}


def main() -> None:
    OVERRIDE_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in OVERRIDES.items():
        path = OVERRIDE_DIR / name
        path.write_text(text, encoding="utf-8")
        print(f"Wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()

