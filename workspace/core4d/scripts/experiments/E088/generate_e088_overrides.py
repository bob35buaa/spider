#!/usr/bin/env python3
"""Generate E088 override YAMLs."""

from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
OVERRIDE_DIR = REPO / "examples/config/override"


GATE_COMMON = """cem_safety_gate_enabled: true
cem_safety_gate_mode: elite_filter
cem_safety_gate_min_sdf_m: -0.005
cem_safety_gate_max_violation_pct: 0.0
cem_safety_gate_min_valid_frac: 0.02
cem_safety_gate_fallback: least_violation
"""


OVERRIDES = {
    "core4d_E088A_m10_gate_main.yaml": f"""# @package _global_
# E088A: 10kg Box021 raw-target baseline plus hard CEM upper-body safety gate.
defaults:
  - core4d_E087B_m10_rawtarget_main
  - _self_

task: d003_box021_20231018_029_p2_upperobj_e083_m10_e087
{GATE_COMMON}""",
    "core4d_E088B_m10_gate_low_main.yaml": f"""# @package _global_
# E088B: hard gate plus low contact/object reward weights.
defaults:
  - core4d_E088A_m10_gate_main
  - _self_

task: d003_box021_20231018_029_p2_upperobj_e083_m10_e087
contact_hdmi_gain: 1.0
task_obj_use_exp: true
task_obj_pos_rew_scale: 0.1
task_obj_rot_rew_scale: 0.1
""",
    "core4d_E088C_m10_gate_clearance_main.yaml": f"""# @package _global_
# E088C: hard gate, low contact/object rewards, and absolute object clearance shaping.
defaults:
  - core4d_E088B_m10_gate_low_main
  - _self_

task: d003_box021_20231018_029_p2_upperobj_e083_m10_e087
object_clearance_rew_scale: 1.0
object_clearance_penalty_scale: 10.0
object_clearance_floor_z: 0.0
object_clearance_min_m: 0.04
object_clearance_max_m: 0.18
object_clearance_sigma: 0.04
object_clearance_above_weight: 0.25
object_clearance_gate_source: contact_mask
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
