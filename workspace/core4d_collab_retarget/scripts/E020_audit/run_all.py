#!/usr/bin/env python3
from audit_common import (
    build_attribution_panels,
    run_anchor_vs_raw,
    run_decide_root_cause,
    run_keyframes,
    run_mask_vs_raw,
    run_ref_physics,
    run_sim_ref_overlay,
)


if __name__ == "__main__":
    print(run_anchor_vs_raw())
    print(run_ref_physics())
    print(run_mask_vs_raw())
    print(run_sim_ref_overlay())
    print(run_decide_root_cause())
    print(run_keyframes())
    print(build_attribution_panels())
