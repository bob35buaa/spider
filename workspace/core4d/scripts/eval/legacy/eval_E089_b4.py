#!/usr/bin/env python3
"""Eval E089 B4 SPIDER smoke results for top-2 B-path repaired box021 cases."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

from pathlib import Path
import json
import sys
sys.path.insert(0, "/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider/workspace/core4d/scripts/eval")
from eval_E089 import compute_sim_metrics, BASELINES  # type: ignore

REPO = Path("/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider")

CASES = [
    ("E089B1_box021_20231018_031_p2_btop", "d003_box021_20231018_031_p2_btop_upperobj_e089b"),
    ("E089B2_box021_20231020_020_p1_btop", "d003_box021_20231020_020_p1_btop_upperobj_e089b"),
]

print("=" * 110)
print(f"{'variant':<40} {'T':>3}  {'cont':>6} {'obj_mean':>8} {'pelv_min':>8} {'head':>6} {'upper':>6} {'LH_fl':>6} {'RH_fl':>6}")
print("-" * 110)
results = {}
for v, td in CASES:
    npz = REPO / f"workspace/core4d/results/E089/B4/{v}_outdir_smoke/trajectory_mjwp_act.npz"
    scene = REPO / f"example_datasets/processed/core4d/unitree_g1/humanoid_object/{td}/scene_act.xml"
    m = compute_sim_metrics(npz, scene)
    results[v] = m
    print(f"{v:<40} {m['T']:>3}  {m['contact_frac_either']*100:>5.1f}% {m['obj_err_mean_m']:>7.3f}m {m['pelvis_min_m']:>7.3f}m {m['head_pen_frac']*100:>5.1f}% {m['upper_pen_frac']*100:>5.1f}% {m['handL_floor_lt_5cm_frac']*100:>5.1f}% {m['handR_floor_lt_5cm_frac']*100:>5.1f}%")

print("-" * 110)
print("Reference: E089A box021_person1 FULL (gate-pass case under same reward stack):")
print(f"{'E089A_box021_person1_FULL':<40} {88:>3}  {60.2:>5.1f}% {0.013:>7.3f}m {0.687:>7.3f}m {0.0:>5.1f}% {0.0:>5.1f}% {0.0:>5.1f}% {0.0:>5.1f}%")
print("Reference: E082-E088 box021_D003_18029_p2 (all FAILED on dynamic CEM):")
for n, b in BASELINES.items():
    print(f"{n:<40} {'-':>3}  {b['contact']*100:>5.1f}% {b['obj_mean_m']:>7.3f}m {b['pelvis_min_m']:>7.3f}m {b['head_pen']*100:>5.1f}% {b['upper_pen']*100:>5.1f}% {b['hand_floor_max']*100:>5.1f}% {'-':>6}")
print()

out = REPO / "workspace/core4d/results/E089/eval_summary.json"
existing = json.loads(out.read_text()) if out.exists() else {}
existing.update({k + "_SMOKE": v for k, v in results.items()})
out.write_text(json.dumps(existing, indent=2))
print(f"saved {out}")
