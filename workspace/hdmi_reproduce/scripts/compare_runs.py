#!/usr/bin/env python3
"""Compare two HDMI runs by analyzing trajectory differences.

Usage:
    python workspace/hdmi_reproduce/scripts/compare_runs.py \
        workspace/hdmi_reproduce/results/R008/trajectory_hdmi.npz \
        workspace/hdmi_reproduce/results/R011/trajectory_hdmi.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_run(npz_path: str) -> dict:
    """Load trajectory data."""
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def compare(path_a: str, path_b: str) -> None:
    """Compare two runs side-by-side."""
    a = load_run(path_a)
    b = load_run(path_b)
    name_a = Path(path_a).parent.name
    name_b = Path(path_b).parent.name

    print(f"Comparing: {name_a} vs {name_b}")
    print("=" * 60)

    # Overall metrics
    print(f"\n{'Metric':25s} {name_a:>12s} {name_b:>12s} {'Delta':>12s}")
    print("-" * 61)
    for key in ["rew_mean", "tracking_mean", "object_tracking_mean"]:
        va = a[key].mean()
        vb = b[key].mean()
        delta = vb - va
        pct = delta / va * 100 if va != 0 else 0
        print(f"{key:25s} {va:12.3f} {vb:12.3f} {delta:+12.3f} ({pct:+.1f}%)")

    print(f"{'opt_steps':25s} {a['opt_steps'].mean():12.1f} {b['opt_steps'].mean():12.1f}")

    # Per-quarter
    n = min(len(a["rew_mean"]), len(b["rew_mean"]))
    q = n // 4
    print(f"\nPer-quarter rew_mean:")
    for i in range(4):
        s, e = i * q, (i + 1) * q
        va = a["rew_mean"][s:e].mean()
        vb = b["rew_mean"][s:e].mean()
        print(f"  Q{i+1}: {va:8.2f} -> {vb:8.2f} ({vb-va:+.2f})")

    # Pelvis Z comparison
    qpos_a = a["qpos"][:, 0, 2]
    qpos_b = b["qpos"][:, 0, 2]
    print(f"\nPelvis Z height:")
    print(f"  {'Time':>6s} {name_a:>10s} {name_b:>10s} {'Delta':>10s}")
    for t in [0, 1, 2, 3, 3.5, 4, 4.5, 5]:
        tick_a = min(int(t / 0.04), len(qpos_a) - 1)
        tick_b = min(int(t / 0.04), len(qpos_b) - 1)
        za = qpos_a[tick_a]
        zb = qpos_b[tick_b]
        print(f"  {t:6.1f} {za:10.3f} {zb:10.3f} {zb-za:+10.3f}")

    # Ctrl tracking error
    if "ctrl" in a and "ctrl_ref" in a and "ctrl" in b and "ctrl_ref" in b:
        err_a = np.abs(a["ctrl"] - a["ctrl_ref"]).mean()
        err_b = np.abs(b["ctrl"] - b["ctrl_ref"]).mean()
        print(f"\nCtrl tracking MAE: {name_a}={err_a:.4f}, {name_b}={err_b:.4f}")

    # Suitcase position (contact guidance nq=42)
    nq_a = a["qpos"].shape[-1]
    nq_b = b["qpos"].shape[-1]
    if nq_a == 42 and nq_b == 42:
        print(f"\nSuitcase slide position:")
        print(f"  {'Time':>6s} {name_a + ' [x,y,z]':>30s} {name_b + ' [x,y,z]':>30s}")
        for t in [0, 2, 3, 4, 5]:
            tick_a = min(int(t / 0.04), a["qpos"].shape[0] - 1)
            tick_b = min(int(t / 0.04), b["qpos"].shape[0] - 1)
            sa = a["qpos"][tick_a, 0, 36:39]
            sb = b["qpos"][tick_b, 0, 36:39]
            print(
                f"  {t:6.1f} [{sa[0]:7.3f},{sa[1]:7.3f},{sa[2]:7.3f}]"
                f"  [{sb[0]:7.3f},{sb[1]:7.3f},{sb[2]:7.3f}]"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two HDMI trajectory runs")
    parser.add_argument("run_a", help="Path to first trajectory_hdmi.npz")
    parser.add_argument("run_b", help="Path to second trajectory_hdmi.npz")
    args = parser.parse_args()

    compare(args.run_a, args.run_b)


if __name__ == "__main__":
    main()
