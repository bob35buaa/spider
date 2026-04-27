#!/usr/bin/env python3
"""Analyze HDMI retargeting results from trajectory_hdmi.npz.

Usage:
    python workspace/hdmi_reproduce/scripts/eval_metrics.py \
        workspace/hdmi_reproduce/results/R011/trajectory_hdmi.npz

    # Compare multiple runs:
    python workspace/hdmi_reproduce/scripts/eval_metrics.py \
        workspace/hdmi_reproduce/results/R008/trajectory_hdmi.npz \
        workspace/hdmi_reproduce/results/R011/trajectory_hdmi.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def analyze_run(npz_path: str) -> dict:
    """Load and analyze a single run's trajectory data."""
    d = np.load(npz_path, allow_pickle=True)
    result: dict = {"path": npz_path}

    # Reward metrics
    rew = d["rew_mean"]
    tracking = d["tracking_mean"]
    obj_track = d["object_tracking_mean"]
    opt_steps = d["opt_steps"]

    result["rew_mean"] = float(rew.mean())
    result["rew_std"] = float(rew.std())
    result["tracking"] = float(tracking.mean())
    result["obj_track"] = float(obj_track.mean())
    result["opt_steps"] = float(opt_steps.mean())

    # Per-quarter analysis
    n = len(rew)
    q = n // 4
    for i in range(4):
        s, e = i * q, (i + 1) * q
        result[f"Q{i+1}_rew"] = float(rew[s:e].mean())
        result[f"Q{i+1}_tracking"] = float(tracking[s:e].mean())
        result[f"Q{i+1}_obj_track"] = float(obj_track[s:e].mean())

    # Pelvis Z trajectory (qpos[:, :, 2])
    qpos = d["qpos"]  # (ticks, ctrl_steps, nq)
    pelvis_z = qpos[:, 0, 2]  # first ctrl_step, z-axis
    result["pelvis_z_mean"] = float(pelvis_z.mean())
    for t in [0, 1, 2, 3, 3.5, 4, 4.5, 5]:
        tick = min(int(t / 0.04), len(pelvis_z) - 1)
        result[f"pelvis_z_t{t}s"] = float(pelvis_z[tick])

    # Suitcase slide joints (last 6 DOFs of qpos for contact guidance model)
    nq = qpos.shape[-1]
    if nq == 42:
        # Contact guidance: qpos[36:39] = suitcase pos (slide), [39:42] = rpy (hinge)
        suitcase_pos = qpos[:, 0, 36:39]
        for t in [0, 2, 3, 4, 5]:
            tick = min(int(t / 0.04), len(suitcase_pos) - 1)
            result[f"suitcase_t{t}s"] = suitcase_pos[tick].tolist()

    return result


def print_results(results: list[dict]) -> None:
    """Print comparison table."""
    hf_ref = {"rew_mean": 5.90, "tracking": 2.19, "obj_track": 3.71}

    # Header
    names = [Path(r["path"]).parent.name for r in results]
    header = f"{'Metric':20s}" + "".join(f"{n:>12s}" for n in names) + f"{'HF':>12s}"
    print(header)
    print("-" * len(header))

    # Main metrics
    for key, hf_val in [
        ("rew_mean", hf_ref["rew_mean"]),
        ("tracking", hf_ref["tracking"]),
        ("obj_track", hf_ref["obj_track"]),
        ("opt_steps", None),
    ]:
        row = f"{key:20s}"
        for r in results:
            row += f"{r.get(key, 0):>12.2f}"
        if hf_val is not None:
            row += f"{hf_val:>12.2f}"
        print(row)

    print()

    # Per-quarter
    print("Per-quarter reward:")
    for qi in range(1, 5):
        key = f"Q{qi}_rew"
        row = f"  {key:18s}"
        for r in results:
            row += f"{r.get(key, 0):>12.2f}"
        print(row)

    print()

    # Pelvis Z
    print("Pelvis Z height:")
    for t in [0, 1, 2, 3, 3.5, 4, 4.5, 5]:
        key = f"pelvis_z_t{t}s"
        row = f"  t={t:<4.1f}s           "
        for r in results:
            row += f"{r.get(key, 0):>12.3f}"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze HDMI trajectory results")
    parser.add_argument("npz_files", nargs="+", help="Path(s) to trajectory_hdmi.npz")
    args = parser.parse_args()

    results = []
    for path in args.npz_files:
        r = analyze_run(path)
        results.append(r)

    print_results(results)


if __name__ == "__main__":
    main()
