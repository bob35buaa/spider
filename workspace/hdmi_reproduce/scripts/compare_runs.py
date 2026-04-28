#!/usr/bin/env python3
"""Compare two HDMI runs by analyzing trajectory differences.

Usage:
    python workspace/hdmi_reproduce/scripts/compare_runs.py \
        workspace/hdmi_reproduce/results/R012/trajectory_hdmi.npz \
        workspace/hdmi_reproduce/results/R013/trajectory_hdmi.npz

    # With scene XML for wrist jitter delta:
    python workspace/hdmi_reproduce/scripts/compare_runs.py \
        --scene "example_datasets/.../scene/mjlab scene.xml" \
        workspace/hdmi_reproduce/results/R012/trajectory_hdmi.npz \
        workspace/hdmi_reproduce/results/R013/trajectory_hdmi.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _extract_final_iter(metric_2d: np.ndarray, opt_steps: np.ndarray) -> np.ndarray:
    """Extract the final iteration's value for each control tick."""
    T, max_iter = metric_2d.shape
    idx = np.clip(opt_steps.astype(int) - 1, 0, max_iter - 1)
    return metric_2d[np.arange(T), idx]


def compare(path_a: str, path_b: str, scene_xml: str | None = None) -> None:
    """Compare two runs side-by-side."""
    a = {k: v for k, v in np.load(path_a, allow_pickle=True).items()}
    b = {k: v for k, v in np.load(path_b, allow_pickle=True).items()}
    name_a = Path(path_a).parent.name
    name_b = Path(path_b).parent.name

    opt_a = a["opt_steps"].ravel()
    opt_b = b["opt_steps"].ravel()

    print(f"Comparing: {name_a} vs {name_b}")
    print("=" * 72)

    # Overall metrics (final iteration)
    print(f"\n{'Metric':25s} {name_a:>12s} {name_b:>12s} {'Delta':>12s}")
    print("-" * 61)
    for key in ["rew_mean", "tracking_mean", "object_tracking_mean"]:
        va = float(_extract_final_iter(a[key], opt_a).mean())
        vb = float(_extract_final_iter(b[key], opt_b).mean())
        delta = vb - va
        pct = delta / va * 100 if va != 0 else 0
        print(f"{key:25s} {va:12.3f} {vb:12.3f} {delta:+12.3f} ({pct:+.1f}%)")

    print(f"{'opt_steps':25s} {opt_a.mean():12.1f} {opt_b.mean():12.1f}")

    # Per-quarter
    T = min(len(opt_a), len(opt_b))
    q = T // 4
    rew_a = _extract_final_iter(a["rew_mean"][:T], opt_a[:T])
    rew_b = _extract_final_iter(b["rew_mean"][:T], opt_b[:T])

    print(f"\nPer-quarter rew_mean:")
    for i in range(4):
        s = i * q
        e = (i + 1) * q if i < 3 else T
        va = float(rew_a[s:e].mean())
        vb = float(rew_b[s:e].mean())
        print(f"  Q{i+1}: {va:8.2f} -> {vb:8.2f} ({vb-va:+.2f})")

    # Pelvis Z comparison (flattened sim steps)
    sim_dt = 0.02
    qa_flat = a["qpos"].reshape(-1, a["qpos"].shape[-1])
    qb_flat = b["qpos"].reshape(-1, b["qpos"].shape[-1])
    pz_a = qa_flat[:, 2]
    pz_b = qb_flat[:, 2]

    print(f"\nPelvis Z height:")
    print(f"  {'t(s)':>6s} {name_a:>10s} {name_b:>10s} {'Delta':>10s}")
    for t in [0, 1, 1.4, 1.8, 2, 3, 4, 5, 8, 10]:
        sa = min(int(t / sim_dt), len(pz_a) - 1)
        sb = min(int(t / sim_dt), len(pz_b) - 1)
        za, zb = pz_a[sa], pz_b[sb]
        print(f"  {t:6.1f} {za:10.4f} {zb:10.4f} {zb-za:+10.4f}")

    # Pelvis stability (first 2s)
    n_2s_a = min(int(2.0 / sim_dt), len(pz_a))
    n_2s_b = min(int(2.0 / sim_dt), len(pz_b))
    print(f"\nPelvis stability (first 2s):")
    print(f"  std:  {name_a}={pz_a[:n_2s_a].std():.4f}, {name_b}={pz_b[:n_2s_b].std():.4f}")
    print(f"  min:  {name_a}={pz_a[:n_2s_a].min():.4f}, {name_b}={pz_b[:n_2s_b].min():.4f}")

    # Ctrl tracking error
    if "ctrl" in a and "ctrl_ref" in a and "ctrl" in b and "ctrl_ref" in b:
        err_a = float(np.abs(a["ctrl"] - a["ctrl_ref"]).mean())
        err_b = float(np.abs(b["ctrl"] - b["ctrl_ref"]).mean())
        print(f"\nCtrl tracking MAE: {name_a}={err_a:.4f}, {name_b}={err_b:.4f}")

    # Wrist jitter comparison
    if scene_xml is not None:
        import mujoco

        model = mujoco.MjModel.from_xml_path(scene_xml)
        n_2s = min(int(2.0 / sim_dt), len(qa_flat), len(qb_flat))

        print(f"\nWrist jitter (first 2s, frame-diff std, degrees):")
        print(f"  {'joint':>25s} {name_a:>10s} {name_b:>10s} {'reduction':>10s}")
        for ji in range(model.njnt):
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, ji)
            if not jname or "wrist" not in jname:
                continue
            qadr = model.jnt_qposadr[ji]
            if qadr >= qa_flat.shape[1] or qadr >= qb_flat.shape[1]:
                continue
            std_a = float(np.degrees(np.std(np.diff(qa_flat[:n_2s, qadr]))))
            std_b = float(np.degrees(np.std(np.diff(qb_flat[:n_2s, qadr]))))
            reduction = (1 - std_b / std_a) * 100 if std_a > 0 else 0
            short = jname.replace("robot/", "")
            print(f"  {short:>25s} {std_a:10.4f} {std_b:10.4f} {reduction:9.1f}%")

    # Suitcase position
    nq_a, nq_b = qa_flat.shape[1], qb_flat.shape[1]
    if nq_a == 42 and nq_b == 42:
        print(f"\nSuitcase slide position:")
        print(f"  {'t(s)':>6s} {name_a + ' [x,y,z]':>30s} {name_b + ' [x,y,z]':>30s}")
        for t in [0, 2, 3, 4, 5]:
            sa = min(int(t / sim_dt), len(qa_flat) - 1)
            sb = min(int(t / sim_dt), len(qb_flat) - 1)
            pa = qa_flat[sa, 36:39]
            pb = qb_flat[sb, 36:39]
            print(
                f"  {t:6.1f} [{pa[0]:7.3f},{pa[1]:7.3f},{pa[2]:7.3f}]"
                f"  [{pb[0]:7.3f},{pb[1]:7.3f},{pb[2]:7.3f}]"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two HDMI trajectory runs")
    parser.add_argument("run_a", help="Path to first trajectory_hdmi.npz")
    parser.add_argument("run_b", help="Path to second trajectory_hdmi.npz")
    parser.add_argument(
        "--scene", type=str, default=None,
        help="Scene XML path for wrist jitter comparison",
    )
    args = parser.parse_args()

    compare(args.run_a, args.run_b, scene_xml=args.scene)


if __name__ == "__main__":
    main()
