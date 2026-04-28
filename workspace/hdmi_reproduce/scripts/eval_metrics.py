#!/usr/bin/env python3
"""Analyze HDMI retargeting results from trajectory_hdmi.npz.

Usage:
    # Single run:
    python workspace/hdmi_reproduce/scripts/eval_metrics.py \
        workspace/hdmi_reproduce/results/R013/trajectory_hdmi.npz

    # Compare multiple runs:
    python workspace/hdmi_reproduce/scripts/eval_metrics.py \
        workspace/hdmi_reproduce/results/R012/trajectory_hdmi.npz \
        workspace/hdmi_reproduce/results/R013/trajectory_hdmi.npz \
        workspace/hdmi_reproduce/results/R013b/trajectory_hdmi.npz

    # With scene XML for wrist jitter analysis:
    python workspace/hdmi_reproduce/scripts/eval_metrics.py \
        --scene "example_datasets/.../scene/mjlab scene.xml" \
        workspace/hdmi_reproduce/results/R013/trajectory_hdmi.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _extract_final_iter(metric_2d: np.ndarray, opt_steps: np.ndarray) -> np.ndarray:
    """Extract the final iteration's value for each control tick.

    metric_2d: (T, max_iter)  — per-iteration metric values
    opt_steps: (T,)           — number of optimizer iterations actually run
    Returns:   (T,)           — final iteration's value per tick
    """
    T, max_iter = metric_2d.shape
    idx = np.clip(opt_steps.astype(int) - 1, 0, max_iter - 1)
    return metric_2d[np.arange(T), idx]


def analyze_run(npz_path: str, scene_xml: str | None = None) -> dict:
    """Load and analyze a single run's trajectory data."""
    d = np.load(npz_path, allow_pickle=True)
    result: dict = {"path": npz_path}

    opt_steps = d["opt_steps"].ravel()
    T = len(opt_steps)
    result["opt_steps"] = float(opt_steps.mean())

    # Extract final-iteration metrics
    rew = _extract_final_iter(d["rew_mean"], opt_steps)
    tracking = _extract_final_iter(d["tracking_mean"], opt_steps)
    obj_track = _extract_final_iter(d["object_tracking_mean"], opt_steps)

    result["rew_mean"] = float(rew.mean())
    result["rew_std"] = float(rew.std())
    result["tracking"] = float(tracking.mean())
    result["obj_track"] = float(obj_track.mean())

    # Per-quarter analysis
    q = T // 4
    for i in range(4):
        s = i * q
        e = (i + 1) * q if i < 3 else T
        result[f"Q{i+1}_rew"] = float(rew[s:e].mean())
        result[f"Q{i+1}_tracking"] = float(tracking[s:e].mean())
        result[f"Q{i+1}_obj_track"] = float(obj_track[s:e].mean())

    # Pelvis Z trajectory — flatten (ticks, ctrl_steps, nq) → (sim_steps, nq)
    qpos = d["qpos"]
    nq = qpos.shape[-1]
    qpos_flat = qpos.reshape(-1, nq)
    pelvis_z = qpos_flat[:, 2]
    result["pelvis_z_mean"] = float(pelvis_z.mean())
    sim_dt = 0.02  # physics step * decimation
    for t in [0, 1, 1.4, 1.8, 2, 3, 4, 5, 8, 10]:
        step = min(int(t / sim_dt), len(pelvis_z) - 1)
        result[f"pelvis_z_t{t}s"] = float(pelvis_z[step])

    # Pelvis stability (first 2s) — std of pelvis_z
    n_2s = min(int(2.0 / sim_dt), len(pelvis_z))
    result["pelvis_z_std_0_2s"] = float(pelvis_z[:n_2s].std())
    result["pelvis_z_min_0_2s"] = float(pelvis_z[:n_2s].min())

    # Wrist jitter analysis (requires scene XML for joint name lookup)
    if scene_xml is not None:
        result["wrist_jitter"] = _analyze_wrist_jitter(qpos_flat, scene_xml)

    # Suitcase position (contact guidance nq=42)
    if nq == 42:
        for t in [0, 2, 3, 4, 5]:
            step = min(int(t / sim_dt), len(qpos_flat) - 1)
            result[f"suitcase_t{t}s"] = qpos_flat[step, 36:39].tolist()

    return result


def _analyze_wrist_jitter(
    qpos_flat: np.ndarray, scene_xml: str
) -> dict[str, float]:
    """Compute wrist joint jitter (frame-diff std in degrees) for first 2s."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(scene_xml)
    sim_dt = 0.02
    n_2s = min(int(2.0 / sim_dt), len(qpos_flat))

    jitter: dict[str, float] = {}
    for ji in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, ji)
        if not jname or "wrist" not in jname:
            continue
        qadr = model.jnt_qposadr[ji]
        if qadr >= qpos_flat.shape[1]:
            continue
        diff = np.diff(qpos_flat[:n_2s, qadr])
        std_deg = float(np.degrees(np.std(diff)))
        short_name = jname.replace("robot/", "")
        jitter[short_name] = std_deg
    return jitter


def print_results(results: list[dict]) -> None:
    """Print comparison table."""
    hf_ref = {"rew_mean": 5.90, "tracking": 2.19, "obj_track": 3.71}
    names = [Path(r["path"]).parent.name for r in results]

    # Header
    header = f"{'Metric':25s}" + "".join(f"{n:>12s}" for n in names) + f"{'HF':>12s}"
    print(header)
    print("-" * len(header))

    # Main metrics
    for key, hf_val in [
        ("rew_mean", hf_ref["rew_mean"]),
        ("tracking", hf_ref["tracking"]),
        ("obj_track", hf_ref["obj_track"]),
        ("opt_steps", None),
    ]:
        row = f"{key:25s}"
        for r in results:
            row += f"{r.get(key, 0):>12.2f}"
        if hf_val is not None:
            row += f"{hf_val:>12.2f}"
        print(row)

    print()

    # Per-quarter reward
    print("Per-quarter reward:")
    for qi in range(1, 5):
        key = f"Q{qi}_rew"
        row = f"  {key:23s}"
        for r in results:
            row += f"{r.get(key, 0):>12.2f}"
        print(row)

    print()

    # Pelvis Z
    print("Pelvis Z height:")
    for t in [0, 1, 1.4, 1.8, 2, 3, 4, 5]:
        key = f"pelvis_z_t{t}s"
        row = f"  t={t:<5.1f}s              "
        for r in results:
            row += f"{r.get(key, 0):>12.4f}"
        print(row)

    # Pelvis stability (first 2s)
    print()
    print("Pelvis stability (first 2s):")
    for key in ["pelvis_z_std_0_2s", "pelvis_z_min_0_2s"]:
        row = f"  {key:23s}"
        for r in results:
            row += f"{r.get(key, 0):>12.4f}"
        print(row)

    # Wrist jitter
    has_jitter = any("wrist_jitter" in r for r in results)
    if has_jitter:
        print()
        print("Wrist jitter (first 2s, frame-diff std, degrees):")
        # Collect all joint names
        all_joints: list[str] = []
        for r in results:
            if "wrist_jitter" in r:
                for j in r["wrist_jitter"]:
                    if j not in all_joints:
                        all_joints.append(j)
        all_joints.sort()
        for jname in all_joints:
            row = f"  {jname:23s}"
            for r in results:
                val = r.get("wrist_jitter", {}).get(jname, float("nan"))
                row += f"{val:>12.4f}"
            print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze HDMI trajectory results")
    parser.add_argument("npz_files", nargs="+", help="Path(s) to trajectory_hdmi.npz")
    parser.add_argument(
        "--scene", type=str, default=None,
        help="Scene XML path for wrist jitter analysis",
    )
    args = parser.parse_args()

    results = []
    for path in args.npz_files:
        r = analyze_run(path, scene_xml=args.scene)
        results.append(r)

    print_results(results)


if __name__ == "__main__":
    main()
