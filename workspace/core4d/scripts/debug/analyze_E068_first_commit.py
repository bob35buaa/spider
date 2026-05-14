#!/usr/bin/env python3
"""E068: analyze the first committed controls in existing MJWP results."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_E068_init_drift import (  # noqa: E402
    _build_config,
    _convert_scene_act_ref,
    _normalize_angle_deg,
    _quat_angle_deg,
)
from spider.io import load_data  # noqa: E402


RESULTS = {
    "E062": "workspace/core4d/results/E062/E062_box023_sphere_autopalm.npz",
    "E063": "workspace/core4d/results/E063/E063_box023.npz",
    "E065A": "workspace/core4d/results/E065/E065A_box023.npz",
    "E066A": "workspace/core4d/results/E066/E066A_box023.npz",
    "E067N": "workspace/core4d/results/E067/E067N_box023.npz",
}


def _yaw_deg(quat_wxyz: np.ndarray) -> float:
    q = quat_wxyz
    return float(R.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz", degrees=True)[2])


def main() -> None:
    out_dir = ROOT / "workspace/core4d/results/E068"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = _build_config("core4d_e062_box023", "box023_person1", "cpu", 4)
    qpos_ref, qvel_ref, ctrl_ref, _contact, _contact_pos = load_data(
        config, config.data_path
    )
    qpos_ref, qvel_ref, ctrl_ref = _convert_scene_act_ref(
        config, qpos_ref, qvel_ref, ctrl_ref
    )
    qpos_ref_np = qpos_ref.detach().cpu().numpy()
    ctrl_ref_np = ctrl_ref.detach().cpu().numpy()

    trace_rows = []
    ctrl_rows = []
    for label, rel_path in RESULTS.items():
        path = ROOT / rel_path
        if not path.is_file():
            continue
        data = np.load(path)
        qpos = data["qpos"]
        ctrl = data["ctrl"]
        n_inner = qpos.shape[1]

        for outer in range(min(5, qpos.shape[0])):
            for inner in range(n_inner):
                sim_idx_after = outer * n_inner + inner + 1
                sim_idx_same = outer * n_inner + inner
                q = qpos[outer, inner]
                yaw = _yaw_deg(q[3:7])
                ref_yaw = _yaw_deg(qpos_ref_np[sim_idx_after, 3:7])
                trace_rows.append(
                    {
                        "label": label,
                        "outer": outer,
                        "inner": inner,
                        "time": float(data["time"][outer, inner]),
                        "sim_idx_after": sim_idx_after,
                        "yaw_deg": yaw,
                        "ref_yaw_deg": ref_yaw,
                        "yaw_err_vs_ref_after_deg": _normalize_angle_deg(
                            yaw - ref_yaw
                        ),
                        "quat_err_vs_ref_after_deg": _quat_angle_deg(
                            q[3:7], qpos_ref_np[sim_idx_after, 3:7]
                        ),
                        "pelvis_z": float(q[2]),
                    }
                )

                c = ctrl[outer, inner]
                ref_same = ctrl_ref_np[sim_idx_same]
                diff = c - ref_same
                robot_diff = diff[:29]
                obj_diff = diff[29:]
                top_idx = int(np.argmax(np.abs(robot_diff)))
                ctrl_rows.append(
                    {
                        "label": label,
                        "outer": outer,
                        "inner": inner,
                        "time": float(data["time"][outer, inner]),
                        "sim_idx_same": sim_idx_same,
                        "ctrl_l2": float(np.linalg.norm(diff)),
                        "ctrl_abs_mean": float(np.mean(np.abs(diff))),
                        "robot_ctrl_max_abs": float(np.max(np.abs(robot_diff))),
                        "object_ctrl_max_abs": float(np.max(np.abs(obj_diff))),
                        "top_robot_idx": top_idx,
                        "top_robot_delta": float(robot_diff[top_idx]),
                        "top_robot_ctrl": float(c[top_idx]),
                        "top_robot_ref": float(ref_same[top_idx]),
                    }
                )

    trace_csv = out_dir / "first_commit_trace.csv"
    ctrl_csv = out_dir / "first_ctrl_delta.csv"
    with trace_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trace_rows)
    with ctrl_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ctrl_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ctrl_rows)

    print(f"Wrote {trace_csv}")
    print(f"Wrote {ctrl_csv}")
    for row in trace_rows[:10]:
        print(
            f"{row['label']} t={row['time']:.3f} "
            f"yaw_err={row['yaw_err_vs_ref_after_deg']:.2f} "
            f"quat_err={row['quat_err_vs_ref_after_deg']:.2f} "
            f"pelvis_z={row['pelvis_z']:.3f}"
        )
    for row in ctrl_rows[:4]:
        print(
            f"{row['label']} ctrl t={row['time']:.3f} "
            f"robot_max={row['robot_ctrl_max_abs']:.3f} "
            f"obj_max={row['object_ctrl_max_abs']:.3f} "
            f"top_idx={row['top_robot_idx']} delta={row['top_robot_delta']:.3f}"
        )


if __name__ == "__main__":
    main()
