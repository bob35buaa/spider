#!/usr/bin/env python3
"""E055 Path B: Generate hand-snap warmstart for box023_person1.

Reads E054 case_tier csv to get intent_window + dominant_hand, then projects
both palms onto the box mesh surface using `spider.preprocess.hand_snap_ik`.

Outputs:
  workspace/core4d/results/E055/box023_person1/warmstart_qpos.npz
    - qpos_ref:    (T, 43) original mocap-IK trajectory
    - qpos_snap:   (T, 43) snapped trajectory (only arm joints differ)
    - intent_window: (t_start, t_end)
    - snap_mask:   (T,) bool, True for frames where snap was applied
  workspace/core4d/results/E055/box023_person1/snap_diagnostics.csv
    one row per (frame, hand) inside intent window

Run:
  .venv/bin/python workspace/core4d/scripts/E055/snap_box023.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from spider.preprocess.hand_snap_ik import find_object_mesh, snap_hands_to_object  # noqa: E402

CASE = "box023_person1"
SCENE = REPO / f"example_datasets/processed/core4d/unitree_g1/humanoid_object/{CASE}/scene.xml"
TRAJ = REPO / f"example_datasets/processed/core4d/unitree_g1/humanoid_object/{CASE}/0/trajectory_kinematic.npz"
OUT_DIR = REPO / f"workspace/core4d/results/E055/{CASE}"


def find_intent_window(qpos_ref: np.ndarray, scene_xml: Path) -> tuple[int, int]:
    """Re-derive intent window for box023 by mirroring E054's v3 detector logic
    on this case alone. Avoids cross-script coupling."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    L_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_palm")
    R_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")

    T = qpos_ref.shape[0]
    obj_pos = np.zeros((T, 3))
    L_pos = np.zeros((T, 3))
    R_pos = np.zeros((T, 3))
    for t in range(T):
        data.qpos[:] = qpos_ref[t]
        mujoco.mj_forward(model, data)
        obj_pos[t] = data.xpos[obj_bid]
        L_pos[t] = data.site_xpos[L_sid]
        R_pos[t] = data.site_xpos[R_sid]

    # min hand-obj dist per frame (matches E054 hand_obj distance)
    L_dist = np.linalg.norm(L_pos - obj_pos, axis=1)
    R_dist = np.linalg.norm(R_pos - obj_pos, axis=1)
    d_min = np.minimum(L_dist, R_dist)
    d_p20 = float(np.percentile(d_min, 20))
    band = d_min < d_p20 + 0.07

    # lifted: obj_z > obj_z_p10 + max(0.10, 0.33 * obj_z_amp)
    obj_z = obj_pos[:, 2]
    z_p10 = float(np.percentile(obj_z, 10))
    z_amp = float(obj_z.max() - obj_z.min())
    lift_thr = z_p10 + max(0.10, 0.33 * z_amp)
    lifted = obj_z > lift_thr

    # slow_rel: |v_hand - v_obj|_smooth < 0.30 m/s (5-frame moving avg)
    dt = 1 / 30.0
    v_obj = np.gradient(obj_pos, dt, axis=0)
    closer = L_pos if L_dist.mean() < R_dist.mean() else R_pos
    v_hand = np.gradient(closer, dt, axis=0)
    v_rel = np.linalg.norm(v_hand - v_obj, axis=1)
    kernel = np.ones(5) / 5
    v_rel_s = np.convolve(v_rel, kernel, mode="same")
    slow_rel = v_rel_s < 0.30

    intent = band & (slow_rel | lifted)
    # morphological closing k=3
    for _ in range(3):
        intent = intent | np.r_[False, intent[:-1]] | np.r_[intent[1:], False]
    for _ in range(3):
        intent = intent & np.r_[True, intent[:-1]] & np.r_[intent[1:], True]

    # take longest contiguous True window
    runs: list[tuple[int, int]] = []
    in_run = False
    s = 0
    for i, v in enumerate(intent):
        if v and not in_run:
            s = i
            in_run = True
        elif not v and in_run:
            runs.append((s, i - 1))
            in_run = False
    if in_run:
        runs.append((s, len(intent) - 1))
    if not runs:
        raise RuntimeError("No intent window detected for box023")
    longest = max(runs, key=lambda x: x[1] - x[0])
    return longest


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[E055] case={CASE}")
    print(f"[E055] scene={SCENE}")
    print(f"[E055] traj={TRAJ}")

    qpos_ref = np.load(TRAJ)["qpos"]
    T, nq = qpos_ref.shape
    print(f"[E055] loaded ref qpos: T={T}, nq={nq}")

    intent = find_intent_window(qpos_ref, SCENE)
    print(f"[E055] intent_window={intent} ({intent[1] - intent[0] + 1} frames)")

    mesh_path = find_object_mesh(str(SCENE))
    print(f"[E055] object mesh: {mesh_path}")

    qpos_snap, diag = snap_hands_to_object(
        scene_xml=str(SCENE),
        qpos_ref=qpos_ref,
        intent_window=intent,
        dominant_hand="both",
        object_mesh_path=str(mesh_path),
        # palm site is at wrist+8cm (mid-hand, near fingertips). G1 hand mesh
        # spans ~10cm from wrist to fingertips. Without orientation IK, we
        # cannot guarantee the hand is parallel to the box surface, so the
        # back of the hand can extend ~5cm inward of the palm site. Use 5cm
        # offset = half the hand length so a worst-case orthogonal hand mesh
        # still barely touches the box from outside. E056 (Path B-CEM) will
        # add orientation IK to tighten this and bring the contact closer.
        surface_offset=0.05,
        approach_blend_frames=10,
    )
    print(f"[E055] snapped {len(diag)} (frame, hand) pairs")

    snap_mask = np.zeros(T, dtype=bool)
    snap_mask[intent[0]:intent[1] + 1] = True

    out_npz = OUT_DIR / "warmstart_qpos.npz"
    np.savez(
        out_npz,
        qpos_ref=qpos_ref,
        qpos_snap=qpos_snap,
        intent_window=np.array(intent),
        snap_mask=snap_mask,
    )
    print(f"[E055] wrote {out_npz}")

    out_csv = OUT_DIR / "snap_diagnostics.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame", "hand", "in_intent",
            "palm_to_surface_init_m", "palm_to_surface_final_m",
            "ik_iterations", "ik_residual_m",
            "joint_in_limits", "target_x", "target_y", "target_z",
        ])
        for d in diag:
            w.writerow([
                d.frame, d.hand, d.in_intent,
                f"{d.palm_to_surface_init:.4f}",
                f"{d.palm_to_surface_final:.4f}",
                d.ik_iterations, f"{d.ik_residual:.4f}",
                d.joint_in_limits,
                f"{d.target_xyz[0]:.4f}", f"{d.target_xyz[1]:.4f}", f"{d.target_xyz[2]:.4f}",
            ])
    print(f"[E055] wrote {out_csv}")

    # summary stats
    init_d = np.array([d.palm_to_surface_init for d in diag])
    final_d = np.array([d.palm_to_surface_final for d in diag])
    in_lim = np.array([d.joint_in_limits for d in diag])
    print()
    print("=== Snap summary ===")
    print(f"  palm-to-surface init:  mean={init_d.mean()*100:.1f}cm, max={init_d.max()*100:.1f}cm")
    print(f"  palm-to-surface final: mean={final_d.mean()*100:.2f}cm, max={final_d.max()*100:.2f}cm")
    print(f"  joints within limits:  {in_lim.sum()}/{len(in_lim)}")
    print()


if __name__ == "__main__":
    main()
