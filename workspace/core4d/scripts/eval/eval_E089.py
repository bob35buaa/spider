#!/usr/bin/env python3
"""E089 eval: compute headline metrics for A path (box021_person1) rollout.

Compares to E088A/B/C baselines on box021_D003_18029_p2 to test if
"G1-Feasibility gate pass" → "SPIDER head/upper penetration much lower".
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import argparse
from pathlib import Path
import numpy as np
import json
import mujoco
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R

REPO = Path("/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider")
TASKS = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"

EEF_OFFSET = np.array([0.05, 0.0, 0.0])

# E082-E088 baselines for box021_D003_18029_p2 (from log 110 + 109)
BASELINES = {
    "E085A_raw_main (29kg)":    {"contact": 0.829, "obj_mean_m": 0.665, "pelvis_min_m": 0.659, "head_pen": 0.186, "upper_pen": 0.535, "hand_floor_max": 0.736},
    "E087A_5kg_raw_main":       {"contact": 0.829, "obj_mean_m": 0.782, "pelvis_min_m": 0.574, "head_pen": 0.891, "upper_pen": 0.891, "hand_floor_max": 0.814},
    "E087B_10kg_raw_main":      {"contact": 0.829, "obj_mean_m": 0.735, "pelvis_min_m": 0.629, "head_pen": 0.698, "upper_pen": 0.760, "hand_floor_max": 0.744},
    "E088A_m10_gate":           {"contact": 0.822, "obj_mean_m": 0.695, "pelvis_min_m": 0.181, "head_pen": 0.279, "upper_pen": 0.550, "hand_floor_max": 0.108},
    "E088B_m10_gate_low":       {"contact": 0.705, "obj_mean_m": 0.722, "pelvis_min_m": 0.643, "head_pen": 0.713, "upper_pen": 0.752, "hand_floor_max": 0.0},
    "E088C_m10_gate_clear":     {"contact": 0.736, "obj_mean_m": 0.612, "pelvis_min_m": 0.173, "head_pen": 0.194, "upper_pen": 0.806, "hand_floor_max": 0.403},
}


def parse_box_half(scene_xml):
    tree = ET.parse(scene_xml); root = tree.getroot()
    for body in root.iter("body"):
        if body.attrib.get("name") == "object":
            for g in body.iter("geom"):
                if g.attrib.get("type") == "box" and "collision" in g.attrib.get("name", "").lower():
                    return np.array([float(x) for x in g.attrib["size"].split()])
    return None


def compute_sim_metrics(npz_path: Path, scene_xml: Path) -> dict:
    """Compute headline metrics directly from rollout qpos.

    Returns: dict with keys head_pen_frac, upper_pen_frac, hand_floor_frac (per-hand),
             contact_frac (sim wrist near object), obj_mean/max_err_m (vs ref), pelvis_min_m, ...
    """
    d = dict(np.load(npz_path))
    qpos = d["qpos"]
    # qpos shape (T, 2, 42): the second dim is (sim, ref)
    T = qpos.shape[0]
    sim_q = qpos[:, 0]; ref_q = qpos[:, 1]
    half = parse_box_half(scene_xml)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    md = mujoco.MjData(model)
    obj_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")

    # Bodies for penetration tests
    head_b   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head_link")
    upper_bs = []
    for n in ["torso_link", "left_shoulder_yaw_link", "right_shoulder_yaw_link",
              "left_shoulder_pitch_link", "right_shoulder_pitch_link", "pelvis_contour_link",
              "left_elbow_link", "right_elbow_link"]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        if bid != -1: upper_bs.append(bid)
    bidL = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    bidR = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")

    pelvis_z = sim_q[:, 2]
    head_in = np.zeros(T, dtype=bool)
    upper_in = np.zeros(T, dtype=bool)
    handL_floor_z = np.zeros(T)
    handR_floor_z = np.zeros(T)
    handL_world = np.zeros((T, 3))
    handR_world = np.zeros((T, 3))
    obj_xyz_sim = np.zeros((T, 3))
    obj_xyz_ref = np.zeros((T, 3))

    def in_box(pos_world, obj_pos, obj_mat):
        local = obj_mat.T @ (pos_world - obj_pos)
        return bool(np.all(np.abs(local) < half))

    for t in range(T):
        md.qpos[:] = sim_q[t]; mujoco.mj_forward(model, md)
        op = md.xpos[obj_b].copy(); om = md.xmat[obj_b].reshape(3, 3)
        obj_xyz_sim[t] = op
        head_in[t]  = in_box(md.xpos[head_b].copy(), op, om) if head_b != -1 else False
        upper_in[t] = any(in_box(md.xpos[b].copy(), op, om) for b in upper_bs)
        # hand with eef offset
        for bid, store_w in [(bidL, handL_world), (bidR, handR_world)]:
            p = md.xpos[bid].copy(); q = md.xquat[bid].copy()
            rot = R.from_quat([q[1], q[2], q[3], q[0]])
            store_w[t] = p + rot.apply(EEF_OFFSET)
        handL_floor_z[t] = handL_world[t, 2]
        handR_floor_z[t] = handR_world[t, 2]
        # Ref object pose
        md.qpos[:] = ref_q[t]; mujoco.mj_forward(model, md)
        obj_xyz_ref[t] = md.xpos[obj_b].copy()

    obj_err = np.linalg.norm(obj_xyz_sim - obj_xyz_ref, axis=1)
    # Contact frac: define as wrist within 8cm of any box face (signed-dist > -0.02 and inside expanded box)
    def hand_contact_frac(handw):
        cnt = 0
        for t in range(T):
            md.qpos[:] = sim_q[t]; mujoco.mj_forward(model, md)
            op = md.xpos[obj_b].copy(); om = md.xmat[obj_b].reshape(3, 3)
            local = om.T @ (handw[t] - op)
            outside = np.maximum(np.abs(local) - half, 0.0)
            d_outside = float(np.linalg.norm(outside))
            if d_outside < 0.08:
                cnt += 1
        return cnt / T
    cL = hand_contact_frac(handL_world)
    cR = hand_contact_frac(handR_world)
    return {
        "T": T,
        "pelvis_min_m": float(pelvis_z.min()),
        "head_pen_frac": float(head_in.mean()),
        "upper_pen_frac": float(upper_in.mean()),
        "handL_floor_lt_5cm_frac": float((handL_floor_z < 0.05).mean()),
        "handR_floor_lt_5cm_frac": float((handR_floor_z < 0.05).mean()),
        "obj_err_mean_m": float(obj_err.mean()),
        "obj_err_max_m": float(obj_err.max()),
        "contact_frac_L_8cm": cL,
        "contact_frac_R_8cm": cR,
        "contact_frac_either": float(max(cL, cR)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="E089A_box021_person1_upperobj")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default="workspace/core4d/results/E089/eval_summary.json")
    args = parser.parse_args()
    suffix = "_smoke" if args.smoke else ""
    npz = REPO / f"workspace/core4d/results/E089/A/{args.variant}_outdir{suffix}/trajectory_mjwp_act.npz"
    # The rollout uses scene_act (nq=42); load scene_act to match qpos dim
    scene = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object/box021_person1_upperobj_e089/scene_act.xml"
    print(f"Reading {npz}")
    m = compute_sim_metrics(npz, scene)
    label = f"E089A_box021_person1{'_SMOKE' if args.smoke else '_FULL'}"
    print("\n" + "=" * 100)
    print(f"{'variant':<35} {'T':>3}  {'cont_max':>8} {'obj_mean':>8} {'pelv_min':>8} {'head_pen':>8} {'upper_pen':>9} {'LH_floor':>8} {'RH_floor':>8}")
    print("-" * 100)
    print(f"{label:<35} {m['T']:>3}  {m['contact_frac_either']*100:>7.1f}% {m['obj_err_mean_m']:>7.3f}m {m['pelvis_min_m']:>7.3f}m {m['head_pen_frac']*100:>7.1f}% {m['upper_pen_frac']*100:>8.1f}% {m['handL_floor_lt_5cm_frac']*100:>7.1f}% {m['handR_floor_lt_5cm_frac']*100:>7.1f}%")
    print("-" * 100)
    print("Baselines on box021_D003_18029_p2 (from logs 109/110):")
    for name, b in BASELINES.items():
        print(f"{name:<35} {'-':>3}  {b['contact']*100:>7.1f}% {b['obj_mean_m']:>7.3f}m {b['pelvis_min_m']:>7.3f}m {b['head_pen']*100:>7.1f}% {b['upper_pen']*100:>8.1f}% {b['hand_floor_max']*100:>7.1f}% {'-':>8}")
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    j = json.loads(out_path.read_text()) if out_path.exists() else {}
    j[label] = m
    out_path.write_text(json.dumps(j, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
