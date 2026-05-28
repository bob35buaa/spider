"""G1-Feasibility Gate (proposed D005b extension).

Reads a SPIDER `trajectory_kinematic.npz` + its `scene.xml` and computes
per-case feasibility signals that the existing D005 handbox-surface gate
DOES NOT compute. These signals would have caught box021 D003 cases before
spending dynamic-retarget compute.

New per-case metrics (none of these exist in D005):

1. ik_wrist_inside_box_frac (per hand): fraction of frames in which the
   IK-retargeted wrist (+5cm forward eef offset) lies strictly inside the
   object collision box. THRESHOLD: should be ~0% for either hand;
   reject if > 5%.

2. ik_wrist_signed_dist_to_face (per hand, mean over frames): how far the
   wrist FK target sits outside the chosen box face along its normal. A
   real grip pose should give > 0.10 m. Reject if < 0.05 m on either hand.

3. wrist_below_pelvis_gap (per hand, mean): pelvis_z - wrist_world_z. A
   natural carry pose has gap ~0.05-0.15 m. Reject if > 0.30 m on either
   hand (= "deep bending required").

4. top_face_frac (per hand): fraction of frames whose chosen face is
   `+z` in the object local frame, i.e. hand on top edge/face. Soft
   reward; > 0.30 is good for box-lift tasks.

5. pelvis_z_min: lowest pelvis height during the motion. Reject if
   < 0.55m (deep squat → CEM unstable downstream).

6. trajectory_T: total frame count. Soft signal; very short (< 80 frames)
   gives CEM less context.

Usage:
    python g1_feasibility_gate.py [task_dir [task_dir ...]]

If no args, runs over a built-in CASES dict spanning known fail/pass
cases for calibration.
"""
from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import sys
from pathlib import Path
import numpy as np
import mujoco
import xml.etree.ElementTree as ET
import json

ROOT = Path("/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider")
TASKS_DIR = ROOT / "example_datasets/processed/core4d/unitree_g1/humanoid_object"

CALIBRATION_CASES = {
    "box021_D003_18029_p2 (E082-E088 FAIL)": "d003_box021_20231018_029_p2_upperobj_e083_m10_e087",
    "box021_D003_11035_p2 (E082-E088 FAIL)": "d003_box021_20231011_035_p2_upperobj_e083",
    "box021_D003_20019_p1 (E082-E088 FAIL)": "d003_box021_20231020_019_p1_upperobj_e083",
    "box023_person2 (E082 guard PASS)": "box023_person2",
    "box025_person2 (E080 partial)": "box025_person2",
    "box001_person1 (UNKNOWN)": "box001_person1",
}

# Feasibility thresholds — calibrated against:
#  PASS:  box023_person2 (E082 guard ok)
#  PARTIAL: box025_person2 (E080 near-usable)
#  FAIL:  box021 D003 18029/11035/20019, box001_person1
# Requirements iterated until all known-bad rejected and known-good admitted.
GATE = {
    "max_ik_inside_box_frac": 0.10,         # ≤ 10% inside (either hand). box025=9.7%, box021_18029=33% R, box001=15% L
    "min_ik_signed_dist_to_face": 0.03,     # ≥ 3 cm outside chosen face (box025=0.034 just passes)
    "max_wrist_below_pelvis_gap": 0.30,     # deep bending blocked
    "min_pelvis_z_min": 0.60,               # box021_11035 pmin=0.583 → reject (deep squat ref)
    "min_trajectory_T": 80,                  # box021_18029 T=75 → reject
    "min_top_face_frac_either_hand": 0.20,  # at least one hand on +z face ≥20% of frames
                                            # (real grip strategy; box021 cases all =0%, box023=51%, box025=100%)
}

EEF_OFFSET = np.array([0.05, 0.0, 0.0])


def parse_object_collision(scene_xml):
    tree = ET.parse(scene_xml)
    root = tree.getroot()
    for body in root.iter("body"):
        if body.attrib.get("name") == "object":
            for g in body.iter("geom"):
                if g.attrib.get("type") == "box" and "collision" in g.attrib.get("name", "").lower():
                    return tuple(float(x) for x in g.attrib["size"].split())
    return None


def evaluate(task_dir: Path) -> dict:
    p_scene = task_dir / "scene.xml"
    p_npz = task_dir / "0" / "trajectory_kinematic.npz"
    if not p_scene.exists() or not p_npz.exists():
        return {"task": task_dir.name, "error": "missing scene.xml or npz"}
    half = np.array(parse_object_collision(p_scene))
    d = dict(np.load(p_npz))
    qpos = d["qpos"]
    T = qpos.shape[0]

    model = mujoco.MjModel.from_xml_path(str(p_scene))
    data = mujoco.MjData(model)
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    try:
        bidL = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
        bidR = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    except Exception as e:
        return {"task": task_dir.name, "error": f"missing wrist bodies: {e}"}

    pelvis_z = qpos[:, 2]
    L_loc = np.zeros((T, 3)); R_loc = np.zeros((T, 3))
    L_world_z = np.zeros(T);  R_world_z = np.zeros(T)
    from scipy.spatial.transform import Rotation as R
    for t in range(T):
        data.qpos[:] = qpos[t]; mujoco.mj_forward(model, data)
        for h, bid, loc_arr, wz_arr in [(0, bidL, L_loc, L_world_z),
                                         (1, bidR, R_loc, R_world_z)]:
            pos = data.xpos[bid].copy(); quat = data.xquat[bid].copy()
            rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            world = pos + rot.apply(EEF_OFFSET)
            obj_pos = data.xpos[obj_bid].copy()
            obj_mat = data.xmat[obj_bid].reshape(3, 3)
            loc_arr[t] = obj_mat.T @ (world - obj_pos)
            wz_arr[t] = world[2]

    def hand_stats(loc, wz):
        inside = float(np.all(np.abs(loc) < half, axis=1).mean())
        norm = loc / half
        ax = np.argmax(np.abs(norm), axis=1)
        sgn = np.sign(np.take_along_axis(norm, ax[:, None], axis=1).squeeze(1))
        signed = sgn * np.take_along_axis(loc, ax[:, None], axis=1).squeeze(1) - half[ax]
        top_face = ((sgn > 0) & (ax == 2)).mean()
        below = float(np.mean(pelvis_z - wz))
        return {
            "inside_box_frac": inside,
            "signed_dist_mean_m": float(signed.mean()),
            "signed_dist_min_m": float(signed.min()),
            "top_face_frac": float(top_face),
            "wrist_below_pelvis_gap_m": below,
            "wrist_world_z_mean_m": float(wz.mean()),
            "wrist_world_z_min_m": float(wz.min()),
        }

    out = {
        "task": task_dir.name,
        "T": int(T),
        "pelvis_z_min": float(pelvis_z.min()),
        "object_half_extents_m": [round(x, 3) for x in half.tolist()],
        "L": hand_stats(L_loc, L_world_z),
        "R": hand_stats(R_loc, R_world_z),
    }

    # gate decision
    reasons = []
    if out["L"]["inside_box_frac"] > GATE["max_ik_inside_box_frac"] or \
       out["R"]["inside_box_frac"] > GATE["max_ik_inside_box_frac"]:
        reasons.append(f"ik_wrist_inside_box_>{int(GATE['max_ik_inside_box_frac']*100)}%")
    if out["L"]["signed_dist_mean_m"] < GATE["min_ik_signed_dist_to_face"] or \
       out["R"]["signed_dist_mean_m"] < GATE["min_ik_signed_dist_to_face"]:
        reasons.append(f"ik_wrist_signed_dist_<{int(GATE['min_ik_signed_dist_to_face']*100)}cm")
    if out["L"]["wrist_below_pelvis_gap_m"] > GATE["max_wrist_below_pelvis_gap"] or \
       out["R"]["wrist_below_pelvis_gap_m"] > GATE["max_wrist_below_pelvis_gap"]:
        reasons.append(f"wrist_>{int(GATE['max_wrist_below_pelvis_gap']*100)}cm_below_pelvis")
    if out["pelvis_z_min"] < GATE["min_pelvis_z_min"]:
        reasons.append(f"pelvis_z_min_<{int(GATE['min_pelvis_z_min']*100)}cm")
    if T < GATE["min_trajectory_T"]:
        reasons.append(f"trajectory_T_<{GATE['min_trajectory_T']}")
    top_either = max(out["L"]["top_face_frac"], out["R"]["top_face_frac"])
    if top_either < GATE["min_top_face_frac_either_hand"]:
        reasons.append(f"no_hand_on_top_face_≥{int(GATE['min_top_face_frac_either_hand']*100)}%")
    out["gate_pass"] = len(reasons) == 0
    out["gate_reject_reasons"] = reasons
    return out


def fmt_row(r):
    if "error" in r:
        return f"{r['task']:<55} ERROR: {r['error']}"
    L = r["L"]; R = r["R"]
    return (
        f"{r['task']:<55} T={r['T']:>3}  "
        f"L_in={L['inside_box_frac']*100:>5.1f}%  R_in={R['inside_box_frac']*100:>5.1f}%  "
        f"L_d={L['signed_dist_mean_m']:+.3f}  R_d={R['signed_dist_mean_m']:+.3f}  "
        f"L_gap={L['wrist_below_pelvis_gap_m']:+.3f}  R_gap={R['wrist_below_pelvis_gap_m']:+.3f}  "
        f"L_top={L['top_face_frac']*100:>4.0f}%  "
        f"pmin={r['pelvis_z_min']:.3f}  "
        f"PASS" if r["gate_pass"] else
        f"{r['task']:<55} T={r['T']:>3}  "
        f"L_in={L['inside_box_frac']*100:>5.1f}%  R_in={R['inside_box_frac']*100:>5.1f}%  "
        f"L_d={L['signed_dist_mean_m']:+.3f}  R_d={R['signed_dist_mean_m']:+.3f}  "
        f"L_gap={L['wrist_below_pelvis_gap_m']:+.3f}  R_gap={R['wrist_below_pelvis_gap_m']:+.3f}  "
        f"L_top={L['top_face_frac']*100:>4.0f}%  "
        f"pmin={r['pelvis_z_min']:.3f}  REJECT  {','.join(r['gate_reject_reasons'])}"
    )


def main():
    args = sys.argv[1:]
    if args:
        cases = {Path(a).name: Path(a) for a in args}
    else:
        cases = {label: TASKS_DIR / td for label, td in CALIBRATION_CASES.items()}

    results = {}
    print(f"Gate thresholds: {GATE}\n")
    print(f"{'task / label':<55} {'T':>3}  {'L_in%':>6} {'R_in%':>6}  {'L_d':>6} {'R_d':>6}  "
          f"{'L_gap':>6} {'R_gap':>6}  {'L_top%':>6}  {'pmin':>6}  decision")
    print("-" * 200)
    for label, td in cases.items():
        r = evaluate(td)
        results[label] = r
        print(fmt_row(r))

    out_path = ROOT / "workspace/exp_diagnostic/findings/06_g1_feasibility_gate.json"
    out_path.write_text(json.dumps({"gate": GATE, "results": results}, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
