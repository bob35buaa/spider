#!/usr/bin/env python3
"""Compute qpos_jerk_l2_p95 + trackbody_jerk_p95 for the main-ablation chain
on the same 3-case set, using the EXACT same math as eval_E166/eval_E167.

Run with the project venv (has mujoco+numpy):
    .venv/bin/python workspace/core4d/report/0624/compute_chain_jerk.py

Validation: E163 reproduces the E167 cem_metrics eval baseline (qpos≈3038, trackbody≈569).
qpos_jerk uses the full flattened (T, 2*nq) array (matches eval_E166.qpos_kinematic_health);
trackbody_jerk uses FK on the sim channel arr[:,0,:] (matches core_metrics.npz_qpos).
"""
import numpy as np, mujoco
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # repo root
TRACK = ["left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
CASES = ["box023_person2", "box021_029_p2", "box004_083_p2"]
SCENE = {
    "box023_person2": "example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person2_legobj/scene_act_E147_rubber_hull.xml",
    "box021_029_p2": "example_datasets/processed/core4d/unitree_g1/humanoid_object/d003_box021_20231018_029_p2_e107_clean/scene_act_E148_rubber_hull.xml",
    "box004_083_p2": "example_datasets/processed/core4d/unitree_g1/humanoid_object/e091_box004_20231003_2_083_p2_e092_dyn/scene_act_E148_rubber_hull.xml",
}
NPZ = {
    "rubberhand": ["workspace/core4d/results/E147/rubber_hand_collision/cem/full/E147_{c}_rubber_hull_outdir_full/trajectory_mjwp_act.npz",
                   "workspace/core4d/results/E148/e143_24case_rubber_hand_collision/cem/full/E148_{c}_rubber_hull_outdir_full/trajectory_mjwp_act.npz"],
    "gateA": ["workspace/core4d/results/E156/clean8_gate_decay/cem/full/E156_{c}_gateA_outdir_full/trajectory_mjwp_act.npz"],
    "surfaceBandA": ["workspace/core4d/results/E158/gate_surface_band/cem/full/E158_{c}_gateA_surfaceBandA_outdir_full/trajectory_mjwp_act.npz"],
    "surfaceBandA2": ["workspace/core4d/results/E159/gate_surface_band_A2/cem/full/E159_{c}_gateA_surfaceBandA2_outdir_full/trajectory_mjwp_act.npz"],
    "postureRerank": ["workspace/core4d/results/E160/posture_rerank/cem/full/E160_{c}_gateA_postureRerankA_outdir_full/trajectory_mjwp_act.npz"],
    "releaseDecay": ["workspace/core4d/results/E161/surface_release_ablation/cem/full/E161_{c}_releaseDecay_outdir_full/trajectory_mjwp_act.npz"],
    "E163": ["workspace/core4d/results/E163/narrow_surface_band/cem/full/E163_{c}_narrowSurfaceBand_outdir_full/trajectory_mjwp_act.npz"],
}


def fps_of(d, n):
    fps = 50.0
    if "time" in d.files:
        t = np.asarray(d["time"], dtype=np.float64).reshape(n, -1)[:, 0]
        dt = np.diff(t); dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size:
            fps = float(1.0 / np.median(dt))
    return fps


def jerks(npz_path, scene_path):
    d = np.load(npz_path, allow_pickle=True)
    q = np.asarray(d["qpos"], dtype=np.float64)
    n = q.shape[0]; fps = fps_of(d, n)
    flat = q.reshape(n, -1)
    qj = float(np.percentile(np.linalg.norm(np.diff(flat, n=3, axis=0), axis=1) * fps**3, 95))
    sim = q[:, 0, :] if q.ndim == 3 else q
    model = mujoco.MjModel.from_xml_path(scene_path); data = mujoco.MjData(model)
    ids = [i for nm in TRACK if (i := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)) >= 0]
    pos = []
    for qq in sim:
        data.qpos[:] = qq; data.qvel[:] = 0.0; mujoco.mj_forward(model, data)
        pos.append(data.xpos[ids].copy())
    tr = np.asarray(pos, dtype=np.float64)
    tj = float(np.percentile(np.linalg.norm(np.diff(tr, n=3, axis=0), axis=-1) * fps**3, 95))
    return qj, tj


def resolve(m, c):
    for p in NPZ[m]:
        fp = ROOT / p.format(c=c)
        if fp.is_file():
            return str(fp)
    raise FileNotFoundError(f"{m} {c}")


if __name__ == "__main__":
    print(f"{'method':14s} {'qpos_jerk_p95':>13s} {'trackbody_jerk_p95':>18s}")
    for m in ["rubberhand", "gateA", "surfaceBandA", "surfaceBandA2", "postureRerank", "releaseDecay", "E163"]:
        qs, ts = [], []
        for c in CASES:
            qj, tj = jerks(resolve(m, c), str(ROOT / SCENE[c]))
            qs.append(qj); ts.append(tj)
        print(f"{m:14s} {sum(qs)/3:13.1f} {sum(ts)/3:18.1f}")
    print("# E167A (z-only) from results/E167/holosoma_zonly/eval/cem_metrics: qpos 3182.4 / trackbody 580.0 (3-case)")
    print("# validation: E163 == E167 eval baseline (qpos 3038 / trackbody 569)")
