#!/usr/bin/env python3
"""E066 evaluation — task_obj_rew form ablation on box023.

NEW B1/B2 metrics target the pre-contact foot-lift failure mode (log 82):
  B1: pre-contact (t=0..2.0s) max(Lf_z, Rf_z) — should be ≤ 0.10m
  B2: single-foot-support frames (one foot z>0.05 while other z<0.02 for ≥5
      consecutive frames) — should be 0

Also writes:
  workspace/core4d/results/E066/eval_summary.csv
  workspace/core4d/results/E066/foot_z_trace_E066A_box023.png
  workspace/core4d/results/E066/foot_z_trace_E066D_box023.png
  workspace/core4d/results/E066/pelvis_obj_E066A_box023.png  (vs E062 baseline)
  workspace/core4d/results/E066/pelvis_obj_E066D_box023.png

Reads:
  workspace/core4d/results/E066/E065{A,D}_box023.npz
  workspace/core4d/results/E063/E063_box023.npz   (E062 baseline shipped here for diff)
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[4]
RESULTS = REPO / "workspace/core4d/results/E066"
E063_RESULTS = REPO / "workspace/core4d/results/E063"  # has E062 baseline npz
DATASET_DIR = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"

# trace_ref in MJWP result npz only stores object (B=1). Foot references must
# come from the dataset's trajectory_kinematic.npz (full ref qpos, nq=43).

# Per-case config: (name, case_dir, intent_src, src_T, expected faces, baseline_name)
CASES = [
    {
        "name": "E066A_box023",
        "variant": "A",
        "case_dir": "box023_person1",
        "intent_src": (21, 78),
        "src_T": 136,
        "expected_L": "-xy",
        "expected_R": "-yz",
        "baseline_name": "E063_box023",  # nearest existing E062 result
    },
    {
        "name": "E066D_box023",
        "variant": "D",
        "case_dir": "box023_person1",
        "intent_src": (21, 78),
        "src_T": 136,
        "expected_L": "-xy",
        "expected_R": "-yz",
        "baseline_name": "E063_box023",
    },
]

PELVIS_OK = 0.50
CONTACT_DIST = 0.05
FACE_NAMES = ["+yz", "-yz", "+xz", "-xz", "+xy", "-xy"]
MAIN_FACE_DIST = 0.07
FACE_STABILITY = 0.60

# B1/B2 thresholds (log 83 §4.3)
PRE_CONTACT_T_END = 2.0  # seconds
PRE_CONTACT_FOOT_MAX = 0.10  # m (allow walking micro-lift)
SINGLE_LIFT_HI = 0.05  # m (one foot raised)
SINGLE_LIFT_LO = 0.02  # m (other foot grounded)
SINGLE_LIFT_MIN_LEN = 5  # frames
FPS = 30.0


def load_qpos(npz_path: Path) -> np.ndarray:
    d = np.load(npz_path, allow_pickle=True)
    qpos = d["qpos"]
    if qpos.ndim == 3:
        qpos = qpos[:, -1, :]
    return qpos


def load_obj_ref_from_result(npz_path: Path) -> np.ndarray:
    """trace_ref body 0 = object, shape (T, 3)."""
    d = np.load(npz_path, allow_pickle=True)
    tr = d["trace_ref"]
    return tr[:, 0, 0, 0, 0, :].copy()


def load_foot_ref_from_dataset(case_dir_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run mj_forward on ref qpos to get LF/RF/pelvis z over time. Returns (lf_z, rf_z, pelvis_z)."""
    ref_npz = DATASET_DIR / case_dir_name / "0" / "trajectory_kinematic.npz"
    if not ref_npz.exists():
        return np.array([]), np.array([]), np.array([])
    d = np.load(ref_npz, allow_pickle=True)
    qpos_ref = d["qpos"]  # (T, 43) for freejoint scene
    scene = DATASET_DIR / case_dir_name / "scene.xml"
    m = mujoco.MjModel.from_xml_path(str(scene))
    md = mujoco.MjData(m)
    LF_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    RF_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
    T = qpos_ref.shape[0]
    lf_z = np.zeros(T)
    rf_z = np.zeros(T)
    pz = np.zeros(T)
    for t in range(T):
        md.qpos[:] = qpos_ref[t]
        mujoco.mj_forward(m, md)
        lf_z[t] = md.site_xpos[LF_sid][2]
        rf_z[t] = md.site_xpos[RF_sid][2]
        pz[t] = md.qpos[2]
    return lf_z, rf_z, pz


def load_for_scene(scene_xml: Path):
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    d = mujoco.MjData(m)
    L_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_palm")
    R_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
    LF_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    RF_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
    obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    half = None
    for gid in range(m.ngeom):
        if (m.geom_bodyid[gid] == obj_bid
                and m.geom_contype[gid] != 0
                and m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX):
            half = m.geom_size[gid].copy()
            break
    if half is None:
        raise RuntimeError(f"No BOX collision geom on object body in {scene_xml}")
    return m, d, L_sid, R_sid, LF_sid, RF_sid, obj_bid, half


def adapt_qpos(qpos: np.ndarray, model_nq: int) -> np.ndarray:
    if qpos.shape[1] == model_nq:
        return qpos
    if qpos.shape[1] == model_nq - 1:
        from scipy.spatial.transform import Rotation as R
        nq_robot = 36
        out = np.zeros((qpos.shape[0], model_nq), dtype=qpos.dtype)
        out[:, :nq_robot] = qpos[:, :nq_robot]
        out[:, nq_robot:nq_robot + 3] = qpos[:, nq_robot:nq_robot + 3]
        eulers = qpos[:, nq_robot + 3:nq_robot + 6]
        quats_xyzw = R.from_euler("xyz", eulers).as_quat()
        out[:, nq_robot + 3] = quats_xyzw[:, 3]
        out[:, nq_robot + 4:nq_robot + 7] = quats_xyzw[:, :3]
        return out
    raise ValueError(f"Cannot adapt qpos shape {qpos.shape} to model_nq={model_nq}")


def pick_scene_for_qpos(case_dir_name: str, qpos_nq: int) -> Path:
    base = REPO / f"example_datasets/processed/core4d/unitree_g1/humanoid_object/{case_dir_name}"
    if qpos_nq == 42 and (base / "scene_act.xml").exists():
        return base / "scene_act.xml"
    return base / "scene.xml"


def compute_per_frame(qpos, m, d, L_sid, R_sid, LF_sid, RF_sid, obj_bid, half):
    T = qpos.shape[0]
    L_signed = np.zeros((T, 6))
    R_signed = np.zeros((T, 6))
    pelvis_z = np.zeros(T)
    obj_xyz_sim = np.zeros((T, 3))
    lf_z_sim = np.zeros(T)
    rf_z_sim = np.zeros(T)
    for t in range(T):
        d.qpos[:] = qpos[t]
        mujoco.mj_forward(m, d)
        obj_pos = d.xpos[obj_bid].copy()
        obj_R = d.xmat[obj_bid].copy().reshape(3, 3)
        obj_xyz_sim[t] = obj_pos
        pelvis_z[t] = qpos[t, 2]
        lf_z_sim[t] = d.site_xpos[LF_sid][2]
        rf_z_sim[t] = d.site_xpos[RF_sid][2]
        for sid, store in [(L_sid, L_signed), (R_sid, R_signed)]:
            local = obj_R.T @ (d.site_xpos[sid] - obj_pos)
            store[t] = [
                local[0] - half[0], -local[0] - half[0],
                local[1] - half[1], -local[1] - half[1],
                local[2] - half[2], -local[2] - half[2],
            ]
    return L_signed, R_signed, pelvis_z, obj_xyz_sim, lf_z_sim, rf_z_sim


def find_main_face(signed_in_window):
    abs_dist = np.abs(signed_in_window)
    med = np.median(abs_dist, axis=0)
    frac = (abs_dist <= MAIN_FACE_DIST).mean(axis=0)
    order = np.argsort(med)
    for idx in order:
        if frac[idx] >= FACE_STABILITY and med[idx] <= MAIN_FACE_DIST:
            return FACE_NAMES[idx], float(med[idx] * 100), float(frac[idx])
    return None, float(med[order[0]] * 100), float(frac[order[0]])


def compute_b1_b2(lf_z, rf_z, fps=FPS, t_end_s=PRE_CONTACT_T_END):
    """B1: max foot z in pre-contact window. B2: count of single-foot-support runs ≥ MIN_LEN."""
    end_idx = int(t_end_s * fps) + 1
    end_idx = min(end_idx, len(lf_z))
    pre_lf = lf_z[:end_idx]
    pre_rf = rf_z[:end_idx]
    b1 = float(max(pre_lf.max(), pre_rf.max()))
    # single-foot-support: (lf>HI & rf<LO) or (rf>HI & lf<LO), look at full episode
    lf_high = lf_z > SINGLE_LIFT_HI
    rf_low = rf_z < SINGLE_LIFT_LO
    rf_high = rf_z > SINGLE_LIFT_HI
    lf_low = lf_z < SINGLE_LIFT_LO
    single = (lf_high & rf_low) | (rf_high & lf_low)
    # count runs of length ≥ MIN_LEN
    n_runs = 0
    i = 0
    while i < len(single):
        if single[i]:
            j = i
            while j < len(single) and single[j]:
                j += 1
            run_len = j - i
            if run_len >= SINGLE_LIFT_MIN_LEN:
                n_runs += 1
            i = j
        else:
            i += 1
    return b1, n_runs


def evaluate_one(cfg, npz_path: Path, label: str):
    if not npz_path.exists():
        print(f"[!] missing {npz_path}")
        return None
    qpos_raw = load_qpos(npz_path)
    scene_xml = pick_scene_for_qpos(cfg["case_dir"], qpos_raw.shape[1])
    print(f"[{label}] using scene = {scene_xml.name}")
    m, d, L_sid, R_sid, LF_sid, RF_sid, obj_bid, half = load_for_scene(scene_xml)
    qpos = adapt_qpos(qpos_raw, m.nq)
    obj_xyz_ref = load_obj_ref_from_result(npz_path)
    lf_ref_z, rf_ref_z, pelvis_ref_z = load_foot_ref_from_dataset(cfg["case_dir"])
    L_s, R_s, pelvis_z, obj_xyz_sim, lf_z, rf_z = compute_per_frame(
        qpos, m, d, L_sid, R_sid, LF_sid, RF_sid, obj_bid, half)
    T = qpos.shape[0]
    ratio = T / cfg["src_T"]
    intent_lo = int(round(cfg["intent_src"][0] * ratio))
    intent_hi = int(round(cfg["intent_src"][1] * ratio))
    in_intent = slice(intent_lo, intent_hi + 1)
    print(f"[{label}] T={T}  intent (rescaled) = ({intent_lo}, {intent_hi})")

    L_min = np.abs(L_s).min(axis=1)
    R_min = np.abs(R_s).min(axis=1)
    L_contact = (L_min[in_intent] <= CONTACT_DIST).mean()
    R_contact = (R_min[in_intent] <= CONTACT_DIST).mean()
    both_contact = ((L_min[in_intent] <= CONTACT_DIST)
                    & (R_min[in_intent] <= CONTACT_DIST)).mean()
    stable_intent = (pelvis_z[in_intent] >= PELVIS_OK).mean()
    L_face, L_med, L_pct = find_main_face(L_s[in_intent])
    R_face, R_med, R_pct = find_main_face(R_s[in_intent])

    obj_err = np.linalg.norm(obj_xyz_sim - obj_xyz_ref, axis=1)
    final_pos_err = float(obj_err[-1])

    b1, b2 = compute_b1_b2(lf_z, rf_z)

    return {
        "label": label,
        "T": T, "intent": (intent_lo, intent_hi),
        "L_contact_pct": L_contact, "R_contact_pct": R_contact,
        "both_contact_pct": both_contact,
        "stable_intent_pct": stable_intent,
        "pelvis_min": float(pelvis_z.min()),
        "pelvis_min_intent": float(pelvis_z[in_intent].min()),
        "pelvis_mean_intent": float(pelvis_z[in_intent].mean()),
        "L_main_face": L_face, "L_med_cm": L_med, "L_close_pct": L_pct,
        "R_main_face": R_face, "R_med_cm": R_med, "R_close_pct": R_pct,
        "final_obj_pos_err": final_pos_err,
        "obj_err_mean": float(obj_err.mean()),
        "B1_pre_contact_max_foot_z": b1,
        "B2_single_foot_runs": b2,
        "lf_z_sim": lf_z, "rf_z_sim": rf_z,
        "lf_z_ref": lf_ref_z, "rf_z_ref": rf_ref_z,
        "pelvis_ref_z": pelvis_ref_z,
        "pelvis_z": pelvis_z,
        "obj_xyz_ref": obj_xyz_ref,
        "obj_xyz_sim": obj_xyz_sim,
    }


def render_foot_z_trace(row, out: Path):
    """Plot Lf_z and Rf_z (sim vs ref) over time. Highlight pre-contact window."""
    if row is None:
        return
    T = row["T"]
    x = np.arange(T) / FPS
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(x, row["lf_z_sim"], label="sim Lf_z", color="#1f77b4", lw=1.5)
    axes[0].plot(x, row["lf_z_ref"], label="ref Lf_z", color="#1f77b4", lw=1.0, ls="--", alpha=0.7)
    axes[0].plot(x, row["rf_z_sim"], label="sim Rf_z", color="#d62728", lw=1.5)
    axes[0].plot(x, row["rf_z_ref"], label="ref Rf_z", color="#d62728", lw=1.0, ls="--", alpha=0.7)
    axes[0].axhline(PRE_CONTACT_FOOT_MAX, color="orange", lw=0.8, ls="--",
                    label=f"B1 thresh {PRE_CONTACT_FOOT_MAX}m")
    axes[0].axvspan(0, PRE_CONTACT_T_END, color="green", alpha=0.08, label="pre-contact window")
    axes[0].set_ylabel("foot z (m)")
    axes[0].set_title(
        f"{row['label']} — B1={row['B1_pre_contact_max_foot_z']:.3f}m "
        f"(thresh ≤ {PRE_CONTACT_FOOT_MAX}m), B2={row['B2_single_foot_runs']} runs"
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8, ncol=3)

    axes[1].plot(x, row["pelvis_z"], label="sim pelvis_z", color="black", lw=1.5)
    axes[1].axhline(PELVIS_OK, color="red", lw=0.8, ls="--", label=f"PELVIS_OK={PELVIS_OK}m")
    axes[1].axvspan(row["intent"][0]/FPS, row["intent"][1]/FPS, color="orange", alpha=0.10)
    axes[1].set_ylabel("pelvis z (m)")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    fig.tight_layout()
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {out}")


def render_pelvis_obj(rows_for_case, out: Path, title: str):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for r in rows_for_case:
        if r is None:
            continue
        T = r["T"]
        x = np.arange(T) / FPS
        axes[0].plot(x, r["pelvis_z"], label=r["label"], lw=1.5)
        obj_err = np.linalg.norm(r["obj_xyz_sim"] - r["obj_xyz_ref"], axis=1)
        axes[1].plot(x, obj_err * 100, label=r["label"], lw=1.5)
        axes[0].axvspan(r["intent"][0]/FPS, r["intent"][1]/FPS, color="orange", alpha=0.10)
        axes[1].axvspan(r["intent"][0]/FPS, r["intent"][1]/FPS, color="orange", alpha=0.10)
    axes[0].axhline(PELVIS_OK, color="red", lw=0.8, ls="--", label=f"PELVIS_OK={PELVIS_OK}m")
    axes[0].set_ylabel("pelvis_z (m)")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].set_ylabel("obj pos err (cm)")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    fig.tight_layout()
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {out}")


def main() -> None:
    e065_rows = []
    all_rows_per_case = {}  # case name → [baseline row, e065 row]
    for cfg in CASES:
        baseline_npz = E063_RESULTS / f"{cfg['baseline_name']}.npz"
        e065_npz = RESULTS / f"{cfg['name']}.npz"
        baseline_row = evaluate_one(cfg, baseline_npz, label=f"{cfg['baseline_name']} (baseline)")
        e065_row = evaluate_one(cfg, e065_npz, label=cfg["name"])
        if e065_row is not None:
            e065_rows.append((cfg, e065_row))
        all_rows_per_case[cfg["name"]] = [baseline_row, e065_row]

    if not e065_rows:
        print("No E065 npz to evaluate.")
        return

    csv_path = RESULTS / "eval_summary.csv"
    cols = [
        "label", "T", "intent_start", "intent_end",
        "L_contact_pct", "R_contact_pct", "both_contact_pct",
        "stable_intent_pct",
        "pelvis_min", "pelvis_min_intent", "pelvis_mean_intent",
        "L_main_face", "L_med_cm", "L_close_pct",
        "R_main_face", "R_med_cm", "R_close_pct",
        "final_obj_pos_err_cm", "obj_err_mean_cm",
        "B1_pre_contact_max_foot_z_m", "B2_single_foot_runs",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for case_rows in all_rows_per_case.values():
            for r in case_rows:
                if r is None:
                    continue
                w.writerow([
                    r["label"], r["T"], r["intent"][0], r["intent"][1],
                    f"{r['L_contact_pct']:.3f}", f"{r['R_contact_pct']:.3f}",
                    f"{r['both_contact_pct']:.3f}",
                    f"{r['stable_intent_pct']:.3f}",
                    f"{r['pelvis_min']:.3f}", f"{r['pelvis_min_intent']:.3f}",
                    f"{r['pelvis_mean_intent']:.3f}",
                    r["L_main_face"] or "", f"{r['L_med_cm']:+.2f}", f"{r['L_close_pct']:.2f}",
                    r["R_main_face"] or "", f"{r['R_med_cm']:+.2f}", f"{r['R_close_pct']:.2f}",
                    f"{r['final_obj_pos_err']*100:.2f}",
                    f"{r['obj_err_mean']*100:.2f}",
                    f"{r['B1_pre_contact_max_foot_z']:.3f}",
                    r["B2_single_foot_runs"],
                ])
    print(f"\n→ {csv_path}")

    for cfg, r in e065_rows:
        render_foot_z_trace(r, RESULTS / f"foot_z_trace_{r['label']}.png")
        render_pelvis_obj(all_rows_per_case[cfg["name"]],
                          RESULTS / f"pelvis_obj_{r['label']}.png",
                          title=f"{cfg['name']}: pelvis_z + obj err vs time (baseline vs E065-{cfg['variant']})")

    # Pretty table
    print()
    pairs = []
    for case_rows in all_rows_per_case.values():
        for r in case_rows:
            if r is not None:
                pairs.append(r)
    print(f"{'metric':<32} " + " ".join(f"{r['label'][:30]:>30}" for r in pairs))
    print("-" * (33 + 31 * len(pairs)))
    for k, label, scale, unit in [
        ("B1_pre_contact_max_foot_z", "B1 max foot_z [0-2s] (m)", 1, ""),
        ("B2_single_foot_runs", "B2 single-foot runs (#)", 1, ""),
        ("pelvis_min_intent", "C1 pelvis_min_intent (m)", 1, ""),
        ("pelvis_mean_intent", "C2 pelvis_mean_intent (m)", 1, ""),
        ("stable_intent_pct", "C3 stable% intent", 100, "%"),
        ("final_obj_pos_err", "C4 final_obj_pos_err (cm)", 100, ""),
        ("L_contact_pct", "L palm contact %", 100, "%"),
        ("R_contact_pct", "R palm contact %", 100, "%"),
    ]:
        vals = []
        for r in pairs:
            v = r[k]
            if isinstance(v, (int, np.integer)) and scale == 1:
                vals.append(f"{v:>29d}")
            else:
                vals.append(f"{v*scale:>29.2f}{unit}")
        print(f"{label:<32} " + " ".join(vals))

    # Pass criteria for each E065 variant on box023
    print()
    print("=" * 70)
    print("E066 PASS criteria (log 83 §4.3):")
    for cfg, r in e065_rows:
        c4_thresh = 30.0 if cfg["variant"] == "A" else 20.0  # A drops task_obj
        b1_pass = r["B1_pre_contact_max_foot_z"] <= PRE_CONTACT_FOOT_MAX
        b2_pass = r["B2_single_foot_runs"] == 0
        c1 = r["pelvis_min_intent"] >= 0.50
        c2 = r["pelvis_mean_intent"] >= 0.50
        c3 = r["stable_intent_pct"] >= 0.80
        c4 = (r["final_obj_pos_err"] * 100) <= c4_thresh
        verdict_main = b1_pass and c1
        print(f"  [{cfg['name']}] B1 max foot_z {r['B1_pre_contact_max_foot_z']:.3f}m ≤ {PRE_CONTACT_FOOT_MAX}: {b1_pass}")
        print(f"  [{cfg['name']}] B2 single-foot runs {r['B2_single_foot_runs']} == 0: {b2_pass}")
        print(f"  [{cfg['name']}] C1 pelvis_min_intent {r['pelvis_min_intent']:.3f}m ≥ 0.50: {c1}")
        print(f"  [{cfg['name']}] C2 pelvis_mean_intent {r['pelvis_mean_intent']:.3f}m ≥ 0.50: {c2}")
        print(f"  [{cfg['name']}] C3 stable% intent {r['stable_intent_pct']*100:.1f}% ≥ 80: {c3}")
        print(f"  [{cfg['name']}] C4 final obj err {r['final_obj_pos_err']*100:.2f}cm ≤ {c4_thresh}: {c4}")
        print(f"  [{cfg['name']}] verdict (B1+C1 main): {'PASS' if verdict_main else 'FAIL'} "
              f"(C5 vis check separately)")
        print()


if __name__ == "__main__":
    main()
