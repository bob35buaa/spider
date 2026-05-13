#!/usr/bin/env python3
"""E058 evaluation: compare baseline vs warmstart CEM rollouts on bucket005_s2.

Reads:
  workspace/core4d/results/E058/E058_baseline.npz
  workspace/core4d/results/E058/E058_warm.npz

Computes per-trajectory:
  - contact% (intent: palm-bucket distance < 5cm fraction)
  - stability% (pelvis_z >= 0.5m fraction)
  - main face L/R inside intent window (matches E057 verify_snap_face logic)
  - pelvis_min, pelvis_min_intent

Outputs:
  workspace/core4d/results/E058/eval_summary.csv     (1 row per trajectory)
  workspace/core4d/results/E058/face_dist_E058.png   (2x2 face dist time-series)
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[4]
CASE = "bucket005_s2_person1"
SCENE = REPO / f"example_datasets/processed/core4d/unitree_g1/humanoid_object/{CASE}/scene.xml"
RESULTS = REPO / "workspace/core4d/results/E058"

INTENT = (20, 107)               # source-resolution intent window from E056/E057
PELVIS_OK = 0.50                 # pelvis_z stability threshold (m)
CONTACT_DIST = 0.05              # palm-to-bucket-surface contact threshold (m)
FACE_NAMES = ["+yz", "-yz", "+xz", "-xz", "+xy", "-xy"]
FACE_COLORS = ["#d62728", "#ff9896", "#2ca02c", "#98df8a", "#1f77b4", "#9ecae1"]
MAIN_FACE_DIST = 0.07
FACE_STABILITY = 0.60
EXPECTED_L = "-yz"
EXPECTED_R = "+yz"


def load_qpos(npz_path: Path) -> np.ndarray:
    """Load qpos from CEM-rollout npz. Shape is (T, n_dyn, nq) where n_dyn=2
    holds the [physics, kinematic] envs; we take the physics one."""
    d = np.load(npz_path, allow_pickle=True)
    if "qpos" not in d.files:
        raise KeyError(f"No qpos key in {npz_path} (keys: {d.files})")
    qpos = d["qpos"]
    if qpos.ndim == 3:
        qpos = qpos[:, 0, :]   # take physics env
    return qpos


def load_for_scene(scene_xml: Path):
    """Build mj model+data + grab the bucket BOX collision half-sizes."""
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    d = mujoco.MjData(m)
    L_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_palm")
    R_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
    obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    half = None
    for gid in range(m.ngeom):
        if m.geom_bodyid[gid] == obj_bid and m.geom_contype[gid] != 0 \
                and m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
            half = m.geom_size[gid].copy()
            break
    if half is None:
        raise RuntimeError(f"No BOX collision geom on object body in {scene_xml}")
    return m, d, L_sid, R_sid, obj_bid, half


def adapt_qpos(qpos: np.ndarray, model_nq: int) -> np.ndarray:
    """CEM rollout qpos may be 42-dim (scene_act, object as euler). Convert to 43-dim
    quat layout by inserting object orientation as quaternion. We use scene.xml
    (43-dim freejoint) for analysis, so need to recover quat from euler."""
    if qpos.shape[1] == model_nq:
        return qpos
    if qpos.shape[1] == model_nq - 1:  # 42 -> 43 (euler -> quat)
        from scipy.spatial.transform import Rotation as R
        nq_robot = 36                  # 7 pelvis + 29 G1
        out = np.zeros((qpos.shape[0], model_nq), dtype=qpos.dtype)
        out[:, :nq_robot] = qpos[:, :nq_robot]
        out[:, nq_robot : nq_robot + 3] = qpos[:, nq_robot : nq_robot + 3]
        eulers = qpos[:, nq_robot + 3 : nq_robot + 6]
        quats_xyzw = R.from_euler("xyz", eulers).as_quat()
        # mujoco uses wxyz
        out[:, nq_robot + 3] = quats_xyzw[:, 3]
        out[:, nq_robot + 4 : nq_robot + 7] = quats_xyzw[:, :3]
        return out
    raise ValueError(f"Cannot adapt qpos shape {qpos.shape} to model_nq={model_nq}")


def compute_face_distances(qpos: np.ndarray, m, d, L_sid, R_sid, obj_bid, half):
    T = qpos.shape[0]
    L_signed = np.zeros((T, 6))
    R_signed = np.zeros((T, 6))
    pelvis_z = np.zeros(T)
    for t in range(T):
        d.qpos[:] = qpos[t]
        mujoco.mj_forward(m, d)
        obj_pos = d.xpos[obj_bid].copy()
        obj_R = d.xmat[obj_bid].copy().reshape(3, 3)
        pelvis_z[t] = qpos[t, 2]
        for sid, store in [(L_sid, L_signed), (R_sid, R_signed)]:
            local = obj_R.T @ (d.site_xpos[sid] - obj_pos)
            store[t] = [
                local[0] - half[0], -local[0] - half[0],
                local[1] - half[1], -local[1] - half[1],
                local[2] - half[2], -local[2] - half[2],
            ]
    return L_signed, R_signed, pelvis_z


def find_main_face(signed_in_window: np.ndarray):
    abs_dist = np.abs(signed_in_window)
    med = np.median(abs_dist, axis=0)
    frac = (abs_dist <= MAIN_FACE_DIST).mean(axis=0)
    order = np.argsort(med)
    for idx in order:
        if frac[idx] >= FACE_STABILITY and med[idx] <= MAIN_FACE_DIST:
            return FACE_NAMES[idx], float(med[idx]*100), float(frac[idx])
    return None, float(med[order[0]]*100), float(frac[order[0]])


def evaluate_one(name: str, npz_path: Path, m, d, L_sid, R_sid, obj_bid, half):
    qpos_raw = load_qpos(npz_path)
    qpos = adapt_qpos(qpos_raw, m.nq)
    print(f"[{name}] qpos shape: {qpos_raw.shape} -> adapted {qpos.shape}")
    L_s, R_s, pelvis_z = compute_face_distances(qpos, m, d, L_sid, R_sid, obj_bid, half)
    T = qpos.shape[0]
    # CEM trajectory may be 2x source-rate (sim_dt < ref_dt); rescale intent boundaries
    src_T = 148
    ratio = T / src_T
    intent_lo = int(round(INTENT[0] * ratio))
    intent_hi = int(round(INTENT[1] * ratio))
    print(f"[{name}] T={T}, intent (rescaled) = ({intent_lo}, {intent_hi})")
    # palm-to-surface = max negative signed dist? Use min |signed|
    L_min = np.abs(L_s).min(axis=1)
    R_min = np.abs(R_s).min(axis=1)
    in_intent = slice(intent_lo, intent_hi + 1)
    L_contact = (L_min[in_intent] <= CONTACT_DIST).mean()
    R_contact = (R_min[in_intent] <= CONTACT_DIST).mean()
    both_contact = ((L_min[in_intent] <= CONTACT_DIST) & (R_min[in_intent] <= CONTACT_DIST)).mean()
    stable = (pelvis_z >= PELVIS_OK).mean()
    stable_intent = (pelvis_z[in_intent] >= PELVIS_OK).mean()
    L_face, L_med, L_pct = find_main_face(L_s[in_intent])
    R_face, R_med, R_pct = find_main_face(R_s[in_intent])
    return {
        "name": name, "T": T, "intent": (intent_lo, intent_hi),
        "L_contact_pct": L_contact, "R_contact_pct": R_contact,
        "both_contact_pct": both_contact,
        "stable_pct": stable, "stable_intent_pct": stable_intent,
        "pelvis_min": float(pelvis_z.min()),
        "pelvis_min_intent": float(pelvis_z[in_intent].min()),
        "L_main_face": L_face, "L_med_cm": L_med, "L_close_pct": L_pct,
        "R_main_face": R_face, "R_med_cm": R_med, "R_close_pct": R_pct,
        "L_signed": L_s, "R_signed": R_s,
    }, qpos


def render_face_dist(rows, intent_pairs, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=False)
    panels = [
        ("BASELINE L", rows[0]["L_signed"], rows[0]["L_main_face"], intent_pairs[0], axes[0, 0]),
        ("BASELINE R", rows[0]["R_signed"], rows[0]["R_main_face"], intent_pairs[0], axes[0, 1]),
        ("WARM    L", rows[1]["L_signed"], rows[1]["L_main_face"], intent_pairs[1], axes[1, 0]),
        ("WARM    R", rows[1]["R_signed"], rows[1]["R_main_face"], intent_pairs[1], axes[1, 1]),
    ]
    for title, signed, main_face, intent, ax in panels:
        for i, fname in enumerate(FACE_NAMES):
            highlight = fname == main_face
            ax.plot(signed[:, i] * 100, label=fname, color=FACE_COLORS[i],
                    lw=3 if highlight else 1.2,
                    ls="-" if fname[0] == "+" else "--")
        ax.axhline(0, color="black", lw=0.5)
        ax.axvspan(intent[0], intent[1], color="orange", alpha=0.15)
        ax.set_ylabel("signed dist (cm)")
        ax.set_title(f"{title}  main = {main_face}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7, ncol=4)
        ax.set_ylim(-50, 30)
    axes[1, 0].set_xlabel("frame")
    axes[1, 1].set_xlabel("frame")
    fig.tight_layout()
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {out}")


def main() -> None:
    m, d, L_sid, R_sid, obj_bid, half = load_for_scene(SCENE)
    print(f"box half = ({half[0]*100:.1f}, {half[1]*100:.1f}, {half[2]*100:.1f}) cm")

    rows = []
    for name in ("E058_baseline", "E058_warm"):
        npz_path = RESULTS / f"{name}.npz"
        if not npz_path.exists():
            print(f"[!] missing {npz_path}, skipping")
            continue
        row, _ = evaluate_one(name, npz_path, m, d, L_sid, R_sid, obj_bid, half)
        rows.append(row)

    if not rows:
        print("No npz to evaluate.")
        return

    csv_path = RESULTS / "eval_summary.csv"
    cols = [
        "name", "T", "intent_start", "intent_end",
        "L_contact_pct", "R_contact_pct", "both_contact_pct",
        "stable_pct", "stable_intent_pct",
        "pelvis_min", "pelvis_min_intent",
        "L_main_face", "L_med_cm", "L_close_pct",
        "R_main_face", "R_med_cm", "R_close_pct",
        "expected_L", "expected_R", "L_match", "R_match",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([
                r["name"], r["T"], r["intent"][0], r["intent"][1],
                f"{r['L_contact_pct']:.3f}", f"{r['R_contact_pct']:.3f}",
                f"{r['both_contact_pct']:.3f}",
                f"{r['stable_pct']:.3f}", f"{r['stable_intent_pct']:.3f}",
                f"{r['pelvis_min']:.3f}", f"{r['pelvis_min_intent']:.3f}",
                r["L_main_face"] or "", f"{r['L_med_cm']:+.2f}", f"{r['L_close_pct']:.2f}",
                r["R_main_face"] or "", f"{r['R_med_cm']:+.2f}", f"{r['R_close_pct']:.2f}",
                EXPECTED_L, EXPECTED_R,
                str(r["L_main_face"] == EXPECTED_L), str(r["R_main_face"] == EXPECTED_R),
            ])
    print(f"\n→ {csv_path}")

    if len(rows) == 2:
        render_face_dist(rows, [r["intent"] for r in rows], RESULTS / "face_dist_E058.png")

    # Print a console summary
    print()
    print(f"{'metric':<28} {'baseline':>14} {'warm':>14} {'Δ':>10}")
    print("-" * 70)
    for k, label in [
        ("L_contact_pct", "L palm contact %"),
        ("R_contact_pct", "R palm contact %"),
        ("both_contact_pct", "both palm contact %"),
        ("stable_pct", "stable % (full)"),
        ("stable_intent_pct", "stable % (intent)"),
        ("pelvis_min", "pelvis_min (m)"),
        ("pelvis_min_intent", "pelvis_min_intent (m)"),
    ]:
        b = rows[0][k] if len(rows) >= 1 else None
        w = rows[1][k] if len(rows) >= 2 else None
        if b is None or w is None:
            continue
        scale = 100 if k.endswith("_pct") else 1
        unit = "%" if k.endswith("_pct") else ""
        print(f"{label:<28} {b*scale:>13.2f}{unit} {w*scale:>13.2f}{unit} {(w-b)*scale:>+9.2f}{unit}")
    if len(rows) == 2:
        print(f"{'L main face':<28} {str(rows[0]['L_main_face']):>14} {str(rows[1]['L_main_face']):>14}")
        print(f"{'R main face':<28} {str(rows[0]['R_main_face']):>14} {str(rows[1]['R_main_face']):>14}")
        print(f"{'expected':<28} {EXPECTED_L:>14}/{EXPECTED_R} (L/R)")


if __name__ == "__main__":
    main()
