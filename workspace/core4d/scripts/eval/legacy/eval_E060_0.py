#!/usr/bin/env python3
"""E060.0 evaluation: baseline (E041c, no warmstart) on box023 + bucket005_s2
after data-layer fix (3-box hand + box023 margin 0.90).

Reads:
  workspace/core4d/results/E060/E060_0_box023.npz
  workspace/core4d/results/E060/E060_0_bucket005_s2.npz

Computes per-trajectory:
  - L/R/both palm contact% (within intent window)
  - stability% (full + intent)
  - main face L/R (E057 verify_snap_face logic)
  - pelvis_min, pelvis_min_intent

Outputs:
  workspace/core4d/results/E060/eval_summary.csv      (1 row per trajectory)
  workspace/core4d/results/E060/face_dist_E060_box023.png
  workspace/core4d/results/E060/face_dist_E060_bucket005_s2.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[4]
RESULTS = REPO / "workspace/core4d/results/E060"

# Per-case config: case_dir, intent (source frames), expected main faces.
# intent + expected from E055/E056 grasp diagnosis.
CASES = [
    {
        "name": "E060_0_box023",
        "case_dir": "box023_person1",
        "intent_src": (21, 78),    # 58 frames; src_T=148? actually ref is 136
        "src_T": 136,
        "expected_L": "-xy",       # 垂直 grasp: L 在 box 底面
        "expected_R": "-yz",       # R 在远侧
    },
    {
        "name": "E060_0_bucket005_s2",
        "case_dir": "bucket005_s2_person1",
        "intent_src": (19, 107),   # 88 frames; from E057 plan
        "src_T": 148,
        "expected_L": "-yz",       # 对侧 grasp
        "expected_R": "+yz",
    },
]

PELVIS_OK = 0.50
CONTACT_DIST = 0.05
FACE_NAMES = ["+yz", "-yz", "+xz", "-xz", "+xy", "-xy"]
FACE_COLORS = ["#d62728", "#ff9896", "#2ca02c", "#98df8a", "#1f77b4", "#9ecae1"]
MAIN_FACE_DIST = 0.07
FACE_STABILITY = 0.60


def load_qpos(npz_path: Path) -> np.ndarray:
    d = np.load(npz_path, allow_pickle=True)
    if "qpos" not in d.files:
        raise KeyError(f"No qpos key in {npz_path} (keys: {d.files})")
    qpos = d["qpos"]
    if qpos.ndim == 3:
        qpos = qpos[:, 0, :]
    return qpos


def load_for_scene(scene_xml: Path):
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    d = mujoco.MjData(m)
    L_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_palm")
    R_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
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
    return m, d, L_sid, R_sid, obj_bid, half


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


def compute_face_distances(qpos, m, d, L_sid, R_sid, obj_bid, half):
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


def find_main_face(signed_in_window):
    abs_dist = np.abs(signed_in_window)
    med = np.median(abs_dist, axis=0)
    frac = (abs_dist <= MAIN_FACE_DIST).mean(axis=0)
    order = np.argsort(med)
    for idx in order:
        if frac[idx] >= FACE_STABILITY and med[idx] <= MAIN_FACE_DIST:
            return FACE_NAMES[idx], float(med[idx] * 100), float(frac[idx])
    return None, float(med[order[0]] * 100), float(frac[order[0]])


def evaluate_one(cfg):
    scene = REPO / f"example_datasets/processed/core4d/unitree_g1/humanoid_object/{cfg['case_dir']}/scene.xml"
    npz = RESULTS / f"{cfg['name']}.npz"
    if not npz.exists():
        print(f"[!] missing {npz}")
        return None
    m, d, L_sid, R_sid, obj_bid, half = load_for_scene(scene)
    print(f"[{cfg['name']}] box half = ({half[0]*100:.1f}, {half[1]*100:.1f}, {half[2]*100:.1f}) cm")
    qpos_raw = load_qpos(npz)
    qpos = adapt_qpos(qpos_raw, m.nq)
    L_s, R_s, pelvis_z = compute_face_distances(qpos, m, d, L_sid, R_sid, obj_bid, half)
    T = qpos.shape[0]
    ratio = T / cfg["src_T"]
    intent_lo = int(round(cfg["intent_src"][0] * ratio))
    intent_hi = int(round(cfg["intent_src"][1] * ratio))
    print(f"[{cfg['name']}] T={T}  intent (rescaled) = ({intent_lo}, {intent_hi})")

    L_min = np.abs(L_s).min(axis=1)
    R_min = np.abs(R_s).min(axis=1)
    in_intent = slice(intent_lo, intent_hi + 1)
    L_contact = (L_min[in_intent] <= CONTACT_DIST).mean()
    R_contact = (R_min[in_intent] <= CONTACT_DIST).mean()
    both_contact = ((L_min[in_intent] <= CONTACT_DIST)
                    & (R_min[in_intent] <= CONTACT_DIST)).mean()
    stable = (pelvis_z >= PELVIS_OK).mean()
    stable_intent = (pelvis_z[in_intent] >= PELVIS_OK).mean()
    L_face, L_med, L_pct = find_main_face(L_s[in_intent])
    R_face, R_med, R_pct = find_main_face(R_s[in_intent])
    return {
        "name": cfg["name"], "T": T, "intent": (intent_lo, intent_hi),
        "L_contact_pct": L_contact, "R_contact_pct": R_contact,
        "both_contact_pct": both_contact,
        "stable_pct": stable, "stable_intent_pct": stable_intent,
        "pelvis_min": float(pelvis_z.min()),
        "pelvis_min_intent": float(pelvis_z[in_intent].min()),
        "L_main_face": L_face, "L_med_cm": L_med, "L_close_pct": L_pct,
        "R_main_face": R_face, "R_med_cm": R_med, "R_close_pct": R_pct,
        "expected_L": cfg["expected_L"], "expected_R": cfg["expected_R"],
        "L_signed": L_s, "R_signed": R_s,
    }


def render_face_dist(row, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    panels = [
        (f"{row['name']} L (expected {row['expected_L']})",
         row["L_signed"], row["L_main_face"], axes[0]),
        (f"{row['name']} R (expected {row['expected_R']})",
         row["R_signed"], row["R_main_face"], axes[1]),
    ]
    for title, signed, main_face, ax in panels:
        for i, fname in enumerate(FACE_NAMES):
            highlight = fname == main_face
            ax.plot(signed[:, i] * 100, label=fname, color=FACE_COLORS[i],
                    lw=3 if highlight else 1.2,
                    ls="-" if fname[0] == "+" else "--")
        ax.axhline(0, color="black", lw=0.5)
        ax.axvspan(row["intent"][0], row["intent"][1], color="orange", alpha=0.15)
        ax.set_ylabel("signed dist (cm)")
        ax.set_xlabel("frame")
        ax.set_title(f"{title}  main = {main_face}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7, ncol=4)
        ax.set_ylim(-50, 30)
    fig.tight_layout()
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {out}")


def main() -> None:
    rows = []
    for cfg in CASES:
        row = evaluate_one(cfg)
        if row is not None:
            rows.append((cfg, row))
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
        "pelvis_min_intent_pass",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for _cfg, r in rows:
            pmi_pass = r["pelvis_min_intent"] >= 0.40
            w.writerow([
                r["name"], r["T"], r["intent"][0], r["intent"][1],
                f"{r['L_contact_pct']:.3f}", f"{r['R_contact_pct']:.3f}",
                f"{r['both_contact_pct']:.3f}",
                f"{r['stable_pct']:.3f}", f"{r['stable_intent_pct']:.3f}",
                f"{r['pelvis_min']:.3f}", f"{r['pelvis_min_intent']:.3f}",
                r["L_main_face"] or "", f"{r['L_med_cm']:+.2f}", f"{r['L_close_pct']:.2f}",
                r["R_main_face"] or "", f"{r['R_med_cm']:+.2f}", f"{r['R_close_pct']:.2f}",
                r["expected_L"], r["expected_R"],
                str(r["L_main_face"] == r["expected_L"]),
                str(r["R_main_face"] == r["expected_R"]),
                str(pmi_pass),
            ])
    print(f"\n→ {csv_path}")

    for _cfg, r in rows:
        out = RESULTS / f"face_dist_{r['name']}.png"
        render_face_dist(r, out)

    # Summary table
    print()
    print(f"{'metric':<28} " + " ".join(f"{r['name']:>22}" for _, r in rows))
    print("-" * (29 + 23 * len(rows)))
    for k, label, scale, unit in [
        ("L_contact_pct", "L palm contact %", 100, "%"),
        ("R_contact_pct", "R palm contact %", 100, "%"),
        ("both_contact_pct", "both palm contact %", 100, "%"),
        ("stable_pct", "stable % (full)", 100, "%"),
        ("stable_intent_pct", "stable % (intent)", 100, "%"),
        ("pelvis_min", "pelvis_min (m)", 1, ""),
        ("pelvis_min_intent", "pelvis_min_intent (m)", 1, ""),
    ]:
        vals = [f"{r[k]*scale:>21.2f}{unit}" for _, r in rows]
        print(f"{label:<28} " + " ".join(vals))
    print(f"{'L main face':<28} " + " ".join(f"{str(r['L_main_face']):>22}" for _, r in rows))
    print(f"{'R main face':<28} " + " ".join(f"{str(r['R_main_face']):>22}" for _, r in rows))
    print(f"{'expected (L/R)':<28} " + " ".join(
        f"{r['expected_L']+'/'+r['expected_R']:>22}" for _, r in rows))
    print()
    print("E060.0 PASS criteria:")
    for _cfg, r in rows:
        pmi_pass = r["pelvis_min_intent"] >= 0.40
        face_pass = (r["L_main_face"] == r["expected_L"]
                     and r["R_main_face"] == r["expected_R"])
        verdict = "PASS" if (pmi_pass and face_pass) else (
            "PARTIAL (站住但握姿错)" if pmi_pass else "FAIL (摔)")
        print(f"  {r['name']}: pelvis_min_intent={r['pelvis_min_intent']:.3f}m "
              f"(>=0.40 = {pmi_pass}), face_match={face_pass} → {verdict}")


if __name__ == "__main__":
    main()
