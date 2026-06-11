#!/usr/bin/env python3
"""E063 evaluation: Tier 1 (stability_penalty=1.0 + task_obj_rew=0.5) vs E062.

Reads:
  workspace/core4d/results/E063/E063_box023.npz
  workspace/core4d/results/E063/E063_box025.npz
  (and E062 npz from workspace/core4d/results/E062/ for diff vs baseline)

Computes per-trajectory:
  - L/R/both palm contact% (within intent window)
  - stability% (full + intent)
  - main face L/R
  - pelvis_min, pelvis_min_intent, pelvis_mean_intent
  - **NEW**: per-frame ref vs sim object xyz error, final pos/rot err

Outputs:
  workspace/core4d/results/E063/eval_summary.csv
  workspace/core4d/results/E063/obj_trace_E063_box023.csv  (full ref vs sim trace)
  workspace/core4d/results/E063/obj_trace_E063_box025.csv
  workspace/core4d/results/E063/face_dist_E063_box023.png
  workspace/core4d/results/E063/face_dist_E063_box025.png
  workspace/core4d/results/E063/pelvis_z_E063_box023.png
  workspace/core4d/results/E063/pelvis_z_E063_box025.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[4]
RESULTS = REPO / "workspace/core4d/results/E063"
E062_RESULTS = REPO / "workspace/core4d/results/E062"

# Per-case config: (name, case_dir, intent_src, src_T, expected faces, e062_npz_name)
CASES = [
    {
        "name": "E063_box023",
        "case_dir": "box023_person1",
        "intent_src": (21, 78),
        "src_T": 136,
        "expected_L": "-xy",
        "expected_R": "-yz",
        "e062_name": "E062_box023_sphere_autopalm",
    },
    {
        "name": "E063_box025",
        "case_dir": "box025_person1",
        "intent_src": (20, 100),  # box025 motion ~3.3s, intent middle ~0.67-3.33s
        "src_T": 124,
        "expected_L": "-yz",
        "expected_R": "+yz",
        "e062_name": "E062_box025_sphere_autopalm",
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
    qpos = d["qpos"]
    if qpos.ndim == 3:
        qpos = qpos[:, -1, :]  # last substep
    return qpos


def load_trace_ref_obj(npz_path: Path) -> np.ndarray:
    """Return ref object xyz over time, shape (T, 3). body 0 = object, horizon idx 0."""
    d = np.load(npz_path, allow_pickle=True)
    tr = d["trace_ref"]
    return tr[:, 0, 0, 0, 0, :].copy()


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


def pick_scene_for_qpos(case_dir_name: str, qpos_nq: int) -> Path:
    """Choose scene XML matching qpos layout.

    spider/MJWP runs save qpos under scene_act.xml's joint structure
    (slider + hinge actuators on the object, nq=42). scene.xml uses a freejoint
    (nq=43). Loading nq=42 into the freejoint scene gives garbage object xpos.
    """
    base = REPO / f"example_datasets/processed/core4d/unitree_g1/humanoid_object/{case_dir_name}"
    if qpos_nq == 42 and (base / "scene_act.xml").exists():
        return base / "scene_act.xml"
    return base / "scene.xml"


def quat_angle_diff(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> float:
    """Angular distance (radians) between two quaternions in wxyz form."""
    dot = abs(float(np.dot(q1_wxyz, q2_wxyz)))
    dot = min(1.0, max(-1.0, dot))
    return 2.0 * np.arccos(dot)


def compute_per_frame(qpos, m, d, L_sid, R_sid, obj_bid, half):
    T = qpos.shape[0]
    L_signed = np.zeros((T, 6))
    R_signed = np.zeros((T, 6))
    pelvis_z = np.zeros(T)
    obj_xyz_sim = np.zeros((T, 3))
    obj_quat_sim = np.zeros((T, 4))  # wxyz
    for t in range(T):
        d.qpos[:] = qpos[t]
        mujoco.mj_forward(m, d)
        obj_pos = d.xpos[obj_bid].copy()
        obj_R = d.xmat[obj_bid].copy().reshape(3, 3)
        obj_xyz_sim[t] = obj_pos
        obj_quat_sim[t] = qpos[t, -4:]
        pelvis_z[t] = qpos[t, 2]
        for sid, store in [(L_sid, L_signed), (R_sid, R_signed)]:
            local = obj_R.T @ (d.site_xpos[sid] - obj_pos)
            store[t] = [
                local[0] - half[0], -local[0] - half[0],
                local[1] - half[1], -local[1] - half[1],
                local[2] - half[2], -local[2] - half[2],
            ]
    return L_signed, R_signed, pelvis_z, obj_xyz_sim, obj_quat_sim


def find_main_face(signed_in_window):
    abs_dist = np.abs(signed_in_window)
    med = np.median(abs_dist, axis=0)
    frac = (abs_dist <= MAIN_FACE_DIST).mean(axis=0)
    order = np.argsort(med)
    for idx in order:
        if frac[idx] >= FACE_STABILITY and med[idx] <= MAIN_FACE_DIST:
            return FACE_NAMES[idx], float(med[idx] * 100), float(frac[idx])
    return None, float(med[order[0]] * 100), float(frac[order[0]])


def evaluate_one(cfg, npz_path: Path, label: str):
    if not npz_path.exists():
        print(f"[!] missing {npz_path}")
        return None
    qpos_raw = load_qpos(npz_path)
    scene_xml = pick_scene_for_qpos(cfg["case_dir"], qpos_raw.shape[1])
    print(f"[{label}] using scene = {scene_xml.name}")
    m, d, L_sid, R_sid, obj_bid, half = load_for_scene(scene_xml)
    print(f"[{label}] box half = ({half[0]*100:.1f}, {half[1]*100:.1f}, {half[2]*100:.1f}) cm")
    qpos = adapt_qpos(qpos_raw, m.nq)
    obj_xyz_ref = load_trace_ref_obj(npz_path)
    L_s, R_s, pelvis_z, obj_xyz_sim, obj_quat_sim = compute_per_frame(
        qpos, m, d, L_sid, R_sid, obj_bid, half)
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
    stable = (pelvis_z >= PELVIS_OK).mean()
    stable_intent = (pelvis_z[in_intent] >= PELVIS_OK).mean()
    L_face, L_med, L_pct = find_main_face(L_s[in_intent])
    R_face, R_med, R_pct = find_main_face(R_s[in_intent])

    obj_err = np.linalg.norm(obj_xyz_sim - obj_xyz_ref, axis=1)
    final_pos_err = float(obj_err[-1])

    return {
        "label": label,
        "T": T, "intent": (intent_lo, intent_hi),
        "L_contact_pct": L_contact, "R_contact_pct": R_contact,
        "both_contact_pct": both_contact,
        "stable_pct": stable, "stable_intent_pct": stable_intent,
        "pelvis_min": float(pelvis_z.min()),
        "pelvis_min_intent": float(pelvis_z[in_intent].min()),
        "pelvis_mean_intent": float(pelvis_z[in_intent].mean()),
        "pelvis_max": float(pelvis_z.max()),
        "L_main_face": L_face, "L_med_cm": L_med, "L_close_pct": L_pct,
        "R_main_face": R_face, "R_med_cm": R_med, "R_close_pct": R_pct,
        "expected_L": cfg["expected_L"], "expected_R": cfg["expected_R"],
        "final_obj_pos_err": final_pos_err,
        "obj_err_mean": float(obj_err.mean()),
        "obj_err_max": float(obj_err.max()),
        "L_signed": L_s, "R_signed": R_s,
        "pelvis_z": pelvis_z,
        "obj_xyz_ref": obj_xyz_ref,
        "obj_xyz_sim": obj_xyz_sim,
    }


def render_face_dist(row, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    panels = [
        (f"{row['label']} L (expected {row['expected_L']})",
         row["L_signed"], row["L_main_face"], axes[0]),
        (f"{row['label']} R (expected {row['expected_R']})",
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


def render_pelvis_obj(rows_for_case, out: Path, title: str):
    """Two-panel: pelvis_z (E063 vs E062) + obj_err (E063 vs E062)."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for r in rows_for_case:
        if r is None:
            continue
        T = r["T"]
        x = np.arange(T) / 30.0
        axes[0].plot(x, r["pelvis_z"], label=r["label"], lw=1.5)
        obj_err = np.linalg.norm(r["obj_xyz_sim"] - r["obj_xyz_ref"], axis=1)
        axes[1].plot(x, obj_err * 100, label=r["label"], lw=1.5)
        axes[0].axvspan(r["intent"][0]/30.0, r["intent"][1]/30.0,
                        color="orange", alpha=0.10)
        axes[1].axvspan(r["intent"][0]/30.0, r["intent"][1]/30.0,
                        color="orange", alpha=0.10)
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


def write_obj_trace_csv(row, out: Path):
    """Per-frame ref vs sim obj xyz + err."""
    if row is None:
        return
    T = row["T"]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_s",
                    "ref_x", "ref_y", "ref_z",
                    "sim_x", "sim_y", "sim_z",
                    "err_cm", "pelvis_z"])
        for t in range(T):
            err = float(np.linalg.norm(row["obj_xyz_sim"][t] - row["obj_xyz_ref"][t]))
            w.writerow([t, f"{t/30.0:.3f}",
                        f"{row['obj_xyz_ref'][t][0]:.4f}",
                        f"{row['obj_xyz_ref'][t][1]:.4f}",
                        f"{row['obj_xyz_ref'][t][2]:.4f}",
                        f"{row['obj_xyz_sim'][t][0]:.4f}",
                        f"{row['obj_xyz_sim'][t][1]:.4f}",
                        f"{row['obj_xyz_sim'][t][2]:.4f}",
                        f"{err*100:.2f}",
                        f"{row['pelvis_z'][t]:.4f}"])
    print(f"[csv] wrote {out}")


def main() -> None:
    e063_rows = []
    all_rows_per_case = {}  # case name → [E062 row, E063 row]
    for cfg in CASES:
        e062_npz = E062_RESULTS / f"{cfg['e062_name']}.npz"
        e063_npz = RESULTS / f"{cfg['name']}.npz"
        e062_row = evaluate_one(cfg, e062_npz, label=f"{cfg['e062_name']} (E062)")
        e063_row = evaluate_one(cfg, e063_npz, label=cfg["name"])
        if e063_row is not None:
            e063_rows.append((cfg, e063_row))
        all_rows_per_case[cfg["name"]] = [e062_row, e063_row]

    if not e063_rows:
        print("No E063 npz to evaluate.")
        return

    # CSV summary
    csv_path = RESULTS / "eval_summary.csv"
    cols = [
        "label", "T", "intent_start", "intent_end",
        "L_contact_pct", "R_contact_pct", "both_contact_pct",
        "stable_pct", "stable_intent_pct",
        "pelvis_min", "pelvis_min_intent", "pelvis_mean_intent", "pelvis_max",
        "L_main_face", "L_med_cm", "L_close_pct",
        "R_main_face", "R_med_cm", "R_close_pct",
        "final_obj_pos_err_cm", "obj_err_mean_cm", "obj_err_max_cm",
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
                    f"{r['stable_pct']:.3f}", f"{r['stable_intent_pct']:.3f}",
                    f"{r['pelvis_min']:.3f}", f"{r['pelvis_min_intent']:.3f}",
                    f"{r['pelvis_mean_intent']:.3f}", f"{r['pelvis_max']:.3f}",
                    r["L_main_face"] or "", f"{r['L_med_cm']:+.2f}", f"{r['L_close_pct']:.2f}",
                    r["R_main_face"] or "", f"{r['R_med_cm']:+.2f}", f"{r['R_close_pct']:.2f}",
                    f"{r['final_obj_pos_err']*100:.2f}",
                    f"{r['obj_err_mean']*100:.2f}",
                    f"{r['obj_err_max']*100:.2f}",
                ])
    print(f"\n→ {csv_path}")

    # face plots, pelvis/obj plots, obj trace csv
    for cfg, r in e063_rows:
        render_face_dist(r, RESULTS / f"face_dist_{r['label']}.png")
        write_obj_trace_csv(r, RESULTS / f"obj_trace_{r['label']}.csv")
        render_pelvis_obj(all_rows_per_case[cfg["name"]],
                          RESULTS / f"pelvis_obj_{r['label']}.png",
                          title=f"{cfg['name']}: pelvis_z + obj err vs time (E062 vs E063)")

    # Pretty table
    print()
    pairs = []
    for case_rows in all_rows_per_case.values():
        for r in case_rows:
            if r is not None:
                pairs.append(r)
    print(f"{'metric':<28} " + " ".join(f"{r['label'][:30]:>30}" for r in pairs))
    print("-" * (29 + 31 * len(pairs)))
    for k, label, scale, unit in [
        ("L_contact_pct", "L palm contact %", 100, "%"),
        ("R_contact_pct", "R palm contact %", 100, "%"),
        ("both_contact_pct", "both palm contact %", 100, "%"),
        ("stable_pct", "stable % (full)", 100, "%"),
        ("stable_intent_pct", "stable % (intent)", 100, "%"),
        ("pelvis_min", "pelvis_min (m)", 1, ""),
        ("pelvis_min_intent", "pelvis_min_intent (m)", 1, ""),
        ("pelvis_mean_intent", "pelvis_mean_intent (m)", 1, ""),
        ("pelvis_max", "pelvis_max (m)", 1, ""),
        ("final_obj_pos_err", "final_obj_pos_err (cm)", 100, ""),
        ("obj_err_mean", "obj_err_mean (cm)", 100, ""),
    ]:
        vals = [f"{r[k]*scale:>29.2f}{unit}" for r in pairs]
        print(f"{label:<28} " + " ".join(vals))

    # Pass criteria for E063 box023
    print()
    print("=" * 70)
    print("E063 PASS criteria (log 79 §5):")
    for cfg, r in e063_rows:
        if cfg["name"] == "E063_box023":
            c1 = r["pelvis_min_intent"] >= 0.50
            c2 = r["pelvis_mean_intent"] >= 0.50
            c3 = r["stable_intent_pct"] >= 0.80
            c4 = (r["final_obj_pos_err"] * 100) <= 20.0
            allp = c1 and c2 and c3 and c4
            print(f"  [box023] C1 pelvis_min_intent {r['pelvis_min_intent']:.3f}m >= 0.50: {c1}")
            print(f"  [box023] C2 pelvis_mean_intent {r['pelvis_mean_intent']:.3f}m >= 0.50: {c2}")
            print(f"  [box023] C3 stable% intent {r['stable_intent_pct']*100:.1f}% >= 80: {c3}")
            print(f"  [box023] C4 final obj err {r['final_obj_pos_err']*100:.2f}cm <= 20: {c4}")
            print(f"  [box023] verdict (C1+C2+C3+C4): {'PASS' if allp else 'FAIL'} "
                  f"(C5 vis check separately)")
        elif cfg["name"] == "E063_box025":
            r1 = r["pelvis_min"] >= 0.65
            r2 = r["stable_pct"] >= 0.99
            r3 = (r["final_obj_pos_err"] * 100) <= 25.0
            allp = r1 and r2 and r3
            print(f"  [box025] R1 pelvis_min {r['pelvis_min']:.3f}m >= 0.65: {r1}")
            print(f"  [box025] R2 stable% {r['stable_pct']*100:.1f}% >= 99: {r2}")
            print(f"  [box025] R3 final obj err {r['final_obj_pos_err']*100:.2f}cm <= 25: {r3}")
            print(f"  [box025] regression guard (R1+R2+R3): {'PASS' if allp else 'FAIL'}")


if __name__ == "__main__":
    main()
