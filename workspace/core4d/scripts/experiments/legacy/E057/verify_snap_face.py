#!/usr/bin/env python3
"""E057 C6 verification: check that snap-modified qpos preserves the
E056-diagnosed contact faces (L on -yz, R on +yz for bucket005_s2_person1).

Reads:
  workspace/core4d/results/E057/bucket005_s2_person1/warmstart_qpos.npz
Outputs:
  workspace/core4d/results/E057/bucket005_s2_person1/face_verification.csv
  workspace/core4d/results/E057/bucket005_s2_person1/face_dist_snap.png

Pass criteria (logged at end):
  snap_L_main_face == E056 expected (-yz)
  snap_R_main_face == E056 expected (+yz)
  median |dist| ≤ 7 cm AND ≥ 60% frames close
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
RESULTS = REPO / f"workspace/core4d/results/E057/{CASE}"

FACE_NAMES = ["+yz", "-yz", "+xz", "-xz", "+xy", "-xy"]
FACE_COLORS = ["#d62728", "#ff9896", "#2ca02c", "#98df8a", "#1f77b4", "#9ecae1"]
MAIN_FACE_DIST_THRESHOLD = 0.07
FACE_STABILITY_FRAC = 0.60

EXPECTED_L_FACE = "-yz"
EXPECTED_R_FACE = "+yz"


def compute_face_distances(qpos: np.ndarray, scene_xml: Path):
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(m)
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

    T = qpos.shape[0]
    L_signed = np.zeros((T, 6))
    R_signed = np.zeros((T, 6))
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(m, data)
        obj_pos = data.xpos[obj_bid].copy()
        obj_R = data.xmat[obj_bid].copy().reshape(3, 3)
        for sid, store in [(L_sid, L_signed), (R_sid, R_signed)]:
            local = obj_R.T @ (data.site_xpos[sid] - obj_pos)
            store[t] = [
                local[0] - half[0], -local[0] - half[0],
                local[1] - half[1], -local[1] - half[1],
                local[2] - half[2], -local[2] - half[2],
            ]
    return L_signed, R_signed, half


def find_main_face(signed_in_window: np.ndarray):
    abs_dist = np.abs(signed_in_window)
    med = np.median(abs_dist, axis=0)
    frac_close = (abs_dist <= MAIN_FACE_DIST_THRESHOLD).mean(axis=0)
    order = np.argsort(med)
    for idx in order:
        if frac_close[idx] >= FACE_STABILITY_FRAC and med[idx] <= MAIN_FACE_DIST_THRESHOLD:
            return FACE_NAMES[idx], float(med[idx] * 100), float(frac_close[idx])
    return None, float(med[order[0]] * 100), float(frac_close[order[0]])


def main() -> None:
    npz = np.load(RESULTS / "warmstart_qpos.npz")
    qpos_ref = npz["qpos_ref"]
    qpos_snap = npz["qpos_snap"]
    intent = tuple(int(x) for x in npz["intent_window"])
    print(f"[verify] T={qpos_ref.shape[0]}, intent={intent}")

    print("[verify] computing face distances on REF...")
    L_ref, R_ref, half = compute_face_distances(qpos_ref, SCENE)
    print("[verify] computing face distances on SNAP...")
    L_snap, R_snap, _ = compute_face_distances(qpos_snap, SCENE)
    print(f"[verify] box half=({half[0]*100:.1f},{half[1]*100:.1f},{half[2]*100:.1f})cm")

    L_ref_in = L_ref[intent[0]:intent[1] + 1]
    R_ref_in = R_ref[intent[0]:intent[1] + 1]
    L_snap_in = L_snap[intent[0]:intent[1] + 1]
    R_snap_in = R_snap[intent[0]:intent[1] + 1]

    L_ref_face, L_ref_med, L_ref_pct = find_main_face(L_ref_in)
    R_ref_face, R_ref_med, R_ref_pct = find_main_face(R_ref_in)
    L_snap_face, L_snap_med, L_snap_pct = find_main_face(L_snap_in)
    R_snap_face, R_snap_med, R_snap_pct = find_main_face(R_snap_in)

    csv_path = RESULTS / "face_verification.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "trajectory", "hand", "main_face",
            "median_abs_dist_cm", "frac_close", "expected_face", "match",
        ])
        for traj, hand, face, med, pct, expected in [
            ("ref",  "L", L_ref_face,  L_ref_med,  L_ref_pct,  EXPECTED_L_FACE),
            ("ref",  "R", R_ref_face,  R_ref_med,  R_ref_pct,  EXPECTED_R_FACE),
            ("snap", "L", L_snap_face, L_snap_med, L_snap_pct, EXPECTED_L_FACE),
            ("snap", "R", R_snap_face, R_snap_med, R_snap_pct, EXPECTED_R_FACE),
        ]:
            w.writerow([traj, hand, face or "", f"{med:+.2f}",
                        f"{pct:.2f}", expected, str(face == expected)])
    print(f"[verify] wrote {csv_path}")

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    titles = [
        ("REF L palm",  L_ref,  L_ref_face,  axes[0, 0]),
        ("REF R palm",  R_ref,  R_ref_face,  axes[0, 1]),
        ("SNAP L palm", L_snap, L_snap_face, axes[1, 0]),
        ("SNAP R palm", R_snap, R_snap_face, axes[1, 1]),
    ]
    for title, signed, main_face, ax in titles:
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
    out = RESULTS / "face_dist_snap.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[verify] wrote {out}")

    print()
    print("=== C6 face verification ===")
    print(f"  REF  : L={L_ref_face} (med {L_ref_med:+.1f}cm, {L_ref_pct*100:.0f}%) | "
          f"R={R_ref_face} (med {R_ref_med:+.1f}cm, {R_ref_pct*100:.0f}%)")
    print(f"  SNAP : L={L_snap_face} (med {L_snap_med:+.1f}cm, {L_snap_pct*100:.0f}%) | "
          f"R={R_snap_face} (med {R_snap_med:+.1f}cm, {R_snap_pct*100:.0f}%)")
    print(f"  Expected: L={EXPECTED_L_FACE}, R={EXPECTED_R_FACE}")
    L_ok = L_snap_face == EXPECTED_L_FACE
    R_ok = R_snap_face == EXPECTED_R_FACE
    if L_ok and R_ok:
        print("  C6 PASS ✓ snap preserves E056 contact faces")
    else:
        print(f"  C6 FAIL ✗ L_match={L_ok}, R_match={R_ok}")


if __name__ == "__main__":
    main()
