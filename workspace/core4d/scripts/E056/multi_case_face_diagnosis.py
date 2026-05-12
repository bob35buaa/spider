#!/usr/bin/env python3
"""E056: Multi-case hand-face diagnostic on the 6 B+C cases from E054.

For each case, compute L/R palm signed distance to each box face across all
frames, identify the "main face" each hand contacts inside the intent window,
and classify the grasp pattern as one of:

  对侧 (opposite): L on +F, R on -F (F in {yz, xz}) — force-closure possible
  垂直 (perpendicular): one on ±xy (top/bottom), the other on ±yz/±xz
  错位 (adjacent): two adjacent faces — force not closed (box023 case)
  同面 (same): both hands on the same face — anomaly
  单手 (single): only one hand engaged (e.g., bucket001 dominant_hand=L)

Outputs:
  workspace/core4d/results/E056/{case}_face_dist.png    one per case
  workspace/core4d/results/E056/case_grasp_type_summary.csv
  workspace/core4d/results/E056/E056_summary.md

Run:
  .venv/bin/python workspace/core4d/scripts/E056/multi_case_face_diagnosis.py
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[4]
PROC = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
OUT = REPO / "workspace/core4d/results/E056"

CASES = [
    ("box021_person1",       "both"),
    ("box023_person1",       "both"),   # known: 错位 (from E055)
    ("bucket001_person1",    "L"),       # single-hand from E054
    ("bucket005_s2_person1", "both"),
    ("bucket007_person1",    "both"),
    ("desk021_person1",      "both"),
]

FACE_NAMES = ["+yz", "-yz", "+xz", "-xz", "+xy", "-xy"]
FACE_COLORS = ["#d62728", "#ff9896", "#2ca02c", "#98df8a", "#1f77b4", "#9ecae1"]

# Threshold for "hand is on this face": median |signed_dist| over intent window
# below this → considered the "main face"
MAIN_FACE_DIST_THRESHOLD = 0.07  # 7 cm — accommodates G1 hand_collision sphere half-size

# Face stability threshold: median ≤ MAIN, AND ≥ 60% of frames have |dist| ≤ 7cm
FACE_STABILITY_FRAC = 0.60


@dataclass
class CaseDiagnosis:
    case_name: str
    n_frames: int
    intent_window: tuple[int, int]
    intent_len: int
    L_main_face: str | None
    L_main_med_cm: float
    L_main_pct: float          # fraction of intent frames with |dist| ≤ 7cm on main face
    R_main_face: str | None
    R_main_med_cm: float
    R_main_pct: float
    grasp_type: str            # 对侧 / 垂直 / 错位 / 同面 / 单手 / 未识别
    grasp_valid: bool
    dominant_hand: str
    notes: str


def find_intent_window(qpos: np.ndarray, scene_xml: Path) -> tuple[int, int]:
    """Re-derive intent window using E055-equivalent v3 detector."""
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(m)
    obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    L_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_palm")
    R_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_palm")

    T = qpos.shape[0]
    obj_pos = np.zeros((T, 3))
    L_pos = np.zeros((T, 3)); R_pos = np.zeros((T, 3))
    for t in range(T):
        data.qpos[:] = qpos[t]; mujoco.mj_forward(m, data)
        obj_pos[t] = data.xpos[obj_bid]
        L_pos[t] = data.site_xpos[L_sid]; R_pos[t] = data.site_xpos[R_sid]

    L_dist = np.linalg.norm(L_pos - obj_pos, axis=1)
    R_dist = np.linalg.norm(R_pos - obj_pos, axis=1)
    d_min = np.minimum(L_dist, R_dist)
    d_p20 = float(np.percentile(d_min, 20))
    band = d_min < d_p20 + 0.07

    obj_z = obj_pos[:, 2]
    z_p10 = float(np.percentile(obj_z, 10))
    z_amp = float(obj_z.max() - obj_z.min())
    lift_thr = z_p10 + max(0.10, 0.33 * z_amp)
    lifted = obj_z > lift_thr

    dt = 1 / 30.0
    v_obj = np.gradient(obj_pos, dt, axis=0)
    closer = L_pos if L_dist.mean() < R_dist.mean() else R_pos
    v_hand = np.gradient(closer, dt, axis=0)
    v_rel = np.linalg.norm(v_hand - v_obj, axis=1)
    v_rel_s = np.convolve(v_rel, np.ones(5) / 5, mode="same")
    slow_rel = v_rel_s < 0.30

    intent = band & (slow_rel | lifted)
    for _ in range(3):
        intent = intent | np.r_[False, intent[:-1]] | np.r_[intent[1:], False]
    for _ in range(3):
        intent = intent & np.r_[True, intent[:-1]] & np.r_[intent[1:], True]

    runs = []
    in_run, s = False, 0
    for i, v in enumerate(intent):
        if v and not in_run: s, in_run = i, True
        elif not v and in_run: runs.append((s, i - 1)); in_run = False
    if in_run: runs.append((s, len(intent) - 1))
    if not runs:
        return (0, len(intent) - 1)
    return max(runs, key=lambda x: x[1] - x[0])


def compute_face_distances(qpos: np.ndarray, scene_xml: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns L_signed (T, 6), R_signed (T, 6), half_sizes (3,)."""
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(m)
    L_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_palm")
    R_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
    obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")

    # find object's collision geom box size
    half = None
    for gid in range(m.ngeom):
        if m.geom_bodyid[gid] == obj_bid and m.geom_contype[gid] != 0 \
                and m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
            half = m.geom_size[gid].copy()
            break
    if half is None:
        raise RuntimeError(f"No BOX collision geom on object body in {scene_xml}")

    T = qpos.shape[0]
    L_signed = np.zeros((T, 6)); R_signed = np.zeros((T, 6))
    for t in range(T):
        data.qpos[:] = qpos[t]; mujoco.mj_forward(m, data)
        obj_pos = data.xpos[obj_bid].copy()
        obj_R = data.xmat[obj_bid].copy().reshape(3, 3)
        for sid, store in [(L_sid, L_signed), (R_sid, R_signed)]:
            local = obj_R.T @ (data.site_xpos[sid] - obj_pos)
            store[t] = [
                local[0] - half[0], -local[0] - half[0],   # +yz, -yz
                local[1] - half[1], -local[1] - half[1],   # +xz, -xz
                local[2] - half[2], -local[2] - half[2],   # +xy, -xy
            ]
    return L_signed, R_signed, half


def find_main_face(signed_in_window: np.ndarray) -> tuple[str | None, float, float]:
    """Pick the face with smallest median |signed_dist| in window, IF
    that face has ≥ FACE_STABILITY_FRAC frames with |dist| ≤ MAIN_FACE_DIST_THRESHOLD.

    Returns (face_name | None, median_cm, frac)."""
    abs_dist = np.abs(signed_in_window)        # (T_window, 6)
    med = np.median(abs_dist, axis=0)          # (6,)
    frac_close = (abs_dist <= MAIN_FACE_DIST_THRESHOLD).mean(axis=0)
    # rank by median
    order = np.argsort(med)
    for idx in order:
        if frac_close[idx] >= FACE_STABILITY_FRAC and med[idx] <= MAIN_FACE_DIST_THRESHOLD:
            return FACE_NAMES[idx], float(med[idx] * 100), float(frac_close[idx])
    # nothing qualifies
    return None, float(med[order[0]] * 100), float(frac_close[order[0]])


def classify_grasp(L_face: str | None, R_face: str | None, dominant: str) -> tuple[str, bool]:
    """Return (grasp_type_label, is_valid).

    valid = 对侧 ∨ 垂直 ∨ (单手 with main face on a useful face)
    """
    # single-hand cases: only the dominant hand counts
    if dominant == "L":
        if L_face is None:
            return "单手 (L 未贴面)", False
        return f"单手 (L on {L_face})", True
    if dominant == "R":
        if R_face is None:
            return "单手 (R 未贴面)", False
        return f"单手 (R on {R_face})", True

    # both hands required for valid grasp
    if L_face is None or R_face is None:
        missing = []
        if L_face is None: missing.append("L")
        if R_face is None: missing.append("R")
        return f"未识别 ({','.join(missing)} 未贴面)", False
    if L_face == R_face:
        return f"同面 (both on {L_face})", False
    # Opposite pair check: same plane, opposite signs
    L_plane, L_sign = L_face[1:], L_face[0]
    R_plane, R_sign = R_face[1:], R_face[0]
    if L_plane == R_plane and L_sign != R_sign:
        return f"对侧 ({L_face} / {R_face})", True
    # Perpendicular: one face uses xy (top/bot), the other uses yz/xz
    L_uses_xy = L_plane == "xy"
    R_uses_xy = R_plane == "xy"
    if L_uses_xy != R_uses_xy:
        return f"垂直 ({L_face} / {R_face})", True
    # Same kind (both xy or both side) but different planes → adjacent
    return f"错位 ({L_face} / {R_face})", False


def diagnose_case(case_name: str, dominant: str) -> tuple[CaseDiagnosis, np.ndarray, np.ndarray, tuple[int, int]]:
    scene = PROC / case_name / "scene.xml"
    traj = PROC / case_name / "0" / "trajectory_kinematic.npz"
    qpos = np.load(traj)["qpos"]
    intent = find_intent_window(qpos, scene)
    L_signed, R_signed, half = compute_face_distances(qpos, scene)

    L_in = L_signed[intent[0]:intent[1] + 1]
    R_in = R_signed[intent[0]:intent[1] + 1]
    L_face, L_med, L_frac = find_main_face(L_in)
    R_face, R_med, R_frac = find_main_face(R_in)
    grasp_type, valid = classify_grasp(L_face, R_face, dominant)

    notes = f"box half=({half[0]*100:.0f},{half[1]*100:.0f},{half[2]*100:.0f})cm"

    diag = CaseDiagnosis(
        case_name=case_name, n_frames=qpos.shape[0],
        intent_window=intent, intent_len=intent[1] - intent[0] + 1,
        L_main_face=L_face, L_main_med_cm=L_med, L_main_pct=L_frac,
        R_main_face=R_face, R_main_med_cm=R_med, R_main_pct=R_frac,
        grasp_type=grasp_type, grasp_valid=valid,
        dominant_hand=dominant, notes=notes,
    )
    return diag, L_signed, R_signed, intent


def render_case_plot(diag: CaseDiagnosis, L_signed: np.ndarray, R_signed: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    intent = diag.intent_window
    for ax, signed, hand in [(axes[0], L_signed, "L"), (axes[1], R_signed, "R")]:
        for i, fname in enumerate(FACE_NAMES):
            highlight = (hand == "L" and fname == diag.L_main_face) or \
                        (hand == "R" and fname == diag.R_main_face)
            ax.plot(signed[:, i] * 100, label=fname, color=FACE_COLORS[i],
                    lw=3 if highlight else 1.2,
                    ls="-" if fname[0] == "+" else "--")
        ax.axhline(0, color="black", lw=0.5)
        ax.axvspan(intent[0], intent[1], color="orange", alpha=0.15, label="intent window")
        ax.set_ylabel(f"{hand} palm signed dist (cm)")
        ax.grid(True, alpha=0.3); ax.legend(loc="upper right", fontsize=8, ncol=4)
        ax.set_ylim(-50, 30)
    axes[0].set_title(
        f"{diag.case_name}  |  intent {intent}  |  grasp_type = {diag.grasp_type}  "
        f"|  valid = {diag.grasp_valid}\n"
        f"L main = {diag.L_main_face} (med {diag.L_main_med_cm:+.1f}cm, {diag.L_main_pct*100:.0f}% close)  "
        f"R main = {diag.R_main_face} (med {diag.R_main_med_cm:+.1f}cm, {diag.R_main_pct*100:.0f}% close)"
    )
    axes[1].set_xlabel("frame")
    fig.tight_layout()
    out = OUT / f"{diag.case_name}_face_dist.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.relative_to(REPO)}")


def write_summary(diags: list[CaseDiagnosis]) -> None:
    csv_path = OUT / "case_grasp_type_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "case_name", "dominant_hand", "n_frames",
            "intent_start", "intent_end", "intent_len",
            "L_main_face", "L_main_med_cm", "L_main_pct_close",
            "R_main_face", "R_main_med_cm", "R_main_pct_close",
            "grasp_type", "grasp_valid", "notes",
        ])
        for d in diags:
            w.writerow([
                d.case_name, d.dominant_hand, d.n_frames,
                d.intent_window[0], d.intent_window[1], d.intent_len,
                d.L_main_face or "", f"{d.L_main_med_cm:+.1f}", f"{d.L_main_pct:.2f}",
                d.R_main_face or "", f"{d.R_main_med_cm:+.1f}", f"{d.R_main_pct:.2f}",
                d.grasp_type, d.grasp_valid, d.notes,
            ])
    print(f"\n→ {csv_path.relative_to(REPO)}")

    md_path = OUT / "E056_summary.md"
    valid = [d for d in diags if d.grasp_valid]
    invalid = [d for d in diags if not d.grasp_valid]

    lines = [
        "# E056: 6 case Hand-Face Diagnostic — Summary",
        "",
        f"Date: 2026-05-12",
        f"Cases: {len(diags)} total, {len(valid)} grasp_valid, {len(invalid)} invalid",
        "",
        "## Classification table",
        "",
        "| Case | dominant | intent | L main | L med | L close% | R main | R med | R close% | grasp_type | valid |",
        "|------|----------|--------|--------|-------|----------|--------|-------|----------|------------|-------|",
    ]
    for d in diags:
        lines.append(
            f"| {d.case_name} | {d.dominant_hand} | "
            f"{d.intent_window[0]}–{d.intent_window[1]} ({d.intent_len}f) | "
            f"{d.L_main_face or '–'} | {d.L_main_med_cm:+.1f}cm | {d.L_main_pct*100:.0f}% | "
            f"{d.R_main_face or '–'} | {d.R_main_med_cm:+.1f}cm | {d.R_main_pct*100:.0f}% | "
            f"**{d.grasp_type}** | {'✅' if d.grasp_valid else '❌'} |"
        )
    lines += ["", "## Recommendation", ""]
    if valid:
        valid_sorted = sorted(valid, key=lambda d: -d.intent_len)
        rec = valid_sorted[0]
        lines.append(
            f"**E057 推荐起点**: `{rec.case_name}` ({rec.grasp_type}, intent {rec.intent_len} frames)"
        )
        lines.append("")
        lines.append("路线 A: 在合理 case 上重做 E055 snap, 验证 snap 方法在 ref 合理时能产出物理有效 warmstart")
    else:
        lines.append(
            "**E057 启动 fallback**: 6 case 全部 grasp_valid=False, 加 face-prior 强制对侧选面"
        )
        lines.append("")
        lines.append("路线 B: 改 hand_snap_ik.py 加 `face_filter` 强制 L 选 +F, R 选 -F (F ∈ {yz, xz})")
    lines.append("")
    lines.append("## Per-case time-series plots")
    lines.append("")
    for d in diags:
        lines.append(f"- `workspace/core4d/results/E056/{d.case_name}_face_dist.png`")
    md_path.write_text("\n".join(lines))
    print(f"→ {md_path.relative_to(REPO)}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    diags = []
    for case_name, dominant in CASES:
        print(f"[E056] {case_name} (dominant={dominant})")
        diag, L_s, R_s, intent = diagnose_case(case_name, dominant)
        render_case_plot(diag, L_s, R_s)
        print(f"   L main={diag.L_main_face} ({diag.L_main_med_cm:+.1f}cm, {diag.L_main_pct*100:.0f}% close)")
        print(f"   R main={diag.R_main_face} ({diag.R_main_med_cm:+.1f}cm, {diag.R_main_pct*100:.0f}% close)")
        print(f"   grasp_type={diag.grasp_type}  valid={diag.grasp_valid}")
        diags.append(diag)
    write_summary(diags)
    valid = [d.case_name for d in diags if d.grasp_valid]
    invalid = [d.case_name for d in diags if not d.grasp_valid]
    print()
    print(f"=== summary ===")
    print(f"  valid grasps:   {valid}")
    print(f"  invalid grasps: {invalid}")


if __name__ == "__main__":
    main()
