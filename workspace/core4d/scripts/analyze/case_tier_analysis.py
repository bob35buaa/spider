"""E054: CORE4D Case 几何可达性 + Mocap 数据质量联合分析.

Analyzes 21 processed CORE4D cases and outputs:
- Tier (1/2/3) based on geometry + partner force contribution
- recommended_method (B+C / C-only / dual-robot / drop) based on mocap quality
- dominant_hand (L/R/both) for Path B IK-snap strategy

Run:
    # All 21 cases (default):
    uv run python workspace/core4d/scripts/analyze/case_tier_analysis.py

    # Subset of cases:
    uv run python workspace/core4d/scripts/analyze/case_tier_analysis.py \
        --cases box023_person1,bucket001_person1

    # Custom output dir:
    uv run python workspace/core4d/scripts/analyze/case_tier_analysis.py \
        --output-dir workspace/core4d/results/E054/

Outputs (under --output-dir):
    case_tier_classification.csv     — 27-column table, 1 row per case
    E054_case_tier_summary.md        — markdown grouped by recommended_method

Companion script (visual verification of dominant_hand):
    bash workspace/core4d/scripts/analyze/extract_case_keyframes.sh \
        box023_person1 bucket001_person1 ...
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass

import numpy as np
import trimesh
import tyro

PROCESSED_DIR = "example_datasets/processed/core4d/unitree_g1/humanoid_object"
MESH_DIR = "workspace/core4d/object_models/object_models"
FPS = 30  # CORE4D processed data is at 30 fps

# G1 anatomy (from CLAUDE.md / OmniRetarget retarget background)
G1_HEIGHT = 1.32
G1_TWO_ARM_SPAN = 1.00
G1_SINGLE_ARM_REACH = 0.50
G1_MAX_REACH_HEIGHT = 1.50

# Tier thresholds
TIER1_DIM_MAX = 0.50
TIER2_DIM_MAX = 0.70
PARTNER_TIER1 = 0.30
PARTNER_TIER2 = 0.50

# B-friendly thresholds
B_OBJ_Z_AMP_MIN = 0.05
B_INTENT_WIN_MIN = 1


@dataclass
class CaseResult:
    case_name: str
    person_id: str  # "person1" / "person2" / "s2_person1"
    n_frames: int
    duration_s: float

    # Geometry
    object_dim_max: float
    object_height: float
    object_mesh_volume: float
    object_mass_est: float

    # Object motion
    obj_z_amplitude: float
    obj_pos_traveled: float
    obj_acc_z_snr_db: float

    # Partner
    has_partner: bool
    partner_hand_obj_dist_min: float
    partner_force_contribution: float

    # Hand-obj kinematics (G1 retargeted hand vs object)
    hand_obj_dist_min: float
    hand_obj_dist_p20: float
    intent_window_count: int
    intent_window_total_frames: int

    # Hand dominance during intent windows (informs Path B: single-hand vs dual-hand snap).
    # See compute_hand_obj_intent for definition of symmetry_ratio.
    dominant_hand: str  # "L" / "R" / "both"
    dominant_hand_L_frac: float  # L_mean / (L_mean + R_mean) — for transparency
    hand_symmetry_ratio: float  # min(L_mean, R_mean) / max(L_mean, R_mean), <0.55 = single
    L_mean_dist_in_intent: float  # m, mean L-hand-to-object distance during intent windows
    R_mean_dist_in_intent: float  # m, mean R-hand-to-object distance during intent windows

    # Locomotion
    pelvis_xy_traveled: float

    # Classification
    tier: int
    recommended_method: str
    notes: str


# ---------- Helpers ----------


def parse_case(case_dir: str) -> tuple[str, str]:
    """`box023_person1` → (`box023`, `person1`); `box025_s2_person1` → (`box025_s2`, `person1`)."""
    m = re.match(r"^(.*?)_(person[12])$", case_dir)
    if not m:
        raise ValueError(f"Can't parse case dir: {case_dir}")
    return m.group(1), m.group(2)


def object_category(case_base: str) -> str:
    """`box023` → `box`; `box025_s2` → `box`; `bucket010` → `bucket`."""
    m = re.match(r"^([a-z]+)\d", case_base)
    return m.group(1) if m else case_base


def find_mesh_path(case_base: str) -> str:
    """`box023` → `.../box/box023_m.obj`; `box025_s2` → `.../box/box025_m.obj`."""
    obj_id = re.match(r"^([a-z]+\d+)", case_base).group(1)
    cat = object_category(case_base)
    return os.path.join(MESH_DIR, cat, f"{obj_id}_m.obj")


def find_partner_case(case_base: str, person_id: str) -> str | None:
    """For `box025` + `person1`, partner is `box025_person2` (if exists).

    For `box025_s2_person1`, partner is `box025_s2_person2` (likely missing) — return None.
    """
    other = "person2" if person_id == "person1" else "person1"
    cand = f"{case_base}_{other}"
    if os.path.isdir(os.path.join(PROCESSED_DIR, cand)):
        return cand
    return None


# ---------- Per-case computations ----------


def compute_geometry(case_base: str) -> dict:
    mesh_path = find_mesh_path(case_base)
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Object mesh missing: {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh")
    extents = mesh.bounding_box.extents  # full lengths along x/y/z
    return {
        "object_dim_max": float(np.max(extents)),
        "object_height": float(extents[2]),
        "object_mesh_volume": float(mesh.volume),
        # Crude mass estimate: mesh volume * 500 kg/m³ (loose pack avg). Real values
        # will come from CORE4D metadata; this is only a fallback for tier scoring.
        "object_mass_est": float(mesh.volume * 500.0),
    }


def load_traj(case_dir: str) -> dict:
    npz = np.load(
        os.path.join(PROCESSED_DIR, case_dir, "0", "trajectory_kinematic.npz"),
        allow_pickle=True,
    )
    qpos = npz["qpos"]  # (T, 43) = root7 + g1_29 + obj7
    contact_pos = npz["contact_pos"]  # (T, 2, 3) — left, right hand sites
    return {"qpos": qpos, "contact_pos": contact_pos}


def compute_obj_motion(qpos: np.ndarray) -> dict:
    obj_pos = qpos[:, 36:39]  # T x 3
    obj_z = obj_pos[:, 2]
    obj_z_amp = float(obj_z.max() - obj_z.min())
    diffs = np.linalg.norm(np.diff(obj_pos, axis=0), axis=1)
    traveled = float(diffs.sum())

    # Acceleration & SNR via low-pass (5 Hz cutoff at fps=30 → simple moving avg ~6 frames)
    win = 6
    if len(obj_z) < 3 * win:
        snr_db = float("nan")
    else:
        v_z = np.gradient(obj_z, 1.0 / FPS)
        a_z = np.gradient(v_z, 1.0 / FPS)
        kernel = np.ones(win) / win
        a_z_smooth = np.convolve(a_z, kernel, mode="same")
        noise = a_z - a_z_smooth
        var_signal = float(np.var(a_z_smooth))
        var_noise = float(np.var(noise)) + 1e-12
        snr_db = float(10.0 * np.log10(var_signal / var_noise))
    return {
        "obj_z_amplitude": obj_z_amp,
        "obj_pos_traveled": traveled,
        "obj_acc_z_snr_db": snr_db,
    }


def compute_hand_obj_intent(qpos: np.ndarray, contact_pos: np.ndarray) -> dict:
    """Detect intent windows.

    Two iterations of diagnosis:
      v1: absolute hand velocity gate (0.3 m/s) → killed everything because G1 walks
          with hand swing of 0.3-0.7 m/s.
      v2: hand-obj RELATIVE velocity gate (0.3 m/s) → still misses lift phases on cases
          like desk021 where the object wobbles during carry (v_rel noisy).
      v3 (current): use obj_z elevation as primary "physical-necessity" signal —
          if the object is lifted, SOMETHING must be supporting it, regardless of how
          jittery v_rel looks.

    Intent gate (sustained ≥ 10 frames):
      band(t)         := d_min(t) < d_p20 + 0.05 m            (hand near object)
      slow_rel(t)     := |v_hand - v_obj|_smoothed < 0.30 m/s
      lifted(t)       := obj_z(t) > obj_z_p10 + 0.05 m         (object above resting)
      intent(t)       := band(t) AND (slow_rel(t) OR lifted(t))

    The OR captures both:
      - lift tasks: intent = "near & object up" (slow_rel may be noisy, doesn't matter)
      - non-lift tasks (push/slide): intent = "near & relatively still"

    A 3-frame morphological closing fills tiny gaps so contiguous windows aren't broken
    by single-frame jitter.
    """
    T = qpos.shape[0]
    obj_pos = qpos[:, 36:39]  # T x 3
    hand_obj = np.linalg.norm(contact_pos - obj_pos[:, None, :], axis=-1)  # T x 2
    d_min = hand_obj.min(axis=1)
    closest_hand_idx = hand_obj.argmin(axis=1)  # which hand is closest each frame

    d_min_overall = float(d_min.min())
    d_p20 = float(np.percentile(d_min, 20))

    # Relative velocity between the closest hand and the object, smoothed with a
    # 5-frame moving average to suppress finite-difference spikes (peaks of 5-15 m/s
    # are pure numerical noise).
    if T < 6:
        v_rel = np.zeros(T)
    else:
        closest_hand_pos = contact_pos[np.arange(T), closest_hand_idx]  # T x 3
        rel_pos = closest_hand_pos - obj_pos  # T x 3
        d_rel = np.diff(rel_pos, axis=0) * FPS
        v = np.linalg.norm(d_rel, axis=-1)
        kernel = np.ones(5) / 5
        v_smooth = np.convolve(v, kernel, mode="same")
        v_rel = np.concatenate([[v_smooth[0]], v_smooth])

    # Object-elevation signal: physical necessity (lifted ⇒ external support exists).
    # Threshold scales with the case's own amplitude so a noisy 5 cm obj_z drift on a
    # push task can't masquerade as a "lift". Only cases with genuinely large lift
    # (≥ 33 % of their own amplitude AND ≥ 10 cm absolute) trigger this branch;
    # everything else falls back to the strict (band ∩ slow_rel) gate of v2.
    obj_z = obj_pos[:, 2]
    obj_z_baseline = float(np.percentile(obj_z, 10))
    obj_z_amp = float(obj_z.max() - obj_z.min())
    lift_thr = max(0.10, 0.33 * obj_z_amp)
    lifted = obj_z > (obj_z_baseline + lift_thr)

    # Intent: near object AND (relatively still OR object is genuinely being lifted)
    band = d_min < (d_p20 + 0.07)  # 7 cm tolerance (was 5 — gives small breathing room)
    slow_rel = v_rel < 0.30
    in_intent = band & (slow_rel | lifted)

    # Morphological closing: fill gaps ≤ 3 frames so jitter doesn't fragment a window
    # Implemented as: dilate by 3 then erode by 3 (over the boolean signal)
    def _close(b: np.ndarray, k: int = 3) -> np.ndarray:
        if k <= 0:
            return b
        # dilation
        d = b.copy()
        for off in range(1, k + 1):
            d[off:] |= b[:-off]
            d[:-off] |= b[off:]
        # erosion (only filled gaps, not extending edges)
        e = d.copy()
        for off in range(1, k + 1):
            e[off:] &= d[:-off]
            e[:-off] &= d[off:]
        # Restore original True frames so we don't shrink (closing should not erode original)
        return e | b
    in_intent = _close(in_intent, k=3)

    # Find contiguous True segments
    windows: list[tuple[int, int]] = []
    i = 0
    while i < T:
        if in_intent[i]:
            j = i
            while j < T and in_intent[j]:
                j += 1
            if j - i >= 10:
                windows.append((i, j))
            i = j
        else:
            i += 1
    total_frames = sum(j - i for i, j in windows)

    # Hand dominance: within intent windows, compare each hand's MEAN distance to the
    # object. The previous version used "fraction of frames where each hand was closest"
    # but that's just a tiebreaker — when both hands are at similar distance (e.g. both
    # at ~28cm carrying a box), the slightly-closer one wins ~100% of frames, yielding
    # a misleading "single-hand" verdict.
    #
    # Visual verification (E054 video frames) confirmed: only bucket001 has one hand
    # truly far from the object during carry (R_mean ≈ 1.7× L_mean). All other "B+C"
    # cases have both hands within ~10% of the same distance — clearly two-handed.
    #
    # Discriminator: symmetry_ratio = min(L_mean, R_mean) / max(L_mean, R_mean), in [0,1]
    #   ≈ 1.0  → both hands equally close to object  → both-hand
    #   < 0.55 → one hand systematically far          → single-hand (snap the closer one)
    if total_frames > 0:
        intent_mask = np.zeros(T, dtype=bool)
        for s, e in windows:
            intent_mask[s:e] = True
        L_mean_in = float(hand_obj[intent_mask, 0].mean())
        R_mean_in = float(hand_obj[intent_mask, 1].mean())
        symmetry = min(L_mean_in, R_mean_in) / max(L_mean_in, R_mean_in, 1e-6)
        L_frac = float(L_mean_in / (L_mean_in + R_mean_in)) if (L_mean_in + R_mean_in) > 0 else 0.5
    else:
        L_mean_in = R_mean_in = float("nan")
        symmetry = 1.0
        L_frac = 0.5
    if symmetry < 0.55:
        # One hand significantly farther from object → single-handed
        dominant = "L" if L_mean_in < R_mean_in else "R"
    else:
        dominant = "both"

    return {
        "hand_obj_dist_min": d_min_overall,
        "hand_obj_dist_p20": d_p20,
        "intent_window_count": len(windows),
        "intent_window_total_frames": int(total_frames),
        "dominant_hand": dominant,
        "dominant_hand_L_frac": L_frac,  # kept for backward compat / transparency
        "hand_symmetry_ratio": symmetry,  # the new actual discriminator
        "L_mean_dist_in_intent": L_mean_in,
        "R_mean_dist_in_intent": R_mean_in,
    }


def compute_locomotion(qpos: np.ndarray) -> dict:
    pelvis_xy = qpos[:, :2]
    diffs = np.linalg.norm(np.diff(pelvis_xy, axis=0), axis=1)
    return {"pelvis_xy_traveled": float(diffs.sum())}


def compute_partner(case_base: str, person_id: str, qpos: np.ndarray) -> dict:
    """Look up paired case (other person) and compute their hand-obj proximity.

    partner_force_contribution heuristic:
        - If partner hand stays within 30 cm of object for >50% of frames → 0.5 (likely co-carrier)
        - Else proportional to fraction of frames within 30 cm
        - Single-person cases → 0.0
    """
    partner_case = find_partner_case(case_base, person_id)
    if partner_case is None:
        return {
            "has_partner": False,
            "partner_hand_obj_dist_min": float("nan"),
            "partner_force_contribution": 0.0,
        }
    p = load_traj(partner_case)
    p_qpos, p_contact = p["qpos"], p["contact_pos"]
    # Use OUR object trajectory as the reference object pose (both retargets share the
    # same underlying mocap object; small numerical differences are fine for this heuristic).
    T = min(qpos.shape[0], p_qpos.shape[0])
    obj_pos = qpos[:T, 36:39]
    p_hand = p_contact[:T]  # T x 2 x 3
    p_hand_obj = np.linalg.norm(p_hand - obj_pos[:, None, :], axis=-1).min(axis=1)
    near = p_hand_obj < 0.30
    frac_near = float(near.mean())
    return {
        "has_partner": True,
        "partner_hand_obj_dist_min": float(p_hand_obj.min()),
        "partner_force_contribution": min(0.5, frac_near),
    }


def classify(geom: dict, motion: dict, intent: dict, partner: dict) -> tuple[int, str, str]:
    """Returns (tier, recommended_method, notes)."""
    notes_parts = []

    dim = geom["object_dim_max"]
    pf = partner["partner_force_contribution"]

    # Tier
    if dim <= TIER1_DIM_MAX and pf < PARTNER_TIER1:
        tier = 1
    elif dim <= TIER2_DIM_MAX and pf < PARTNER_TIER2:
        tier = 2
    else:
        tier = 3

    # Method
    if tier == 3:
        if dim > 0.70:
            method = "drop"
            notes_parts.append(f"object_dim_max={dim:.2f}m > G1 dual-arm grip limit 0.70m")
        else:
            method = "dual-robot"
            notes_parts.append(f"partner force contribution ≈{pf:.0%}, requires dual-robot")
    else:
        b_friendly = (
            motion["obj_z_amplitude"] >= B_OBJ_Z_AMP_MIN
            and intent["intent_window_count"] >= B_INTENT_WIN_MIN
        )
        method = "B+C" if b_friendly else "C-only"
        if not b_friendly:
            if motion["obj_z_amplitude"] < B_OBJ_Z_AMP_MIN:
                notes_parts.append(f"obj_z_amp={motion['obj_z_amplitude']*100:.1f}cm < 5cm")
            if intent["intent_window_count"] < B_INTENT_WIN_MIN:
                notes_parts.append("no intent window detected")

    return tier, method, "; ".join(notes_parts)


def analyze(case_dir: str) -> CaseResult:
    case_base, person_id = parse_case(case_dir)
    geom = compute_geometry(case_base)
    traj = load_traj(case_dir)
    qpos = traj["qpos"]
    contact_pos = traj["contact_pos"]

    motion = compute_obj_motion(qpos)
    intent = compute_hand_obj_intent(qpos, contact_pos)
    loco = compute_locomotion(qpos)
    partner = compute_partner(case_base, person_id, qpos)
    tier, method, notes = classify(geom, motion, intent, partner)

    T = qpos.shape[0]
    return CaseResult(
        case_name=case_base,
        person_id=person_id,
        n_frames=int(T),
        duration_s=float(T / FPS),
        object_dim_max=geom["object_dim_max"],
        object_height=geom["object_height"],
        object_mesh_volume=geom["object_mesh_volume"],
        object_mass_est=geom["object_mass_est"],
        obj_z_amplitude=motion["obj_z_amplitude"],
        obj_pos_traveled=motion["obj_pos_traveled"],
        obj_acc_z_snr_db=motion["obj_acc_z_snr_db"],
        has_partner=partner["has_partner"],
        partner_hand_obj_dist_min=partner["partner_hand_obj_dist_min"],
        partner_force_contribution=partner["partner_force_contribution"],
        hand_obj_dist_min=intent["hand_obj_dist_min"],
        hand_obj_dist_p20=intent["hand_obj_dist_p20"],
        intent_window_count=intent["intent_window_count"],
        intent_window_total_frames=intent["intent_window_total_frames"],
        dominant_hand=intent["dominant_hand"],
        dominant_hand_L_frac=intent["dominant_hand_L_frac"],
        hand_symmetry_ratio=intent["hand_symmetry_ratio"],
        L_mean_dist_in_intent=intent["L_mean_dist_in_intent"],
        R_mean_dist_in_intent=intent["R_mean_dist_in_intent"],
        pelvis_xy_traveled=loco["pelvis_xy_traveled"],
        tier=tier,
        recommended_method=method,
        notes=notes,
    )


# ---------- Output ----------


def write_csv(rows: list[CaseResult], path: str) -> None:
    import csv

    keys = list(asdict(rows[0]).keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            row = asdict(r)
            for k, v in row.items():
                if isinstance(v, float):
                    row[k] = f"{v:.4f}" if not np.isnan(v) else ""
            w.writerow(row)


def write_summary_md(rows: list[CaseResult], path: str) -> None:
    by_method: dict[str, list[CaseResult]] = {}
    for r in rows:
        by_method.setdefault(r.recommended_method, []).append(r)

    lines = ["# E054 Case Tier Analysis — Summary", ""]
    lines.append(f"Total cases analyzed: **{len(rows)}**")
    lines.append("")

    # Tier distribution
    tier_counts = {1: 0, 2: 0, 3: 0}
    for r in rows:
        tier_counts[r.tier] += 1
    lines.append("## Tier 分布")
    lines.append("")
    lines.append("| Tier | 数量 | 含义 |")
    lines.append("|---|---|---|")
    lines.append(f"| 1 | {tier_counts[1]} | 单臂可达 + 单人足以承担 |")
    lines.append(f"| 2 | {tier_counts[2]} | 双臂可达 + 单人勉强 |")
    lines.append(f"| 3 | {tier_counts[3]} | 双人协作必须 / 几何不可解 |")
    lines.append("")

    # Method distribution
    lines.append("## 推荐方法分布")
    lines.append("")
    lines.append("| Method | 数量 |")
    lines.append("|---|---|")
    for m in ["B+C", "C-only", "dual-robot", "drop"]:
        lines.append(f"| {m} | {len(by_method.get(m, []))} |")
    lines.append("")

    # Per-method case lists
    for m in ["B+C", "C-only", "dual-robot", "drop"]:
        cases = by_method.get(m, [])
        if not cases:
            continue
        lines.append(f"## {m}")
        lines.append("")
        lines.append(
            "| case | person | tier | obj_dim | obj_z_amp | intent_win | dom_hand | "
            "L_mean | R_mean | sym | partner_pf | notes |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in cases:
            l_str = (
                f"{r.L_mean_dist_in_intent*100:.0f}cm"
                if not np.isnan(r.L_mean_dist_in_intent) else "-"
            )
            r_str = (
                f"{r.R_mean_dist_in_intent*100:.0f}cm"
                if not np.isnan(r.R_mean_dist_in_intent) else "-"
            )
            lines.append(
                f"| {r.case_name} | {r.person_id} | {r.tier} | "
                f"{r.object_dim_max:.2f} m | {r.obj_z_amplitude*100:.1f} cm | "
                f"{r.intent_window_count} | **{r.dominant_hand}** | "
                f"{l_str} | {r_str} | {r.hand_symmetry_ratio:.2f} | "
                f"{r.partner_force_contribution:.2f} | {r.notes} |"
            )
        lines.append("")

    # Recommended first-batch B+C cases
    bc_cases = sorted(
        by_method.get("B+C", []),
        key=lambda r: (r.tier, -r.obj_z_amplitude, r.object_dim_max),
    )
    lines.append("## 推荐首批验证 case (Path B)")
    lines.append("")
    if not bc_cases:
        lines.append("*无 B-friendly case，直接走 Path C-only。*")
    else:
        lines.append(
            "按 (tier↑, obj_z_amp↓, obj_dim↑) 排序。**dom_hand** 决定 IK-snap 策略："
            "`L`/`R` = 单手 snap，`both` = 双手 snap。"
        )
        lines.append("")
        for i, r in enumerate(bc_cases[:3], start=1):
            snap_strategy = (
                f"snap **{r.dominant_hand}** hand only"
                if r.dominant_hand in ("L", "R")
                else "snap **both** hands"
            )
            lines.append(
                f"{i}. **{r.case_name}_{r.person_id}** "
                f"— Tier {r.tier}, obj_z_amp={r.obj_z_amplitude*100:.1f}cm, "
                f"intent_win={r.intent_window_count} ({r.intent_window_total_frames} frames), "
                f"dim_max={r.object_dim_max:.2f}m → {snap_strategy} "
                f"(symmetry={r.hand_symmetry_ratio:.2f})"
            )
    lines.append("")

    # C1/C2 verification
    lines.append("## Plan Claims 验证")
    lines.append("")
    box023 = next((r for r in rows if r.case_name == "box023"), None)
    box025 = next((r for r in rows if r.case_name == "box025" and r.person_id == "person1"), None)
    if box023:
        ok = box023.tier == 1
        lines.append(
            f"- **C1 (box023 → Tier 1)**: {'✅' if ok else '❌'} actual tier = {box023.tier}, "
            f"dim_max={box023.object_dim_max:.2f}m, partner_pf={box023.partner_force_contribution:.2f}"
        )
    if box025:
        ok = box025.tier == 3
        lines.append(
            f"- **C2 (box025 → Tier 3)**: {'✅' if ok else '❌'} actual tier = {box025.tier}, "
            f"dim_max={box025.object_dim_max:.2f}m, partner_pf={box025.partner_force_contribution:.2f}"
        )
    lines.append(
        f"- **C3 (≥3 B-friendly)**: {'✅' if len(by_method.get('B+C', [])) >= 3 else '❌'} "
        f"actual = {len(by_method.get('B+C', []))}"
    )
    lines.append(
        f"- **C4 (≥1 C-only)**: {'✅' if len(by_method.get('C-only', [])) >= 1 else '❌'} "
        f"actual = {len(by_method.get('C-only', []))}"
    )
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------- Main ----------


def main(
    output_dir: str = "workspace/core4d/results/E054",
    cases: str = "all",  # "all" or comma-separated list of case dirs
):
    os.makedirs(output_dir, exist_ok=True)

    if cases == "all":
        case_dirs = sorted(
            d for d in os.listdir(PROCESSED_DIR) if os.path.isdir(os.path.join(PROCESSED_DIR, d))
        )
    else:
        case_dirs = [c.strip() for c in cases.split(",")]

    print(f"Analyzing {len(case_dirs)} cases...")
    rows: list[CaseResult] = []
    for cd in case_dirs:
        try:
            r = analyze(cd)
            rows.append(r)
            print(
                f"  [{r.tier}] {cd:<28} dim={r.object_dim_max:.2f}m "
                f"obj_z_amp={r.obj_z_amplitude*100:5.1f}cm "
                f"intent={r.intent_window_count} "
                f"pf={r.partner_force_contribution:.2f} "
                f"→ {r.recommended_method}"
            )
        except Exception as e:
            print(f"  [!] {cd} failed: {e}")

    csv_path = os.path.join(output_dir, "case_tier_classification.csv")
    md_path = os.path.join(output_dir, "E054_case_tier_summary.md")
    write_csv(rows, csv_path)
    write_summary_md(rows, md_path)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    tyro.cli(main)
