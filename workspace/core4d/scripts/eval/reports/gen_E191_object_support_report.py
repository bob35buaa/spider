#!/usr/bin/env python3
"""E191: turn the object-support audit into pre-registered hypothesis verdicts.

Consumes `results/E191/audit/e191_object_support_audit.tsv` (written by
`runners/eval_E191_object_support_audit.py`) and emits:

  * `E191_object_support_report.md` — per-object table + H1..H6 verdicts
  * `e191_config_provenance.tsv`    — resolved Hydra chain and every
    metric-length reward/gate parameter, one row per case

The hypotheses are fixed in `HYPOTHESES` below and were registered in the plan
before the numbers were produced. A refuted hypothesis is reported as REFUTE;
thresholds are never adjusted to fit the data.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
OUT_DIR = REPO_ROOT / "workspace/core4d/results/E191/audit"
AUDIT_TSV = OUT_DIR / "e191_object_support_audit.tsv"
REPORT_MD = OUT_DIR / "E191_object_support_report.md"
PROVENANCE_TSV = OUT_DIR / "e191_config_provenance.tsv"
OVERRIDE_DIR = REPO_ROOT / "examples/config/override"

PERMUTATIONS = 20_000
RNG_SEED = 0
KP_POS = 500.0  # init_pos_actuator_gain, constant across every audited case
LIFTED_MIN_FRAC = 0.20  # below this the object is dragged, not carried

# Metric-length / scale-sensitive reward and gate parameters. These are the
# fields whose appropriateness depends on object size but which are byte
# identical across the whole box+bucket+desk range.
SCALE_SENSITIVE = (
    "surface_band_width_m",
    "surface_band_min_sdf_m",
    "surface_band_sigma",
    "surface_band_penetration_tol_m",
    "cem_hand_gate_min_sdf_m",
    "cem_hand_gate_max_violation_pct",
    "cem_hand_gate_hard_floor_m",
    "cem_leg_gate_min_sdf_m",
    "cem_leg_gate_hard_floor_m",
    "cem_safety_gate_min_sdf_m",
    "cem_posture_gate_mean_z_err_m",
    "cem_posture_gate_terminal_z_err_m",
    "cem_posture_gate_max_z_drop_m",
    "robot_object_penalty_margin_m",
    "robot_object_penalty_deep_threshold_m",
    "leg_object_penalty_margin_m",
    "hand_object_deep_penalty_scale",
    "hand_object_deep_penalty_threshold_m",
    "object_lift_sigma",
    "object_floor_margin_m",
    "object_clearance_min_m",
    "object_clearance_max_m",
    "task_obj_pos_sigma",
    "task_obj_rot_sigma",
    "init_pos_actuator_gain",
    "init_rot_actuator_gain",
    "partner_force_scale",
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def num(row: dict[str, str], key: str) -> float:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return math.nan
    return value


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Spearman rho with a deterministic permutation p-value.

    scipy is not a dependency of this repo, so ranks and the null distribution
    are computed directly. Ties get average ranks.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = int(x.size)
    if n < 4:
        return math.nan, math.nan, n

    def rank(values: np.ndarray) -> np.ndarray:
        order = values.argsort()
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        # average ranks for ties
        for value in np.unique(values):
            hit = values == value
            if hit.sum() > 1:
                ranks[hit] = ranks[hit].mean()
        return ranks

    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return math.nan, math.nan, n
    rho = float(np.corrcoef(rx, ry)[0, 1])
    # Deterministic permutation null, vectorised: rank-Pearson reduces to a dot
    # product once both rank vectors are centred and scaled.
    rng = np.random.default_rng(RNG_SEED)
    cx = (rx - rx.mean()) / (np.linalg.norm(rx - rx.mean()) or 1.0)
    cy = (ry - ry.mean()) / (np.linalg.norm(ry - ry.mean()) or 1.0)
    shuffled = np.array([rng.permutation(cy) for _ in range(PERMUTATIONS)])
    null = shuffled @ cx
    p = float((np.abs(null) >= abs(rho) - 1e-12).mean())
    return rho, p, n


def group_by(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(row.get(key, ""), []).append(row)
    return out


def mean_of(rows: list[dict[str, str]], key: str) -> float:
    values = np.array([num(r, key) for r in rows])
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else math.nan


def fmt(value: float, digits: int = 3) -> str:
    return "—" if not np.isfinite(value) else f"{value:.{digits}f}"


# ---------------------------------------------------------------------------
# Config provenance (A6)
# ---------------------------------------------------------------------------


def resolve_chain(name: str, seen: list[str] | None = None) -> list[str]:
    """Resolve a Hydra override's `defaults:` list into a flat ordered chain."""
    seen = seen or []
    if name in seen:
        return seen
    path = OVERRIDE_DIR / f"{name}.yaml"
    if not path.is_file():
        return [*seen, f"{name}<MISSING>"]
    seen = [*seen, name]
    parents: list[str] = []
    in_defaults = False
    for line in path.read_text().splitlines():
        if re.match(r"^defaults:\s*(#.*)?$", line):
            in_defaults = True
            continue
        if not in_defaults:
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not stripped.startswith("- "):
            break
        parent = stripped[2:].strip().strip("\"'")
        if parent != "_self_":
            parents.append(parent)
    for parent in parents:
        seen = resolve_chain(parent, seen)
    return seen


def case_scope(node: str) -> str:
    """Which object a chain node is case-scoped to, if any."""
    match = re.search(r"(box\d{3}|bucket\d{3}|desk\d{3})", node)
    return match.group(1) if match else ""


def read_config_act(path_raw: str) -> dict[str, str]:
    path = Path(path_raw)
    if not path.is_absolute():
        path = REPO_ROOT / path_raw
    if not path.is_file():
        alt = re.search(r"(core4d/results/.*)$", path_raw or "")
        if not alt:
            return {}
        path = REPO_ROOT / "workspace" / alt.group(1)
        if not path.is_file():
            return {}
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        match = re.match(r"^(\w+):\s*(.+?)\s*$", line)
        if match and match.group(1) in SCALE_SENSITIVE:
            out[match.group(1)] = match.group(2)
    return out


def build_provenance(rows: list[dict[str, str]], source_rows: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        variant = row.get("variant", "")
        chain = resolve_chain(f"core4d_{variant}") if variant else []
        foreign = [n for n in chain[1:] if (s := case_scope(n)) and s != row.get("object_key", "")]
        record: dict[str, Any] = {
            "case_id": row.get("case_id", ""),
            "object_key": row.get("object_key", ""),
            "exp": row.get("exp", ""),
            "variant": variant,
            "chain_len": len(chain),
            "foreign_case_scoped_nodes": len(foreign),
            "foreign_nodes": ",".join(foreign),
            "chain": " <- ".join(chain),
        }
        src = source_rows.get((row.get("exp", ""), row.get("case_id", "")), {})
        record.update(read_config_act(src.get("config_act", "")))
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Hypotheses (pre-registered in the plan)
# ---------------------------------------------------------------------------

HYPOTHESES = """
H1 servo sag dominates the object position error
H2 side asymmetry is driven by the grasp lever arm (across objects)
H3 hand penetration is mostly inherited from the kinematic reference
H4 box024 saturates the CEM hand-gate hard floor more than box004/box001
H5 an OmniRetarget upstream control does not share the sag artifact
H6 within-object lever regression separates mechanism (b) from (a)
""".strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=AUDIT_TSV)
    args = parser.parse_args()
    if not args.audit.is_file():
        print(f"missing audit TSV: {args.audit}", file=sys.stderr)
        return 1
    rows = read_tsv(args.audit)

    # Source rows keyed by (exp, case_id) so provenance can reach config_act.
    source_rows: dict[tuple[str, str], dict[str, str]] = {}
    for exp, tsv in (
        ("E172", "workspace/core4d/results/E172/s6_downstream/eval/full/e171_case_metrics.tsv"),
        ("E173", "workspace/core4d/results/E173/s6_downstream/eval/full/e173_case_metrics.tsv"),
        ("E174", "workspace/core4d/results/E174/s6_downstream/eval/full/e174_case_metrics.tsv"),
        ("E189", "workspace/core4d/results/E189/s6_downstream/eval/full/e189_case_metrics.tsv"),
    ):
        path = REPO_ROOT / tsv
        if path.is_file():
            for src in read_tsv(path):
                source_rows[(exp, src["case_id"])] = src

    lines: list[str] = ["# E191 object-support audit — results", ""]
    lines.append(f"Rows scored: **{len(rows)}** from {len(set(r['exp'] for r in rows))} source experiments.")
    lines.append("")
    lines.append("> E191 ran no physics. Every number below is a re-scoring of already-completed")
    lines.append("> E172/E173/E174/E189 rollouts with additive columns; all pre-existing columns")
    lines.append("> reproduce bit-identically (see `e191_coverage.json`).")
    lines.append("")

    # ---- per-object table -------------------------------------------------
    lines.append("## Per-object summary")
    lines.append("")
    header = (
        "| exp | object | n | mass (kg) | max half-extent (m) | grip far arm (m) | lifted frac "
        "| obj pos err (cm) | z err lifted (cm) | mg/kp predicted (cm) | z share | near Δz (cm) "
        "| far Δz (cm) | asym (cm) | servo τ p95 (N·m) | ref pen | run pen | gate-floor sat |"
    )
    lines.append(header)
    lines.append("|" + "---|" * 18)
    per_object: list[tuple[str, str, list[dict[str, str]]]] = []
    for exp, obj in sorted({(r["exp"], r["object_key"]) for r in rows}):
        per_object.append((exp, obj, [r for r in rows if r["exp"] == exp and r["object_key"] == obj]))
    for exp, obj, subset in per_object:
        weight = mean_of(subset, "object_weight_N")
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                exp,
                obj,
                len(subset),
                fmt(mean_of(subset, "object_mass_kg"), 2),
                fmt(mean_of(subset, "object_max_half_extent_m")),
                fmt(mean_of(subset, "grip_far_arm_m")),
                fmt(mean_of(subset, "obj_lifted_frame_frac"), 2),
                fmt(mean_of(subset, "track_obj_pos_err_cm_mean"), 2),
                fmt(mean_of(subset, "track_obj_z_err_m_lifted_mean") * 100.0, 2),
                fmt(-weight / KP_POS * 100.0, 2),
                fmt(mean_of(subset, "track_obj_z_err_share_lifted"), 2),
                fmt(mean_of(subset, "obj_side_near_z_err_m") * 100.0, 2),
                fmt(mean_of(subset, "obj_side_far_z_err_m") * 100.0, 2),
                fmt(mean_of(subset, "obj_side_z_asym_cm"), 2),
                fmt(mean_of(subset, "object_guidance_torque_Nm_p95"), 1),
                fmt(mean_of(subset, "ref_hand_geom_penetration_frac")),
                fmt(mean_of(subset, "hand_geom_penetration_frac")),
                fmt(mean_of(subset, "hand_gate_floor_saturation_frac"), 4),
            )
        )
    lines.append("")
    lines.append(
        "Object mass is **not** a single constant: the boxes are all pinned at 5.0 kg, but the "
        "bucket/desk objects introduced by E174 span 2.0-120.4 kg. That variation is independent "
        "of object size and is exploited by H1b below."
    )
    lines.append("")

    # ---- H1 ---------------------------------------------------------------
    lines.append("## Pre-registered hypotheses")
    lines.append("")
    lines.append("```")
    lines.append(HYPOTHESES)
    lines.append("```")
    lines.append("")
    verdicts: list[str] = []

    shares = []
    zs = []
    for _exp, obj, subset in per_object:
        shares.append((obj, mean_of(subset, "track_obj_z_err_share_lifted"), mean_of(subset, "track_obj_z_err_m_lifted_mean")))
        zs.append(mean_of(subset, "track_obj_z_err_m_lifted_mean"))
    z_in_band = [(-0.13 <= z <= -0.07) for z in zs if np.isfinite(z)]
    share_ok = [s for _o, s, _z in shares if np.isfinite(s) and s >= 0.60]
    h1 = "PASS" if all(z_in_band) and len(share_ok) == len([s for _o, s, _z in shares if np.isfinite(s)]) else "REFUTE"
    verdicts.append(
        f"**H1 = {h1}** — lifted-frame object z error per group: "
        + ", ".join(f"{o} {fmt(z)}m (z share {fmt(s, 2)})" for o, s, z in shares)
        + f". Registered pass band was z ∈ [−0.13, −0.07] m AND z share ≥ 0.60 for every group."
    )

    # ---- H1b (post-hoc): does the sag magnitude track m*g/kp? -------------
    # NOT pre-registered. It only became possible once the audit revealed that
    # object mass varies 2.0-120.4 kg across E174's bucket/desk objects while
    # kp_pos stays at 500 N/m for every case. Objects that are dragged rather
    # than carried are excluded, since a floor-supported object never loads the
    # servo with its full weight.
    h1b_rows = []
    for _exp, obj, subset in per_object:
        lifted_frac = mean_of(subset, "obj_lifted_frame_frac")
        if not np.isfinite(lifted_frac) or lifted_frac < LIFTED_MIN_FRAC:
            continue
        predicted = -mean_of(subset, "object_weight_N") / KP_POS
        observed = mean_of(subset, "track_obj_z_err_m_lifted_mean")
        h1b_rows.append((obj, mean_of(subset, "object_mass_kg"), predicted, observed))
    excluded = [
        (o, mean_of(s, "obj_lifted_frame_frac"), mean_of(s, "object_mass_kg"))
        for _e, o, s in per_object
        if not (np.isfinite(mean_of(s, "obj_lifted_frame_frac")) and mean_of(s, "obj_lifted_frame_frac") >= LIFTED_MIN_FRAC)
    ]
    if len(h1b_rows) >= 4:
        pred = np.array([r[2] for r in h1b_rows])
        obs = np.array([r[3] for r in h1b_rows])
        rho1b, p1b, n1b = spearman(pred, obs)
        ratio = obs / pred
        # Sensitivity: groups whose predicted sag exceeds the object's own scale
        # cannot be servo-supported at all (the floor carries them), so they sit
        # in a different regime. Reported separately, the rule is NOT applied to
        # the headline number above.
        keep = pred > -0.30
        sens = ""
        if keep.sum() >= 4 and keep.sum() < len(h1b_rows):
            rho_s, p_s, n_s = spearman(pred[keep], obs[keep])
            ratio_s = obs[keep] / pred[keep]
            dropped = [h1b_rows[i][0] for i in range(len(h1b_rows)) if not keep[i]]
            sens = (
                f" **Sensitivity** excluding {', '.join(sorted(set(dropped)))} "
                f"(predicted sag > 30 cm, i.e. never actually servo-lifted): "
                f"ρ = {fmt(rho_s, 2)} (p={fmt(p_s, 4)}, n={n_s}), "
                f"observed/predicted median {fmt(float(np.median(ratio_s)), 2)}, "
                f"range [{fmt(float(ratio_s.min()), 2)}, {fmt(float(ratio_s.max()), 2)}]."
            )
        verdicts.append(
            f"**H1b (post-hoc, not pre-registered) ρ = {fmt(rho1b, 2)} (p={fmt(p1b, 4)}, n={n1b})** — "
            f"observed lifted-frame sag vs the m·g/kp_pos prediction, kp_pos={KP_POS:g} N/m held constant. "
            f"observed/predicted ratio: median {fmt(float(np.median(ratio)), 2)}, "
            f"range [{fmt(float(ratio.min()), 2)}, {fmt(float(ratio.max()), 2)}]." + sens + " Per group: "
            + ", ".join(f"{o} {m:g}kg pred {fmt(p * 100, 1)}cm obs {fmt(x * 100, 1)}cm" for o, m, p, x in h1b_rows)
            + (
                ". Excluded as dragged-not-carried: "
                + ", ".join(f"{o} (lifted {fmt(f, 2)}, {m:g}kg)" for o, f, m in excluded)
                if excluded
                else ""
            )
        )

    # ---- H2: cross-object lever regression --------------------------------
    obj_lever = np.array([mean_of(s, "object_max_half_extent_m") for _e, _o, s in per_object])
    obj_asym = np.array([mean_of(s, "obj_side_z_asym_cm") for _e, _o, s in per_object])
    rho2, p2, n2 = spearman(obj_lever, obj_asym)
    h2 = "PASS" if np.isfinite(rho2) and rho2 >= 0.8 else ("REFUTE" if np.isfinite(rho2) and rho2 < 0.5 else "INCONCLUSIVE")
    verdicts.append(f"**H2 = {h2}** — across {n2} exp×object groups, Spearman ρ(max half-extent, side asymmetry) = {fmt(rho2, 2)} (p={fmt(p2, 4)}). Registered: PASS ρ≥0.8, REFUTE ρ<0.5.")

    # ---- H3 ---------------------------------------------------------------
    h3_rows = []
    for _exp, obj, subset in per_object:
        ref_pen = mean_of(subset, "ref_hand_geom_penetration_frac")
        run_pen = mean_of(subset, "hand_geom_penetration_frac")
        h3_rows.append((obj, ref_pen, run_pen))
    h3 = "PASS" if all(np.isfinite(a) and a >= 0.45 and a >= b for _o, a, b in h3_rows) else "REFUTE"
    verdicts.append(
        f"**H3 = {h3}** — ref/run hand penetration fraction: "
        + ", ".join(f"{o} {fmt(a, 2)}/{fmt(b, 2)}" for o, a, b in h3_rows)
        + ". Registered: every group ref ≥ 0.45 AND ref ≥ run."
    )

    # ---- H4 ---------------------------------------------------------------
    by_obj = group_by([r for r in rows if r["exp"] in ("E172", "E173", "E189")], "object_key")
    sat = {o: mean_of(g, "hand_gate_floor_saturation_frac") for o, g in by_obj.items()}
    others = [sat.get("box004", math.nan), sat.get("box001", math.nan)]
    others = [v for v in others if np.isfinite(v)]
    b24 = sat.get("box024", math.nan)
    h4 = "PASS" if others and np.isfinite(b24) and b24 >= 2.0 * max(others) and max(others) > 0 else "REFUTE"
    verdicts.append(
        f"**H4 = {h4}** — hand-gate hard-floor saturation fraction: "
        + ", ".join(f"{o} {fmt(v, 4)}" for o, v in sorted(sat.items()))
        + ". Registered: box024 ≥ 2× box004/box001."
    )

    # ---- H5 ---------------------------------------------------------------
    verdicts.append(
        "**H5 = NOT TESTABLE** — the audit found no OmniRetarget upstream rollout set. "
        "`E174` was cited upstream as the \"OmniRetarget bucket control\", but its rows carry "
        "`method=E174_E170PRG_nonbox_candidate_r1` and `plan/190` describes it as SPIDER's own "
        "bucket/desk extension of the frozen E170 PRG recipe. There is therefore no Omni-side "
        "upstream table in this repo, and the symmetric-audit gap is wider than R018 assumed."
    )

    # ---- H6: within-object lever regression -------------------------------
    lines_h6 = []
    any_significant = False
    any_low_power = False
    for exp, obj in sorted({(r["exp"], r["object_key"]) for r in rows}):
        subset = [r for r in rows if r["exp"] == exp and r["object_key"] == obj]
        if len(subset) < 6:
            continue
        lever = np.array([num(r, "grip_far_arm_m") for r in subset])
        asym = np.array([num(r, "obj_side_z_asym_cm") for r in subset])
        spread = float(np.nanmax(lever) - np.nanmin(lever)) if np.isfinite(lever).any() else math.nan
        rho, p, n = spearman(lever, asym)
        low_power = np.isfinite(spread) and spread < 0.10
        any_low_power |= low_power
        if np.isfinite(rho) and rho >= 0.6 and p < 0.05:
            any_significant = True
        lines_h6.append(
            f"| {exp} | {obj} | {n} | {fmt(spread, 3)} | {fmt(rho, 2)} | {fmt(p, 4)} | "
            f"{'low-power (<0.10 m spread)' if low_power else ''} |"
        )
    h6 = "PASS" if any_significant else ("LOW POWER" if any_low_power else "REFUTE")
    verdicts.append(
        f"**H6 = {h6}** — within-object regression of side asymmetry on the per-case grip lever "
        "(object geometry and every threshold held constant, so only mechanism (b) can move it):"
    )

    for verdict in verdicts:
        lines.append(f"- {verdict}")
        lines.append("")
    lines.append("| exp | object | n | lever spread (m) | ρ | p | note |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.extend(lines_h6)
    lines.append("")

    # ---- A6 provenance ----------------------------------------------------
    provenance = build_provenance(rows, source_rows)
    if provenance:
        fields: list[str] = []
        for record in provenance:
            for key in record:
                if key not in fields:
                    fields.append(key)
        with PROVENANCE_TSV.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(provenance)
        distinct = {
            key: sorted({str(r.get(key, "")) for r in provenance if r.get(key, "") != ""})
            for key in SCALE_SENSITIVE
        }
        varying = {k: v for k, v in distinct.items() if len(v) > 1}
        constant = {k: v[0] for k, v in distinct.items() if len(v) == 1}
        lines.append("## Config provenance (A6)")
        lines.append("")
        lines.append(f"Wrote `{PROVENANCE_TSV.name}` ({len(provenance)} cases).")
        lines.append("")
        lines.append(
            f"Of {len(SCALE_SENSITIVE)} tracked metric-length / scale-sensitive parameters, "
            f"**{len(constant)} are identical across every object** and {len(varying)} vary."
        )
        lines.append("")
        lines.append("| parameter | value(s) |")
        lines.append("|---|---|")
        for key, value in sorted(constant.items()):
            lines.append(f"| `{key}` | `{value}` (constant) |")
        for key, values in sorted(varying.items()):
            lines.append(f"| `{key}` | {', '.join(f'`{v}`' for v in values)} |")
        lines.append("")
        foreign = sorted({(r["object_key"], r["foreign_case_scoped_nodes"]) for r in provenance})
        lines.append("Foreign (other-object) case-scoped nodes in each object's defaults chain: "
                     + ", ".join(f"{o} {n}" for o, n in foreign) + ".")
        lines.append("")

    REPORT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {REPORT_MD}")
    print(f"wrote {PROVENANCE_TSV}")
    for verdict in verdicts:
        print(" -", re.sub(r"\*\*", "", verdict.split(" — ")[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
