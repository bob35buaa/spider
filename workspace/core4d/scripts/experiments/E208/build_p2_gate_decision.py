#!/usr/bin/env python3
"""E208: derive the P2 gate decision (G1-G4) from the artifacts, not from prose.

``p2_gate_decision.json`` is read by the eval runner -- ``G4.verdict`` decides
whether C4 is judged per-case or only at distribution level -- so it has to be a
machine-readable file derived from measurements, not a number transcribed out of
the log.  G1/G2/G3 were observed during P2/P3/P4 but only ever written down in
prose; this recomputes all four from the TSVs that recorded them.

**G4 / probe R6 is measured here for the first time**, at zero compute cost, out
of the concurrent-runner incident: 24 rot npz were quarantined rather than
deleted, and every one of them has a clean single-instance re-run counterpart.
Both were produced from the same seeded ``_original``, the same case file and the
same six env knobs, in two different processes -- which is exactly the question
R6 asks.

The asymmetry in what that can prove is important and is recorded in the output:

* bit-identical  -> conclusive.  The aug IK is reproducible across processes, so
  a per-case aug-vs-orig delta carries no IK noise and C4 can judge per case.
* differing      -> NOT conclusive.  The quarantined copies came from a run that
  overlapped a second instance racing on the per-object files ``pipeline.sh``
  rewrites, so a difference could be that race rather than nondeterminism.  It
  would only bound determinism from above, and C4 would degrade to
  distribution-level under plan238 R7b.

Also separates two things the feasibility TSV conflates.  ``pass2-rescue`` reruns
all five aug configs in the v2 tree because that tree only has ``_original`` to
short-circuit against, so 20 rows are marked ``rescued_v2`` while only the
variants that pass1 could not produce are actually adopted.  ``built`` means "this
pass produced an npz"; ``adopted`` means "this npz is in the experiment".  Anyone
computing C2 or a rescue yield off ``built`` alone gets 124 instead of 110.

Usage:
    .venv/bin/python .../E208/build_p2_gate_decision.py
    ... --annotate-feasibility     # add the derived `adopted` column to the TSV
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e208_common as C  # noqa: E402

QUARANTINE = C.DP / "quarantine/rot_npz_2026-09-05_concurrent_runners"

# plan238 G1: per-case retarget budget above which the holosoma opt-in patch
# (skip the rot configs) would have been worth asking the user for.
G1_BAR_MIN = 25.0


def adoption() -> dict[tuple[str, str], str]:
    """Which retarget variant each (case, variant) is actually taken from.

    Same rule ``build_augmented_tasks.effective_variants`` applies: pass1 wins,
    a rescue only fills a variant pass1 could not produce.
    """
    chosen: dict[tuple[str, str], str] = {}
    for row in C.read_tsv(C.FEASIBILITY_TSV):
        if row["rescue_state"] not in C.BUILT_STATES:
            continue
        key = (row["case_id"], row["variant"])
        if key not in chosen or row["pass"] == "pass1":
            chosen[key] = row["retarget_variant"]
    return chosen


def gate_g1(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Retarget cost per case, all five variants.

    ``wall_s`` is the WHOLE CASE's pipeline wall time, written identically onto
    each of that case's variant rows -- one ``pipeline.sh`` invocation produces
    all five.  Summing the rows would overcount 5x, so the per-case value is read
    off one row and the equality is asserted rather than assumed.
    """
    per_case: dict[str, set[float]] = {}
    for row in rows:
        if row["pass"] != "pass1":
            continue
        per_case.setdefault(row["case_id"], set()).add(float(row["wall_s"] or 0))
    inconsistent = {c: sorted(v) for c, v in per_case.items() if len(v) > 1}
    mins = sorted(next(iter(v)) / 60.0 for v in per_case.values() if len(v) == 1)
    return {
        "gate": "G1 retarget cost",
        "bar_min_per_case": G1_BAR_MIN,
        "n_cases": len(per_case),
        "median_min": round(statistics.median(mins), 2) if mins else None,
        "max_min": round(max(mins), 2) if mins else None,
        "wall_s_semantics": "per case (one pipeline.sh run emits all five variants)",
        "cases_with_inconsistent_wall_s": inconsistent,
        "verdict": "pass" if mins and not inconsistent and max(mins) <= G1_BAR_MIN else "fail",
        "note": ("Under the bar, so the holosoma opt-in patch to skip the rot configs was "
                 "never needed -- and after F6 the rot configs turned out to be the point."),
    }


def gate_g2(rows: list[dict[str, str]], chosen: dict[tuple[str, str], str]) -> dict[str, Any]:
    by_state: dict[str, int] = {}
    for row in rows:
        by_state[row["rescue_state"]] = by_state.get(row["rescue_state"], 0) + 1
    adopted_by_variant: dict[str, int] = {}
    for (_case, _variant), rv in chosen.items():
        adopted_by_variant[rv] = adopted_by_variant.get(rv, 0) + 1

    n_adopted = len(chosen)
    tier, action = C.l_tier(n_adopted, C.POTENTIAL_AUG_TASKS)
    return {
        "gate": "G2 aug yield",
        "rows_in_feasibility_tsv": len(rows),
        "rows_with_npz": sum(1 for r in rows if r["built"] == "1"),
        "adopted": n_adopted,
        "potential": C.POTENTIAL_AUG_TASKS,
        "adopted_by_retarget_variant": adopted_by_variant,
        "rescue_state_counts": by_state,
        "adopted_rescues": sum(
            1 for (case, _v), rv in chosen.items()
            if rv == "omnirt_v2" and case not in C.SOURCE_V2_CASES
        ),
        "recomputed_but_not_adopted": sum(1 for r in rows if r["built"] == "1") - n_adopted,
        "min_to_escalate": C.MIN_AUG_TASKS_ESCALATE,
        "l_tier": tier, "l_tier_action": action,
        "verdict": "pass" if n_adopted >= C.MIN_AUG_TASKS_ESCALATE else "fail",
        "note": ("`rows_with_npz` counts every npz any pass produced; `adopted` counts the "
                 "ones in the experiment. They differ because pass2-rescue recomputes all "
                 "five configs in the v2 tree (only `_original` is there to short-circuit "
                 "against) while only pass1's failures are taken from it."),
    }


def gate_g3(arts: list[dict[str, str]]) -> dict[str, Any]:
    built = [a for a in arts if a.get("status", "").startswith("built")]
    trans_yaw = [float(a["approach_yaw_deg_max"]) for a in built if a["aug_variant"].startswith("trans")]
    rot_yaw = [float(a["approach_yaw_deg_max"]) for a in built if a["aug_variant"].startswith("rot")]
    pair_bad = [
        a["target_task"] for a in built
        if int(a["compiled_robot_object_pair_count"]) != 18 * int(a["object_geom_count"])
    ]
    return {
        "gate": "G3 build correctness",
        "n_built": len(built),
        "trans_yaw_deg_max": round(max(trans_yaw), 4) if trans_yaw else None,
        "rot_yaw_deg_max": round(max(rot_yaw), 4) if rot_yaw else None,
        "rot_yaw_deg_min": round(min(rot_yaw), 4) if rot_yaw else None,
        "pair_count_mismatches": pair_bad,
        "verdict": "pass" if not pair_bad and (not trans_yaw or max(trans_yaw) <= 1.0) else "fail",
        "note": ("trans variants must stay yaw-free (<=1 deg); rot variants carry the "
                 "45 deg yaw, decayed by however late trim_start lands (F7)."),
    }


def gate_g4(*, tol: float = 0.0) -> dict[str, Any]:
    """R6: aug-IK reproducibility across processes, from the quarantined re-run."""
    pairs: list[dict[str, Any]] = []
    for q in sorted(QUARANTINE.rglob("*_rot_*.npz")):
        live = C.DP / q.relative_to(QUARANTINE)
        if not live.is_file():
            pairs.append({"npz": C.rel(q), "status": "no_live_counterpart"})
            continue
        with np.load(q, allow_pickle=True) as d:
            a = np.asarray(d["qpos"], dtype=np.float64)
        with np.load(live, allow_pickle=True) as d:
            b = np.asarray(d["qpos"], dtype=np.float64)
        row: dict[str, Any] = {
            "npz": C.rel(live),
            "sha_equal": int(C.sha256(q) == C.sha256(live)),
            "shape_equal": int(a.shape == b.shape),
        }
        if a.shape == b.shape:
            row["max_abs_dqpos"] = float(np.abs(a - b).max())
            row["max_abs_dobj_pos_m"] = float(np.abs(a[:, 36:39] - b[:, 36:39]).max())
        row["status"] = "ok"
        pairs.append(row)

    comparable = [p for p in pairs if "max_abs_dqpos" in p]
    worst = max((p["max_abs_dqpos"] for p in comparable), default=None)
    identical = all(p["sha_equal"] == 1 for p in comparable) and bool(comparable)
    return {
        "gate": "G4 aug IK determinism (probe R6)",
        "source": ("quarantined rot npz from the 2026-09-05 concurrent-runner incident vs "
                   "the clean single-instance re-run of the same variants"),
        "n_pairs": len(pairs), "n_comparable": len(comparable),
        "all_bit_identical": identical,
        "max_abs_dqpos": worst,
        "tolerance": tol,
        "verdict": "deterministic" if identical else "nondeterministic_or_raced",
        "c4_granularity": "per_case" if identical else "distribution_level",
        "note": (
            "Bit-identical is conclusive: same seeded `_original`, same case file, same six "
            "env knobs, two different processes. A DIFFERENCE would not be, because the "
            "quarantined copies overlapped a second instance racing on the per-object files "
            "pipeline.sh rewrites -- it would only bound determinism from above, and C4 "
            "would degrade to distribution-level per plan238 R7b."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotate-feasibility", action="store_true",
                    help="add the derived `adopted` column to the feasibility TSV")
    ap.add_argument("--out", type=Path, default=C.P2_GATE_JSON)
    args = ap.parse_args()

    rows = C.read_tsv(C.FEASIBILITY_TSV)
    arts = C.read_tsv(C.ARTIFACTS_TSV)
    chosen = adoption()

    gates = {
        "G1": gate_g1(rows),
        "G2": gate_g2(rows, chosen),
        "G3": gate_g3(arts),
        "G4": gate_g4(),
    }
    payload = {
        "experiment": C.EXP_ID, "run_id": C.RUN_ID, "generated_at": C.now(),
        "derived_from": [C.rel(C.FEASIBILITY_TSV), C.rel(C.ARTIFACTS_TSV), C.rel(QUARANTINE)],
        "excluded_objects": list(C.EXCLUDED_OBJECT_KEYS),
        "excluded_reason": C.EXCLUDED_REASON,
        "note": ("G1-G3 were observed during P2-P4 but only recorded in prose; recomputed "
                 "here from the TSVs. G4 is measured for the first time."),
        **gates,
    }
    C.write_json(args.out, payload)

    if args.annotate_feasibility:
        for row in rows:
            key = (row["case_id"], row["variant"])
            row["adopted"] = int(row["built"] == "1" and chosen.get(key) == row["retarget_variant"])
        fields = list(rows[0].keys())
        C.write_tsv(C.FEASIBILITY_TSV, rows, fields)
        n = sum(r["adopted"] for r in rows)
        print(f"annotated {C.rel(C.FEASIBILITY_TSV)}: adopted={n} of {len(rows)} rows")

    for key in ("G1", "G2", "G3", "G4"):
        g = gates[key]
        print(f"[{g['verdict']:24s}] {g['gate']}")
    print(f"  G1 median {gates['G1']['median_min']} min/case, max {gates['G1']['max_min']} "
          f"(bar {G1_BAR_MIN})")
    print(f"  G2 adopted {gates['G2']['adopted']}/{gates['G2']['potential']} "
          f"({gates['G2']['adopted_rescues']} rescued, "
          f"{gates['G2']['recomputed_but_not_adopted']} recomputed-not-adopted) "
          f"-> {gates['G2']['l_tier']}")
    print(f"  G3 built {gates['G3']['n_built']}, trans yaw<={gates['G3']['trans_yaw_deg_max']} deg, "
          f"rot yaw {gates['G3']['rot_yaw_deg_min']}-{gates['G3']['rot_yaw_deg_max']} deg")
    print(f"  G4 {gates['G4']['n_comparable']} pairs, bit-identical="
          f"{gates['G4']['all_bit_identical']}, max|dqpos|={gates['G4']['max_abs_dqpos']} "
          f"-> C4 granularity {gates['G4']['c4_granularity']}")
    print(f"-> {C.rel(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
