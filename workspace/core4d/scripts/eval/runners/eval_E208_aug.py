#!/usr/bin/env python3
"""E208 evaluation: score every aug run and pair it against its SAME-CASE E206 orig.

Three rulers, in the order they can invalidate each other:

  1. build correctness -- carried over from P5 (offset, yaw, pair count); this
     script only joins it on, it does not recompute it
  2. absolute gates -- the same six thresholds E199/E202 used, copied verbatim so
     a "pass" means the same thing across the whole augmentation line
  3. paired delta vs the same case's orig -- the only ruler that can answer
     "did augmenting cost anything", because it differences out per-case
     difficulty

Statistics are pre-registered (plan238 C4) and computed with the estimator the
data shape calls for, not the convenient one:

* **Hodges-Lehmann** median of pairwise Walsh averages as the point estimate.
  n is small (pooled 105, smallest stratum 1) and ``track_obj_pos_err_cm_mean``
  is heavy-tailed -- a mean difference gets dragged by a single failed run.
* **Wilcoxon signed-rank** as the primary test, with an **exact paired sign test**
  reported alongside wherever n < 10, because Wilcoxon's normal approximation is
  not trustworthy there.
* **McNemar exact** for the paired pass-rate comparison.

Stratified pooled / per-variant / per-object / per-(object,variant) and -- new in
E208 -- **per offset band**.  The band stratification is the point of including
the weak variants at all: it answers, for the first time in this line of work, at
what augmentation amplitude tracking accuracy actually starts to degrade.

``--rescore-orig`` (default on) re-scores E206's own PRG rollouts through THIS
script's ``core_metrics`` and asserts agreement with E206's published numbers.
That checks scoring reproducibility -- the same npz scored twice -- and is NOT a
check on retarget reproducibility (F15), which P1 removed by construction.  The
two must not be conflated in the write-up.

Usage:
    .venv/bin/python .../eval/runners/eval_E208_aug.py
    ... --no-rescore-orig        # trust E206's published per-case numbers
    ... --jobs 8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/E208"))

from eval.core.core_metrics import (  # noqa: E402
    EVAL_METRIC_STANDARD_ID,
    EvalConfig,
    evaluate_sequence,
)
from eval.core.motion_health import run_health  # noqa: E402

import e208_common as C  # noqa: E402

METHOD = C.E206.BASE_REWARD_METHOD          # E167A_zOnlyBody
HAND_VARIANT = "rubber_hull"                # E206's hand patch, same for orig and aug

E206_CASE_METRICS = C.E206.S6_DIR / "eval/two_arm/e206_arm_case_metrics.tsv"

KEY_METRICS = (
    "track_obj_pos_err_cm_mean", "track_obj_ori_err_deg_mean", "track_obj_z_abs_err_cm_mean",
    "track_root_pos_err_cm_mean", "track_root_ori_err_deg_mean",
    "track_eef_pos_err_cm_mean", "track_eef_ori_err_deg_mean", "body_z_err_p95_m",
    "hand_object_physics_contact_in_mask_frac", "hand_object_physics_penetration_3mm_frame_frac",
    "leg_penetration_frac", "fall_flag",
)
GATE_THRESHOLDS = {
    "object_pos": 20.0, "object_ori": 10.0, "contact": 0.50,
    "hand_penetration": 0.30, "lower_body": 0.10,
}

# plan238 C4, pre-registered before any aug result was scored.
C4_MAX_PASS_RATE_DROP = 0.15
C4_MAX_OBJ_POS_HL_CM = 5.0
C4_PRIMARY_METRIC = "track_obj_pos_err_cm_mean"

RESCORE_TOL = 1e-6
INDICATIVE_MIN_N = 3


def finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def person_idx(case_id: str) -> int:
    return 0 if case_id.lower().endswith("_p1") else 1


def gate_set(item: dict[str, Any]) -> dict[str, bool]:
    return {
        "fall": not bool(item.get("fall_flag")),
        "object_pos": finite(item.get("track_obj_pos_err_cm_mean")) <= GATE_THRESHOLDS["object_pos"],
        "object_ori": finite(item.get("track_obj_ori_err_deg_mean")) <= GATE_THRESHOLDS["object_ori"],
        "contact": finite(item.get("hand_object_physics_contact_in_mask_frac")) >= GATE_THRESHOLDS["contact"],
        "hand_penetration": finite(item.get("hand_object_physics_penetration_3mm_frame_frac"))
        <= GATE_THRESHOLDS["hand_penetration"],
        "lower_body": finite(item.get("leg_penetration_frac")) <= GATE_THRESHOLDS["lower_body"],
    }


def score(row: dict[str, str], cfg: EvalConfig, *, npz_key: str, scene_key: str) -> dict[str, Any]:
    scoring = dict(row)
    scoring.setdefault("spider_method_id", METHOD)
    scoring.setdefault("hand_collision_variant_id", HAND_VARIANT)
    paths = {
        "qpos": C.repo_path(row[npz_key]),
        "scene": C.repo_path(row[scene_key]),
        "trajectory": C.repo_path(row["trajectory"]),
        "mask": C.repo_path(row["contact_mask"]),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{row['case_id']}/{row.get('aug_variant')}:{label}:{path}")
    item = evaluate_sequence(
        row=scoring, method=METHOD, hand_collision_variant_id=HAND_VARIANT,
        qpos_path=paths["qpos"], scene_xml=paths["scene"], config=cfg,
        kin_ref_path=paths["trajectory"], contact_mask_path=paths["mask"],
        person_idx=person_idx(row["case_id"]),
    )
    item.update(run_health(paths["qpos"], paths["scene"], cfg))
    gates = gate_set(item)
    for name, ok in gates.items():
        item[f"{name}_gate_pass"] = ok
    item["all_gates_pass"] = all(gates.values())
    item["numeric_release_pass"] = all(gates.values())
    item["numeric_failure_modes"] = ",".join(n for n, ok in gates.items() if not ok)
    item.update({
        "case_id": row["case_id"], "object_key": row["object_key"],
        "aug_variant": row["aug_variant"], "target_task": row["target_task"],
        "effective_retarget_variant": row.get("effective_retarget_variant", ""),
        "rescue_state": row.get("rescue_state", ""),
        "offset_band": row.get("offset_band", ""),
        "approach_trans_offset_m_max": row.get("approach_trans_offset_m_max", ""),
        "metric_standard_id": EVAL_METRIC_STANDARD_ID,
        "outdir_npz": C.rel(paths["qpos"]), "scene_act": C.rel(paths["scene"]),
        "video": row.get("video", ""),
        "f15_divergent": row.get("f15_divergent", "0"),
        "contact_mask_blind_frac": C.BLIND3CM_BY_OBJECT.get(row["object_key"], ""),
    })
    return item


def e206_published() -> dict[str, dict[str, str]]:
    """E206's per-case PRG numbers, keyed by case_id."""
    if not E206_CASE_METRICS.is_file():
        return {}
    out = {}
    for row in C.read_tsv(E206_CASE_METRICS):
        if row.get("arm", "").lower() != "prg":
            continue
        out[row["case_id"]] = row
    return out


def rescore_check(scored: list[dict[str, Any]]) -> dict[str, Any]:
    """C7(c): the same npz scored twice must agree to 1e-6."""
    published = e206_published()
    mismatches: list[dict[str, Any]] = []
    checked = 0
    for item in scored:
        ref = published.get(item["case_id"])
        if ref is None:
            mismatches.append({"case_id": item["case_id"], "issue": "no_published_row"})
            continue
        checked += 1
        for metric in KEY_METRICS:
            a, b = finite(item.get(metric)), finite(ref.get(metric))
            if math.isnan(a) and math.isnan(b):
                continue
            if math.isnan(a) or math.isnan(b) or abs(a - b) > RESCORE_TOL:
                mismatches.append({"case_id": item["case_id"], "metric": metric,
                                   "rescored": a, "published": b,
                                   "abs_delta": abs(a - b) if not (math.isnan(a) or math.isnan(b)) else None})
    return {
        "enabled": True, "n_checked": checked, "tolerance": RESCORE_TOL,
        "n_mismatches": len(mismatches), "mismatches": mismatches[:40],
        "verdict": "pass" if not mismatches else "fail",
        "scope": ("scoring reproducibility -- the same rollout npz scored twice. NOT "
                  "retarget reproducibility (F15), which P1 removed by seeding `_original` "
                  "byte-for-byte from E206."),
    }


PUBLISHED_GATE_COL = {
    "fall": "fall_gate_pass", "object_pos": "object_pos_gate_pass",
    "object_ori": "object_ori_gate_pass", "contact": "contact_gate_pass",
    "hand_penetration": "hand_penetration_gate_pass", "lower_body": "lower_body_gate_pass",
}


def criterion_crosscheck(case_ids: set[str]) -> dict[str, Any]:
    """Measure how this script's 6 gates differ from E206's own published gates.

    E206 released on TWELVE gates; the augmentation line (E199/E202/E208) uses a
    six-gate subset, and at least one threshold genuinely differs -- E206's
    lower_body gate passes at ~0.20 leg_penetration_frac while the augmentation
    standard is 0.10.  That is fine for C4, which applies one criterion to orig
    and aug alike, but it means three different "pass rates" exist for the same
    22 cases (E206 human review, E206 12-gate release, E208 6-gate).  Reporting
    the disagreement makes them impossible to conflate by accident.
    """
    published = e206_published()
    disagreements: list[dict[str, Any]] = []
    n = 0
    for cid, ref in published.items():
        if cid not in case_ids:
            continue
        n += 1
        mine = gate_set({m: ref.get(m) for m in KEY_METRICS} | {
            "fall_flag": str(ref.get("fall_flag", "")).strip().lower() in {"true", "1"}})
        for gate, col in PUBLISHED_GATE_COL.items():
            if col not in ref:
                continue
            pub = str(ref[col]).strip().lower() in {"true", "1"}
            if mine[gate] != pub:
                disagreements.append({"case_id": cid, "gate": gate,
                                      "e208_6gate": mine[gate], "e206_published": pub})
    return {
        "n_cases": n,
        "gates": sorted(PUBLISHED_GATE_COL),
        "thresholds": GATE_THRESHOLDS,
        "n_disagreements": len(disagreements),
        "disagreements": disagreements,
        "note": ("E206 released on 12 gates; this is the 6-gate augmentation-line subset "
                 "(E199/E202), applied identically to orig and aug. Any disagreement "
                 "listed here is a THRESHOLD difference, not a scoring bug -- E206's "
                 "lower_body gate passes at ~0.20 leg_penetration_frac vs 0.10 here."),
    }


def orig_from_published(case_ids: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cid, row in e206_published().items():
        if cid not in case_ids:
            continue
        item: dict[str, Any] = {
            "case_id": cid, "aug_variant": "orig",
            "object_key": row.get("object_key", cid.split("_")[0]),
            "offset_band": "orig",
        }
        # Copy every published numeric column, not just KEY_METRICS. The review
        # feed's 12-gate view needs metrics C4 does not use (e.g.
        # hand_object_release_false_contact_3mm_frac); dropping them here made
        # orig rows fail gates they actually pass, biasing the side-by-side the
        # human reviewer sees against orig.
        for col, val in row.items():
            if col in {"case_id", "arm", "object_key", "variant"} or col.endswith("_gate_pass"):
                continue
            num = finite(val)
            item.setdefault(col, num if math.isfinite(num) else val)
        for metric in KEY_METRICS:
            item[metric] = finite(row.get(metric))
        gates = gate_set({**item, "fall_flag": str(row.get("fall_flag", "")).strip().lower()
                          in {"true", "1"}})
        item["fall_flag"] = 0.0 if gates["fall"] else 1.0
        item["all_gates_pass"] = all(gates.values())
        item["numeric_failure_modes"] = ",".join(n for n, ok in gates.items() if not ok)
        out.append(item)
    return out


# --------------------------------------------------------------------------
# Paired statistics
# --------------------------------------------------------------------------
def hodges_lehmann(diffs: np.ndarray) -> float:
    """Median of all pairwise Walsh averages -- the robust paired location shift."""
    d = diffs[np.isfinite(diffs)]
    if d.size == 0:
        return math.nan
    if d.size == 1:
        return float(d[0])
    i, j = np.triu_indices(d.size, k=0)
    return float(np.median((d[i] + d[j]) / 2.0))


def paired_tests(diffs: np.ndarray) -> dict[str, Any]:
    d = diffs[np.isfinite(diffs)]
    out: dict[str, Any] = {"n": int(d.size), "hl_shift": hodges_lehmann(d)}
    if d.size == 0:
        return out
    out["mean_delta"] = float(d.mean())
    out["std_delta"] = float(d.std(ddof=1)) if d.size > 1 else 0.0
    out["worst_delta"] = float(d.max())
    nz = d[d != 0]
    try:
        from scipy import stats

        if nz.size >= 1:
            out["wilcoxon_p"] = float(stats.wilcoxon(nz, alternative="two-sided").pvalue)
        n_pos = int((nz > 0).sum())
        if nz.size:
            out["sign_test_p"] = float(
                stats.binomtest(n_pos, nz.size, 0.5, alternative="two-sided").pvalue
            )
            out["n_worse"] = n_pos
            out["n_better"] = int(nz.size - n_pos)
        if d.size < 10:
            out["primary_test"] = "sign_test_exact"
            out["note"] = "n<10: Wilcoxon's normal approximation is not trustworthy here"
        else:
            out["primary_test"] = "wilcoxon"
    except Exception as exc:  # noqa: BLE001
        out["test_error"] = f"{type(exc).__name__}: {exc}"
    return out


def mcnemar(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    """Paired pass-rate comparison. b = orig pass & aug fail, c = the reverse."""
    b = sum(1 for o, a in pairs if o and not a)
    c = sum(1 for o, a in pairs if a and not o)
    n = len(pairs)
    out: dict[str, Any] = {
        "n_pairs": n, "orig_pass": sum(1 for o, _ in pairs if o),
        "aug_pass": sum(1 for _, a in pairs if a),
        "b_orig_only": b, "c_aug_only": c,
        "orig_pass_rate": (sum(1 for o, _ in pairs if o) / n) if n else math.nan,
        "aug_pass_rate": (sum(1 for _, a in pairs if a) / n) if n else math.nan,
    }
    out["pass_rate_drop"] = out["orig_pass_rate"] - out["aug_pass_rate"]
    try:
        from scipy import stats

        if b + c:
            out["exact_p"] = float(stats.binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)
        else:
            out["exact_p"] = 1.0
    except Exception as exc:  # noqa: BLE001
        out["test_error"] = f"{type(exc).__name__}: {exc}"
    return out


def stratify(deltas: list[dict[str, Any]]) -> dict[str, Any]:
    """pooled / per-variant / per-object / per-(object,variant) / per-offset-band."""
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        "pooled": {"all": deltas},
        "by_variant": {},
        "by_object": {},
        "by_object_variant": {},
        "by_offset_band": {},
    }
    for row in deltas:
        groups["by_variant"].setdefault(row["aug_variant"], []).append(row)
        groups["by_object"].setdefault(row["object_key"], []).append(row)
        groups["by_object_variant"].setdefault(
            f"{row['object_key']}|{row['aug_variant']}", []).append(row)
        groups["by_offset_band"].setdefault(row["offset_band"], []).append(row)

    out: dict[str, Any] = {}
    for level, cells in groups.items():
        out[level] = {}
        for key, rows in sorted(cells.items()):
            cell: dict[str, Any] = {
                "n": len(rows),
                "indicative": len(rows) < INDICATIVE_MIN_N,
                "gates": mcnemar([(bool(r["orig_all_gates_pass"]), bool(r["aug_all_gates_pass"]))
                                  for r in rows]),
                "metrics": {},
            }
            for metric in KEY_METRICS:
                d = np.array([finite(r[f"delta_{metric}"]) for r in rows], dtype=float)
                cell["metrics"][metric] = paired_tests(d)
            out[level][key] = cell
    return out


def deltas_vs_orig(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(r["case_id"], r["aug_variant"]): r for r in rows}
    out: list[dict[str, Any]] = []
    for (case_id, variant), item in sorted(by_key.items()):
        if variant == "orig":
            continue
        base = by_key.get((case_id, "orig"))
        if base is None:
            continue
        row: dict[str, Any] = {
            "object_key": item["object_key"], "case_id": case_id, "aug_variant": variant,
            "offset_band": item.get("offset_band", ""),
            "approach_trans_offset_m_max": item.get("approach_trans_offset_m_max", ""),
            "effective_retarget_variant": item.get("effective_retarget_variant", ""),
            "f15_divergent": item.get("f15_divergent", "0"),
            "contact_mask_blind_frac": item.get("contact_mask_blind_frac", ""),
            "orig_all_gates_pass": bool(base.get("all_gates_pass")),
            "aug_all_gates_pass": bool(item.get("all_gates_pass")),
            "aug_failure_modes": item.get("numeric_failure_modes", ""),
        }
        for metric in KEY_METRICS:
            b, a = finite(base.get(metric)), finite(item.get(metric))
            row[f"orig_{metric}"] = b
            row[f"aug_{metric}"] = a
            row[f"delta_{metric}"] = a - b if math.isfinite(a) and math.isfinite(b) else math.nan
            row[f"pct_{metric}"] = (
                (a - b) / b * 100.0
                if math.isfinite(a) and math.isfinite(b) and abs(b) > 1e-9 else math.nan
            )
        out.append(row)
    return out


def c4_verdict(strata: dict[str, Any]) -> dict[str, Any]:
    pooled = strata["pooled"]["all"]
    drop = pooled["gates"]["pass_rate_drop"]
    hl = pooled["metrics"][C4_PRIMARY_METRIC]["hl_shift"]
    checks = {
        "pass_rate_drop": {"value": drop, "bar": C4_MAX_PASS_RATE_DROP,
                           "verdict": "pass" if drop <= C4_MAX_PASS_RATE_DROP else "fail"},
        f"hl_{C4_PRIMARY_METRIC}": {"value": hl, "bar": C4_MAX_OBJ_POS_HL_CM,
                                    "verdict": "pass" if hl <= C4_MAX_OBJ_POS_HL_CM else "fail"},
    }
    return {
        "granularity": "per_case",
        "granularity_basis": ("G4 measured the aug IK bit-identical across processes "
                              "(24/24, max|dqpos|=0), so plan238 R7b's degradation to "
                              "distribution-level does not trigger"),
        "pre_registered": True,
        "checks": checks,
        "verdict": "pass" if all(c["verdict"] == "pass" for c in checks.values()) else "fail",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=C.PRIORITY_MANIFEST,
                    help="live run state; the CEM runner owns this file")
    ap.add_argument("--authority", type=Path, default=C.AUTHORITY_TSV,
                    help="static orig partners (reused_e206 rows)")
    ap.add_argument("--out-dir", type=Path, default=C.EVAL_DIR)
    ap.add_argument("--rescore-orig", dest="rescore", action="store_true", default=True)
    ap.add_argument("--no-rescore-orig", dest="rescore", action="store_false")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = EvalConfig()
    # Two files, two authorities.  The runner owns the priority manifest and is
    # the only place a live `status` exists; the authority TSV's aug rows are a
    # freeze-time snapshot and would report every run as unstarted.  The orig
    # partners live only in the authority TSV and the runner never touches them.
    aug_rows = [r for r in C.read_tsv(C.repo_path(args.manifest))
                if r.get("status") == "cem_ok"]
    orig_rows = [r for r in C.read_tsv(C.repo_path(args.authority))
                 if r["aug_variant"] == "orig"]
    if args.limit:
        aug_rows = aug_rows[: args.limit]
    if not aug_rows:
        raise SystemExit("no aug rows with status=cem_ok yet -- is the CEM queue finished?")

    scored: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for i, row in enumerate(aug_rows, 1):
        try:
            scored.append(score(row, cfg, npz_key="outdir_npz", scene_key="scene_act"))
        except Exception as exc:  # noqa: BLE001
            failed.append({"case_id": row["case_id"], "aug_variant": row["aug_variant"],
                           "error": f"{type(exc).__name__}: {exc}"})
        if i % 10 == 0:
            print(f"  scored {i}/{len(aug_rows)} aug", flush=True)

    rescore = {"enabled": False}
    if args.rescore:
        orig_scored: list[dict[str, Any]] = []
        for i, row in enumerate(orig_rows, 1):
            try:
                item = score({**row, "aug_variant": "orig"}, cfg,
                             npz_key="result_npz", scene_key="scene_act")
                item["offset_band"] = "orig"
                orig_scored.append(item)
            except Exception as exc:  # noqa: BLE001
                failed.append({"case_id": row["case_id"], "aug_variant": "orig",
                               "error": f"{type(exc).__name__}: {exc}"})
            if i % 7 == 0:
                print(f"  rescored {i}/{len(orig_rows)} orig", flush=True)
        rescore = rescore_check(orig_scored)
        orig_items = orig_scored
    else:
        orig_items = orig_from_published({r["case_id"] for r in orig_rows})

    all_items = scored + orig_items
    deltas = deltas_vs_orig(all_items)
    strata = stratify(deltas)

    out_dir = C.repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # aug and orig items carry different key sets (orig from the published table
    # is a subset), so the column list is their union in first-seen order --
    # letting write_tsv infer from row 0 would silently drop the rest.
    def union_fields(items: list[dict[str, Any]]) -> list[str]:
        seen: dict[str, None] = {}
        for item in items:
            for key in item:
                seen.setdefault(key, None)
        return list(seen)

    C.write_tsv(out_dir / "e208_aug_rollout.tsv", all_items, union_fields(all_items))
    C.write_tsv(out_dir / "e208_aug_deltas.tsv", deltas, union_fields(deltas))

    summary = {
        "experiment": C.EXP_ID, "run_id": C.RUN_ID, "generated_at": C.now(),
        "metric_standard_id": EVAL_METRIC_STANDARD_ID,
        "gate_thresholds": GATE_THRESHOLDS,
        "key_metrics": list(KEY_METRICS),
        "n_aug_scored": len(scored), "n_orig": len(orig_items),
        "n_pairs": len(deltas), "n_failed": len(failed), "failures": failed,
        "excluded_objects": list(C.EXCLUDED_OBJECT_KEYS),
        "excluded_reason": C.EXCLUDED_REASON,
        "offset_band_counts": dict(Counter(d["offset_band"] for d in deltas)),
        "rescore_orig": rescore,
        "gate_criterion_vs_e206": criterion_crosscheck({r["case_id"] for r in orig_rows}),
        "strata": strata,
        "C4": c4_verdict(strata),
        "caveats": {
            "chair006_blind3cm": {
                "value": C.BLIND3CM_BY_OBJECT.get("chair006"),
                "note": ("orig and aug share the same 3cm mask AND the same lowgeom proxy, "
                         "so the contact bias is common-mode and the delta stays valid; the "
                         "absolute contact numbers are not comparable across objects"),
            },
            "indicative_cells": (
                f"any stratum with n < {INDICATIVE_MIN_N} is marked indicative: point "
                "estimate only, no inference"
            ),
            "f15_watch": sorted(C.F15_DIVERGENT_CASES),
        },
    }
    C.write_json(out_dir / "e208_aug_eval_summary.json", summary)

    pooled = strata["pooled"]["all"]
    print(f"\nscored {len(scored)} aug + {len(orig_items)} orig -> {len(deltas)} pairs "
          f"({len(failed)} failed)")
    print(f"  pass rate orig {pooled['gates']['orig_pass_rate']:.3f} -> aug "
          f"{pooled['gates']['aug_pass_rate']:.3f} (drop {pooled['gates']['pass_rate_drop']:+.3f}, "
          f"bar {C4_MAX_PASS_RATE_DROP})")
    hl = pooled["metrics"][C4_PRIMARY_METRIC]
    print(f"  {C4_PRIMARY_METRIC}: HL {hl['hl_shift']:+.3f} cm (bar +{C4_MAX_OBJ_POS_HL_CM}), "
          f"n={hl['n']}, {hl.get('primary_test')} p={hl.get('wilcoxon_p', hl.get('sign_test_p'))}")
    for band, cell in sorted(strata["by_offset_band"].items()):
        m = cell["metrics"][C4_PRIMARY_METRIC]
        print(f"  band {band:8s} n={cell['n']:3d} HL {m['hl_shift']:+.3f} cm"
              + ("  [indicative]" if cell["indicative"] else ""))
    print(f"  rescore_orig: {rescore.get('verdict', 'skipped')} "
          f"({rescore.get('n_mismatches', '-')} mismatches)")
    print(f"  C4 verdict: {summary['C4']['verdict']}")
    print(f"-> {C.rel(out_dir)}")
    return 0 if not failed and summary["C4"]["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
