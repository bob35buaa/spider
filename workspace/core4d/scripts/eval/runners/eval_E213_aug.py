#!/usr/bin/env python3
"""E213 evaluation: score every selected-arm aug run and pair it against its
SAME-CASE SELECTED-ARM orig rollout.

The orig baseline here is NOT E206 PRG -- it is the case's own E212-selected-arm
rollout (the ``cem_result_npz`` in the paired export TSV, e.g. the E209 G1 / E211
gc08 / E212 gc06 run).  That is the only honest baseline: aug and orig then differ
in exactly one thing -- the object approach perturbation -- on the SAME arm.  Both
sides are scored fresh through this script's ``core_metrics`` so the comparison
is same-evaluator, same-gates.

Three rulers (same as the augmentation line E199/E202/E208):
  1. build correctness -- carried on from the manifest (offset/band), not recomputed.
  2. absolute 6-gate release standard -- copied verbatim so "pass" means the same.
  3. paired delta vs the same case's selected-arm orig -- differences out per-case
     difficulty; the only ruler that answers "did augmenting cost anything".

Pre-registered stats (report mean+std+worst, never cherry-pick):
  * Hodges-Lehmann point estimate (heavy-tailed, small n).
  * Wilcoxon signed-rank primary; exact sign test alongside when n<10.
  * McNemar exact for paired pass-rate.
  * case-level companion test (21 independent cases) because the ~5 variants per
    case share one orig rollout -> row-level p-values are anti-conservative.

The PRG-selected case (desk007_034_p1) reuses E208's PRG aug rollouts as its aug
runs (its selected arm IS PRG), paired against its E206 PRG orig.

Usage:
    .venv/bin/python workspace/core4d/scripts/eval/runners/eval_E213_aug.py
    ... --limit N --jobs 1
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
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/E213"))

from eval.core.core_metrics import (  # noqa: E402
    EVAL_METRIC_STANDARD_ID,
    EvalConfig,
    evaluate_sequence,
)
from eval.core.motion_health import run_health  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_E187_e178_compat import body_z_p95  # noqa: E402

import e213_common as C  # noqa: E402

METHOD = C.E208.E206.BASE_REWARD_METHOD    # E167A_zOnlyBody
HAND_VARIANT = "rubber_hull"               # E206 hand patch; same for orig and aug

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

# pre-registered C4 (plan243), before any aug result was scored.
C4_MAX_PASS_RATE_DROP = 0.15
C4_MAX_OBJ_POS_HL_CM = 5.0
C4_PRIMARY_METRIC = "track_obj_pos_err_cm_mean"
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


def score(row: dict[str, Any], cfg: EvalConfig, *, npz_key: str, scene_key: str) -> dict[str, Any]:
    scoring = dict(row)
    scoring.setdefault("spider_method_id", METHOD)
    scoring.setdefault("hand_collision_variant_id", HAND_VARIANT)
    scoring.setdefault(
        "variant",
        f"{C.EXP_ID}_{row['case_id']}_{row.get('aug_variant', '')}_{row.get('arm', '')}",
    )
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
    try:
        item["body_z_err_p95_m"] = body_z_p95(paths["qpos"], paths["scene"], paths["trajectory"])
    except Exception as exc:  # noqa: BLE001
        item["body_z_err_p95_m"] = math.nan
        item["body_z_error"] = f"{type(exc).__name__}: {exc}"
    gates = gate_set(item)
    for name, ok in gates.items():
        item[f"{name}_gate_pass"] = ok
    item["all_gates_pass"] = all(gates.values())
    item["numeric_release_pass"] = all(gates.values())
    item["numeric_failure_modes"] = ",".join(n for n, ok in gates.items() if not ok)
    item.update({
        "case_id": row["case_id"], "object_key": row["object_key"],
        "aug_variant": row["aug_variant"], "arm": row.get("arm", ""),
        "target_task": row.get("target_task", ""),
        "offset_band": row.get("offset_band", ""),
        "approach_trans_offset_m_max": row.get("approach_trans_offset_m_max", ""),
        "metric_standard_id": EVAL_METRIC_STANDARD_ID,
        "outdir_npz": C.rel(paths["qpos"]), "scene_act": C.rel(paths["scene"]),
    })
    return item


# --------------------------------------------------------------------------
# Row assembly
# --------------------------------------------------------------------------
def aug_rows() -> list[dict[str, str]]:
    """cem_ok rows from the 4 shard manifests (or the merged main) + PRG-reuse rows."""
    rows: list[dict[str, str]] = []
    shard_files = [C.manifest_path(s) for s in C.SHARDS if C.manifest_path(s).is_file()]
    if shard_files:
        for f in shard_files:
            rows += [r for r in C.read_tsv(f) if r.get("status") == "cem_ok"]
    elif C.SOURCE_MANIFEST.is_file():
        rows += [r for r in C.read_tsv(C.SOURCE_MANIFEST) if r.get("status") == "cem_ok"]

    # PRG-selected case: its aug runs are E208's PRG aug rollouts (selected arm = PRG).
    cases = {c["case_id"]: c for c in C.load_cases()}
    aug = C.aug_rows_by_case()
    for cid, case in cases.items():
        if not C.is_prg_case(case):
            continue
        for a in aug.get(cid, []):
            npz = C.e208_prg_result_npz(cid, a["aug_variant"])
            if not npz.is_file():
                continue
            rows.append({
                "case_id": cid, "object_key": case["object_key"], "aug_variant": a["aug_variant"],
                "arm": "PRG", "target_task": a["target_task"],
                "outdir_npz": C.rel(npz),
                "selected_scene_act": a["scene_act"],  # E208 aug PRG scene
                "trajectory": a["trajectory"], "contact_mask": a["contact_mask"],
                "offset_band": a.get("offset_band", ""),
                "approach_trans_offset_m_max": a.get("approach_trans_offset_m_max", ""),
                "status": "cem_ok_reuse_e208",
            })
    return rows


def orig_rows() -> list[dict[str, dict[str, str]]]:
    """One selected-arm orig row per case, from the paired export TSV."""
    out: list[dict[str, str]] = []
    for c in C.load_cases():
        out.append({
            "case_id": c["case_id"], "object_key": c["object_key"], "aug_variant": "orig",
            "arm": c["arm"], "offset_band": "orig",
            "cem_result_npz": c["orig_result_npz"],
            "orig_scene": C.rel(C.orig_selected_scene(c)),
            "trajectory": c["orig_trajectory"], "contact_mask": c["orig_contact_mask"],
        })
    return out


# --------------------------------------------------------------------------
# Paired statistics (identical estimators to the E208 augmentation line)
# --------------------------------------------------------------------------
def hodges_lehmann(diffs: np.ndarray) -> float:
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
            out["sign_test_p"] = float(stats.binomtest(n_pos, nz.size, 0.5, alternative="two-sided").pvalue)
            out["n_worse"] = n_pos
            out["n_better"] = int(nz.size - n_pos)
        out["primary_test"] = "sign_test_exact" if d.size < 10 else "wilcoxon"
    except Exception as exc:  # noqa: BLE001
        out["test_error"] = f"{type(exc).__name__}: {exc}"
    return out


def mcnemar(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    b = sum(1 for o, a in pairs if o and not a)
    c = sum(1 for o, a in pairs if a and not o)
    n = len(pairs)
    out: dict[str, Any] = {
        "n_pairs": n, "orig_pass": sum(1 for o, _ in pairs if o), "aug_pass": sum(1 for _, a in pairs if a),
        "b_orig_only": b, "c_aug_only": c,
        "orig_pass_rate": (sum(1 for o, _ in pairs if o) / n) if n else math.nan,
        "aug_pass_rate": (sum(1 for _, a in pairs if a) / n) if n else math.nan,
    }
    out["pass_rate_drop"] = out["orig_pass_rate"] - out["aug_pass_rate"]
    try:
        from scipy import stats
        out["exact_p"] = float(stats.binomtest(b, b + c, 0.5, alternative="two-sided").pvalue) if b + c else 1.0
    except Exception as exc:  # noqa: BLE001
        out["test_error"] = f"{type(exc).__name__}: {exc}"
    return out


def deltas_vs_orig(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(r["case_id"], r["aug_variant"]): r for r in rows}
    orig_by_case = {r["case_id"]: r for r in rows if r["aug_variant"] == "orig"}
    out: list[dict[str, Any]] = []
    for (case_id, variant), item in sorted(by_key.items()):
        if variant == "orig":
            continue
        base = orig_by_case.get(case_id)
        if base is None:
            continue
        row: dict[str, Any] = {
            "object_key": item["object_key"], "case_id": case_id, "aug_variant": variant,
            "arm": item.get("arm", ""), "offset_band": item.get("offset_band", ""),
            "approach_trans_offset_m_max": item.get("approach_trans_offset_m_max", ""),
            "orig_all_gates_pass": bool(base.get("all_gates_pass")),
            "aug_all_gates_pass": bool(item.get("all_gates_pass")),
            "aug_failure_modes": item.get("numeric_failure_modes", ""),
        }
        for metric in KEY_METRICS:
            b, a = finite(base.get(metric)), finite(item.get(metric))
            row[f"orig_{metric}"] = b
            row[f"aug_{metric}"] = a
            row[f"delta_{metric}"] = a - b if math.isfinite(a) and math.isfinite(b) else math.nan
        out.append(row)
    return out


def stratify(deltas: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        "pooled": {"all": deltas}, "by_variant": {}, "by_object": {},
        "by_arm": {}, "by_object_variant": {}, "by_offset_band": {},
    }
    for row in deltas:
        groups["by_variant"].setdefault(row["aug_variant"], []).append(row)
        groups["by_object"].setdefault(row["object_key"], []).append(row)
        groups["by_arm"].setdefault(row.get("arm", "?"), []).append(row)
        groups["by_object_variant"].setdefault(f"{row['object_key']}|{row['aug_variant']}", []).append(row)
        groups["by_offset_band"].setdefault(row["offset_band"], []).append(row)
    out: dict[str, Any] = {}
    for level, cells in groups.items():
        out[level] = {}
        for key, rows in sorted(cells.items()):
            n_cases = len({r["case_id"] for r in rows})
            cell: dict[str, Any] = {
                "n": len(rows), "n_cases": n_cases,
                "indicative": n_cases < INDICATIVE_MIN_N,
                "gates": mcnemar([(bool(r["orig_all_gates_pass"]), bool(r["aug_all_gates_pass"]))
                                  for r in rows]),
                "metrics": {m: paired_tests(np.array([finite(r[f"delta_{m}"]) for r in rows], dtype=float))
                            for m in KEY_METRICS},
            }
            out[level][key] = cell
    return out


def cluster_check(deltas: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in deltas:
        by_case.setdefault(row["case_id"], []).append(row)
    case_pass_delta: list[float] = []
    case_metric_delta: list[float] = []
    for rows in by_case.values():
        orig = 1.0 if bool(rows[0]["orig_all_gates_pass"]) else 0.0
        aug = sum(1.0 for r in rows if r["aug_all_gates_pass"]) / len(rows)
        case_pass_delta.append(orig - aug)
        vals = [finite(r[f"delta_{C4_PRIMARY_METRIC}"]) for r in rows]
        vals = [v for v in vals if math.isfinite(v)]
        if vals:
            case_metric_delta.append(float(np.median(vals)))
    out: dict[str, Any] = {
        "n_rows": len(deltas), "n_cases": len(by_case),
        "why": ("each case's ~5 variants share one orig rollout, so row-level "
                "Wilcoxon/McNemar are anti-conservative; point estimates unaffected"),
        "case_level_mean_pass_rate_drop": float(np.mean(case_pass_delta)) if case_pass_delta else math.nan,
        "case_level_median_pass_rate_drop": float(np.median(case_pass_delta)) if case_pass_delta else math.nan,
        "case_level_n_cases_worse": int(sum(1 for v in case_pass_delta if v > 0)),
        "case_level_n_cases_better": int(sum(1 for v in case_pass_delta if v < 0)),
        "case_level_n_cases_unchanged": int(sum(1 for v in case_pass_delta if v == 0)),
        "case_level_metric": paired_tests(np.array(case_metric_delta, dtype=float)),
    }
    try:
        from scipy import stats
        nz = [v for v in case_pass_delta if v != 0]
        if nz:
            out["case_level_pass_sign_test_p"] = float(
                stats.binomtest(sum(1 for v in nz if v > 0), len(nz), 0.5, alternative="two-sided").pvalue)
    except Exception as exc:  # noqa: BLE001
        out["test_error"] = f"{type(exc).__name__}: {exc}"
    return out


def c4_verdict(strata: dict[str, Any], clustering: dict[str, Any]) -> dict[str, Any]:
    pooled = strata["pooled"]["all"]
    drop = pooled["gates"]["pass_rate_drop"]
    hl = pooled["metrics"][C4_PRIMARY_METRIC]["hl_shift"]
    checks = {
        "pass_rate_drop_row_level": {"value": drop, "bar": C4_MAX_PASS_RATE_DROP,
                                     "verdict": "pass" if drop <= C4_MAX_PASS_RATE_DROP else "fail"},
        f"hl_{C4_PRIMARY_METRIC}": {"value": hl, "bar": C4_MAX_OBJ_POS_HL_CM,
                                    "verdict": "pass" if (math.isnan(hl) or hl <= C4_MAX_OBJ_POS_HL_CM) else "fail"},
    }
    return {
        "granularity": "per_case",
        "pre_registered": True,
        "primary_metric": C4_PRIMARY_METRIC,
        "checks": checks,
        "case_level_pass_rate_drop": clustering["case_level_mean_pass_rate_drop"],
        "verdict": "pass" if all(c["verdict"] == "pass" for c in checks.values()) else "fail",
        "note": ("row-level pass-rate drop is anti-conservative (clustered by case); "
                 "read the case-level companion in `clustering` as the honest test"),
    }


def union_fields(items: list[dict[str, Any]]) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        for key in item:
            seen.setdefault(key, None)
    return list(seen)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=C.EVAL_DIR / "aug")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = EvalConfig()
    augs = aug_rows()
    origs = orig_rows()
    if args.limit:
        augs = augs[: args.limit]
    if not augs:
        raise SystemExit("no aug rows with status=cem_ok yet -- is the CEM queue finished?")

    scored: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for i, row in enumerate(augs, 1):
        try:
            scored.append(score(row, cfg, npz_key="outdir_npz", scene_key="selected_scene_act"))
        except Exception as exc:  # noqa: BLE001
            failed.append({"case_id": row["case_id"], "aug_variant": row["aug_variant"],
                           "error": f"{type(exc).__name__}: {exc}"})
        if i % 10 == 0:
            print(f"  scored {i}/{len(augs)} aug", flush=True)

    # only score origs for cases that actually have >=1 scored aug (so the pooled
    # pass-rate is over comparable pairs, not padded by unpaired origs)
    scored_cases = {r["case_id"] for r in scored}
    orig_scored: list[dict[str, Any]] = []
    for row in origs:
        if row["case_id"] not in scored_cases:
            continue
        try:
            orig_scored.append(score(row, cfg, npz_key="cem_result_npz", scene_key="orig_scene"))
        except Exception as exc:  # noqa: BLE001
            failed.append({"case_id": row["case_id"], "aug_variant": "orig",
                           "error": f"{type(exc).__name__}: {exc}"})

    all_items = scored + orig_scored
    deltas = deltas_vs_orig(all_items)
    strata = stratify(deltas)
    clustering = cluster_check(deltas)

    out_dir = C.repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    C.write_tsv(out_dir / "e213_aug_rollout.tsv", all_items, union_fields(all_items))
    C.write_tsv(out_dir / "e213_aug_deltas.tsv", deltas, union_fields(deltas))

    summary = {
        "experiment": C.EXP_ID, "run_id": C.RUN_ID, "generated_at": C.now(),
        "metric_standard_id": EVAL_METRIC_STANDARD_ID,
        "gate_thresholds": GATE_THRESHOLDS, "key_metrics": list(KEY_METRICS),
        "orig_baseline": "each case's E212-selected-arm rollout (cem_result_npz), scored fresh",
        "n_aug_scored": len(scored), "n_orig_scored": len(orig_scored),
        "n_pairs": len(deltas), "n_failed": len(failed), "failures": failed,
        "offset_band_counts": dict(Counter(d["offset_band"] for d in deltas)),
        "arm_counts": dict(Counter(d["arm"] for d in deltas)),
        "strata": strata, "clustering": clustering,
        "C4": c4_verdict(strata, clustering),
    }
    C.write_json(out_dir / "e213_aug_eval_summary.json", summary)

    pooled = strata["pooled"]["all"]
    print(f"\nscored {len(scored)} aug + {len(orig_scored)} orig -> {len(deltas)} pairs ({len(failed)} failed)")
    print(f"  row-level pass rate orig {pooled['gates']['orig_pass_rate']:.3f} -> aug "
          f"{pooled['gates']['aug_pass_rate']:.3f} (drop {pooled['gates']['pass_rate_drop']:+.3f}, bar {C4_MAX_PASS_RATE_DROP})")
    hl = pooled["metrics"][C4_PRIMARY_METRIC]
    print(f"  {C4_PRIMARY_METRIC}: HL {hl['hl_shift']:+.3f} cm (bar +{C4_MAX_OBJ_POS_HL_CM}), "
          f"mean {hl.get('mean_delta', float('nan')):+.3f} std {hl.get('std_delta', float('nan')):.3f} "
          f"worst {hl.get('worst_delta', float('nan')):+.3f}, n={hl['n']}")
    print(f"  case-level pass drop {clustering['case_level_mean_pass_rate_drop']:+.3f} "
          f"(worse/better/same={clustering['case_level_n_cases_worse']}/"
          f"{clustering['case_level_n_cases_better']}/{clustering['case_level_n_cases_unchanged']})")
    for arm, cell in sorted(strata["by_arm"].items()):
        m = cell["metrics"][C4_PRIMARY_METRIC]
        print(f"  arm {arm:4s} n={cell['n']:3d} ({cell['n_cases']} cases) obj_pos HL {m['hl_shift']:+.3f} cm"
              + ("  [indicative]" if cell["indicative"] else ""))
    print(f"  C4 verdict: {summary['C4']['verdict']}")
    print(f"-> {C.rel(out_dir)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
