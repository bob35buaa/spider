#!/usr/bin/env python3
"""E215 evaluation: score every rot-aug rollout and pair it against the SAME
case+arm orig rollout, through the public ``eval.core.core_metrics``.

The orig baseline is each case's own selected-arm orig CEM (located by
preflight_baseline_audit's candidate manifests): aug and orig then differ in
exactly one thing -- the +/-45 deg object yaw (+ 0.2 m lateral) approach
perturbation -- on the identical arm, scored by the same evaluator and gates.

Three rulers, mirroring the E199/E202/E208/E213 augmentation line:
  1. build correctness (approach yaw / degraded flag) -- carried from the manifest.
  2. absolute 6-gate release standard -- verbatim, so "pass" means the same.
  3. paired delta vs the same case+arm orig -- the only ruler that answers "did
     rotating cost anything".  Reported mean+std+worst, never cherry-picked.

C4 (plan247): rot's obj-ori error rise is the inherent cost of rotating and is
reported but NOT counted as degradation; degradation = a NEW fall/diverge or an
obj-pos/ori mean increase > 25% vs the same case+arm orig.

Usage:
    .venv/bin/python workspace/core4d/scripts/eval/runners/eval_E215_rot_augmentation.py
    ... --limit N
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

_RUNNERS = Path(__file__).resolve().parent
sys.path.insert(0, str(_RUNNERS.parents[1]))                      # scripts/
sys.path.insert(0, str(_RUNNERS.parents[1] / "experiments/E215"))
sys.path.insert(0, str(_RUNNERS))

from eval.core.core_metrics import (  # noqa: E402
    EVAL_METRIC_STANDARD_ID, EvalConfig, evaluate_sequence,
)
from eval.core.motion_health import run_health  # noqa: E402
from eval_E187_e178_compat import body_z_p95  # noqa: E402

import e215_common as C  # noqa: E402
import preflight_baseline_audit as BASE  # noqa: E402

METHOD = "E167A_zOnlyBody"
HAND_VARIANT = "rubber_hull"

KEY_METRICS = (
    "track_obj_pos_err_cm_mean", "track_obj_ori_err_deg_mean", "track_obj_z_abs_err_cm_mean",
    "track_root_pos_err_cm_mean", "track_root_ori_err_deg_mean",
    "track_eef_pos_err_cm_mean", "track_eef_ori_err_deg_mean", "body_z_err_p95_m",
    "hand_object_physics_contact_in_mask_frac", "hand_object_physics_penetration_3mm_frame_frac",
    "leg_penetration_frac", "fall_flag",
)
GATE_THRESHOLDS = {"object_pos": 20.0, "object_ori": 10.0, "contact": 0.50,
                   "hand_penetration": 0.30, "lower_body": 0.10}

# pre-registered C4 (plan247): rot obj-ori rise is inherent, reported not gated.
C4_MAX_MEAN_INCREASE_PCT = 0.25
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
    scoring.setdefault("variant", f"{C.EXP_ID}_{row['case_id']}_{row.get('aug_variant', '')}_{row.get('arm_group', '')}")
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
        "aug_variant": row["aug_variant"], "arm_group": row.get("arm_group", ""),
        "base_variant": row.get("base_variant", ""),
        "target_task": row.get("target_task", ""),
        "approach_yaw_deg_max": row.get("approach_yaw_deg_max", ""),
        "degraded_yaw": row.get("degraded_yaw", ""),
        "metric_standard_id": EVAL_METRIC_STANDARD_ID,
        "outdir_npz": C.rel(paths["qpos"]), "scene_act": C.rel(paths["scene"]),
    })
    return item


# --------------------------------------------------------------------------
# Row assembly
# --------------------------------------------------------------------------
def aug_rows() -> list[dict[str, str]]:
    """cem-done rows from the 3 shard manifests (CEM wrote status there), else main."""
    done = {"run_complete_pending_eval", "cem_ok"}
    import string
    shards = [C.MANIFEST_DIR / f"e215_priority_manifest.shard{s}.tsv" for s in string.ascii_uppercase[:3]]
    files = [f for f in shards if f.is_file()] or [C.PRIORITY_MANIFEST]
    rows: list[dict[str, str]] = []
    for f in files:
        rows += [r for r in C.read_tsv(f) if r.get("status") in done]
    return rows


def _baseline_full_row(case: dict[str, str]) -> dict[str, str] | None:
    """The full orig baseline row (result_npz + scene_act + trajectory + mask)."""
    b = BASE.find_baseline(case)
    if not (b["npz_exists"] and b["scene_act"] and b["trajectory"] and b["contact_mask"]):
        return None
    return {
        "case_id": case["case_id"], "object_key": case["object_key"],
        "aug_variant": "orig", "arm_group": case["arm_group"], "base_variant": case["base_variant"],
        "result_npz": b["result_npz"], "scene_act": b["scene_act"],
        "trajectory": b["trajectory"], "contact_mask": b["contact_mask"],
        "retarget_variant_id": b["retarget_variant_id"],
        "orig_arm_note": b["arm_note"],
    }


# --------------------------------------------------------------------------
# Paired statistics (identical estimators to the E208/E213 line)
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
    orig_pass = sum(1 for o, _ in pairs if o)
    aug_pass = sum(1 for _, a in pairs if a)
    out: dict[str, Any] = {
        "n_pairs": n, "orig_pass": orig_pass, "aug_pass": aug_pass,
        "b_orig_only": b, "c_aug_only": c,
        "orig_pass_rate": (orig_pass / n) if n else math.nan,
        "aug_pass_rate": (aug_pass / n) if n else math.nan,
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
            "arm_group": item.get("arm_group", ""), "base_variant": item.get("base_variant", ""),
            "approach_yaw_deg_max": item.get("approach_yaw_deg_max", ""),
            "degraded_yaw": item.get("degraded_yaw", ""),
            "orig_all_gates_pass": bool(base.get("all_gates_pass")),
            "aug_all_gates_pass": bool(item.get("all_gates_pass")),
            "orig_fall": bool(base.get("fall_flag")), "aug_fall": bool(item.get("fall_flag")),
            "aug_failure_modes": item.get("numeric_failure_modes", ""),
        }
        for metric in KEY_METRICS:
            b, a = finite(base.get(metric)), finite(item.get(metric))
            row[f"orig_{metric}"], row[f"aug_{metric}"] = b, a
            row[f"delta_{metric}"] = a - b if math.isfinite(a) and math.isfinite(b) else math.nan
        out.append(row)
    return out


def _cell(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_cases = len({r["case_id"] for r in rows})
    new_falls = sum(1 for r in rows if r["aug_fall"] and not r["orig_fall"])
    cell: dict[str, Any] = {
        "n": len(rows), "n_cases": n_cases, "indicative": n_cases < INDICATIVE_MIN_N,
        "new_falls": new_falls,
        "gates": mcnemar([(bool(r["orig_all_gates_pass"]), bool(r["aug_all_gates_pass"])) for r in rows]),
        "metrics": {m: paired_tests(np.array([finite(r[f"delta_{m}"]) for r in rows], dtype=float))
                    for m in KEY_METRICS},
    }
    # relative mean increase for obj pos/ori (C4)
    for m in ("track_obj_pos_err_cm_mean", "track_obj_ori_err_deg_mean"):
        origs = np.array([finite(r[f"orig_{m}"]) for r in rows], dtype=float)
        augs = np.array([finite(r[f"aug_{m}"]) for r in rows], dtype=float)
        om, am = np.nanmean(origs), np.nanmean(augs)
        cell[f"{m}_orig_mean"] = float(om)
        cell[f"{m}_aug_mean"] = float(am)
        cell[f"{m}_mean_increase_pct"] = float((am - om) / om) if om > 1e-9 else math.nan
    return cell


def stratify(deltas: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        "pooled": {"all": deltas}, "by_variant": {}, "by_object": {}, "by_arm_group": {},
        "by_object_variant": {}, "by_base_variant": {},
    }
    for row in deltas:
        groups["by_variant"].setdefault(row["aug_variant"], []).append(row)
        groups["by_object"].setdefault(row["object_key"], []).append(row)
        groups["by_arm_group"].setdefault(row.get("arm_group", "?"), []).append(row)
        groups["by_object_variant"].setdefault(f"{row['object_key']}|{row['aug_variant']}", []).append(row)
        groups["by_base_variant"].setdefault(row.get("base_variant", "?"), []).append(row)
    return {level: {key: _cell(rows) for key, rows in sorted(cells.items())}
            for level, cells in groups.items()}


def c4_verdict(strata: dict[str, Any]) -> dict[str, Any]:
    """Per arm group: no NEW falls, obj-pos mean increase <=25%. obj-ori reported."""
    checks: dict[str, Any] = {}
    for group, cell in strata["by_arm_group"].items():
        pos_pct = cell.get("track_obj_pos_err_cm_mean_mean_increase_pct", math.nan)
        pos_ok = math.isnan(pos_pct) or pos_pct <= C4_MAX_MEAN_INCREASE_PCT
        checks[group] = {
            "n": cell["n"], "n_cases": cell["n_cases"], "indicative": cell["indicative"],
            "new_falls": cell["new_falls"],
            "obj_pos_mean_increase_pct": pos_pct,
            "obj_ori_mean_increase_pct": cell.get("track_obj_ori_err_deg_mean_mean_increase_pct", math.nan),
            "pass_rate_drop": cell["gates"]["pass_rate_drop"],
            "verdict": "pass" if (cell["new_falls"] == 0 and pos_ok) else "fail",
        }
    return {
        "granularity": "per_arm_group", "pre_registered": True,
        "primary_metric": C4_PRIMARY_METRIC, "max_mean_increase_pct": C4_MAX_MEAN_INCREASE_PCT,
        "note": ("obj-ori rise is rot's inherent cost -- reported, not gated; degradation = a "
                 "new fall or obj-pos mean increase > 25% vs the same case+arm orig"),
        "by_arm_group": checks,
        "verdict": "pass" if all(c["verdict"] == "pass" for c in checks.values()) else "fail",
    }


def union_fields(items: list[dict[str, Any]]) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        for key in item:
            seen.setdefault(key, None)
    return list(seen)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=C.EVAL_DIR)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = EvalConfig()
    augs = aug_rows()
    if args.limit:
        augs = augs[: args.limit]
    if not augs:
        raise SystemExit("no aug rows with a done status yet -- is the CEM queue finished?")

    scored: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for i, row in enumerate(augs, 1):
        try:
            scored.append(score(row, cfg, npz_key="outdir_npz", scene_key="scene_act"))
        except Exception as exc:  # noqa: BLE001
            failed.append({"case_id": row["case_id"], "aug_variant": row["aug_variant"],
                           "error": f"{type(exc).__name__}: {exc}"})
        if i % 10 == 0:
            print(f"  scored {i}/{len(augs)} aug", flush=True)

    # orig baselines only for cases with >=1 scored aug
    scored_cases = {r["case_id"] for r in scored}
    cases_by_id = {c["case_id"]: c for c in C.load_e215_cases()}
    orig_scored: list[dict[str, Any]] = []
    orig_missing: list[str] = []
    for case_id in sorted(scored_cases):
        base_row = _baseline_full_row(cases_by_id[case_id])
        if base_row is None:
            orig_missing.append(case_id)
            continue
        try:
            orig_scored.append(score(base_row, cfg, npz_key="result_npz", scene_key="scene_act"))
        except Exception as exc:  # noqa: BLE001
            failed.append({"case_id": case_id, "aug_variant": "orig", "error": f"{type(exc).__name__}: {exc}"})

    all_items = scored + orig_scored
    deltas = deltas_vs_orig(all_items)
    strata = stratify(deltas)

    out_dir = C.repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    C.write_tsv(out_dir / "e215_rot_rollout.tsv", all_items, union_fields(all_items))
    C.write_tsv(out_dir / "e215_rot_deltas.tsv", deltas, union_fields(deltas))

    summary = {
        "experiment": C.EXP_ID, "run_id": C.RUN_ID, "generated_at": C.now(),
        "metric_standard_id": EVAL_METRIC_STANDARD_ID,
        "gate_thresholds": GATE_THRESHOLDS, "key_metrics": list(KEY_METRICS),
        "orig_baseline": "each case's same-arm orig CEM (preflight candidate manifests), scored fresh",
        "n_aug_scored": len(scored), "n_orig_scored": len(orig_scored),
        "n_pairs": len(deltas), "n_failed": len(failed), "failures": failed,
        "orig_baseline_missing": orig_missing,
        "arm_group_counts": dict(Counter(d["arm_group"] for d in deltas)),
        "strata": strata, "C4": c4_verdict(strata),
    }
    C.write_json(out_dir / "e215_rot_eval_summary.json", summary)

    print(f"\nscored {len(scored)} aug + {len(orig_scored)} orig -> {len(deltas)} pairs ({len(failed)} failed)")
    if orig_missing:
        print(f"  orig baseline MISSING for: {orig_missing}")
    for group, cell in sorted(strata["by_arm_group"].items()):
        m = cell["metrics"][C4_PRIMARY_METRIC]
        print(f"  {group:20s} n={cell['n']:3d} ({cell['n_cases']}c) obj_pos HL {m['hl_shift']:+.3f}cm "
              f"pos+{cell.get('track_obj_pos_err_cm_mean_mean_increase_pct', float('nan')):.0%} "
              f"ori+{cell.get('track_obj_ori_err_deg_mean_mean_increase_pct', float('nan')):.0%} "
              f"newfall={cell['new_falls']}" + ("  [indicative]" if cell["indicative"] else ""))
    print(f"  C4 verdict: {summary['C4']['verdict']}")
    print(f"-> {C.rel(out_dir)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
