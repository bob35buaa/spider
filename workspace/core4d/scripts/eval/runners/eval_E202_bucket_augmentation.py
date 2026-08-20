#!/usr/bin/env python3
"""E202 public-core evaluation: score each bucket aug full-CEM run and compare
every translation variant against its SAME-CASE E178 orig baseline (C5).

Self-contained: uses only the public eval.core API (evaluate_sequence, run_health,
EvalConfig) -- rule 13. Orig baselines are the reused E178 full-CEM rollouts
(retarget confound: orig=omnirt_v1, aug=omnirt_v2 -- collision body identical).
Reports the FULL distribution (mean+std+worst, no cherry-picking), per-case
orig-vs-aug deltas, per-object stratification, and the feasibility distribution
(C6). Reads the E202 authority TSV (queued trans rows + reused_e178 orig rows).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/E202"))

from eval.core.core_metrics import (  # noqa: E402
    EVAL_METRIC_STANDARD_ID,
    EvalConfig,
    evaluate_sequence,
)
from eval.core.motion_health import run_health  # noqa: E402
import e202_common as C  # noqa: E402

METHOD = "E167A_zOnlyBody"
HAND_VARIANT = "rubber_hull"
E178_CASE_METRICS = (
    C.REPO / "workspace/core4d/results/E178/s6_downstream/eval/full/e178_case_metrics.tsv"
)

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


def finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def person_idx(case_id: str) -> int:
    return 0 if case_id.lower().endswith("_p1") else 1


def score(row: dict[str, str], cfg: EvalConfig) -> dict[str, Any]:
    scoring = dict(row)
    scoring.setdefault("spider_method_id", METHOD)
    scoring.setdefault("hand_collision_variant_id", HAND_VARIANT)
    qpos_path = C.repo_path(row["outdir_npz"])
    scene = C.repo_path(row["scene_act"])
    trajectory = C.repo_path(row["trajectory"])
    mask = C.repo_path(row["contact_mask"])
    for label, path in (("outdir_npz", qpos_path), ("scene_act", scene),
                        ("trajectory", trajectory), ("contact_mask", mask)):
        if not path.is_file():
            raise FileNotFoundError(f"{row['case_id']}:{label}:{path}")
    item = evaluate_sequence(row=scoring, method=METHOD, hand_collision_variant_id=HAND_VARIANT,
                             qpos_path=qpos_path, scene_xml=scene, config=cfg, kin_ref_path=trajectory,
                             contact_mask_path=mask, person_idx=person_idx(row["case_id"]))
    item.update(run_health(qpos_path, scene, cfg))
    gates = {
        "fall": not bool(item.get("fall_flag")),
        "object_pos": finite(item.get("track_obj_pos_err_cm_mean")) <= GATE_THRESHOLDS["object_pos"],
        "object_ori": finite(item.get("track_obj_ori_err_deg_mean")) <= GATE_THRESHOLDS["object_ori"],
        "contact": finite(item.get("hand_object_physics_contact_in_mask_frac")) >= GATE_THRESHOLDS["contact"],
        "hand_penetration": finite(item.get("hand_object_physics_penetration_3mm_frame_frac")) <= GATE_THRESHOLDS["hand_penetration"],
        "lower_body": finite(item.get("leg_penetration_frac")) <= GATE_THRESHOLDS["lower_body"],
    }
    for name, ok in gates.items():
        item[f"{name}_gate_pass"] = ok
    item["all_gates_pass"] = all(gates.values())
    item["numeric_failure_modes"] = ",".join(n for n, ok in gates.items() if not ok)
    item.update({
        "object_key": row["object_key"], "case_id": row["case_id"], "aug_variant": row["aug_variant"],
        "target_task": row["target_task"], "tier": row["tier"],
        "metric_standard_id": EVAL_METRIC_STANDARD_ID,
        "result_npz": C.rel(row["result_npz"]), "scene_act": C.rel(scene),
    })
    return item


def load_e178_orig(case_ids: set[str]) -> list[dict[str, Any]]:
    """Orig baseline = E178's canonical full-CEM eval (same public core_metrics).

    Avoids re-scoring the reused rollout (whose task-dir scene_act was overwritten
    by later experiments; the snapshot copy has unresolvable relative mesh paths).
    Pulls per-case KEY_METRICS from e178_case_metrics.tsv and recomputes the same
    gate set so orig/aug are compared identically.
    """
    if not E178_CASE_METRICS.is_file():
        return []
    out: list[dict[str, Any]] = []
    for row in C.read_tsv(E178_CASE_METRICS):
        cid = row.get("case_id", "")
        if cid not in case_ids:
            continue
        item: dict[str, Any] = {"case_id": cid, "aug_variant": "orig",
                                "object_key": row.get("object_key", cid.split("_")[0])}
        for metric in KEY_METRICS:
            item[metric] = finite(row.get(metric)) if metric != "fall_flag" else row.get(metric)
        gates = {
            "fall": str(row.get("fall_flag", "")).strip().lower() not in {"true", "1"},
            "object_pos": finite(row.get("track_obj_pos_err_cm_mean")) <= GATE_THRESHOLDS["object_pos"],
            "object_ori": finite(row.get("track_obj_ori_err_deg_mean")) <= GATE_THRESHOLDS["object_ori"],
            "contact": finite(row.get("hand_object_physics_contact_in_mask_frac")) >= GATE_THRESHOLDS["contact"],
            "hand_penetration": finite(row.get("hand_object_physics_penetration_3mm_frame_frac")) <= GATE_THRESHOLDS["hand_penetration"],
            "lower_body": finite(row.get("leg_penetration_frac")) <= GATE_THRESHOLDS["lower_body"],
        }
        item["fall_flag"] = 0.0 if gates["fall"] else 1.0
        item["all_gates_pass"] = all(gates.values())
        out.append(item)
    return out


def deltas_vs_orig(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-CASE pairing: (case_id, variant) vs (case_id, orig)."""
    by_case_variant = {(r["case_id"], r["aug_variant"]): r for r in rows}
    out: list[dict[str, Any]] = []
    for (case_id, variant), item in sorted(by_case_variant.items()):
        if variant == "orig":
            continue
        base = by_case_variant.get((case_id, "orig"))
        if base is None:
            continue
        row: dict[str, Any] = {"object_key": item["object_key"], "case_id": case_id, "aug_variant": variant,
                               "orig_all_gates_pass": base.get("all_gates_pass", ""),
                               "aug_all_gates_pass": item.get("all_gates_pass", ""),
                               "aug_failure_modes": item.get("numeric_failure_modes", "")}
        for metric in KEY_METRICS:
            b, a = finite(base.get(metric)), finite(item.get(metric))
            row[f"orig_{metric}"] = b
            row[f"aug_{metric}"] = a
            row[f"delta_{metric}"] = a - b if math.isfinite(a) and math.isfinite(b) else math.nan
            row[f"pct_{metric}"] = (a - b) / b * 100.0 if math.isfinite(a) and math.isfinite(b) and abs(b) > 1e-9 else math.nan
        out.append(row)
    return out


def _agg(group: list[dict[str, Any]], tag: str, dist: dict[str, Any]) -> None:
    import numpy as np
    for metric in KEY_METRICS:
        vals = np.array([finite(r.get(metric)) for r in group], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            dist[f"{tag}_{metric}_mean"] = float(vals.mean())
            dist[f"{tag}_{metric}_std"] = float(vals.std())
            dist[f"{tag}_{metric}_worst"] = float(vals.max())
    dist[f"{tag}_gate_pass_frac"] = (
        float(np.mean([1.0 if r.get("all_gates_pass") else 0.0 for r in group])) if group else math.nan
    )


def distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aug = [r for r in rows if r["aug_variant"] != "orig"]
    orig = [r for r in rows if r["aug_variant"] == "orig"]
    dist: dict[str, Any] = {"n_aug": len(aug), "n_orig": len(orig), "by_object": {}}
    _agg(orig, "orig", dist)
    _agg(aug, "aug", dist)
    for obj in sorted({r["object_key"] for r in rows}):
        sub = {"n_aug": 0, "n_orig": 0}
        _agg([r for r in orig if r["object_key"] == obj], "orig", sub)
        _agg([r for r in aug if r["object_key"] == obj], "aug", sub)
        sub["n_aug"] = sum(r["object_key"] == obj for r in aug)
        sub["n_orig"] = sum(r["object_key"] == obj for r in orig)
        dist["by_object"][obj] = sub
    return dist


def feasibility(authority: list[dict[str, str]], scored_ids: set[tuple[str, str]]) -> dict[str, Any]:
    """C6: per-object trans0/1/2 feasible (built+scored) vs infeasible (not built)."""
    out: dict[str, Any] = {}
    for obj in sorted({r["object_key"] for r in authority}):
        cases = {r["case_id"] for r in authority if r["object_key"] == obj and r["aug_variant"] == "orig"}
        per_variant = {}
        for variant in ("trans0", "trans1", "trans2"):
            built = sum(1 for r in authority if r["object_key"] == obj and r["aug_variant"] == variant)
            scored = sum(1 for cid in cases if (cid, variant) in scored_ids)
            per_variant[variant] = {"cases": len(cases), "built": built, "scored": scored}
        out[obj] = per_variant
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()

    authority = C.read_tsv(C.AUTHORITY_TSV)
    cfg = EvalConfig()
    # score only the aug (translation) rows we produced; orig baseline comes from
    # E178's canonical eval (load_e178_orig), not re-scored here.
    aug_authority = [r for r in authority if r.get("aug_variant") != "orig"]
    complete = [r for r in aug_authority
                if r.get("outdir_npz") and C.repo_path(r["outdir_npz"]).is_file()
                and r.get("result_npz") and C.repo_path(r["result_npz"]).is_file()]
    if args.require_all and len(complete) != len(aug_authority):
        raise SystemExit(f"complete aug rows={len(complete)} expected={len(aug_authority)}")

    scored: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for row in complete:
        try:
            scored.append(score(row, cfg))
            print(f"[scored] {row['object_key']} {row['case_id']} {row['aug_variant']}")
        except Exception as exc:  # noqa: BLE001
            errors.append({"case_id": row["case_id"], "aug_variant": row["aug_variant"],
                           "error": f"{type(exc).__name__}: {exc}"})
            print(f"[error] {row['case_id']} {row['aug_variant']}: {errors[-1]['error']}", file=sys.stderr)

    # orig baseline from E178 canonical eval, for the cases that have scored aug
    orig_rows = load_e178_orig({r["case_id"] for r in scored})
    scored = scored + orig_rows

    out = C.RESULTS / "s6_downstream/eval/full_augmentation"
    deltas = deltas_vs_orig(scored)
    scored_ids = {(r["case_id"], r["aug_variant"]) for r in scored if r["aug_variant"] != "orig"}
    C.write_tsv(out / "e202_aug_case_metrics.tsv", scored)
    C.write_tsv(out / "e202_aug_orig_deltas.tsv", deltas)
    if errors:
        C.write_tsv(out / "e202_aug_eval_errors.tsv", errors)
    n_aug_scored = sum(1 for r in scored if r["aug_variant"] != "orig")
    n_orig = sum(1 for r in scored if r["aug_variant"] == "orig")
    summary = {
        "created_at": C.now(),
        "aug_scored": n_aug_scored, "orig_baseline": n_orig,
        "aug_authority_rows": len(aug_authority), "complete": len(complete),
        "errors": len(errors),
        "confound_note": "orig=omnirt_v1 (reused E178 full-CEM); aug=omnirt_v2. Collision body identical (E178).",
        "distribution": distribution(scored),
        "feasibility": feasibility(authority, scored_ids),
        "by_object_scored": dict(Counter(r["object_key"] for r in scored if r["aug_variant"] != "orig")),
        "status": "pass" if not errors else "incomplete",
    }
    C.write_json(out / "e202_aug_eval_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
