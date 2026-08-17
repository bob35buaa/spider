#!/usr/bin/env python3
"""E199 full-scale (plan229) evaluation: score every translation-augmented box
full-CEM run and pair each against its SAME-CASE orig baseline.

Scope difference vs the pilot runner (eval_E199_augmentation.py): there are many
cases per object, and `orig` is NOT re-run -- it is the existing E198 A0/PRG full
CEM rollout. To keep the orig-vs-aug comparison under a single scoring contract
(no metric-standard drift), this re-scores each A0 orig rollout with the SAME
public-core `score()` + `EvalConfig()` used for the aug runs (both land the same
metric_standard_id core4d-e154-physics-contact-v1). Pairing is by case_id.

Reports the full distribution (mean+std+worst, no cherry-picking), per-object
strata, per-case orig-vs-aug deltas, and the translation feasibility distribution
(C5: how many of trans0/1/2 survived upstream IK + initial-overlap gating).
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

RUNNERS = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNNERS.parents[1]))                    # scripts/eval on path (eval.core)
sys.path.insert(0, str(RUNNERS.parents[1] / "experiments/E199"))

from eval.core.core_metrics import EVAL_METRIC_STANDARD_ID, EvalConfig  # noqa: E402
import e199_common as C  # noqa: E402
# reuse the pilot's single-contract scorer + constants (same experiment, public core)
from eval_E199_augmentation import (  # noqa: E402
    GATE_THRESHOLDS,
    KEY_METRICS,
    finite,
    score,
)


def _norm_cid(cid: str) -> str:
    """Canonical case-id for joining aug rows (..._person1/person2, rebuilt from
    task_info) with the E198 A0 arm_cache (..._p1/p2)."""
    return cid.replace("_person", "_p")


def a0_orig_rows() -> dict[str, dict[str, str]]:
    """norm(case_id) -> synthetic scoring row for the reused E198 A0/PRG orig rollout."""
    out: dict[str, dict[str, str]] = {}
    for r in C.read_tsv(C.E198_ARM_CACHE):
        if r.get("arm") != "A0" or r.get("object_key") not in C.BOX_OBJECTS:
            continue
        cid = _norm_cid(r["case_id"])
        if cid in out:
            continue
        out[cid] = {
            "outdir_npz": r.get("qpos_path", "") or r.get("outdir_npz", ""),
            "result_npz": r.get("result_npz", ""),
            "scene_act": r["scene_xml"],
            "trajectory": r["trajectory"],
            "contact_mask": r["contact_mask"],
            "object_key": r["object_key"], "case_id": cid, "aug_variant": "orig",
            "target_task": Path(r["scene_xml"]).parent.name, "tier": "orig",
            # metadata fields evaluate_sequence expects on the row:
            "variant": r.get("variant", "orig"), "method": r.get("method", ""),
        }
    return out


def gate_pack(item: dict[str, Any]) -> dict[str, Any]:
    gates = {
        "fall": not bool(item.get("fall_flag")),
        "object_pos": finite(item.get("track_obj_pos_err_cm_mean")) <= GATE_THRESHOLDS["object_pos"],
        "object_ori": finite(item.get("track_obj_ori_err_deg_mean")) <= GATE_THRESHOLDS["object_ori"],
        "contact": finite(item.get("hand_object_physics_contact_in_mask_frac")) >= GATE_THRESHOLDS["contact"],
        "hand_penetration": finite(item.get("hand_object_physics_penetration_3mm_frame_frac")) <= GATE_THRESHOLDS["hand_penetration"],
        "lower_body": finite(item.get("leg_penetration_frac")) <= GATE_THRESHOLDS["lower_body"],
    }
    return {"all_gates_pass": all(gates.values()),
            "numeric_failure_modes": ",".join(n for n, ok in gates.items() if not ok)}


def stats(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    import numpy as np
    vals = np.array([finite(r.get(metric)) for r in rows], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if not vals.size:
        return {}
    return {"mean": float(vals.mean()), "std": float(vals.std()),
            "worst": float(vals.max()), "n": int(vals.size)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()

    manifest = C.read_tsv(C.FULLSCALE_MANIFEST)
    cfg = EvalConfig()
    complete = [r for r in manifest
                if C.repo_path(r["outdir_npz"]).is_file() and C.repo_path(r["result_npz"]).is_file()]
    print(f"[fullscale-eval] manifest={len(manifest)} complete={len(complete)}", flush=True)
    if args.require_all and len(complete) != len(manifest):
        raise SystemExit(f"complete rows={len(complete)} expected={len(manifest)}")

    # score every completed aug run
    aug_scored: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for row in complete:
        try:
            item = score(row, cfg)
            aug_scored.append(item)
            print(f"[aug] {row['case_id']} {row['aug_variant']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            errors.append({"case_id": row["case_id"], "aug_variant": row["aug_variant"],
                           "error": f"{type(exc).__name__}: {exc}"})
            print(f"[error-aug] {row['case_id']} {row['aug_variant']}: {errors[-1]['error']}", file=sys.stderr)

    # score the reused A0 orig for every case that has >=1 completed aug (same contract)
    orig_pool = a0_orig_rows()
    need_orig = sorted({_norm_cid(r["case_id"]) for r in aug_scored})
    orig_scored: dict[str, dict[str, Any]] = {}
    for cid in need_orig:
        base = orig_pool.get(cid)
        if base is None:
            errors.append({"case_id": cid, "aug_variant": "orig", "error": "no_A0_arm_cache_row"})
            continue
        try:
            item = score(base, cfg)
            orig_scored[cid] = item
            print(f"[orig] {cid}", flush=True)
        except Exception as exc:  # noqa: BLE001
            errors.append({"case_id": cid, "aug_variant": "orig", "error": f"{type(exc).__name__}: {exc}"})
            print(f"[error-orig] {cid}: {errors[-1]['error']}", file=sys.stderr)

    # per-case orig-vs-aug deltas
    deltas: list[dict[str, Any]] = []
    for a in aug_scored:
        cid = _norm_cid(a["case_id"])
        o = orig_scored.get(cid)
        if o is None:
            continue
        row: dict[str, Any] = {
            "object_key": a["object_key"], "case_id": cid, "aug_variant": a["aug_variant"],
            "orig_all_gates_pass": o.get("all_gates_pass", ""),
            "aug_all_gates_pass": a.get("all_gates_pass", ""),
            "aug_failure_modes": a.get("numeric_failure_modes", ""),
        }
        for m in KEY_METRICS:
            b, v = finite(o.get(m)), finite(a.get(m))
            row[f"orig_{m}"] = b
            row[f"aug_{m}"] = v
            row[f"delta_{m}"] = v - b if math.isfinite(v) and math.isfinite(b) else math.nan
            row[f"pct_{m}"] = (v - b) / b * 100.0 if math.isfinite(v) and math.isfinite(b) and abs(b) > 1e-9 else math.nan
        deltas.append(row)

    # distributions: overall + per object
    def dist_for(aug: list[dict[str, Any]], orig: list[dict[str, Any]]) -> dict[str, Any]:
        d: dict[str, Any] = {"n_aug": len(aug), "n_orig": len(orig)}
        for tag, grp in (("aug", aug), ("orig", orig)):
            for m in KEY_METRICS:
                s = stats(grp, m)
                if s:
                    d[f"{tag}_{m}_mean"] = s["mean"]; d[f"{tag}_{m}_std"] = s["std"]; d[f"{tag}_{m}_worst"] = s["worst"]
            import numpy as np
            d[f"{tag}_gate_pass_frac"] = float(np.mean([1.0 if r.get("all_gates_pass") else 0.0 for r in grp])) if grp else math.nan
        return d

    orig_list = list(orig_scored.values())
    overall = dist_for(aug_scored, orig_list)
    by_object: dict[str, Any] = {}
    for obj in sorted(C.BOX_OBJECTS):
        a = [r for r in aug_scored if r["object_key"] == obj]
        o = [r for r in orig_list if r["object_key"] == obj]
        if a or o:
            by_object[obj] = dist_for(a, o)

    # C5 feasibility: per object, per-case count of trans variants present in the manifest
    feas: dict[str, Any] = {}
    per_case_variants: dict[str, set] = defaultdict(set)
    for r in manifest:
        per_case_variants[r["case_id"]].add(r["aug_variant"])
    cases_by_obj = defaultdict(list)
    obj_of_case = {r["case_id"]: r["object_key"] for r in manifest}
    for cid, vs in per_case_variants.items():
        cases_by_obj[obj_of_case[cid]].append(len(vs & {"trans0", "trans1", "trans2"}))
    for obj in sorted(cases_by_obj):
        counts = cases_by_obj[obj]
        feas[obj] = {"cases": len(counts), "trans_variants_total": sum(counts),
                     "full_3of3": sum(1 for c in counts if c == 3),
                     "partial": sum(1 for c in counts if 0 < c < 3),
                     "none": sum(1 for c in counts if c == 0)}

    out = C.RESULTS / "s6_downstream/eval/fullscale_augmentation"
    for r in aug_scored:
        r["group"] = "aug"
    for cid, r in orig_scored.items():
        r["group"] = "orig"
    C.write_tsv(out / "e199_fullscale_case_metrics.tsv", aug_scored + list(orig_scored.values()))
    C.write_tsv(out / "e199_fullscale_orig_deltas.tsv", deltas)
    if errors:
        C.write_tsv(out / "e199_fullscale_eval_errors.tsv", errors)
    summary = {
        "created_at": C.now(), "metric_standard_id": EVAL_METRIC_STANDARD_ID,
        "manifest_rows": len(manifest), "aug_scored": len(aug_scored),
        "orig_scored": len(orig_scored), "paired_deltas": len(deltas), "errors": len(errors),
        "feasibility_by_object": feas,
        "distribution_overall": overall, "distribution_by_object": by_object,
        "status": "pass" if not errors and len(complete) == len(manifest) else "incomplete",
    }
    C.write_json(out / "summary.json", summary)
    import json
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if args.require_all and summary["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
