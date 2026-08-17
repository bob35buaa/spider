#!/usr/bin/env python3
"""E200 per-arm evaluation: score every completed arm-augmentation full-CEM run
under the SAME public-core scoring contract used for E199 (so E200 noPRG /
PRG+G1+A2 are directly comparable to the E199 PRG aug rollouts).

Reuses eval_E199_augmentation.score() verbatim -- same EvalConfig(), same
METHOD=E167A_zOnlyBody + rubber_hull hand, same 6 numeric gates, same
metric_standard_id (core4d-e154-physics-contact-v1). The only difference vs
E199 is the input manifest (an E200 arm manifest) and that each scored row is
tagged with its arm.

Emits per-arm case metrics + the full distribution (mean+std+worst, no
cherry-picking) + per-object strata. Pairing / arm-vs-arm comparison and the
xlsx workbook are done by gen_E200_arm_gate_xlsx.py.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

RUNNERS = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNNERS.parents[1]))                       # scripts/ (eval.core)
sys.path.insert(0, str(RUNNERS.parents[1] / "experiments/E200"))  # e200_common

from eval.core.core_metrics import EVAL_METRIC_STANDARD_ID, EvalConfig  # noqa: E402
import e200_common as C  # noqa: E402
# reuse the E199 single-contract scorer + constants (same public core)
from eval_E199_augmentation import GATE_THRESHOLDS, KEY_METRICS, finite, score  # noqa: E402

# noPRG same-contract orig baseline = the E190 38-case noPRG export (omnirt_v1
# orig, E167A_zOnlyBody + rubber_hull, no PRG pairs). Symmetric to the PRG side,
# which reuses the E198 A0 (also v1) orig. Scoring these with the same score()
# gives the noPRG families their orig anchor so classify_funnel's L3 family
# arbitration matches the PRG side on the 38 cases E190 covers.
E190_RL_EXPORT = C.REPO / "workspace/core4d/results/E190/s6_downstream/rl_export/rl_export_input.tsv"


def e190_noprg_orig_rows() -> list[dict[str, str]]:
    """score()-ready rows for the 38 E190 noPRG orig rollouts (aug_variant=orig)."""
    rows: list[dict[str, str]] = []
    for r in C.read_tsv(E190_RL_EXPORT):
        npz = r["cem_result_npz"]
        rows.append({
            "outdir_npz": npz, "result_npz": npz,
            "scene_act": r["scene_act"], "trajectory": r["trajectory"], "contact_mask": r["contact_mask"],
            "object_key": r["object_key"], "case_id": r["case_id"], "aug_variant": "orig",
            "target_task": Path(r["scene_act"]).parent.name, "tier": "orig",
            "variant": "orig", "method": r.get("spider_method_id", ""),
        })
    return rows


def stats(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    import numpy as np
    vals = np.array([finite(r.get(metric)) for r in rows], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if not vals.size:
        return {}
    return {"mean": float(vals.mean()), "std": float(vals.std()),
            "worst": float(vals.max()), "n": int(vals.size)}


def eval_arm(arm: str, cfg: EvalConfig) -> dict[str, Any]:
    manifest = C.read_tsv(C.manifest_path(arm))
    complete = [{**r, "group": "aug"} for r in manifest
                if C.repo_path(r["outdir_npz"]).is_file() and C.repo_path(r["result_npz"]).is_file()]
    # noPRG: add the E190 same-contract orig baseline (family anchor for 38 cases)
    orig_rows: list[dict[str, str]] = []
    if arm == "noprg":
        for r in e190_noprg_orig_rows():
            if C.repo_path(r["outdir_npz"]).is_file():
                orig_rows.append({**r, "group": "orig"})
        print(f"[{arm}] +E190 noPRG orig rows on disk={len(orig_rows)}", flush=True)
    todo = complete + orig_rows
    print(f"[{arm}] manifest={len(manifest)} aug_complete={len(complete)} total_to_score={len(todo)}", flush=True)

    scored: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for i, row in enumerate(todo, start=1):
        try:
            item = score(row, cfg)
            item["arm"] = arm
            item["group"] = row["group"]
            scored.append(item)
        except Exception as exc:  # noqa: BLE001
            errors.append({"case_id": row["case_id"], "aug_variant": row["aug_variant"],
                           "group": row.get("group", ""), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[error] {arm} {row['case_id']} {row['aug_variant']}: {errors[-1]['error']}",
                  file=sys.stderr, flush=True)
        if i % 25 == 0 or i == len(todo):
            print(f"[{arm}] scored {i}/{len(todo)}", flush=True)

    out = C.RESULTS / "s6_downstream/eval" / arm
    C.write_tsv(out / f"e200_{arm}_case_metrics.tsv", scored)
    if errors:
        C.write_tsv(out / f"e200_{arm}_eval_errors.tsv", errors)

    import numpy as np
    aug_scored = [r for r in scored if r.get("group") == "aug"]  # distribution over aug only
    dist: dict[str, Any] = {"n_aug": len(aug_scored), "n_orig": len(scored) - len(aug_scored)}
    for m in KEY_METRICS:
        s = stats(aug_scored, m)
        if s:
            dist[f"{m}_mean"], dist[f"{m}_std"], dist[f"{m}_worst"] = s["mean"], s["std"], s["worst"]
    dist["gate6_pass_frac"] = float(np.mean([1.0 if r.get("all_gates_pass") else 0.0 for r in aug_scored])) if aug_scored else math.nan
    by_object: dict[str, Any] = {}
    for obj in sorted({r["object_key"] for r in aug_scored}):
        grp = [r for r in aug_scored if r["object_key"] == obj]
        by_object[obj] = {"n": len(grp),
                          "gate6_pass_frac": float(np.mean([1.0 if r.get("all_gates_pass") else 0.0 for r in grp]))}
    summary = {"arm": arm, "created_at": C.now(), "metric_standard_id": EVAL_METRIC_STANDARD_ID,
               "manifest_rows": len(manifest), "aug_complete": len(complete), "orig_scored": len(orig_rows),
               "scored": len(scored), "errors": len(errors),
               "distribution_overall": dist, "distribution_by_object": by_object,
               "status": "pass" if not errors else "incomplete"}
    C.write_json(out / "summary.json", summary)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="noprg")
    args = ap.parse_args()
    cfg = EvalConfig()
    import json
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if arm not in C.ARMS:
            raise SystemExit(f"unknown arm {arm}; expected {C.ARMS}")
        summary = eval_arm(arm, cfg)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
