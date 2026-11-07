#!/usr/bin/env python3
"""E216 C4: the clean 0.1m-vs-0.2m comparison (both omnirt_v2, same per-object arm).

This is E216's core motivation test: is the gentler 0.1m offset at least as
physically clean as the historical 0.2m, so more survives the downstream
pre-filter?  Unlike the aug-vs-orig pairing (88/104 origs are omnirt_v1 -> a
retarget-variant confound), 0.1m vs 0.2m differs in EXACTLY the offset magnitude
on the identical arm, scored by the identical evaluator (eval_E216.score).

Per arm group the 0.2m sibling rollout lives in the arm's own experiment:
    box_prg       box021            E199 fullscale PRG     (case_id person{N})
    box_noprg     box023            E200 noPRG             (case_id person{N})
    box_prg_g1a2  box001/004/024    E200 prg_g1a2          (case_id person{N})
    bucket_prg    bucket003/007     E202 PRG               (case_id p{N})

The 0.1m metrics are read from the already-scored E216 rollout TSV; the 0.2m
rollouts are scored fresh here with the same EvalConfig.  Reports per-arm 6-gate
pass rate and paired metric deltas (0.1m - 0.2m), mean+std+worst, never cherry-
picked.

Usage:
    .venv/bin/python workspace/core4d/scripts/eval/reports/gen_E216_0p1m_vs_0p2m.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))                      # scripts/
sys.path.insert(0, str(_HERE.parents[1] / "experiments/E216"))
sys.path.insert(0, str(_HERE.parents[1] / "eval/runners"))

import e216_common as C  # noqa: E402
import eval_E216_trans_augmentation as EV  # noqa: E402
from eval.core.core_metrics import EvalConfig  # noqa: E402

RES = C.REPO / "workspace/core4d/results"

# (manifest, case_id form, arm_filter) per arm group for the 0.2m sibling.
SOURCES = {
    "box_prg": ("E199/s6_downstream/manifests/e199_fullscale_authority.tsv", "person", None),
    "box_noprg": ("E200/s6_downstream/manifests/e200_noprg_priority_manifest.tsv", "person", None),
    "box_prg_g1a2": ("E200/s6_downstream/manifests/e200_prg_g1a2_priority_manifest.tsv", "person", None),
    "bucket_prg": ("E202/s6_downstream/manifests/e202_bucket_authority.tsv", "p", None),
}
DONE = {"run_complete_pending_eval", "cem_ok", "", "READY_FOR_FULL"}
KEY = ("track_obj_pos_err_cm_mean", "track_obj_ori_err_deg_mean",
       "hand_object_physics_contact_in_mask_frac",
       "hand_object_physics_penetration_3mm_frame_frac", "leg_penetration_frac")


def to_person(case_id: str) -> str:
    return case_id.replace("_p1", "_person1").replace("_p2", "_person2")


def load_0p2m_index() -> dict[str, dict[tuple[str, str], dict[str, str]]]:
    """arm_group -> {(case_id_p, variant): source_row} for the 0.2m siblings."""
    idx: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    for arm, (rel, form, _arm_f) in SOURCES.items():
        tsv = RES / rel
        rows = C.read_tsv(tsv) if tsv.is_file() else []
        by_key: dict[tuple[str, str], dict[str, str]] = {}
        for r in rows:
            cid = r.get("case_id", "")
            # normalize to p{N}
            cid_p = cid.replace("_person1", "_p1").replace("_person2", "_p2")
            v = r.get("aug_variant", "")
            if v in ("trans0", "trans1", "trans2") and (r.get("result_npz") or r.get("outdir_npz")):
                by_key[(cid_p, v)] = r
        idx[arm] = by_key
    return idx


def finite(x: Any) -> float:
    try:
        f = float(x)
        return f if math.isfinite(f) else math.nan
    except (TypeError, ValueError):
        return math.nan


def main() -> int:
    cfg = EvalConfig()
    # 0.1m: already scored
    rollout = C.read_tsv(C.EVAL_DIR / "e216_trans_rollout.tsv")
    aug01 = {(r["case_id"], r["aug_variant"]): r for r in rollout if r.get("aug_variant", "").startswith("trans")}

    idx02 = load_0p2m_index()

    paired: list[dict[str, Any]] = []
    missing02: list[str] = []
    scored02_cache: dict[tuple[str, str], dict[str, Any]] = {}

    for (cid, v), r01 in sorted(aug01.items()):
        arm = r01.get("arm_group", "")
        src = idx02.get(arm, {}).get((cid, v))
        if src is None:
            missing02.append(f"{cid}/{v}({arm})")
            continue
        # score the 0.2m sibling fresh with the same evaluator
        try:
            npz_key = "outdir_npz" if (src.get("outdir_npz") and C.repo_path(src["outdir_npz"]).is_file()) else "result_npz"
            s02 = EV.score(src, cfg, npz_key=npz_key, scene_key="scene_act")
        except Exception as exc:  # noqa: BLE001
            missing02.append(f"{cid}/{v}({arm}):score_err:{type(exc).__name__}")
            continue
        scored02_cache[(cid, v)] = s02
        row = {"case_id": cid, "aug_variant": v, "arm_group": arm,
               "pass_0p1m": r01.get("all_gates_pass"), "pass_0p2m": s02.get("all_gates_pass"),
               "fall_0p1m": r01.get("fall_flag"), "fall_0p2m": s02.get("fall_flag")}
        for m in KEY:
            a, b = finite(r01.get(m)), finite(s02.get(m))
            row[f"{m}_0p1m"], row[f"{m}_0p2m"], row[f"delta_{m}"] = a, b, (a - b)
        paired.append(row)

    # per-arm aggregation
    def truthy(x: Any) -> bool:
        return str(x).lower() in ("true", "1")

    arms = sorted({r["arm_group"] for r in paired})
    summary: dict[str, Any] = {}
    for arm in arms + ["POOLED"]:
        rows = paired if arm == "POOLED" else [r for r in paired if r["arm_group"] == arm]
        if not rows:
            continue
        n = len(rows)
        p01 = sum(truthy(r["pass_0p1m"]) for r in rows)
        p02 = sum(truthy(r["pass_0p2m"]) for r in rows)
        cell = {
            "n": n, "n_cases": len({r["case_id"] for r in rows}),
            "pass_rate_0p1m": p01 / n, "pass_rate_0p2m": p02 / n,
            "pass_rate_delta": (p01 - p02) / n,
            "falls_0p1m": sum(truthy(r["fall_0p1m"]) for r in rows),
            "falls_0p2m": sum(truthy(r["fall_0p2m"]) for r in rows),
        }
        for m in KEY:
            d = np.array([finite(r[f"delta_{m}"]) for r in rows], dtype=float)
            d = d[np.isfinite(d)]
            cell[f"delta_{m}_mean"] = float(d.mean()) if d.size else math.nan
            cell[f"delta_{m}_worst"] = float(d.max()) if d.size else math.nan
        summary[arm] = cell

    out_dir = C.EVAL_DIR
    C.write_tsv(out_dir / "e216_0p1m_vs_0p2m_pairs.tsv", paired, EV.union_fields(paired))
    C.write_json(out_dir / "e216_0p1m_vs_0p2m_summary.json", {
        "experiment": C.EXP_ID, "run_id": C.RUN_ID, "generated_at": C.now(),
        "note": ("0.1m vs 0.2m, both omnirt_v2 same arm, same evaluator; pass = 6-gate all_gates_pass. "
                 "pass_rate_delta>0 => 0.1m passes MORE often than 0.2m."),
        "n_paired": len(paired), "n_missing_0p2m": len(missing02),
        "missing_0p2m": missing02[:40], "by_arm": summary,
    })

    print(f"\n0.1m vs 0.2m: {len(paired)} paired ({len(missing02)} missing 0.2m sibling)")
    print(f"{'arm':14s} {'n':>4s} {'pass0.1':>8s} {'pass0.2':>8s} {'Δpass':>7s} "
          f"{'fall0.1':>7s} {'fall0.2':>7s} {'Δobjpos':>8s} {'Δcontact':>9s} {'Δlegpen':>8s}")
    for arm in arms + ["POOLED"]:
        c = summary.get(arm)
        if not c:
            continue
        print(f"{arm:14s} {c['n']:4d} {c['pass_rate_0p1m']:8.1%} {c['pass_rate_0p2m']:8.1%} "
              f"{c['pass_rate_delta']:+7.1%} {c['falls_0p1m']:7d} {c['falls_0p2m']:7d} "
              f"{c['delta_track_obj_pos_err_cm_mean_mean']:+8.3f} "
              f"{c['delta_hand_object_physics_contact_in_mask_frac_mean']:+9.3f} "
              f"{c['delta_leg_penetration_frac_mean']:+8.3f}")
    if missing02:
        print(f"  missing 0.2m: {missing02[:10]}")
    print(f"-> {C.rel(out_dir)}/e216_0p1m_vs_0p2m_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
