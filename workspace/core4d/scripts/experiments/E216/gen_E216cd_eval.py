#!/usr/bin/env python3
"""E216 chair/desk (Phase 2) eval: score the 59 trans-0.1m rollouts + the clean
0.1m-vs-0.2m comparison against E213's SAME selected-arm 0.2m rollouts.

0.1m rollouts come from the cd priority manifest (3 shards merged).  The 0.2m
siblings come from E213's ``e213_source_arm_cem.tsv`` (same case, same variant,
same per-case selected gravcomp arm) -- a clean single-variable (offset) pairing
on the identical arm, scored by the identical evaluator (eval_E216.score).

Writes:
  e216cd_trans_rollout.tsv       every scored 0.1m rollout (gates + metrics)
  e216cd_0p1m_vs_0p2m_summary.json  per-object + pooled gate pass + metric deltas

Usage:
    .venv/bin/python .../E216/gen_E216cd_eval.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "eval/runners"))

import e216cd_common as C  # noqa: E402
import eval_E216_trans_augmentation as EV  # noqa: E402  (score/gate_set reused)
from eval.core.core_metrics import EvalConfig  # noqa: E402

E213_SOURCE = C.REPO / "workspace/core4d/results/E213/s6_downstream/manifests/e213_source_arm_cem.tsv"
DONE = {"run_complete_pending_eval", "cem_ok"}
KEY = ("track_obj_pos_err_cm_mean", "track_obj_ori_err_deg_mean",
       "hand_object_physics_contact_in_mask_frac",
       "hand_object_physics_penetration_3mm_frame_frac", "leg_penetration_frac")


def finite(x: Any) -> float:
    try:
        f = float(x)
        return f if math.isfinite(f) else math.nan
    except (TypeError, ValueError):
        return math.nan


def truthy(x: Any) -> bool:
    return str(x).lower() in ("true", "1")


def main() -> int:
    cfg = EvalConfig()
    # 0.1m: cd priority manifest (shards already wrote status back into it? no --
    # shards are disjoint; merge all three)
    rows01: list[dict[str, str]] = []
    for s in ("A", "B", "C"):
        f = C.MANIFEST_DIR / f"e216cd_priority_manifest.shard{s}.tsv"
        if f.is_file():
            rows01 += [r for r in C.read_tsv(f) if r.get("status") in DONE]
    print(f"0.1m done rows: {len(rows01)}")

    # 0.2m siblings keyed by (case_id, aug_variant) from E213
    src02: dict[tuple[str, str], dict[str, str]] = {}
    for r in C.read_tsv(E213_SOURCE):
        if r.get("aug_variant", "").startswith("trans") and (r.get("outdir_npz") or r.get("result_npz")):
            src02[(r["case_id"], r["aug_variant"])] = r

    scored01: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    failed: list[str] = []
    for i, r in enumerate(sorted(rows01, key=lambda x: (x["case_id"], x["aug_variant"])), 1):
        cid, v = r["case_id"], r["aug_variant"]
        try:
            s01 = EV.score(r, cfg, npz_key="outdir_npz", scene_key="scene_act")
        except Exception as exc:  # noqa: BLE001
            failed.append(f"0.1m {cid}/{v}: {type(exc).__name__}: {exc}")
            continue
        scored01.append(s01)
        src = src02.get((cid, v))
        if src is None:
            continue
        try:
            npz_key = "outdir_npz" if (src.get("outdir_npz") and C.repo_path(src["outdir_npz"]).is_file()) else "result_npz"
            # E213 row: scene in selected_scene_act
            src = dict(src); src["scene_act"] = src.get("selected_scene_act", src.get("scene_act", ""))
            src.setdefault("object_key", r["object_key"]); src.setdefault("arm_group", r["arm_group"])
            s02 = EV.score(src, cfg, npz_key=npz_key, scene_key="scene_act")
        except Exception as exc:  # noqa: BLE001
            failed.append(f"0.2m {cid}/{v}: {type(exc).__name__}: {exc}")
            continue
        row = {"case_id": cid, "aug_variant": v, "object_key": r["object_key"], "gravcomp": r.get("gravcomp", ""),
               "pass_0p1m": s01.get("all_gates_pass"), "pass_0p2m": s02.get("all_gates_pass"),
               "fall_0p1m": s01.get("fall_flag"), "fall_0p2m": s02.get("fall_flag")}
        for m in KEY:
            a, b = finite(s01.get(m)), finite(s02.get(m))
            row[f"{m}_0p1m"], row[f"{m}_0p2m"], row[f"delta_{m}"] = a, b, (a - b)
        paired.append(row)
        if i % 10 == 0:
            print(f"  scored {i}/{len(rows01)}")

    # per-object + pooled
    objects = sorted({r["object_key"] for r in paired})
    summary: dict[str, Any] = {}
    for obj in objects + ["POOLED"]:
        rs = paired if obj == "POOLED" else [r for r in paired if r["object_key"] == obj]
        if not rs:
            continue
        n = len(rs)
        cell = {"n": n, "n_cases": len({r["case_id"] for r in rs}),
                "pass_rate_0p1m": sum(truthy(r["pass_0p1m"]) for r in rs) / n,
                "pass_rate_0p2m": sum(truthy(r["pass_0p2m"]) for r in rs) / n,
                "falls_0p1m": sum(truthy(r["fall_0p1m"]) for r in rs),
                "falls_0p2m": sum(truthy(r["fall_0p2m"]) for r in rs)}
        cell["pass_rate_delta"] = cell["pass_rate_0p1m"] - cell["pass_rate_0p2m"]
        for m in KEY:
            d = np.array([finite(r[f"delta_{m}"]) for r in rs], float)
            d = d[np.isfinite(d)]
            cell[f"delta_{m}_mean"] = float(d.mean()) if d.size else math.nan
        summary[obj] = cell

    # absolute 0.1m gate pass by object
    abs01: dict[str, Any] = {}
    for obj in objects + ["POOLED"]:
        rs = scored01 if obj == "POOLED" else [r for r in scored01 if r["object_key"] == obj]
        if rs:
            abs01[obj] = {"n": len(rs), "pass": sum(1 for r in rs if r.get("all_gates_pass")),
                          "pass_rate": sum(1 for r in rs if r.get("all_gates_pass")) / len(rs),
                          "falls": sum(1 for r in rs if truthy(r.get("fall_flag")))}

    out = C.EVAL_DIR
    C.write_tsv(out / "e216cd_trans_rollout.tsv", scored01, EV.union_fields(scored01))
    C.write_tsv(out / "e216cd_0p1m_vs_0p2m_pairs.tsv", paired, EV.union_fields(paired))
    C.write_json(out / "e216cd_0p1m_vs_0p2m_summary.json", {
        "experiment": "E216CD", "run_id": C.RUN_ID, "generated_at": C.now(),
        "note": "chair/desk 0.1m vs 0.2m (E213 same selected-arm), same evaluator, 6-gate all_gates_pass.",
        "n_scored_0p1m": len(scored01), "n_paired": len(paired), "n_failed": len(failed), "failed": failed[:30],
        "absolute_0p1m_by_object": abs01, "compare_by_object": summary})

    print(f"\nscored {len(scored01)} 0.1m, {len(paired)} paired ({len(failed)} failed)")
    print("=== absolute 0.1m 6-gate pass by object ===")
    for obj in objects + ["POOLED"]:
        c = abs01.get(obj)
        if c:
            print(f"  {obj:10s} {c['pass']}/{c['n']} = {c['pass_rate']:.1%}  falls={c['falls']}")
    print("=== 0.1m vs 0.2m (same arm) ===")
    print(f"  {'obj':10s} {'n':>3s} {'p0.1':>6s} {'p0.2':>6s} {'Δpp':>6s} {'f0.1':>4s} {'f0.2':>4s} {'Δobjpos':>8s}")
    for obj in objects + ["POOLED"]:
        c = summary.get(obj)
        if c:
            print(f"  {obj:10s} {c['n']:3d} {c['pass_rate_0p1m']:6.1%} {c['pass_rate_0p2m']:6.1%} "
                  f"{c['pass_rate_delta'] * 100:+6.1f} {c['falls_0p1m']:4d} {c['falls_0p2m']:4d} "
                  f"{c['delta_track_obj_pos_err_cm_mean_mean']:+8.3f}")
    for f in failed[:8]:
        print(f"  FAIL {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
