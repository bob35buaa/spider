#!/usr/bin/env python3
"""E204 (noPRG) / E205 (G1A2) vs E178 (PRG) — three-arm 14-gate ablation on 27 buckets.

Scores every arm under the IDENTICAL E201 14-gate funnel (4 hard + 10 banded
wide/narrow) so the three reward arms are compared on one ruler:

  * E178 PRG   -> read the 14 fields from its canonical e178_case_metrics.tsv
                  (same public core_metrics; avoids the task-dir mesh-path rescore
                   gotcha noted in eval_E202).
  * E204 noPRG / E205 G1A2 -> score the fresh CEM rollout with the public
                  evaluate_sequence + run_health, plus body_z_p95 (the E178/E187
                  four-body Z p95) for the body_z hard gate.

All three arms are 'orig' single-arm families (no augmentation), so a narrow-pass
row is L3_auto by the funnel's family rule. Writes a combined three_arm_rollout.tsv
consumed by gen_E204E205_three_arm_workbook.py.

Usage:
    MUJOCO_GL=disable .venv/bin/python \
      workspace/core4d/scripts/eval/runners/eval_E204E205_arm_ablation.py
    ... --limit N          # first N cases (smoke)
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts"))
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/eval/runners"))
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E201"))
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E204_E205"))

from eval.core.core_metrics import EvalConfig, evaluate_sequence  # noqa: E402
from eval.core.motion_health import run_health  # noqa: E402
from eval_E187_e178_compat import body_z_p95  # noqa: E402
import funnel_config as FC  # noqa: E402
import e204e205_common as C  # noqa: E402

METHOD = "E167A_zOnlyBody"
HAND_VARIANT = "rubber_hull"
E178_CASE_METRICS = REPO / "workspace/core4d/results/E178/s6_downstream/eval/full/e178_case_metrics.tsv"
OUT_DIR = REPO / "workspace/core4d/results/E204/s6_downstream/eval/three_arm"

# arm label -> (source kind). PRG is the reused E178 baseline.
ARMS = [("PRG", "e178"), ("noPRG", "e204"), ("G1A2", "e205")]
ARM_TO_KEY = {"noPRG": "noprg_e204", "G1A2": "g1a2_e205"}
# the 14 funnel fields we persist per row
GATE_FIELDS = ["fall_flag", "body_z_err_p95_m", "ankle_jerk_p95", "obj_speed_max"] + \
    [g[1] for g in FC.BANDED_GATES]


def _person_idx(case_id: str) -> int:
    return 0 if case_id.lower().endswith("_p1") else 1


def _finite(v: Any) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else math.nan
    except (TypeError, ValueError):
        return math.nan


def score_rollout(arm_key: str, source: dict, cfg: EvalConfig) -> dict[str, Any]:
    """Score one E204/E205 rollout NPZ into the 14 funnel fields."""
    cid = source["case_id"]
    _tgt, trajectory, mask = C.base.local_authorities(source)
    qpos = C.result_npz(arm_key, cid, "full")
    scene = C.arm_scene_path(arm_key, cid)
    for label, p in (("rollout", qpos), ("scene", scene), ("traj", trajectory), ("mask", mask)):
        if not p.is_file():
            raise FileNotFoundError(f"{cid}:{arm_key}:{label}:{p}")
    row = {"case_id": cid, "variant": "orig", "object_key": source["object_key"],
           "spider_method_id": METHOD, "hand_collision_variant_id": HAND_VARIANT}
    item = evaluate_sequence(row=row, method=METHOD, hand_collision_variant_id=HAND_VARIANT,
                             qpos_path=qpos, scene_xml=scene, config=cfg,
                             kin_ref_path=trajectory, contact_mask_path=mask,
                             person_idx=_person_idx(cid))
    item.update(run_health(qpos, scene, cfg))
    item["body_z_err_p95_m"] = body_z_p95(qpos, scene, trajectory)
    return item


def read_prg_row(tsv_row: dict[str, str]) -> dict[str, Any]:
    """Pull the 14 funnel fields from E178's canonical case_metrics tsv."""
    out: dict[str, Any] = {"fall_flag": str(tsv_row.get("fall_flag", "")).strip()}
    for f in GATE_FIELDS:
        if f == "fall_flag":
            continue
        out[f] = _finite(tsv_row.get(f))
    return out


def classify(metrics: dict[str, Any]) -> dict[str, Any]:
    """Apply the 14-gate funnel to one arm's metrics. orig single-arm family:
    narrow-pass => L3_auto (no other family arms to be inconsistent with)."""
    hard_ok, hard_bad = FC.hard_gate_result(metrics)
    wide_ok, wide_bad = FC.banded_gate_result(metrics, "wide")
    narrow_ok, narrow_bad = FC.banded_gate_result(metrics, "narrow")
    if not hard_ok or not wide_ok:
        layer = "L1_reject"
    elif not narrow_ok:
        layer = "L2_review"
    else:
        layer = "L3_auto"
    return {
        "layer": layer,
        "hard_pass": hard_ok, "hard_failed": ",".join(hard_bad),
        "wide_pass": wide_ok, "wide_failed": ",".join(wide_bad),
        "narrow_pass": narrow_ok, "narrow_failed": ",".join(narrow_bad),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    FC.assert_monotonic()

    sources = C.load_sources()
    if args.limit:
        sources = sources[: args.limit]
    case_ids = [s["case_id"] for s in sources]
    src_by_case = {s["case_id"]: s for s in sources}

    # PRG baseline from E178 tsv
    prg_by_case = {r["case_id"]: read_prg_row(r)
                   for r in C.base.read_tsv(E178_CASE_METRICS) if r["case_id"] in set(case_ids)}
    missing = set(case_ids) - set(prg_by_case)
    if missing:
        print(f"[warn] {len(missing)} cases missing from E178 tsv: {sorted(missing)[:3]}...", file=sys.stderr)

    cfg = EvalConfig()
    rows: list[dict[str, Any]] = []
    for i, cid in enumerate(case_ids, 1):
        src = src_by_case[cid]
        obj = src["object_key"]
        for arm_label, kind in ARMS:
            if kind == "e178":
                m = prg_by_case.get(cid)
                if m is None:
                    continue
            else:
                m = score_rollout(ARM_TO_KEY[arm_label], src, cfg)
            cls = classify(m)
            rec = {"object_key": obj, "case_id": cid, "arm": arm_label}
            for f in GATE_FIELDS:
                rec[f] = m.get(f)
            rec.update(cls)
            rows.append(rec)
        print(f"[{i}/{len(case_ids)}] {cid} scored (3 arms)", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "three_arm_rollout.tsv"
    cols = ["object_key", "case_id", "arm"] + GATE_FIELDS + \
        ["layer", "hard_pass", "hard_failed", "wide_pass", "wide_failed", "narrow_pass", "narrow_failed"]

    def fmt(v: Any) -> Any:
        if isinstance(v, bool):
            return v
        if isinstance(v, float):
            return "" if not math.isfinite(v) else round(v, 4)
        return v

    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({c: fmt(r.get(c, "")) for c in cols})
    print(f"\n[done] wrote {out.relative_to(REPO)} ({len(rows)} rows = {len(case_ids)} cases x 3 arms)")

    # console summary: per-arm narrow/wide/hard/L3 pass counts + per-gate narrow pass
    print("\n=== 14-gate pass summary (per arm) ===")
    for arm_label, _k in ARMS:
        ar = [r for r in rows if r["arm"] == arm_label]
        n = len(ar)
        if not n:
            continue
        hp = sum(r["hard_pass"] for r in ar)
        wp = sum(r["wide_pass"] for r in ar)
        npass = sum(r["narrow_pass"] for r in ar)
        l3 = sum(r["layer"] == "L3_auto" for r in ar)
        print(f"  {arm_label:6s} n={n:2d}  hard={hp:2d}/{n}  wide_all={wp:2d}/{n}  "
              f"narrow_all={npass:2d}/{n} ({npass/n:.0%})  L3_auto={l3:2d}")
    print("\n=== per-gate NARROW pass rate (per arm) ===")
    gate_defs = [("fall", "fall_flag"), ("body_z", "body_z_err_p95_m"),
                 ("ankle_jerk", "ankle_jerk_p95"), ("obj_speed", "obj_speed_max")] + \
        [(g[0], g[1]) for g in FC.BANDED_GATES]
    hdr = "  gate         " + "".join(f"{a:>10s}" for a, _ in ARMS)
    print(hdr)
    for gname, gfield in gate_defs:
        cells = ""
        for arm_label, _k in ARMS:
            ar = [r for r in rows if r["arm"] == arm_label]
            if gname == "fall":
                ok = sum(str(r["fall_flag"]).strip().lower() not in ("true", "1") for r in ar)
            else:
                hard = {h[1]: h for h in FC.HARD_GATES}
                if gfield in hard:
                    _n, _f, op, thr = hard[gfield]
                    ok = sum(FC.passes(op, _finite(r[gfield]), thr) for r in ar)
                else:
                    bg = {g[1]: g for g in FC.BANDED_GATES}[gfield]
                    ok = sum(FC.passes(bg[2], _finite(r[gfield]), bg[3]) for r in ar)
            cells += f"{ok:>10d}"
        print(f"  {gname:12s}{cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
