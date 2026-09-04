#!/usr/bin/env python3
"""E209 P7: paired E206-PRG vs E209-G1 evaluation over 44 rollouts (22 cases x 2 arms).

**Both arms are scored from rollouts through the same code path.** E204 read its
PRG row out of a frozen TSV, so the two arms were not measured under identical
conditions -- a confound in the very comparison the experiment exists to make.
`eval_E206_arm_ablation.py` fixed that; E209 keeps the fix, which is why the PRG
baseline is re-scored here rather than lifted from `e206_two_arm_rollout.tsv`.
The cost is a few CPU-minutes; the benefit is that any gate delta is attributable
to gravcomp and not to evaluator drift.

Why a new runner instead of a third arm in `eval_E206_arm_ablation.py`: that
module binds `e206_common.ARMS` / `SCENE_BY_ARM` / `arm_out_dir` in five places,
and `e206_common` is imported by E208's live contract module. Widening it would
silently change E206's and E208's behaviour. Per E207's precedent
(`eval_E207_g1only.py`), the thresholds and classifiers are imported and reused;
only path resolution is local. All scoring goes through `eval.core.core_metrics`
(rule 13).

Two calibers, reported side by side, exactly as E206:
  * E201 14-gate funnel (4 hard + 10 banded, wide/narrow) -> L1/L2/L3
  * frozen 12-gate (physics 6 + tracking 6)

Statistics follow rule 5: every gate reports mean / std / min / max, never just a
pass count; per-object carries n so chair005 (n=1) cannot be read as a
per-object recommendation.

Usage:
    .venv/bin/python .../eval_E209_g1_gravcomp.py [--jobs 16] [--cases a,b]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
for _p in (
    str(HERE),
    str(HERE.parent.parent),
    str(REPO / "workspace/core4d/scripts/experiments/E201"),
    str(REPO / "workspace/core4d/scripts/experiments/E206"),
    str(REPO / "workspace/core4d/scripts/experiments/E209"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.core.core_metrics import EvalConfig, evaluate_sequence  # noqa: E402
from eval.core.motion_health import run_health  # noqa: E402
import funnel_config as FC  # noqa: E402
import eval_E206_arm_ablation as ABL  # noqa: E402
import e209_common as C  # noqa: E402

#: "prg" = E206 baseline (reused rollout, re-scored); "g1" = E209 gravcomp arm.
ARMS = ("prg", "g1")
ARM_LABEL = {"prg": "E206 PRG (baseline)", "g1": "E209 PRG+gravcomp (G1)"}
OUT_DIR = C.S6_DIR / "eval/two_arm"

# Reused verbatim from the E206 runner -- same thresholds, same classifiers.
GATE_FIELDS = ABL.GATE_FIELDS
PHYSICS_6 = ABL.PHYSICS_6
TRACKING_6 = ABL.TRACKING_6
_finite = ABL._finite
classify_funnel = ABL.classify_funnel
classify_12gate = ABL.classify_12gate
stats = ABL.stats


def arm_paths(arm: str, row: dict[str, str]) -> tuple[Path, Path]:
    case_id = row["case_id"]
    if arm == "g1":
        return C.result_npz(case_id), C.scene_path(row)
    return C.baseline_npz(case_id), C.base_scene_path(row)


def contact_mask_for(arm: str, case_id: str) -> Path:
    """Mask path from the run's own config_act.yaml -- the authority is what ran.

    P2 asserted both arms compose the same `contact_hdmi_mask_path`; reading it
    back per-arm here means that assertion is re-checked against reality rather
    than assumed.
    """
    import yaml

    cfg_path = (
        C.config_act(case_id) if arm == "g1"
        else C.baseline_out_dir(case_id) / "config_act.yaml"
    )
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return REPO / str(cfg["contact_hdmi_mask_path"])


def score_one(arm: str, row: dict[str, str]) -> dict[str, Any]:
    case_id = row["case_id"]
    qpos, scene = arm_paths(arm, row)
    trajectory = C.kinematic_npz(row)
    mask = contact_mask_for(arm, case_id)
    for label, p in (("rollout", qpos), ("scene", scene), ("traj", trajectory), ("mask", mask)):
        if not Path(p).is_file():
            raise FileNotFoundError(f"{case_id}:{arm}:{label}:{p}")
    cfg = EvalConfig()
    meta = {
        "case_id": case_id,
        "variant": "orig",
        "object_key": row["object_key"],
        "spider_method_id": ABL.METHOD,
        "hand_collision_variant_id": ABL.HAND_VARIANT,
    }
    item = evaluate_sequence(
        row=meta,
        method=ABL.METHOD,
        hand_collision_variant_id=ABL.HAND_VARIANT,
        qpos_path=qpos,
        scene_xml=scene,
        config=cfg,
        kin_ref_path=trajectory,
        contact_mask_path=mask,
        person_idx=ABL._person_idx(case_id),
    )
    item.update(run_health(qpos, scene, cfg))
    item["body_z_err_p95_m"] = ABL.body_z_p95(qpos, scene, trajectory)
    item["_mask"] = str(mask)
    return item


def _worker(job: tuple[str, dict[str, str]]) -> tuple[str, str, dict[str, Any] | None, str]:
    arm, row = job
    try:
        return arm, row["case_id"], score_one(arm, row), ""
    except Exception as exc:  # noqa: BLE001 - one bad case must not kill the sweep
        return arm, row["case_id"], None, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--cases", default="")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    FC.assert_monotonic()

    sources = C.sources()
    if args.cases:
        want = {c.strip() for c in args.cases.split(",") if c.strip()}
        sources = [r for r in sources if r["case_id"] in want]
    jobs = [(arm, row) for row in sources for arm in ARMS]
    print(f"E209 eval: {len(sources)} cases x {len(ARMS)} arms = {len(jobs)} rows "
          f"| jobs={args.jobs}", flush=True)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    masks: dict[str, dict[str, str]] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(_worker, j): j for j in jobs}
        for fut in as_completed(futs):
            arm, case_id, m, err = fut.result()
            done += 1
            if m is None:
                errors.append({"case_id": case_id, "arm": arm, "error": err})
                print(f"[{done}/{len(jobs)}] {arm}/{case_id} FAILED {err}",
                      file=sys.stderr, flush=True)
                continue
            masks.setdefault(case_id, {})[arm] = m.pop("_mask", "")
            rec: dict[str, Any] = {
                "object_key": case_id.split("_")[0], "case_id": case_id, "arm": arm,
                "stratum": "S+" if case_id in C.S_PLUS else "S-",
            }
            for f in GATE_FIELDS:
                rec[f] = m.get(f)
            rec.update(classify_funnel(m))
            rec.update(classify_12gate(m))
            rows.append(rec)
            print(f"[{done}/{len(jobs)}] {arm}/{case_id} -> {rec['layer']} "
                  f"12gate={'pass' if rec['gate12_pass'] else 'fail'}", flush=True)

    # Both arms must have read the SAME contact mask, else contact gates are
    # incomparable. P2 asserted this on the composed config; this is the runtime
    # confirmation.
    mask_drift = {c: v for c, v in masks.items() if len(set(v.values())) > 1}
    if mask_drift:
        raise SystemExit(f"contact mask differs between arms: {mask_drift}")

    rows.sort(key=lambda r: (r["object_key"], r["case_id"], r["arm"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def fmt(v: Any) -> Any:
        if isinstance(v, bool):
            return v
        if isinstance(v, float):
            return "" if not math.isfinite(v) else round(v, 4)
        return v

    cols = (["object_key", "case_id", "arm", "stratum"] + GATE_FIELDS +
            ["layer", "hard_pass", "hard_failed", "wide_pass", "wide_failed",
             "narrow_pass", "narrow_failed", "physics6_pass", "tracking6_pass",
             "gate12_pass", "gate12_failed"] +
            [f"g12_{n}" for n, *_ in PHYSICS_6 + TRACKING_6])
    with (args.out_dir / "e209_two_arm_rollout.tsv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({c: fmt(r.get(c, "")) for c in cols})

    by = {(r["arm"], r["case_id"]): r for r in rows}
    paired = [r["case_id"] for r in sources if all((a, r["case_id"]) in by for a in ARMS)]

    per_gate = []
    for name, field in [(g[0], g[1]) for g in FC.BANDED_GATES] + [("body_z", "body_z_err_p95_m")]:
        entry: dict[str, Any] = {"gate": name, "field": field}
        for arm in ARMS:
            for k, v in stats([_finite(by[(arm, c)][field]) for c in paired]).items():
                entry[f"{arm}_{k}"] = v
        # delta = g1 - prg, i.e. the effect of gravcomp.
        d = [_finite(by[("g1", c)][field]) - _finite(by[("prg", c)][field]) for c in paired]
        ds = stats(d)
        entry.update({"paired_delta_mean": ds["mean"], "paired_delta_std": ds["std"],
                      "paired_delta_min": ds["min"], "paired_delta_max": ds["max"]})
        per_gate.append(entry)
    ABL.C.write_tsv(args.out_dir / "e209_per_gate.tsv", per_gate)

    per_object = []
    for obj in sorted({c.split("_")[0] for c in paired}):
        cs = [c for c in paired if c.split("_")[0] == obj]
        row: dict[str, Any] = {"object_key": obj, "n": len(cs),
                               "per_object_recommendation_allowed": len(cs) >= 3}
        for arm in ARMS:
            rs = [by[(arm, c)] for c in cs]
            row[f"{arm}_L3"] = sum(1 for r in rs if r["layer"] == "L3_auto")
            row[f"{arm}_L2"] = sum(1 for r in rs if r["layer"] == "L2_review")
            row[f"{arm}_L1"] = sum(1 for r in rs if r["layer"] == "L1_reject")
            row[f"{arm}_narrow"] = sum(1 for r in rs if r["narrow_pass"])
            row[f"{arm}_gate12"] = sum(1 for r in rs if r["gate12_pass"])
            row[f"{arm}_legpen_mean"] = stats(
                [_finite(r["leg_penetration_frac"]) for r in rs])["mean"]
            row[f"{arm}_contact_mean"] = stats(
                [_finite(r["hand_object_physics_contact_in_mask_frac"]) for r in rs])["mean"]
        per_object.append(row)
    ABL.C.write_tsv(args.out_dir / "e209_per_object.tsv", per_object)

    def arm_summary(arm: str) -> dict[str, Any]:
        rs = [by[(arm, c)] for c in paired]
        return {
            "n": len(rs),
            "L3_auto": sum(1 for r in rs if r["layer"] == "L3_auto"),
            "L2_review": sum(1 for r in rs if r["layer"] == "L2_review"),
            "L1_reject": sum(1 for r in rs if r["layer"] == "L1_reject"),
            "hard_pass": sum(1 for r in rs if r["hard_pass"]),
            "wide_pass": sum(1 for r in rs if r["wide_pass"]),
            "narrow_pass": sum(1 for r in rs if r["narrow_pass"]),
            "gate12_pass": sum(1 for r in rs if r["gate12_pass"]),
            "physics6_pass": sum(1 for r in rs if r["physics6_pass"]),
            "tracking6_pass": sum(1 for r in rs if r["tracking6_pass"]),
        }

    arms_sum = {a: arm_summary(a) for a in ARMS}
    g = C.GATES

    def mean_of(arm: str, field: str) -> float:
        return _finite(stats([_finite(by[(arm, c)][field]) for c in paired])["mean"])

    contact_mean = mean_of("g1", "hand_object_physics_contact_in_mask_frac")
    legpen_mean = mean_of("g1", "leg_penetration_frac")
    release_mean = mean_of("g1", "hand_object_release_false_contact_3mm_frac")
    handpen_mean = mean_of("g1", "hand_object_physics_penetration_3mm_frame_frac")
    claims = {
        "C2_execution": {"n_scored": len(paired), "errors": len(errors),
                         "pass": len(paired) == C.EXPECTED_CASES and not errors},
        "C3_14gate": {
            "g1_narrow_pass": arms_sum["g1"]["narrow_pass"],
            "prg_narrow_pass": arms_sum["prg"]["narrow_pass"],
            "g1_hard_pass": arms_sum["g1"]["hard_pass"],
            "gate_narrow_min": g["C3_narrow_pass_min"],
            "pass": (arms_sum["g1"]["narrow_pass"] >= g["C3_narrow_pass_min"]
                     and arms_sum["g1"]["hard_pass"] == len(paired)),
        },
        "C5_load_bearing": {
            "contact_in_mask_mean": contact_mean, "leg_pen_frac_mean": legpen_mean,
            "prg_contact_in_mask_mean": mean_of("prg", "hand_object_physics_contact_in_mask_frac"),
            "prg_leg_pen_frac_mean": mean_of("prg", "leg_penetration_frac"),
            "pass": contact_mean >= g["C5_contact_in_mask_min"]
            and legpen_mean <= g["C5_leg_pen_frac_max"],
        },
        "C6_side_effects": {
            "release_false_contact_mean": release_mean, "hand_pen_3mm_mean": handpen_mean,
            "prg_release_false_contact_mean": mean_of(
                "prg", "hand_object_release_false_contact_3mm_frac"),
            "prg_hand_pen_3mm_mean": mean_of(
                "prg", "hand_object_physics_penetration_3mm_frame_frac"),
            "pass": release_mean <= g["C6_release_false_contact_max"]
            and handpen_mean <= g["C6_hand_pen_3mm_max"],
        },
    }

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scored_rows": len(rows), "errors": errors, "paired_cases": len(paired),
        "note": "both arms scored from rollouts through ONE code path (no frozen-TSV baseline)",
        "arm_label": ARM_LABEL, "arms": arms_sum, "claims": claims,
    }
    (args.out_dir / "e209_two_arm_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    print("\n" + json.dumps(arms_sum, ensure_ascii=False, indent=2))
    for cid, c in claims.items():
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {cid}")
    print(f"paired_cases={len(paired)}  errors={len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
