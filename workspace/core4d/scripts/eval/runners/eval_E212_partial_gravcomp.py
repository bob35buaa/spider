#!/usr/bin/env python3
"""E212 P8: 5-arm partial-gravcomp sweep evaluation over 20 rollouts (4 desk023 cases x 5 g).

The five arms are the g-curve: 0.0 (E206 PRG), 0.4 / 0.6 / 0.8 (E212 Stage A),
1.0 (E209 G1).  Both endpoints reuse rollouts that already exist -- but they are
**re-scored here through the same code path as the new arms**, never lifted from
E209's TSV.  E204 made that mistake once (frozen-TSV baseline vs freshly-scored
candidate) and it put a confound inside the very comparison the experiment
exists to make.  The cost is CPU-minutes; the benefit is that a gate delta is
attributable to g and not to evaluator drift.  P0 froze the endpoint numbers, so
if this re-score disagrees with them the run stops.

Three things this runner does that E209's does not:

  * **Per-metric direction.** E210 F6: a single `delta > 0 == worse` rule reports
    `contact` (higher is better) backwards and flips the conclusion. Every metric
    carries its own direction and "improved" is computed against it.
  * **Object z.** C1a is a z criterion, so z_bias / z_mae are scored here
    alongside the gates rather than in a separate report that could drift.
  * **Monotonicity.** P2 / P5 are pre-registered as monotone trends in g;
    testing them needs all five arms in one table, which is the whole point of
    scoring the endpoints here.
  * **Gate fallback.** P4 tests whether E211's fallback mechanism extrapolates,
    so `cem_gate_fallback_used` is read off each rollout and carried into the
    per-row table -- desk023's fallback is leg-gate driven, not body/hand.

Note the C1 clause layout differs from E211's runner: desk023 breaks on
eef_pos/hand_pen and never on contact/release, so C1d is (eef_pos AND hand_pen)
and C1e is contact.  Swapping only the GATES dict would judge SUCCESS on the
wrong gates.

All scoring goes through `eval.core.core_metrics`; thresholds and classifiers are
imported from `funnel_config` / `eval_E206_arm_ablation` (rule 13).

Usage:
    .venv/bin/python .../eval_E212_partial_gravcomp.py [--jobs 16] [--cases a,b]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
for _p in (
    str(HERE),
    str(HERE.parent.parent),
    str(REPO / "workspace/core4d/scripts/experiments/E201"),
    str(REPO / "workspace/core4d/scripts/experiments/E206"),
    str(REPO / "workspace/core4d/scripts/experiments/E209"),
    str(REPO / "workspace/core4d/scripts/experiments/E212"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mujoco  # noqa: E402

from eval.core.core_metrics import EvalConfig, evaluate_sequence  # noqa: E402
from eval.core.motion_health import run_health  # noqa: E402
import funnel_config as FC  # noqa: E402
import eval_E206_arm_ablation as ABL  # noqa: E402
import e212_common as C  # noqa: E402


def _load_by_path(name: str, path: Path):
    """Import by explicit path. `e212_common` puts five E2xx dirs on sys.path and
    several of them ship same-named modules, so a plain import can silently bind
    the wrong one."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Z = _load_by_path(
    "e212_z_report", REPO / "workspace/core4d/scripts/eval/reports/gen_E178_object_z_diff_report.py"
)

#: The g-curve, in ascending g. "prg"/"g1" reuse existing rollouts; the rest are
#: E212 Stage A.
ARMS = ("prg", "G04", "G06", "G08", "g1")
ARM_G = {"prg": 0.0, "G04": 0.4, "G06": 0.6, "G08": 0.8, "g1": 1.0}
ARM_LABEL = {
    "prg": "E206 PRG (g=0.0)", "G04": "E212 g=0.4", "G06": "E212 g=0.6",
    "G08": "E212 g=0.8", "g1": "E209 G1 (g=1.0)",
}
BASELINE_ARM = "prg"
OUT_DIR = C.S6_DIR / "eval/g_sweep"

GATE_FIELDS = ABL.GATE_FIELDS
PHYSICS_6 = ABL.PHYSICS_6
TRACKING_6 = ABL.TRACKING_6
_finite = ABL._finite
classify_funnel = ABL.classify_funnel
classify_12gate = ABL.classify_12gate
stats = ABL.stats

#: +1 = higher is better, -1 = lower is better. E210 F6: without this, `contact`
#: is reported with its sign flipped and the headline conclusion inverts.
DIRECTION: dict[str, int] = {
    "fall_flag": -1, "body_z_err_p95_m": -1, "ankle_jerk_p95": -1, "obj_speed_max": -1,
    "track_root_pos_err_cm_mean": -1, "track_root_ori_err_deg_mean": -1,
    "track_eef_pos_err_cm_mean": -1, "track_eef_ori_err_deg_mean": -1,
    "track_obj_pos_err_cm_mean": -1, "track_obj_ori_err_deg_mean": -1,
    "hand_object_physics_contact_in_mask_frac": +1,
    "hand_object_release_false_contact_3mm_frac": -1,
    "hand_object_physics_penetration_3mm_frame_frac": -1,
    "leg_penetration_frac": -1,
    "z_abs_bias_cm": -1, "z_mae_cm": -1,
    # P4's explanatory variable. More gate fallback is worse: a fallback frame
    # discards the reward ranking entirely and picks on penetration depth alone.
    "fallback_frac": -1, "leg_fallback_frac": -1, "gate_valid_frac": +1,
}

#: Extra per-row fields carried alongside the funnel gates (P4 / mechanism).
FALLBACK_FIELDS = ("fallback_frac", "leg_fallback_frac",
                   "posture_fallback_frac", "gate_valid_frac")


def c1_eval(s: dict[str, Any], g: dict[str, float], n_paired: int) -> dict[str, Any]:
    """plan242 section 3 main gate, over one arm's macro summary.

    Module-level and pure so the gate arithmetic can be unit-tested against
    synthetic summaries -- see ``--self-test``.  A gate that has never been shown
    to reject anything is not evidence that it works.

    NOTE the clause layout is NOT E211's.  desk023's G1 failures are eef_ori
    (3/4) plus eef_pos/hand_pen (1/4), and contact/release never break -- so C1d
    is (eef_pos AND hand_pen) and C1e is contact, where E211 had C1d=contact and
    C1e=release.  Reusing E211's function body with a swapped GATES dict would
    judge SUCCESS on gates desk023 never fails.
    """
    sub = {
        "C1a_z": (s["z_abs_bias_cm"] <= g["C1a_abs_z_bias_max_cm"]
                  and s["z_mae_cm"] <= g["C1a_z_mae_max_cm"]),
        "C1b_object": (s["obj_pos_cm"] <= g["C1b_obj_pos_max_cm"]
                       and s["obj_ori_deg"] <= g["C1b_obj_ori_max_deg"]),
        "C1c_eef_ori": s["eef_ori_deg"] <= g["C1c_eef_ori_max_deg"],
        "C1d_eef_pos_hand_pen": (s["eef_pos_cm"] <= g["C1d_eef_pos_max_cm"]
                                 and s["hand_pen"] <= g["C1d_hand_pen_max"]),
        "C1e_contact": s["contact"] >= g["C1e_contact_min"],
        "C1f_narrow": (s["narrow_pass"] >= g["C1f_narrow_pass_min"]
                       and s["hard_pass"] == n_paired),
    }
    return {**sub, "pass": all(sub.values()),
            "failed": [k for k, v in sub.items() if not v]}


def c1_self_test() -> list[str]:
    """Prove each C1 clause can actually reject. Returns the clauses proven live."""
    g = C.GATES
    ok = {  # a synthetic arm that passes every clause with margin
        "z_abs_bias_cm": 0.40, "z_mae_cm": 2.50, "obj_pos_cm": 9.0,
        "obj_ori_deg": 4.5, "eef_ori_deg": 16.0, "eef_pos_cm": 13.0,
        "hand_pen": 0.15, "contact": 0.95, "narrow_pass": 4, "hard_pass": 4,
    }
    base = c1_eval(ok, g, 4)
    if not base["pass"]:
        raise SystemExit(f"C1 self-test: the all-good summary failed {base['failed']}")
    # each perturbation must break exactly its own clause
    breaks = {
        "C1a_z": {"z_mae_cm": 99.0},
        "C1b_object": {"obj_pos_cm": 99.0},
        "C1c_eef_ori": {"eef_ori_deg": 30.0},
        "C1d_eef_pos_hand_pen": {"hand_pen": 0.99},
        "C1e_contact": {"contact": 0.10},
        "C1f_narrow": {"narrow_pass": 1},
    }
    proven = []
    for clause, patch in breaks.items():
        got = c1_eval({**ok, **patch}, g, 4)
        if got["pass"] or got["failed"] != [clause]:
            raise SystemExit(
                f"C1 self-test: perturbing {patch} should break exactly [{clause}], "
                f"got failed={got['failed']} pass={got['pass']}"
            )
        proven.append(clause)
    return proven


def num(value: Any) -> float:
    """Numeric view of a rollout cell, coercing the boolean `fall_flag` to 0/1.

    `_finite` returns NaN for "True"/"False", which would drop the fall gate out
    of every aggregate silently rather than reporting it as 0% / 100% (E209 F8).
    """
    s = str(value).strip().lower()
    if s in ("true", "false"):
        return 1.0 if s == "true" else 0.0
    return _finite(value)


def arm_paths(arm: str, row: dict[str, str]) -> tuple[Path, Path, Path]:
    """(rollout npz, scene xml, config_act yaml) for this arm."""
    case_id = row["case_id"]
    if arm == "prg":
        d = C.prg_out_dir(case_id)
        return d / "trajectory_mjwp_act.npz", C.base_scene_path(row), d / "config_act.yaml"
    if arm == "g1":
        d = C.g1_out_dir(case_id)
        return (d / "trajectory_mjwp_act.npz", C.task_dir(row) / f"{C.G1_SCENE}.xml",
                d / "config_act.yaml")
    return (C.result_npz(case_id, arm), C.scene_path(row, arm), C.config_act(case_id, arm))


def object_z_stats(qpos_path: Path, scene: Path, trajectory: Path) -> dict[str, float]:
    """z bias / MAE of the object against the kinematic reference, in cm."""
    model = mujoco.MjModel.from_xml_path(str(scene))
    run = np.load(qpos_path, allow_pickle=True)["qpos"][:, 0, :]
    kin = np.load(trajectory, allow_pickle=True)["qpos"]
    z_sim, z_ref, _ = Z.object_z_series(run, kin, model)
    diff = (z_sim - z_ref) * 100.0
    return {
        "z_bias_cm": float(np.mean(diff)),
        "z_abs_bias_cm": float(abs(np.mean(diff))),
        "z_mae_cm": float(np.mean(np.abs(diff))),
    }


def gate_fallback_stats(qpos_path: Path) -> dict[str, float]:
    """Per-rollout CEM safety-gate fallback rates -- P4's explanatory variable.

    E211 found posture collapse tracks `cem_gate_fallback_used` rather than g.
    On desk023 the fallback is almost entirely the LEG gate (E211's desk007 was
    body/hand), so the sub-gates are reported separately: aggregating them would
    hide the fact that a different subsystem is doing the work.
    """
    data = np.load(qpos_path, allow_pickle=True)
    keys = {
        "fallback_frac": "cem_gate_fallback_used",
        "leg_fallback_frac": "cem_leg_gate_fallback_used",
        "posture_fallback_frac": "cem_posture_gate_fallback_used",
        "gate_valid_frac": "cem_gate_valid_frac",
    }
    out: dict[str, float] = {}
    for name, key in keys.items():
        if key in data.files:
            out[name] = float(np.nanmean(np.asarray(data[key], dtype=float)))
        else:
            out[name] = float("nan")
    return out


def score_one(arm: str, row: dict[str, str]) -> dict[str, Any]:
    case_id = row["case_id"]
    qpos, scene, cfg_path = arm_paths(arm, row)
    trajectory = C.kinematic_npz(row)
    # Mask from the run's own resolved config -- the authority is what ran, not
    # what the manifest says it should have been.
    mask = REPO / str(yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["contact_hdmi_mask_path"])
    for label, p in (("rollout", qpos), ("scene", scene), ("traj", trajectory), ("mask", mask)):
        if not Path(p).is_file():
            raise FileNotFoundError(f"{case_id}:{arm}:{label}:{p}")
    cfg = EvalConfig()
    meta = {
        "case_id": case_id, "variant": "orig", "object_key": row["object_key"],
        "spider_method_id": ABL.METHOD, "hand_collision_variant_id": ABL.HAND_VARIANT,
    }
    item = evaluate_sequence(
        row=meta, method=ABL.METHOD, hand_collision_variant_id=ABL.HAND_VARIANT,
        qpos_path=qpos, scene_xml=scene, config=cfg, kin_ref_path=trajectory,
        contact_mask_path=mask, person_idx=ABL._person_idx(case_id),
    )
    item.update(run_health(qpos, scene, cfg))
    item["body_z_err_p95_m"] = ABL.body_z_p95(qpos, scene, trajectory)
    item.update(object_z_stats(qpos, scene, trajectory))
    item.update(gate_fallback_stats(qpos))
    item["_mask"] = str(mask)
    return item


def _worker(job: tuple[str, dict[str, str]]):
    arm, row = job
    try:
        return arm, row["case_id"], score_one(arm, row), ""
    except Exception as exc:  # noqa: BLE001 - one bad case must not kill the sweep
        return arm, row["case_id"], None, f"{type(exc).__name__}: {exc}"


def monotone(values: list[float], direction: int) -> dict[str, Any]:
    """Is the metric monotone in g, in the direction the prereg predicted?

    A-P2/A-P3 predict that raising g monotonically *worsens* the robot side. So
    the predicted sign of each successive step is -direction (worse), and we
    report both strict monotonicity and the Spearman-style step agreement, since
    a single 0.01 wobble should not by itself falsify a trend.
    """
    steps = [b - a for a, b in zip(values, values[1:])]
    worse = [-direction * s > 0 for s in steps]
    return {
        "steps": [round(s, 4) for s in steps],
        "monotone_worse_with_g": all(worse),
        "n_steps_worse": sum(worse),
        "n_steps": len(steps),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--cases", default="")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    FC.assert_monotonic()
    proven = c1_self_test()
    print(f"C1 self-test: {len(proven)}/6 clauses proven to reject -> {proven}")

    sources = C.sources()
    if args.cases:
        want = {c.strip() for c in args.cases.split(",") if c.strip()}
        sources = [r for r in sources if r["case_id"] in want]
    jobs = [(arm, row) for row in sources for arm in ARMS]
    print(f"E211 eval: {len(sources)} cases x {len(ARMS)} arms = {len(jobs)} rows "
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
                "object_key": case_id.split("_")[0], "case_id": case_id,
                "arm": arm, "gravcomp": ARM_G[arm],
            }
            for f in (GATE_FIELDS + ["z_bias_cm", "z_abs_bias_cm", "z_mae_cm"]
                      + list(FALLBACK_FIELDS)):
                rec[f] = m.get(f)
            rec.update(classify_funnel(m))
            rec.update(classify_12gate(m))
            rows.append(rec)
            print(f"[{done}/{len(jobs)}] {arm}/{case_id} -> {rec['layer']}", flush=True)

    # Every arm must have read the SAME contact mask, else contact gates are
    # incomparable across the sweep. P3 asserted it on the composed configs; this
    # is the runtime confirmation.
    mask_drift = {c: v for c, v in masks.items() if len(set(v.values())) > 1}
    if mask_drift:
        raise SystemExit(f"contact mask differs between arms: {mask_drift}")

    rows.sort(key=lambda r: (r["case_id"], ARM_G[r["arm"]]))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def fmt(v: Any) -> Any:
        if isinstance(v, bool):
            return v
        if isinstance(v, float):
            return "" if not math.isfinite(v) else round(v, 4)
        return v

    metric_fields = (GATE_FIELDS + ["z_bias_cm", "z_abs_bias_cm", "z_mae_cm"]
                     + list(FALLBACK_FIELDS))
    cols = (["object_key", "case_id", "arm", "gravcomp"] + metric_fields +
            ["layer", "hard_pass", "hard_failed", "wide_pass", "wide_failed",
             "narrow_pass", "narrow_failed", "physics6_pass", "tracking6_pass",
             "gate12_pass", "gate12_failed"] +
            [f"g12_{n}" for n, *_ in PHYSICS_6 + TRACKING_6])
    with (args.out_dir / "e212_g_sweep_rollout.tsv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({c: fmt(r.get(c, "")) for c in cols})

    by = {(r["arm"], r["case_id"]): r for r in rows}
    paired = [r["case_id"] for r in sources if all((a, r["case_id"]) in by for a in ARMS)]

    def macro(arm: str, field: str) -> float:
        """Mean over the cases where THIS metric is finite for EVERY arm.

        Restricting to the common finite set matters for `release`: one case has
        an empty release window (NaN on every arm), and averaging over different
        case sets per arm would compare different populations.
        """
        common = [c for c in paired if all(math.isfinite(num(by[(a, c)][field])) for a in ARMS)]
        if not common:
            return float("nan")
        return float(np.mean([num(by[(arm, c)][field]) for c in common]))

    def common_n(field: str) -> int:
        return sum(
            1 for c in paired
            if all(math.isfinite(num(by[(a, c)][field])) for a in ARMS)
        )

    # --- per-gate table: all 14 funnel gates + the 3 z metrics ---------------
    gate_defs = ([(g[0], g[1], "hard") for g in FC.HARD_GATES]
                 + [(g[0], g[1], "banded") for g in FC.BANDED_GATES]
                 + [("z_bias", "z_bias_cm", "z"), ("z_abs_bias", "z_abs_bias_cm", "z"),
                    ("z_mae", "z_mae_cm", "z")]
                 + [(f, f, "mechanism") for f in FALLBACK_FIELDS])
    per_gate = []
    for name, field, kind in gate_defs:
        direction = DIRECTION.get(field, -1)
        entry: dict[str, Any] = {
            "gate": name, "field": field, "kind": kind,
            "direction": "higher_is_better" if direction > 0 else "lower_is_better",
            "n_common": common_n(field),
        }
        vals = []
        for arm in ARMS:
            common = [c for c in paired
                      if all(math.isfinite(num(by[(a, c)][field])) for a in ARMS)]
            for k, v in stats([num(by[(arm, c)][field]) for c in common]).items():
                entry[f"{arm}_{k}"] = v
            vals.append(macro(arm, field))
            if arm != BASELINE_ARM:
                d = [num(by[(arm, c)][field]) - num(by[(BASELINE_ARM, c)][field])
                     for c in common]
                ds = stats(d)
                entry[f"{arm}_delta_vs_prg_mean"] = ds["mean"]
                entry[f"{arm}_delta_vs_prg_worst"] = (
                    ds["min"] if direction > 0 else ds["max"]
                )
                entry[f"{arm}_n_improved"] = sum(1 for x in d if direction * x > 0)
        if field not in ("z_bias_cm", "posture_fallback_frac"):  # signed bias not monotone-in-badness
            entry.update({f"mono_{k}": v for k, v in monotone(vals, direction).items()})
        per_gate.append(entry)
    ABL.C.write_tsv(args.out_dir / "e212_per_gate.tsv", per_gate)

    # --- per-arm summary -----------------------------------------------------
    def arm_summary(arm: str) -> dict[str, Any]:
        rs = [by[(arm, c)] for c in paired]
        return {
            "gravcomp": ARM_G[arm], "n": len(rs),
            "L3_auto": sum(1 for r in rs if r["layer"] == "L3_auto"),
            "L2_review": sum(1 for r in rs if r["layer"] == "L2_review"),
            "L1_reject": sum(1 for r in rs if r["layer"] == "L1_reject"),
            "hard_pass": sum(1 for r in rs if r["hard_pass"]),
            "wide_pass": sum(1 for r in rs if r["wide_pass"]),
            "narrow_pass": sum(1 for r in rs if r["narrow_pass"]),
            "gate12_pass": sum(1 for r in rs if r["gate12_pass"]),
            "z_bias_cm": macro(arm, "z_bias_cm"),
            "z_abs_bias_cm": abs(macro(arm, "z_bias_cm")),
            "z_mae_cm": macro(arm, "z_mae_cm"),
            "eef_ori_deg": macro(arm, "track_eef_ori_err_deg_mean"),
            "contact": macro(arm, "hand_object_physics_contact_in_mask_frac"),
            "release": macro(arm, "hand_object_release_false_contact_3mm_frac"),
            "obj_pos_cm": macro(arm, "track_obj_pos_err_cm_mean"),
            "obj_ori_deg": macro(arm, "track_obj_ori_err_deg_mean"),
            "root_ori_deg": macro(arm, "track_root_ori_err_deg_mean"),
            "root_pos_cm": macro(arm, "track_root_pos_err_cm_mean"),
            "eef_pos_cm": macro(arm, "track_eef_pos_err_cm_mean"),
            "hand_pen": macro(arm, "hand_object_physics_penetration_3mm_frame_frac"),
            "fallback_frac": macro(arm, "fallback_frac"),
            "leg_fallback_frac": macro(arm, "leg_fallback_frac"),
            "gate_valid_frac": macro(arm, "gate_valid_frac"),
        }

    arms_sum = {a: arm_summary(a) for a in ARMS}

    # --- P0 endpoint reproduction: the run stops if the ruler moved ----------
    drift: list[str] = []
    for arm, frozen_key in (("prg", "prg"), ("g1", "g1")):
        frozen = C.BASELINE_DESK023[frozen_key]
        for key in ("eef_ori_deg", "contact", "obj_pos_cm", "obj_ori_deg",
                    "root_ori_deg", "root_pos_cm", "eef_pos_cm", "hand_pen"):
            got, want = arms_sum[arm][key], frozen[key]
            if abs(got - want) > 5e-3:
                drift.append(f"{arm}.{key}: rescored {got:.4f} != frozen {want}")
        if arms_sum[arm]["narrow_pass"] != int(frozen["narrow_pass"]):
            drift.append(
                f"{arm}.narrow_pass: rescored {arms_sum[arm]['narrow_pass']} "
                f"!= frozen {int(frozen['narrow_pass'])}"
            )

    # --- claims --------------------------------------------------------------
    g = C.GATES

    def macro_ex(arm: str, field: str, exclude: str) -> float:
        """Macro mean excluding one case -- plan242 section 3 anti-cherry-pick rule.

        With n=4 a single case carries 25% of the weight, so 019_p1 alone can
        drive the headline.  Both views are reported; the MAIN GATE always uses
        the all-4 view.  This exists to make the outlier's influence visible, not
        to give a second chance at passing.
        """
        common = [c for c in paired if c != exclude
                  and all(math.isfinite(num(by[(a, c)][field])) for a in ARMS)]
        if not common:
            return float("nan")
        return float(np.mean([num(by[(arm, c)][field]) for c in common]))

    def c1_for(arm: str) -> dict[str, Any]:
        return c1_eval(arms_sum[arm], g, len(paired))

    def c2_c3_for(arm: str) -> dict[str, Any]:
        """Secondary gates: recorded, never decide SUCCESS, but a break must be
        named explicitly in the log (plan242 section 3)."""
        s = arms_sum[arm]
        sub = {
            "C2_root_ori": s["root_ori_deg"] <= g["C2_root_ori_max_deg"],
            "C2_root_pos": s["root_pos_cm"] <= g["C2_root_pos_max_cm"],
            "C3_release": s["release"] <= g["C3_release_max"],
            "C3_body_z_hard": macro(arm, "body_z_err_p95_m") <= g["C3_body_z_p95_max_m"],
            "C3_leg_pen_hard": macro(arm, "leg_penetration_frac") <= g["C3_leg_pen_max"],
            "C3_ankle_jerk_hard": macro(arm, "ankle_jerk_p95") < g["C3_ankle_jerk_max"],
        }
        return {**sub, "failed": [k for k, v in sub.items() if not v]}

    c1 = {a: c1_for(a) for a in ARMS}
    c2c3 = {a: c2_c3_for(a) for a in ARMS}
    winners = [a for a in ARMS if a not in (BASELINE_ARM, "g1") and c1[a]["pass"]]

    gsv = [ARM_G[a] for a in ARMS]
    bias = [arms_sum[a]["z_bias_cm"] for a in ARMS]
    A = np.vstack([np.ones(len(gsv)), gsv]).T
    coef, *_ = np.linalg.lstsq(A, np.array(bias), rcond=None)
    pred = A @ coef
    r2 = float(1 - ((np.array(bias) - pred) ** 2).sum()
               / ((np.array(bias) - np.mean(bias)) ** 2).sum())

    def mono_of(field: str) -> dict[str, Any]:
        return monotone([macro(a, field) for a in ARMS], DIRECTION[field])

    def case_mono(case_id: str, field: str) -> dict[str, Any]:
        """Monotonicity of one case's metric across the g-curve."""
        vals = [num(by[(a, case_id)][field]) for a in ARMS]
        out = monotone(vals, DIRECTION[field])
        out["values"] = [round(v, 4) for v in vals]
        return out

    def pearson(xs: list[float], ys: list[float]) -> float:
        x, y = np.asarray(xs, float), np.asarray(ys, float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
            return float("nan")
        return float(np.corrcoef(x[ok], y[ok])[0, 1])

    # P4: does E211's fallback mechanism extrapolate to desk023?  Correlate
    # per-rollout fallback against eef_ori over all (case, arm) rows, holding g
    # fixed via the partial correlation -- on desk007 this was +0.504 while
    # r(g, fallback) was only +0.016, i.e. two separable variables.
    rows_fb = [(num(r["fallback_frac"]), num(r["track_eef_ori_err_deg_mean"]),
                float(r["gravcomp"])) for r in rows if r["case_id"] in paired]
    fb, eo, gg = [list(t) for t in zip(*rows_fb)] if rows_fb else ([], [], [])
    r_fe, r_fg, r_eg = pearson(fb, eo), pearson(fb, gg), pearson(eo, gg)
    denom = math.sqrt(max(1e-12, (1 - r_fg ** 2) * (1 - r_eg ** 2)))
    partial_fe_g = (r_fe - r_fg * r_eg) / denom if math.isfinite(r_fe) else float("nan")

    p3_case, p5_case = C.PREREG_P3_CASE, C.PREREG_P5_CASE
    new_arms = [a for a in ARMS if a not in (BASELINE_ARM, "g1")]
    p3_narrow = {a: bool(by[(a, p3_case)]["narrow_pass"]) for a in ARMS} \
        if p3_case in paired else {}
    p5 = case_mono(p5_case, "track_eef_ori_err_deg_mean") if p5_case in paired else {}

    prereg = {
        "P1_z_linear": {
            "fit_intercept_cm": round(float(coef[0]), 4),
            "fit_slope_cm_per_g": round(float(coef[1]), 4),
            "r2": round(r2, 4),
            "prereg": C.PREREG_Z_LINEAR,
            "pass": r2 >= C.PREREG_Z_LINEAR["min_r2"],
        },
        "P2_eef_ori_monotone": {
            **mono_of("track_eef_ori_err_deg_mean"),
            "values": [round(macro(a, "track_eef_ori_err_deg_mean"), 4) for a in ARMS],
            # E211 falsified this on desk007; a cross-family re-test.
            "pass": mono_of("track_eef_ori_err_deg_mean")["monotone_worse_with_g"],
        },
        "P3_zero_fallback_case_survives": {
            "case": p3_case,
            "narrow_by_arm": p3_narrow,
            "fallback_prg_g1": [C.PREREG_PER_CASE[p3_case]["fallback_prg"],
                                C.PREREG_PER_CASE[p3_case]["fallback_g1"]],
            "pass": bool(p3_narrow) and all(p3_narrow[a] for a in new_arms),
        },
        "P4_fallback_explains_eef_ori": {
            "r_fallback_eef_ori": round(r_fe, 4),
            "r_fallback_g": round(r_fg, 4),
            "r_eef_ori_g": round(r_eg, 4),
            "partial_r_fallback_eef_ori_given_g": round(partial_fe_g, 4),
            "min_partial_r": C.PREREG_P4_MIN_PARTIAL_R,
            "n_rows": len(rows_fb),
            "e211_desk007_partial_r": 0.504,
            "pass": bool(math.isfinite(partial_fe_g)
                         and partial_fe_g > C.PREREG_P4_MIN_PARTIAL_R),
        },
        "P5_flat_fallback_case_is_load_driven": {
            "case": p5_case,
            **p5,
            "fallback_prg_g1": [C.PREREG_PER_CASE[p5_case]["fallback_prg"],
                                C.PREREG_PER_CASE[p5_case]["fallback_g1"]],
            "pass": bool(p5.get("monotone_worse_with_g", False)),
        },
        "C1_some_arm_passes": {"winners": winners, "pass": bool(winners)},
    }

    # Anti-cherry-pick: the same headline metrics with the outlier removed.
    # Reported for honesty; the gate above already used the all-4 view.
    outlier = C.OUTLIER_CASE
    sens_fields = {
        "z_abs_bias_cm": "z_bias_cm", "z_mae_cm": "z_mae_cm",
        "eef_ori_deg": "track_eef_ori_err_deg_mean",
        "eef_pos_cm": "track_eef_pos_err_cm_mean",
        "hand_pen": "hand_object_physics_penetration_3mm_frame_frac",
        "contact": "hand_object_physics_contact_in_mask_frac",
    }
    sensitivity = {
        "excluded_case": outlier,
        "n_excluded": len([c for c in paired if c != outlier]),
        "all4": {k: {a: round(macro(a, f), 4) for a in ARMS}
                 for k, f in sens_fields.items()},
        f"without_{outlier}": {k: {a: round(macro_ex(a, f, outlier), 4) for a in ARMS}
                               for k, f in sens_fields.items()},
    }
    sensitivity["all4"]["z_abs_bias_cm"] = {
        a: round(abs(macro(a, "z_bias_cm")), 4) for a in ARMS}
    sensitivity[f"without_{outlier}"]["z_abs_bias_cm"] = {
        a: round(abs(macro_ex(a, "z_bias_cm", outlier)), 4) for a in ARMS}

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": {"object_key": C.OBJECT_KEY, "cases": paired, "arms": list(ARMS),
                  "gravcomp": ARM_G},
        "note": "all five arms scored from rollouts through ONE code path; "
                "endpoints re-scored, never lifted from E209's TSV",
        "scored_rows": len(rows), "errors": errors, "paired_cases": len(paired),
        "release_n_common": common_n("hand_object_release_false_contact_3mm_frac"),
        "arm_label": ARM_LABEL, "arms": arms_sum,
        "endpoint_rescore_drift": drift,
        "C1": c1, "C1_self_test_proven": proven, "C2_C3": c2c3, "prereg": prereg, "gates": g,
        "sensitivity_outlier": sensitivity,
    }
    (args.out_dir / "e212_g_sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    # --- console report ------------------------------------------------------
    hdr = ["metric"] + [f"g={ARM_G[a]:.1f}" for a in ARMS]
    print("\n" + " ".join(f"{h:>12s}" for h in hdr))
    for key, label in (
        ("z_bias_cm", "z_bias cm"), ("z_mae_cm", "z_mae cm"),
        ("obj_pos_cm", "obj_pos cm"), ("obj_ori_deg", "obj_ori deg"),
        ("eef_ori_deg", "eef_ori deg"), ("eef_pos_cm", "eef_pos cm"),
        ("root_ori_deg", "root_ori deg"), ("root_pos_cm", "root_pos cm"),
        ("contact", "contact"), ("release", "release"), ("hand_pen", "hand_pen"),
    ):
        print(f"{label:>12s} " + " ".join(f"{arms_sum[a][key]:12.4f}" for a in ARMS))
    for key, label in (("fallback_frac", "fallback"), ("leg_fallback_frac", "leg_fb"),
                       ("gate_valid_frac", "gate_valid")):
        print(f"{label:>12s} " + " ".join(f"{arms_sum[a][key]:12.4f}" for a in ARMS))
    print(f"{'narrow':>12s} " + " ".join(f"{arms_sum[a]['narrow_pass']:8d}/{len(paired)}" for a in ARMS))
    print(f"{'hard':>12s} " + " ".join(f"{arms_sum[a]['hard_pass']:8d}/{len(paired)}" for a in ARMS))

    print("\nC1 (main gate), per arm:")
    for a in ARMS:
        mark = "PASS" if c1[a]["pass"] else "FAIL"
        print(f"  g={ARM_G[a]:.1f} [{mark}] " + (f"failed={c1[a]['failed']}" if c1[a]["failed"] else ""))
    print("\nC2/C3 (secondary, recorded not gating):")
    for a in ARMS:
        f = c2c3[a]["failed"]
        print(f"  g={ARM_G[a]:.1f} " + (f"broken={f}" if f else "all clear"))
    print(f"\nSensitivity (excluding outlier {C.OUTLIER_CASE}, n=3):")
    for k in ("z_abs_bias_cm", "eef_ori_deg", "eef_pos_cm", "hand_pen"):
        a4 = sensitivity["all4"][k]
        ex = sensitivity[f"without_{C.OUTLIER_CASE}"][k]
        print(f"  {k:14s} all4 " + " ".join(f"{a4[a]:8.3f}" for a in ARMS)
              + "   | ex " + " ".join(f"{ex[a]:8.3f}" for a in ARMS))
    print("\nPre-registered predictions:")
    for k, v in prereg.items():
        verdict = v.get("pass")
        tag = "" if verdict is None else ("PASS" if verdict else "FAIL")
        print(f"  {k:26s} {tag:4s} {json.dumps({x: y for x, y in v.items() if x != 'pass'}, default=str)}")

    if drift:
        print("\n".join(f"  ENDPOINT DRIFT {d}" for d in drift))
        raise SystemExit(
            "endpoint re-score disagrees with the P0-frozen baseline -- evaluator "
            "drift, results are not comparable"
        )
    print(f"\npaired_cases={len(paired)}  errors={len(errors)}  -> {args.out_dir.relative_to(REPO)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
