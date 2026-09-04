#!/usr/bin/env python3
"""E206 P9: paired noPRG vs PRG evaluation over the 130 fresh CEM rollouts.

Key improvement over `eval_E204E205_arm_ablation.py`: **both arms are scored
from fresh rollouts with the same code path**. E204 read its PRG row out of the
E178 case-metrics TSV, so the two arms were not measured under identical
conditions -- a confound in the very comparison the experiment exists to make.
Here every row goes through `evaluate_sequence` + `run_health` + `body_z_p95`.

Two calibers, reported side by side (plan236 P9):
  * E201 14-gate funnel (4 hard + 10 banded, wide/narrow) -> L1/L2/L3
  * frozen 12-gate (physics 6 + tracking 6, docs/EVAL_METRICS_12GATE.md)
Their thresholds genuinely differ (e.g. leg_pen narrow 0.20 vs 12-gate 0.10),
so neither is a restatement of the other.

Statistics follow rule 5: every gate reports mean / std / **worst**, never just
a pass count, and the per-object table carries n so that chair020 (n=1) and
chair005 / desk020 (n=2) cannot be read as per-object recommendations.

Usage:
    .venv/bin/python .../eval_E206_arm_ablation.py
    ... --limit 4 --objects desk007 --jobs 16
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E201"))
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E206"))

from eval.core.core_metrics import EvalConfig, evaluate_sequence  # noqa: E402
from eval.core.motion_health import run_health  # noqa: E402
from eval_E187_e178_compat import body_z_p95  # noqa: E402
import funnel_config as FC  # noqa: E402
import e206_common as C  # noqa: E402

METHOD = C.BASE_REWARD_METHOD
HAND_VARIANT = "rubber_hull"
OUT_DIR = C.S6_DIR / "eval/two_arm"

GATE_FIELDS = ["fall_flag", "body_z_err_p95_m", "ankle_jerk_p95", "obj_speed_max"] + \
    [g[1] for g in FC.BANDED_GATES]

# Frozen 12-gate (docs/EVAL_METRICS_12GATE.md) -- NOT the funnel bands.
PHYSICS_6 = [
    ("fall", "fall_flag", "fall", 0.0),
    ("body_z", "body_z_err_p95_m", "<=", 0.20),
    ("contact", "hand_object_physics_contact_in_mask_frac", ">=", 0.50),
    ("release", "hand_object_release_false_contact_3mm_frac", "<=", 0.30),
    ("hand_penetration", "hand_object_physics_penetration_3mm_frame_frac", "<=", 0.30),
    ("lower_body", "leg_penetration_frac", "<=", 0.10),
]
TRACKING_6 = [
    ("root_pos", "track_root_pos_err_cm_mean", "<=", 20.0),
    ("root_ori", "track_root_ori_err_deg_mean", "<=", 20.0),
    ("hand_pos", "track_eef_pos_err_cm_mean", "<=", 20.0),
    ("hand_ori", "track_eef_ori_err_deg_mean", "<=", 20.0),
    ("object_pos", "track_obj_pos_err_cm_mean", "<=", 20.0),
    ("object_ori", "track_obj_ori_err_deg_mean", "<=", 10.0),
]


def _finite(v: Any) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else math.nan
    except (TypeError, ValueError):
        return math.nan


def _person_idx(case_id: str) -> int:
    return 0 if case_id.lower().endswith("_p1") else 1


def target_task_by_case() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("omnirt_v1", "omnirt_v2"):
        m = C.S3_DIR / f"{name}/ref_fk/stage2b_manifest_{name}_ref_fk.tsv"
        if not m.is_file():
            continue
        for row in C.read_tsv(m):
            if row.get("stage2b_status") == "pass":
                out[row["case_id"]] = row["target_task"]
    return out


def contact_mask_for(case_id: str, arm: str) -> Path:
    """Mask path from the run's own config_act.yaml -- the authority is what ran."""
    import yaml
    cfg = yaml.safe_load(
        (C.arm_out_dir(arm, case_id, "full") / "config_act.yaml").read_text(encoding="utf-8"))
    return C.REPO / str(cfg["contact_hdmi_mask_path"])


def score_one(case_id: str, arm: str, task: str) -> dict[str, Any]:
    """Score one arm's fresh rollout into the funnel + 12-gate fields."""
    qpos = C.arm_out_dir(arm, case_id, "full") / "trajectory_mjwp_act.npz"
    tdir = C.PROCESSED_ROOT / task
    scene = tdir / f"{C.SCENE_BY_ARM[arm]}.xml"
    trajectory = tdir / "0/trajectory_kinematic.npz"
    mask = contact_mask_for(case_id, arm)
    for label, p in (("rollout", qpos), ("scene", scene),
                     ("traj", trajectory), ("mask", mask)):
        if not p.is_file():
            raise FileNotFoundError(f"{case_id}:{arm}:{label}:{p}")
    cfg = EvalConfig()
    row = {"case_id": case_id, "variant": "orig",
           "object_key": case_id.split("_")[0],
           "spider_method_id": METHOD, "hand_collision_variant_id": HAND_VARIANT}
    item = evaluate_sequence(row=row, method=METHOD,
                             hand_collision_variant_id=HAND_VARIANT,
                             qpos_path=qpos, scene_xml=scene, config=cfg,
                             kin_ref_path=trajectory, contact_mask_path=mask,
                             person_idx=_person_idx(case_id))
    item.update(run_health(qpos, scene, cfg))
    item["body_z_err_p95_m"] = body_z_p95(qpos, scene, trajectory)
    return item


def _worker(args: tuple[str, str, str]) -> tuple[str, str, dict[str, Any] | None, str]:
    case_id, arm, task = args
    try:
        return case_id, arm, score_one(case_id, arm, task), ""
    except Exception as exc:  # noqa: BLE001 - one bad case must not kill the sweep
        return case_id, arm, None, f"{type(exc).__name__}: {exc}"


def classify_funnel(m: dict[str, Any]) -> dict[str, Any]:
    hard_ok, hard_bad = FC.hard_gate_result(m)
    wide_ok, wide_bad = FC.banded_gate_result(m, "wide")
    narrow_ok, narrow_bad = FC.banded_gate_result(m, "narrow")
    layer = "L1_reject" if (not hard_ok or not wide_ok) else (
        "L2_review" if not narrow_ok else "L3_auto")
    return {"layer": layer,
            "hard_pass": hard_ok, "hard_failed": ",".join(hard_bad),
            "wide_pass": wide_ok, "wide_failed": ",".join(wide_bad),
            "narrow_pass": narrow_ok, "narrow_failed": ",".join(narrow_bad)}


def gate_ok(op: str, value: Any, thr: float) -> bool:
    if op == "fall":
        return not (str(value).lower() in ("true", "1") or value is True)
    v = _finite(value)
    if not math.isfinite(v):
        # release is N/A (auto-pass) when there is no release window; every other
        # non-finite metric is a failure, not a free pass.
        return False
    return v <= thr if op == "<=" else v >= thr


def classify_12gate(m: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    phys, track = [], []
    for name, field, op, thr in PHYSICS_6:
        ok = gate_ok(op, m.get(field), thr)
        if name == "release" and not math.isfinite(_finite(m.get(field))):
            ok = True          # documented N/A => auto-pass
        out[f"g12_{name}"] = ok
        phys.append(ok)
    for name, field, op, thr in TRACKING_6:
        ok = gate_ok(op, m.get(field), thr)
        out[f"g12_{name}"] = ok
        track.append(ok)
    out["physics6_pass"] = all(phys)
    out["tracking6_pass"] = all(track)
    out["gate12_pass"] = all(phys) and all(track)
    out["gate12_failed"] = ",".join(
        k[4:] for k, v in out.items() if k.startswith("g12_") and not v)
    return out


def stats(values: list[float]) -> dict[str, Any]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return {"n": 0, "mean": "", "std": "", "worst": "", "min": "", "max": ""}
    return {"n": len(vals),
            "mean": round(statistics.mean(vals), 4),
            "std": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--objects", default="")
    ap.add_argument("--cases", default="")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    FC.assert_monotonic()

    tasks = target_task_by_case()
    case_ids = sorted(tasks)
    if args.objects:
        want = {o.strip() for o in args.objects.split(",") if o.strip()}
        case_ids = [c for c in case_ids if c.split("_")[0] in want]
    if args.cases:
        want = {c.strip() for c in args.cases.split(",") if c.strip()}
        case_ids = [c for c in case_ids if c in want]
    if args.limit:
        case_ids = case_ids[: args.limit]

    jobs = [(c, arm, tasks[c]) for c in case_ids for arm in C.ARMS]
    print(f"E206 eval: {len(case_ids)} cases x {len(C.ARMS)} arms = {len(jobs)} rows "
          f"| jobs={args.jobs}", flush=True)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(_worker, j): j for j in jobs}
        for fut in as_completed(futs):
            case_id, arm, m, err = fut.result()
            done += 1
            if m is None:
                errors.append({"case_id": case_id, "arm": arm, "error": err})
                print(f"[{done}/{len(jobs)}] {arm}/{case_id} FAILED {err}",
                      file=sys.stderr, flush=True)
                continue
            rec: dict[str, Any] = {"object_key": case_id.split("_")[0],
                                   "case_id": case_id, "arm": arm}
            for f in GATE_FIELDS:
                rec[f] = m.get(f)
            rec.update(classify_funnel(m))
            rec.update(classify_12gate(m))
            rows.append(rec)
            print(f"[{done}/{len(jobs)}] {arm}/{case_id} -> {rec['layer']} "
                  f"12gate={'pass' if rec['gate12_pass'] else 'fail'}", flush=True)

    rows.sort(key=lambda r: (r["object_key"], r["case_id"], r["arm"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def fmt(v: Any) -> Any:
        if isinstance(v, bool):
            return v
        if isinstance(v, float):
            return "" if not math.isfinite(v) else round(v, 4)
        return v

    cols = (["object_key", "case_id", "arm"] + GATE_FIELDS +
            ["layer", "hard_pass", "hard_failed", "wide_pass", "wide_failed",
             "narrow_pass", "narrow_failed", "physics6_pass", "tracking6_pass",
             "gate12_pass", "gate12_failed"] +
            [f"g12_{n}" for n, *_ in PHYSICS_6 + TRACKING_6])
    with (args.out_dir / "e206_two_arm_rollout.tsv").open("w", newline="",
                                                          encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({c: fmt(r.get(c, "")) for c in cols})

    by = {(r["arm"], r["case_id"]): r for r in rows}
    paired = [c for c in case_ids if all((a, c) in by for a in C.ARMS)]

    per_gate = []
    for name, field, *_ in [(g[0], g[1]) for g in FC.BANDED_GATES] + \
                           [("body_z", "body_z_err_p95_m")]:
        entry: dict[str, Any] = {"gate": name, "field": field}
        for arm in C.ARMS:
            s = stats([_finite(by[(arm, c)][field]) for c in paired])
            for k, v in s.items():
                entry[f"{arm}_{k}"] = v
        d = [_finite(by[("prg", c)][field]) - _finite(by[("noprg", c)][field])
             for c in paired]
        ds = stats(d)
        entry.update({"paired_delta_mean": ds["mean"], "paired_delta_std": ds["std"],
                      "paired_delta_min": ds["min"], "paired_delta_max": ds["max"]})
        per_gate.append(entry)
    C.write_tsv(args.out_dir / "e206_per_gate.tsv", per_gate)

    per_object = []
    for obj in sorted({c.split("_")[0] for c in paired}):
        cs = [c for c in paired if c.split("_")[0] == obj]
        row: dict[str, Any] = {"object_key": obj, "n": len(cs),
                               "per_object_recommendation_allowed": len(cs) >= 3}
        for arm in C.ARMS:
            rs = [by[(arm, c)] for c in cs]
            row[f"{arm}_L3"] = sum(1 for r in rs if r["layer"] == "L3_auto")
            row[f"{arm}_L2"] = sum(1 for r in rs if r["layer"] == "L2_review")
            row[f"{arm}_L1"] = sum(1 for r in rs if r["layer"] == "L1_reject")
            row[f"{arm}_gate12"] = sum(1 for r in rs if r["gate12_pass"])
            row[f"{arm}_physics6"] = sum(1 for r in rs if r["physics6_pass"])
            row[f"{arm}_legpen_mean"] = stats(
                [_finite(r["leg_penetration_frac"]) for r in rs])["mean"]
            row[f"{arm}_contact_mean"] = stats(
                [_finite(r["hand_object_physics_contact_in_mask_frac"]) for r in rs])["mean"]
        per_object.append(row)
    C.write_tsv(args.out_dir / "e206_per_object.tsv", per_object)

    summary: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scored_rows": len(rows), "errors": errors,
        "paired_cases": len(paired),
        "note": "both arms scored from FRESH rollouts through one code path",
        "arms": {},
    }
    for arm in C.ARMS:
        rs = [by[(arm, c)] for c in paired]
        summary["arms"][arm] = {
            "n": len(rs),
            "L3_auto": sum(1 for r in rs if r["layer"] == "L3_auto"),
            "L2_review": sum(1 for r in rs if r["layer"] == "L2_review"),
            "L1_reject": sum(1 for r in rs if r["layer"] == "L1_reject"),
            "gate12_pass": sum(1 for r in rs if r["gate12_pass"]),
            "physics6_pass": sum(1 for r in rs if r["physics6_pass"]),
            "tracking6_pass": sum(1 for r in rs if r["tracking6_pass"]),
            "narrow_pass": sum(1 for r in rs if r["narrow_pass"]),
        }
    (args.out_dir / "e206_two_arm_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")
    print("\n" + json.dumps(summary["arms"], ensure_ascii=False, indent=2))
    print(f"paired_cases={len(paired)}  errors={len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
