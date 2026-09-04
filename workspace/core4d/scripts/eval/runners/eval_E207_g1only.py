#!/usr/bin/env python3
"""E207 (G1only) vs E178 (PRG) / E204 (noPRG) / E205 (G1A2) — four-arm 14-gate.

Scores the 9 E207 bucket cases under the IDENTICAL E201 14-gate funnel used by
eval_E204E205_arm_ablation.py, so the fourth arm lands on the same ruler as the
three existing ones. Nothing about the funnel, the metrics, or the PRG baseline
read path is reimplemented here -- all of it is imported from that module
(rule 13); only E207's own rollout/scene path resolution is new.

Arm semantics (see plan237):
    PRG    E178  A0 hand-gate, no gravcomp   <- canonical e178_case_metrics.tsv
    noPRG  E204  leg PRG off,  no gravcomp   <- rescored rollout
    G1A2   E205  A2 hand-gate, gravcomp      <- rescored rollout
    G1only E207  A0 hand-gate, gravcomp      <- rescored rollout (this experiment)

The two single-variable contrasts the funnel table should be read along:
    PRG    -> G1only   isolates gravcomp (G1)
    G1only -> G1A2     isolates the hand-gate (A2), given gravcomp on

Usage:
    MUJOCO_GL=disable .venv/bin/python \
      workspace/core4d/scripts/eval/runners/eval_E207_g1only.py
    ... --limit N     # first N cases (smoke)
    ... --arms PRG,G1only
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
for _p in ("workspace/core4d/scripts",
           "workspace/core4d/scripts/eval/runners",
           "workspace/core4d/scripts/experiments/E201",
           "workspace/core4d/scripts/experiments/E204_E205",
           "workspace/core4d/scripts/experiments/E207"):
    sys.path.insert(0, str(REPO / _p))

from eval.core.core_metrics import EvalConfig, evaluate_sequence  # noqa: E402
from eval.core.motion_health import run_health  # noqa: E402
import funnel_config as FC  # noqa: E402
import eval_E204E205_arm_ablation as ABL  # noqa: E402
import e204e205_common as C205  # noqa: E402
import e207_common as C207  # noqa: E402
from eval_E187_e178_compat import body_z_p95  # noqa: E402

OUT_DIR = C207.RESULTS / "s6_downstream/eval/four_arm"
ARMS = ("PRG", "noPRG", "G1A2", "G1only")


def rel_label(path: Path) -> str:
    """Repo-relative label; `results/` is a symlink off-tree, so compare lexically."""
    try:
        return str((path if path.is_absolute() else REPO / path).relative_to(REPO))
    except ValueError:
        return str(path)


def arm_paths(arm: str, case_id: str) -> tuple[Path, Path]:
    """(rollout npz, scene xml) for a rescored arm."""
    if arm == "G1only":
        return C207.result_npz(case_id, "full"), C207.scene_path(case_id)
    key = {"noPRG": "noprg_e204", "G1A2": "g1a2_e205"}[arm]
    return C205.result_npz(key, case_id, "full"), C205.arm_scene_path(key, case_id)


def score(arm: str, source: dict, cfg: EvalConfig) -> dict[str, Any]:
    case_id = source["case_id"]
    qpos, scene = arm_paths(arm, case_id)
    _target, trajectory, mask = C205.base.local_authorities(source)
    for label, path in (("rollout", qpos), ("scene", scene),
                        ("traj", trajectory), ("mask", mask)):
        if not Path(path).is_file():
            raise FileNotFoundError(f"{case_id}:{arm}:{label}:{path}")
    row = {"case_id": case_id, "variant": "orig", "object_key": source["object_key"],
           "spider_method_id": ABL.METHOD, "hand_collision_variant_id": ABL.HAND_VARIANT}
    item = evaluate_sequence(row=row, method=ABL.METHOD,
                             hand_collision_variant_id=ABL.HAND_VARIANT,
                             qpos_path=qpos, scene_xml=scene, config=cfg,
                             kin_ref_path=trajectory, contact_mask_path=mask,
                             person_idx=ABL._person_idx(case_id))
    item.update(run_health(qpos, scene, cfg))
    item["body_z_err_p95_m"] = body_z_p95(qpos, scene, trajectory)
    return item


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    FC.assert_monotonic()
    arms = [a for a in args.arms.split(",") if a.strip()]
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise SystemExit(f"unknown arms: {sorted(unknown)}")

    C207.audit(verbose=False)
    sources = C207.sources()[: args.limit or None]
    case_ids = [s["case_id"] for s in sources]

    prg = {r["case_id"]: ABL.read_prg_row(r)
           for r in C205.base.read_tsv(ABL.E178_CASE_METRICS) if r["case_id"] in set(case_ids)}
    missing = sorted(set(case_ids) - set(prg))
    if missing:
        raise SystemExit(f"E178 baseline missing for: {missing}")

    cfg = EvalConfig()
    rows: list[dict[str, Any]] = []
    for i, source in enumerate(sources, 1):
        case_id = source["case_id"]
        for arm in arms:
            metrics = prg[case_id] if arm == "PRG" else score(arm, source, cfg)
            record = {"object_key": source["object_key"], "case_id": case_id, "arm": arm}
            for field in ABL.GATE_FIELDS:
                record[field] = metrics.get(field)
            record.update(ABL.classify(metrics))
            rows.append(record)
        print(f"[{i}/{len(sources)}] {case_id} scored ({len(arms)} arms)", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "four_arm_rollout.tsv"
    cols = ["object_key", "case_id", "arm"] + ABL.GATE_FIELDS + [
        "layer", "hard_pass", "hard_failed", "wide_pass", "wide_failed",
        "narrow_pass", "narrow_failed"]

    def fmt(value: Any) -> Any:
        if isinstance(value, bool):
            return value
        if isinstance(value, float):
            return "" if not math.isfinite(value) else round(value, 4)
        return value

    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        for record in rows:
            writer.writerow({c: fmt(record.get(c, "")) for c in cols})
    print(f"\n[done] {rel_label(out)} ({len(rows)} rows = {len(sources)} cases x {len(arms)} arms)")

    print("\n=== 14-gate pass summary (per arm, n={}) ===".format(len(sources)))
    for arm in arms:
        group = [r for r in rows if r["arm"] == arm]
        n = len(group)
        if not n:
            continue
        print(f"  {arm:7s} hard={sum(r['hard_pass'] for r in group):2d}/{n}  "
              f"wide_all={sum(r['wide_pass'] for r in group):2d}/{n}  "
              f"narrow_all={sum(r['narrow_pass'] for r in group):2d}/{n} "
              f"({sum(r['narrow_pass'] for r in group)/n:.0%})  "
              f"L3_auto={sum(r['layer']=='L3_auto' for r in group):2d}")

    print("\n=== per-gate NARROW pass count (per arm) ===")
    gate_defs = [("fall", "fall_flag"), ("body_z", "body_z_err_p95_m"),
                 ("ankle_jerk", "ankle_jerk_p95"), ("obj_speed", "obj_speed_max")] + \
        [(g[0], g[1]) for g in FC.BANDED_GATES]
    print("  gate        " + "".join(f"{a:>9s}" for a in arms))
    hard = {h[1]: h for h in FC.HARD_GATES}
    banded = {g[1]: g for g in FC.BANDED_GATES}
    for gname, gfield in gate_defs:
        cells = ""
        for arm in arms:
            group = [r for r in rows if r["arm"] == arm]
            if gname == "fall":
                ok = sum(str(r["fall_flag"]).strip().lower() not in ("true", "1") for r in group)
            elif gfield in hard:
                _n, _f, op, thr = hard[gfield]
                ok = sum(FC.passes(op, ABL._finite(r[gfield]), thr) for r in group)
            else:
                bg = banded[gfield]
                ok = sum(FC.passes(bg[2], ABL._finite(r[gfield]), bg[3]) for r in group)
            cells += f"{ok:>9d}"
        print(f"  {gname:12s}{cells}")

    print("\n=== paired deltas (single-variable contrasts) ===")
    by = {(r["arm"], r["case_id"]): r for r in rows}
    for a, b, label in (("PRG", "G1only", "isolates gravcomp"),
                        ("G1only", "G1A2", "isolates A2 hand-gate")):
        if a not in arms or b not in arms:
            continue
        wins = sum(by[(b, c)]["narrow_pass"] and not by[(a, c)]["narrow_pass"] for c in case_ids)
        loss = sum(by[(a, c)]["narrow_pass"] and not by[(b, c)]["narrow_pass"] for c in case_ids)
        print(f"  {a:7s} -> {b:7s} ({label}): narrow +{wins} / -{loss}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
