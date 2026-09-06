#!/usr/bin/env python3
"""E210: the 2x2 of {orig, aug} x {PRG, PRG+G1} on the E201 14-gate funnel.

    A  orig x PRG     E178   frozen e178_case_metrics.tsv
    B  orig x PRG+G1  E207   rescored from E207's rollout
    C  aug  x PRG     E202   rescored from E202's rollout
    D  aug  x PRG+G1  E210   rescored from E210's rollout   <- this experiment

Everything about the funnel and the metrics is imported from
eval_E204E205_arm_ablation / E201.funnel_config (rule 13), so all four cells land
on the ruler E207 was judged with. Only cell resolution is new here.

Contrasts, in decreasing order of how much they can be trusted:

  C -> D   strictly single-variable (gravcomp only; same aug trajectory, same
           mask, same proxy, same budget). This is the clean scientific result.
  A -> B   the same intervention on orig -- E207's own result, recomputed here so
           the interaction term below is apples-to-apples.
  (D-C) vs (A-B)  the interaction: does gravcomp cost more on augmented motion?
  B -> D   the delivery question, but it moves TWO axes at once (object
           perturbation AND omnirt_v1->v2 retarget). Labelled as confounded
           everywhere it is printed; never quoted as a causal effect.

Two judgement axes were rewritten relative to plan237, on evidence that postdates
E207 (see plan240):
  * C5a phantom support is read off `eef_ori`, NOT `contact_in_mask` -- E209
    showed contact does not react to phantom support at all (-0.016) while
    eef_ori moved +2.78 deg.
  * C5b hand penetration gets its own axis -- E208 F14 found it carries 13 of 16
    gate flips on augmented desk/chair, and E207 measured gravcomp adding +0.035
    on its own. E210 is the first time both act together.

Usage:
    MUJOCO_GL=disable .venv/bin/python \
      workspace/core4d/scripts/eval/runners/eval_E210_aug_g1only.py
    ... --cells A,C,D --limit 2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
for _p in ("workspace/core4d/scripts",
           "workspace/core4d/scripts/eval/runners",
           "workspace/core4d/scripts/experiments/E201",
           "workspace/core4d/scripts/experiments/E204_E205",
           "workspace/core4d/scripts/experiments/E207",
           "workspace/core4d/scripts/experiments/E210"):
    sys.path.insert(0, str(REPO / _p))

from eval.core.core_metrics import EvalConfig, evaluate_sequence  # noqa: E402
from eval.core.motion_health import run_health  # noqa: E402
import funnel_config as FC  # noqa: E402
import eval_E204E205_arm_ablation as ABL  # noqa: E402
import e204e205_common as C205  # noqa: E402
import e207_common as C207  # noqa: E402
import e210_common as C210  # noqa: E402

OUT_DIR = C210.RESULTS / "s6_downstream/eval/four_cell"
CELLS = ("A_orig_PRG", "B_orig_G1", "C_aug_PRG", "D_aug_G1")
E202_ARTIFACTS = (
    REPO / "workspace/core4d/results/E202/data_preprocess/manifests/e202_bucket_aug_artifacts.tsv"
)
E207_FROZEN = (
    REPO / "workspace/core4d/results/E207/s6_downstream/eval/four_arm/four_arm_rollout.tsv"
)
#: The two axes plan240 promoted to first-class judgement (C5a / C5b).
EEF_ORI = "track_eef_ori_err_deg_mean"
HAND_PEN = "hand_object_physics_penetration_3mm_frame_frac"
CONTACT = "hand_object_physics_contact_3mm_in_mask_frac"


def rel_label(path: Path) -> str:
    """`results/` is a symlink off-tree, so relative_to(REPO) raises (E207 F8)."""
    try:
        return str((path if path.is_absolute() else REPO / path).relative_to(REPO))
    except ValueError:
        return str(path)


def score_paths(case_id: str, object_key: str, variant: str, qpos: Path, scene: Path,
                trajectory: Path, mask: Path, cfg: EvalConfig) -> dict[str, Any]:
    for label, path in (("rollout", qpos), ("scene", scene),
                        ("traj", trajectory), ("mask", mask)):
        if not Path(path).is_file():
            raise FileNotFoundError(f"{case_id}:{variant}:{label}:{path}")
    row = {"case_id": case_id, "variant": variant, "object_key": object_key,
           "spider_method_id": ABL.METHOD, "hand_collision_variant_id": ABL.HAND_VARIANT}
    item = evaluate_sequence(row=row, method=ABL.METHOD,
                             hand_collision_variant_id=ABL.HAND_VARIANT,
                             qpos_path=qpos, scene_xml=scene, config=cfg,
                             kin_ref_path=trajectory, contact_mask_path=mask,
                             person_idx=ABL._person_idx(case_id))
    item.update(run_health(qpos, scene, cfg))
    item["body_z_err_p95_m"] = ABL.body_z_p95(qpos, scene, trajectory)
    return item


def score_manifest_row(row: dict[str, str], cfg: EvalConfig) -> dict[str, Any]:
    """Cells C and D: everything needed is already in the priority-manifest row."""
    return score_paths(
        row["case_id"], row["object_key"], row["aug_variant"],
        C210.repo_path(row["outdir_npz"]), C210.repo_path(row["scene_act"]),
        C210.repo_path(row["trajectory"]), C210.repo_path(row["contact_mask"]), cfg)


def offset_table() -> dict[tuple[str, str], float]:
    """Effective approach displacement, already measured by E202's builder."""
    out = {}
    for r in C210.read_tsv(E202_ARTIFACTS):
        try:
            out[(r["case_id"], r["aug_variant"])] = float(r["posediff_approach_trans_offset_m_max"])
        except (KeyError, ValueError):
            pass
    return out


def band(offset: float) -> str:
    return "full" if offset >= 0.18 else ("partial" if offset >= 0.10 else "weak")


def fmt(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return "" if not math.isfinite(value) else round(value, 4)
    return value


def summarize(name: str, xs: list[float]) -> dict[str, Any]:
    xs = [x for x in xs if math.isfinite(x)]
    if not xs:
        return {"metric": name, "n": 0}
    s = sorted(xs)
    q = lambda p: s[min(len(s) - 1, int(round(p * (len(s) - 1))))]  # noqa: E731
    return {"metric": name, "n": len(s), "mean": round(st.fmean(s), 4),
            "std": round(st.pstdev(s), 4) if len(s) > 1 else 0.0,
            "median": round(st.median(s), 4), "p75": round(q(0.75), 4),
            "p90": round(q(0.90), 4), "min": round(s[0], 4), "max": round(s[-1], 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    FC.assert_monotonic()

    cells = [c for c in args.cells.split(",") if c.strip()]
    if unknown := set(cells) - set(CELLS):
        raise SystemExit(f"unknown cells: {sorted(unknown)}")

    C210.audit(verbose=False)
    d_rows = C210.read_tsv(C210.FULL_MANIFEST)
    c_rows = C210.load_variants()
    if bad := [r["variant"] for r in d_rows if r["status"] != "run_complete_pending_eval"]:
        raise SystemExit(f"E210 manifest not complete: {bad}")

    aug_cases = list(C210.AUG_CASES)[: args.limit or None]
    keep = set(aug_cases)
    d_rows = [r for r in d_rows if r["case_id"] in keep]
    c_rows = [r for r in c_rows if r["case_id"] in keep]
    sources = {s["case_id"]: s for s in C207.sources() if s["case_id"] in keep}
    if set(sources) != keep:
        raise SystemExit(f"E207 source rows missing for {sorted(keep - set(sources))}")

    offsets = offset_table()
    cfg = EvalConfig()
    records: list[dict[str, Any]] = []

    # --- cells A and B: one row per case (orig) ------------------------------
    frozen_e178 = {r["case_id"]: r for r in C205.base.read_tsv(ABL.E178_CASE_METRICS)}
    for i, case_id in enumerate(aug_cases, 1):
        src = sources[case_id]
        for cell in ("A_orig_PRG", "B_orig_G1"):
            if cell not in cells:
                continue
            if cell == "A_orig_PRG":
                if case_id not in frozen_e178:
                    raise SystemExit(f"E178 baseline missing for {case_id}")
                metrics = ABL.read_prg_row(frozen_e178[case_id])
            else:
                _t, trajectory, mask = C205.base.local_authorities(src)
                metrics = score_paths(case_id, src["object_key"], "orig",
                                      C207.result_npz(case_id, "full"),
                                      C207.scene_path(case_id), trajectory, mask, cfg)
            rec = {"cell": cell, "object_key": src["object_key"], "case_id": case_id,
                   "aug_variant": "orig", "approach_offset_m": "", "offset_band": "orig"}
            for f in ABL.GATE_FIELDS:
                rec[f] = metrics.get(f)
            for f in (EEF_ORI, HAND_PEN, CONTACT):
                rec[f] = metrics.get(f)
            rec.update(ABL.classify(metrics))
            records.append(rec)
        print(f"[orig {i}/{len(aug_cases)}] {case_id}", flush=True)

    # --- cells C and D: one row per aug variant ------------------------------
    for cell, rows in (("C_aug_PRG", c_rows), ("D_aug_G1", d_rows)):
        if cell not in cells:
            continue
        for i, row in enumerate(rows, 1):
            metrics = score_manifest_row(row, cfg)
            off = offsets.get((row["case_id"], row["aug_variant"]), float("nan"))
            rec = {"cell": cell, "object_key": row["object_key"], "case_id": row["case_id"],
                   "aug_variant": row["aug_variant"], "approach_offset_m": round(off, 4),
                   "offset_band": band(off)}
            for f in ABL.GATE_FIELDS:
                rec[f] = metrics.get(f)
            for f in (EEF_ORI, HAND_PEN, CONTACT):
                rec[f] = metrics.get(f)
            rec.update(ABL.classify(metrics))
            records.append(rec)
            print(f"[{cell} {i}/{len(rows)}] {row['case_id']}/{row['aug_variant']}", flush=True)

    # --- write ---------------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "four_cell_rollout.tsv"
    cols = (["cell", "object_key", "case_id", "aug_variant", "approach_offset_m", "offset_band"]
            + ABL.GATE_FIELDS + [EEF_ORI, HAND_PEN, CONTACT]
            + ["layer", "hard_pass", "hard_failed", "wide_pass", "wide_failed",
               "narrow_pass", "narrow_failed"])
    seen: list[str] = []
    for c in cols:  # GATE_FIELDS already contains some of the C5 metrics
        if c not in seen:
            seen.append(c)
    cols = seen
    with out.open("w", newline="", encoding="utf-8") as stream:
        w = csv.DictWriter(stream, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in records:
            w.writerow({c: fmt(r.get(c, "")) for c in cols})
    print(f"\n[done] {rel_label(out)} ({len(records)} rows)")

    by_cell = {c: [r for r in records if r["cell"] == c] for c in cells}

    # --- evaluator-drift cross-check (E209's discipline) ---------------------
    drift: list[str] = []
    if "B_orig_G1" in cells and E207_FROZEN.is_file():
        frozen = {r["case_id"]: r for r in C210.read_tsv(E207_FROZEN) if r["arm"] == "G1only"}
        for r in by_cell["B_orig_G1"]:
            fr = frozen.get(r["case_id"])
            if not fr:
                drift.append(f"{r['case_id']}: absent from E207 frozen table")
                continue
            if str(fr["narrow_pass"]).lower() != str(r["narrow_pass"]).lower():
                drift.append(f"{r['case_id']}: narrow_pass {fr['narrow_pass']} -> {r['narrow_pass']}")
        print(f"\n=== evaluator drift check (cell B rescored vs E207 frozen) ===")
        print("  " + ("PASS: 5/5 reproduce E207's narrow verdict" if not drift
                      else "FAIL:\n    " + "\n    ".join(drift)))

    # --- funnel summary ------------------------------------------------------
    print("\n=== 14-gate funnel per cell ===")
    print(f"  {'cell':12s} {'n':>3s} {'hard':>8s} {'wide_all':>9s} {'narrow_all':>12s} {'L3_auto':>8s}")
    for c in cells:
        g = by_cell[c]
        n = len(g)
        if not n:
            continue
        npass = sum(r["narrow_pass"] for r in g)
        print(f"  {c:12s} {n:3d} {sum(r['hard_pass'] for r in g):7d} "
              f"{sum(r['wide_pass'] for r in g):9d} {npass:8d} ({npass/n:3.0%}) "
              f"{sum(r['layer'] == 'L3_auto' for r in g):8d}")

    print("\n=== per-gate NARROW pass count ===")
    gate_defs = [("fall", "fall_flag"), ("body_z", "body_z_err_p95_m"),
                 ("ankle_jerk", "ankle_jerk_p95"), ("obj_speed", "obj_speed_max")] + \
        [(g[0], g[1]) for g in FC.BANDED_GATES]
    hard = {h[1]: h for h in FC.HARD_GATES}
    banded = {g[1]: g for g in FC.BANDED_GATES}
    print("  gate         " + "".join(f"{c.split('_', 1)[1]:>12s}" for c in cells))
    for gname, gfield in gate_defs:
        cells_txt = ""
        for c in cells:
            g = by_cell[c]
            if gname == "fall":
                ok = sum(str(r["fall_flag"]).strip().lower() not in ("true", "1") for r in g)
            elif gfield in hard:
                _n, _f, op, thr = hard[gfield]
                ok = sum(FC.passes(op, ABL._finite(r[gfield]), thr) for r in g)
            else:
                bg = banded[gfield]
                ok = sum(FC.passes(bg[2], ABL._finite(r[gfield]), bg[3]) for r in g)
            cells_txt += f"{ok:>7d}/{len(g):<4d}"
        print(f"  {gname:12s}{cells_txt}")

    # --- paired deltas -------------------------------------------------------
    idx = {(r["cell"], r["case_id"], r["aug_variant"]): r for r in records}
    metrics_of_interest = [EEF_ORI, HAND_PEN, CONTACT, "track_obj_pos_err_cm_mean",
                           "track_obj_ori_err_deg_mean", "track_root_ori_err_deg_mean"]

    def paired(a_cell: str, b_cell: str, per_variant: bool) -> dict[str, Any]:
        """delta = b - a over matched units."""
        pairs = []
        for case_id in aug_cases:
            variants = C210.TRANS_VARIANTS if per_variant else ("orig",)
            for v in variants:
                a_key = (a_cell, case_id, v if a_cell.startswith(("C", "D")) else "orig")
                b_key = (b_cell, case_id, v if b_cell.startswith(("C", "D")) else "orig")
                ra, rb = idx.get(a_key), idx.get(b_key)
                if ra and rb:
                    pairs.append((ra, rb))
        if not pairs:
            return {}
        out: dict[str, Any] = {"n": len(pairs),
                               "narrow_gain": sum(rb["narrow_pass"] and not ra["narrow_pass"]
                                                  for ra, rb in pairs),
                               "narrow_loss": sum(ra["narrow_pass"] and not rb["narrow_pass"]
                                                  for ra, rb in pairs)}
        for m in metrics_of_interest:
            d = [ABL._finite(rb.get(m)) - ABL._finite(ra.get(m)) for ra, rb in pairs]
            d = [x for x in d if math.isfinite(x)]
            if d:
                out[m] = {"delta_mean": round(st.fmean(d), 4),
                          "n_worse": sum(x > 0 for x in d), "n_better": sum(x < 0 for x in d),
                          **{k: v for k, v in summarize(m, d).items()
                             if k in ("p75", "p90", "max", "min")}}
        return out

    contrasts = {
        "C->D (gravcomp on aug, STRICT single-variable)": paired("C_aug_PRG", "D_aug_G1", True),
        "A->B (gravcomp on orig, E207's own effect)": paired("A_orig_PRG", "B_orig_G1", False),
        "B->D (delivery, CONFOUNDED: aug + v1->v2 retarget)": paired("B_orig_G1", "D_aug_G1", True),
        "A->C (aug on PRG, CONFOUNDED: aug + v1->v2 retarget)": paired("A_orig_PRG", "C_aug_PRG", True),
    }
    print("\n=== paired contrasts ===")
    for label, res in contrasts.items():
        if not res:
            continue
        print(f"\n  {label}   n={res['n']}")
        print(f"    narrow: +{res['narrow_gain']} / -{res['narrow_loss']}")
        for m in metrics_of_interest:
            if m in res:
                r = res[m]
                print(f"    {m:52s} d={r['delta_mean']:+8.4f}  worse={r['n_worse']:2d} "
                      f"better={r['n_better']:2d}  p90={r['p90']:+.4f} max={r['max']:+.4f}")

    # --- C5a / C5b / C6 ------------------------------------------------------
    verdicts: dict[str, Any] = {}
    cd, ab = contrasts.get("C->D (gravcomp on aug, STRICT single-variable)", {}), \
        contrasts.get("A->B (gravcomp on orig, E207's own effect)", {})
    if cd and ab and EEF_ORI in cd and EEF_ORI in ab:
        d_aug, d_orig = cd[EEF_ORI]["delta_mean"], ab[EEF_ORI]["delta_mean"]
        verdicts["C5a_phantom_support"] = {
            "eef_ori_delta_aug(C->D)": d_aug, "eef_ori_delta_orig(A->B)": d_orig,
            "excess_on_aug": round(d_aug - d_orig, 4), "gate": "excess <= +1.0 deg",
            "pass": (d_aug - d_orig) <= 1.0,
            "contact_delta_aug(C->D)": cd.get(CONTACT, {}).get("delta_mean"),
            "note": "E209: contact_in_mask does not react to phantom support; eef_ori does.",
        }
    if "C_aug_PRG" in cells and "D_aug_G1" in cells:
        rate = lambda c, f: (sum(  # noqa: E731
            FC.passes(banded[f][2], ABL._finite(r[f]), banded[f][3]) for r in by_cell[c])
            / max(1, len(by_cell[c])))
        hp_c, hp_d = rate("C_aug_PRG", HAND_PEN), rate("D_aug_G1", HAND_PEN)
        hp_b = rate("B_orig_G1", HAND_PEN) if "B_orig_G1" in cells else float("nan")
        verdicts["C5b_hand_penetration"] = {
            "narrow_rate_C_aug_PRG": round(hp_c, 3), "narrow_rate_D_aug_G1": round(hp_d, 3),
            "narrow_rate_B_orig_G1": round(hp_b, 3),
            "gate": "rate(D) >= rate(B) - 0.20",
            "pass": bool(hp_d >= hp_b - 0.20) if math.isfinite(hp_b) else None,
            "delta_dist_C->D": cd.get(HAND_PEN),
            "delta_dist_B->D": contrasts.get(
                "B->D (delivery, CONFOUNDED: aug + v1->v2 retarget)", {}).get(HAND_PEN),
        }
    if "D_aug_G1" in cells:
        bands = Counter(r["offset_band"] for r in by_cell["D_aug_G1"])
        offs = [r["approach_offset_m"] for r in by_cell["D_aug_G1"]
                if isinstance(r["approach_offset_m"], float)]
        verdicts["C6_effective_offset"] = {
            "bands": dict(bands), "below_floor_0.05m": sum(o < 0.05 for o in offs),
            "min": round(min(offs), 4) if offs else None,
            "median": round(st.median(offs), 4) if offs else None,
            "by_band_narrow_rate": {
                b: round(st.fmean([float(r["narrow_pass"]) for r in by_cell["D_aug_G1"]
                                   if r["offset_band"] == b]), 3) for b in bands},
        }
    verdicts["C3_delivery"] = {
        "narrow_rate_B_orig_G1": round(st.fmean([float(r["narrow_pass"])
                                                 for r in by_cell.get("B_orig_G1", [])] or [0]), 3),
        "narrow_rate_D_aug_G1": round(st.fmean([float(r["narrow_pass"])
                                                for r in by_cell.get("D_aug_G1", [])] or [0]), 3),
        "gate": "rate(D) >= rate(B) - 0.15",
        "fall_count_D": sum(str(r["fall_flag"]).strip().lower() in ("true", "1")
                            for r in by_cell.get("D_aug_G1", [])),
    }
    verdicts["C3_delivery"]["pass"] = bool(
        verdicts["C3_delivery"]["narrow_rate_D_aug_G1"]
        >= verdicts["C3_delivery"]["narrow_rate_B_orig_G1"] - 0.15
        and verdicts["C3_delivery"]["fall_count_D"] == 0)

    print("\n=== pre-registered verdicts ===")
    print(json.dumps(verdicts, ensure_ascii=False, indent=2))

    C210.write_json(args.out_dir / "e210_four_cell_summary.json", {
        "created_at": C210.now(), "cells": cells, "rows": len(records),
        "cases": aug_cases, "excluded_cases": C210.EXCLUDED_CASES,
        "evaluator_drift_vs_E207": drift or "none",
        "contrasts": contrasts, "verdicts": verdicts,
        "confound_note": ("cells C/D use omnirt_v2 (E202 had to: v1 is often IK-infeasible "
                          "once the object moves); cells A/B use omnirt_v1. C->D and A->B are "
                          "clean; A->C and B->D move retarget variant as well as augmentation."),
    })
    print(f"\n[done] {rel_label(args.out_dir / 'e210_four_cell_summary.json')}")
    return 1 if drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
