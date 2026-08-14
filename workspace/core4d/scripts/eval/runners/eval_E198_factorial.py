#!/usr/bin/env python3
"""E198 2x2 factorial evaluation: G1 (gravcomp) x A2 (hand-gate) over box004/021/023/024.

Scores all four arms per case with the single public evaluator (core_metrics via
base.score) and estimates the interaction term per object per metric:

    INT = M(G1+A2) - M(A2) - M(G1) + M(A0)

Arm rollout sources (all re-scored by current core_metrics -> single-evaluator C3):
  A0    box024->E173, box004->E172 manifests; box021/023 -> E194 three-arm PRG rows
  G1    box024/004->E194 cem_full (arm G1);  box021/023 -> E194 three-arm G1 rows
  A2    box024/004->E192 cem_full (arm A2);  box021/023 -> E198 manifest (E192-ext)
  G1A2  all -> E198 manifest (arm G1A2)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))                                   # runners/
sys.path.insert(0, str(HERE.parents[2] / "experiments/E198"))         # e198_common
sys.path.insert(0, str(HERE.parents[2]))                              # eval.core

import eval_E194_G1_expansion as base  # noqa: E402
import e198_common as C  # noqa: E402
from eval.core.core_metrics import EVAL_METRIC_STANDARD_ID, EvalConfig  # noqa: E402

REPO = C.REPO
OUT = C.RESULTS_E198 / "s6_downstream/eval/full_factorial"
CACHE = OUT / "e198_arm_cache.tsv"
ARMS = ("A0", "G1", "A2", "G1A2")
# plan227: box001 added as the 5th object (G1 baseline = E196 corrected 21 + E194 clean 7).
OBJECT_ORDER = ("box024", "box021", "box023", "box004", "box001")
OBJ_RANK = {k: i for i, k in enumerate(OBJECT_ORDER)}
N_EXPECTED = 59 + C.OBJECT_COUNTS["box001"]  # 87

E173 = REPO / "workspace/core4d/results/E173/s6_downstream/manifests/cem_full_manifest.tsv"
E172 = REPO / "workspace/core4d/results/E172/s6_downstream/manifests/cem_full_manifest.tsv"
E194_ORIG = REPO / "workspace/core4d/results/E194/s6_downstream/manifests/cem_full_manifest.tsv"
E192 = REPO / "workspace/core4d/results/E192/s6_downstream/manifests/cem_full_manifest.tsv"
THREE_ARM = REPO / "workspace/core4d/results/E194/s6_downstream/eval/full_g1_expansion/e194_three_arm_case_metrics.tsv"
E198_MAN = C.FULL_MANIFEST
# box001 baseline sources
E196_CORR = REPO / "workspace/core4d/report/E196/provenance/eval/e196_reference_fix_case_metrics.tsv"
E194_G1EXP = REPO / "workspace/core4d/results/E194/s6_downstream/manifests/g1_expansion_full_manifest.tsv"
E198_BOX001_MAN = C.manifest_paths("box001")["full"]

BOX2404 = set(C.BOX2404_CASES["box024"]) | set(C.BOX2404_CASES["box004"])


def norm(raw: dict[str, str], *, scene_field: str = "scene_act") -> dict[str, str]:
    """Carry the full source row, normalizing scene_act + defaults base.score needs."""
    row = dict(raw)
    row["scene_act"] = raw.get(scene_field) or raw.get("scene_act") or raw.get("scene_xml")
    # trajectory is the kinematic reference (arm-invariant); repair stale paths that
    # drop the "/0/" data-id subdir (seen in one E192 manifest row).
    traj = row.get("trajectory", "")
    if traj and not C.repo_path(traj).is_file():
        p = Path(traj)
        alt = p.with_name("0") / p.name if p.parent.name != "0" else p
        if C.repo_path(alt).is_file():
            row["trajectory"] = alt.as_posix()
    row.setdefault("variant", raw.get("variant", raw["case_id"]))
    row.setdefault("object_category", raw.get("object_category", "box"))
    row.setdefault("expected_quality", raw.get("expected_quality", ""))
    return row


def arm_rows() -> dict[str, dict[str, dict[str, str]]]:
    """Return {arm: {case_id: normalized row}}."""
    three = [r for r in C.read_tsv(THREE_ARM) if r["object_key"] in ("box021", "box023")]
    three_prg = {r["case_id"]: r for r in three if r.get("arm") == "PRG"}
    three_g1 = {r["case_id"]: r for r in three if r.get("arm") == "G1"}
    e173 = {r["case_id"]: r for r in C.read_tsv(E173)}
    e172 = {r["case_id"]: r for r in C.read_tsv(E172)}
    e194 = {r["case_id"]: r for r in C.read_tsv(E194_ORIG) if r.get("arm") == "G1"}
    e192 = {r["case_id"]: r for r in C.read_tsv(E192) if r.get("arm") == "A2"}
    e198 = C.read_tsv(E198_MAN)
    e198_a2 = {r["case_id"]: r for r in e198 if r["arm"] == "A2"}
    e198_g1a2 = {r["case_id"]: r for r in e198 if r["arm"] == "G1A2"}

    out: dict[str, dict[str, dict[str, str]]] = {a: {} for a in ARMS}
    for cid in C.BOX2404_CASES["box024"]:
        out["A0"][cid] = norm(e173[cid]); out["G1"][cid] = norm(e194[cid])
        out["A2"][cid] = norm(e192[cid]); out["G1A2"][cid] = norm(e198_g1a2[cid])
    for cid in C.BOX2404_CASES["box004"]:
        out["A0"][cid] = norm(e172[cid]); out["G1"][cid] = norm(e194[cid])
        out["A2"][cid] = norm(e192[cid]); out["G1A2"][cid] = norm(e198_g1a2[cid])
    for cid, r in three_prg.items():
        out["A0"][cid] = norm(r, scene_field="scene_xml")
    for cid, r in three_g1.items():
        out["G1"][cid] = norm(r, scene_field="scene_xml")
    for cid, r in e198_a2.items():
        out["A2"][cid] = norm(r)
    for cid, r in e198_g1a2.items():
        if r["object_key"] in ("box021", "box023"):
            out["G1A2"][cid] = norm(r)

    # --- box001 (plan227): A0=E173 PRG, G1=E196 corrected(21)+E194 clean(7),
    #     A2/G1A2 = E198 box001 supplement manifest -------------------------------
    corr = {r["case_id"]: r for r in C.read_tsv(E196_CORR)
            if r.get("object_key") == "box001" and r.get("arm") == "G1_corrected"}
    g1exp = {r["case_id"]: r for r in C.read_tsv(E194_G1EXP)
             if r.get("object_key") == "box001" and r.get("arm") == "G1"}
    b1 = C.read_tsv(E198_BOX001_MAN)
    b1_a2 = {r["case_id"]: r for r in b1 if r["arm"] == "A2"}
    b1_g1a2 = {r["case_id"]: r for r in b1 if r["arm"] == "G1A2"}
    for cid in b1_g1a2:
        out["A0"][cid] = norm(e173[cid])
        # corrected G1 for the 21 Euler-mismatch cases; clean E194 G1 for the other 7
        out["G1"][cid] = norm(corr[cid], scene_field="scene_xml") if cid in corr else norm(g1exp[cid])
        out["A2"][cid] = norm(b1_a2[cid])
        out["G1A2"][cid] = norm(b1_g1a2[cid])
    return out


def load_cache() -> dict[tuple[str, str], dict[str, Any]]:
    if not CACHE.is_file():
        return {}
    return {(r["arm"], r["case_id"]): r for r in C.read_tsv(CACHE)
            if r.get("metric_standard_id") == EVAL_METRIC_STANDARD_ID}


def score_all(rows: dict[str, dict[str, dict[str, str]]]) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, str]]]:
    cache = load_cache()
    scored: dict[tuple[str, str], dict[str, Any]] = dict(cache)
    errors: list[dict[str, str]] = []
    cfg = EvalConfig()
    total = sum(len(v) for v in rows.values())
    done = 0
    for arm in ARMS:
        for cid in sorted(rows[arm], key=lambda c: (OBJ_RANK[rows[arm][c]["object_key"]], c)):
            done += 1
            if (arm, cid) in scored:
                continue
            try:
                item = base.score(rows[arm][cid], arm, cfg)
                item["arm"] = arm
                z = base.finite(item.get("track_obj_z_abs_err_cm_mean"))
                pos = base.finite(item.get("track_obj_pos_err_cm_mean"))
                if not math.isfinite(z) or not math.isfinite(pos) or z > pos + 1e-9:
                    raise ValueError(f"metric_contract z={z} pos={pos}")
                scored[(arm, cid)] = item
                ordered = [scored[(a, c)] for a in ARMS for c in sorted(rows[a]) if (a, c) in scored]
                C.write_tsv(CACHE, ordered)
                print(f"[{done:03d}/{total}] scored {arm} {cid} z={z:.3f}")
            except Exception as exc:  # noqa: BLE001
                errors.append({"arm": arm, "case_id": cid, "error": f"{type(exc).__name__}: {exc}"})
                print(f"[error] {arm} {cid}: {errors[-1]['error']}", file=sys.stderr)
    return scored, errors


def bootstrap_ci(values: list[float], seed: int = 0, n_boot: int = 10000) -> tuple[float, float, float]:
    data = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if data.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, size=(n_boot, data.size))].mean(axis=1)
    return float(data.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar_exact(p2f: int, f2p: int) -> float:
    n = p2f + f2p
    if n == 0:
        return 1.0
    from math import comb
    k = min(p2f, f2p)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * tail))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = arm_rows()
    for arm in ARMS:
        if len(rows[arm]) != N_EXPECTED:
            raise SystemExit(f"arm {arm} has {len(rows[arm])} cases, expected {N_EXPECTED}")
    scored, errors = score_all(rows)
    C.write_tsv(OUT / "e198_arm_eval_errors.tsv", errors or [{"arm": "", "case_id": "", "error": ""}])
    if errors:
        print(f"[abort] {len(errors)} scoring errors; see e198_arm_eval_errors.tsv", file=sys.stderr)
        return 1

    cases = sorted({c for _a, c in scored}, key=lambda c: (OBJ_RANK[scored[("A0", c)]["object_key"]], c))
    obj_of = {c: scored[("A0", c)]["object_key"] for c in cases}

    # per-case factorial rows
    by_case: list[dict[str, Any]] = []
    for c in cases:
        row: dict[str, Any] = {"case_id": c, "object_key": obj_of[c],
                               "retarget_variant_id": scored[("A0", c)].get("retarget_variant_id", "")}
        for m in base.KEY_METRICS:
            v = {a: base.finite(scored[(a, c)].get(m)) for a in ARMS}
            row[f"{m}__A0"] = v["A0"]; row[f"{m}__G1"] = v["G1"]
            row[f"{m}__A2"] = v["A2"]; row[f"{m}__G1A2"] = v["G1A2"]
            row[f"{m}__mainG_atA0"] = v["G1"] - v["A0"]
            row[f"{m}__mainA_atG0"] = v["A2"] - v["A0"]
            row[f"{m}__mainG_atA1"] = v["G1A2"] - v["A2"]
            row[f"{m}__mainA_atG1"] = v["G1A2"] - v["G1"]
            row[f"{m}__INT"] = v["G1A2"] - v["A2"] - v["G1"] + v["A0"]
        by_case.append(row)
    C.write_tsv(OUT / "e198_factorial_by_case.tsv", by_case)

    # per-object interaction + cell means + bootstrap CI
    by_object: list[dict[str, Any]] = []
    for obj in OBJECT_ORDER:
        ocases = [c for c in cases if obj_of[c] == obj]
        rec: dict[str, Any] = {"object_key": obj, "n": len(ocases)}
        for m in base.KEY_METRICS:
            for a in ARMS:
                rec[f"{m}__{a}_mean"] = float(np.nanmean([base.finite(scored[(a, c)].get(m)) for c in ocases]))
            ints = [by_case_of(by_case, c)[f"{m}__INT"] for c in ocases]
            mean, lo, hi = bootstrap_ci(ints)
            rec[f"{m}__INT_mean"] = mean; rec[f"{m}__INT_ci_lo"] = lo; rec[f"{m}__INT_ci_hi"] = hi
        by_object.append(rec)
    C.write_tsv(OUT / "e198_factorial_by_object.tsv", by_object)

    # gate migrations for the four arm transitions
    transitions = [("A0", "A2"), ("A0", "G1"), ("G1", "G1A2"), ("A2", "G1A2")]
    migr: list[dict[str, Any]] = []
    for before, after in transitions:
        for obj in ("ALL", *OBJECT_ORDER):
            sel = cases if obj == "ALL" else [c for c in cases if obj_of[c] == obj]
            for gate in base.ALL_GATES:
                key = f"{gate}_gate_pass"
                bpass = sum(C.truth(scored[(before, c)].get(key)) for c in sel)
                apass = sum(C.truth(scored[(after, c)].get(key)) for c in sel)
                p2f = sum(C.truth(scored[(before, c)].get(key)) and not C.truth(scored[(after, c)].get(key)) for c in sel)
                f2p = sum((not C.truth(scored[(before, c)].get(key))) and C.truth(scored[(after, c)].get(key)) for c in sel)
                migr.append({"transition": f"{before}->{after}", "object_key": obj, "gate": gate,
                             "n": len(sel), "before_pass": bpass, "after_pass": apass,
                             "delta_pp": round(100.0 * (apass - bpass) / len(sel), 2) if sel else 0.0,
                             "p2f": p2f, "f2p": f2p, "mcnemar_p": round(mcnemar_exact(p2f, f2p), 6)})
    C.write_tsv(OUT / "e198_gate_migrations.tsv", migr)

    summary = {
        "created_at": C.now(), "metric_standard_id": EVAL_METRIC_STANDARD_ID,
        "arms": list(ARMS), "n_cases": len(cases), "objects": {o: sum(1 for c in cases if obj_of[c] == o) for o in OBJECT_ORDER},
        "primary_interaction": {
            obj: {m: {"INT_mean": next(r for r in by_object if r["object_key"] == obj)[f"{m}__INT_mean"],
                      "ci": [next(r for r in by_object if r["object_key"] == obj)[f"{m}__INT_ci_lo"],
                             next(r for r in by_object if r["object_key"] == obj)[f"{m}__INT_ci_hi"]]}
                  for m in ("track_obj_z_abs_err_cm_mean", "track_obj_pos_err_cm_mean",
                            "hand_object_physics_penetration_3mm_frame_frac", "leg_penetration_frac")}
            for obj in OBJECT_ORDER},
    }
    C.write_json(OUT / "e198_factorial_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


def by_case_of(by_case: list[dict[str, Any]], case_id: str) -> dict[str, Any]:
    for r in by_case:
        if r["case_id"] == case_id:
            return r
    raise KeyError(case_id)


if __name__ == "__main__":
    raise SystemExit(main())
