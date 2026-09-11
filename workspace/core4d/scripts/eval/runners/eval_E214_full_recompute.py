#!/usr/bin/env python3
"""Recompute the FULL-method metric set for the 43 non-box023 paper cases from
their already-selected SPIDER-CEM rollout NPZ (NO CEM re-run).

Why: the paper cache (report/0908/paper_results/_cache_method_metrics.jsonl) only
stored the original 15 metrics.  The E214 report added new fields
(contact@5/10mm, pen@5/10mm, max phys pen (mm), foot-skate mean/max) that are
absent from the cache, so the full column for those rows previously covered only
the 7 recomputed box023 cases (n=7/50) -- not comparable to the 50-case ablation
columns.  This runner fills the gap: it resolves each non-box023 case's selected
rollout exactly like report/0908/code/gen_paper_results.py (GROUPS / CASE_OVERRIDES
/ SourceIndex), then runs the shared eval_E214_ablation.eval_one on it, producing
every metric (old + new) via the identical evaluate_sequence + run_health path.

The 7 box023 full-stack (E173) cases already have all metrics in
e214_metrics.jsonl (series full__box023) and are NOT recomputed here.

Output (results/E214/eval/):
  e214_full_recompute.jsonl   one record per non-box023 case (series="full")

Old-metric cross-check: recomputed contact_3mm/pen_3mm/etc. should match the
paper cache to floating tolerance (same selected rollout, same evaluator); the
runner prints any |delta|>1e-3 so a mask/scene mismatch would be caught.

Usage:
    .venv/bin/python workspace/core4d/scripts/eval/runners/eval_E214_full_recompute.py
    ... --workers 8 --fresh
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
for _p in ("workspace/core4d/report/0908/code",
           "workspace/core4d/scripts",
           "workspace/core4d/scripts/eval/reports",
           "workspace/core4d/scripts/experiments/E214",
           "workspace/core4d/scripts/eval/runners"):
    sys.path.insert(0, str(REPO / _p))

import gen_paper_results as G  # noqa: E402
G.REPO = REPO  # gen_paper_results computes REPO one level short (parents[4]); fix it.

import e214_common as C  # noqa: E402
from eval_E214_ablation import KEYS, eval_one  # noqa: E402

CACHE = C.EVAL_DIR / "e214_full_recompute.jsonl"
OUT_TSV = C.EVAL_DIR / "e214_full_recompute.tsv"
PAPER_CACHE = REPO / "workspace/core4d/report/0908/paper_results/_cache_method_metrics.jsonl"
# Old metrics also present in the paper cache -> cross-check recompute against them.
XCHECK_KEYS = ["raw_contact", "contact_3mm", "pen_3mm", "geom_2mm",
               "track_root_pos", "track_eef_pos", "track_obj_pos", "fall_flag"]


def _finite(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.nan
    return f if math.isfinite(f) else math.nan


def build_jobs() -> list[dict[str, str]]:
    """One job per non-box023 case: resolve selected rollout via gen_paper_results."""
    idx = G.SourceIndex()
    jobs: list[dict[str, str]] = []
    for case in C.load_cases():
        if C.object_key_of(case) == "box023":
            continue  # full__box023 already computed in e214_metrics.jsonl
        cfg = G.case_cfg(case)
        rl = G.find_rl_row(idx, case, cfg) or {}
        cem = G.resolve(rl.get(G.RL_CEM_NPZ, ""))
        scn = G.resolve(rl.get(G.RL_SCENE_ACT, ""))
        kin = G.resolve(rl.get(G.RL_TRAJECTORY, ""))
        mask = G.resolve(rl.get(G.RL_CONTACT_MASK, ""))
        if cem is None or scn is None:
            jobs.append({"case_id": case, "status": "UNRESOLVED",
                         "cem": "", "scn": "", "kin": "", "mask": ""})
            continue
        jobs.append({"case_id": case, "status": "ok",
                     "cem": str(cem), "scn": str(scn),
                     "kin": str(kin) if kin else "", "mask": str(mask) if mask else ""})
    return jobs


def _worker(job: dict[str, str]) -> dict[str, Any]:
    case = job["case_id"]
    if job["status"] != "ok":
        return {"case_id": case, "series": "full", "object_key": C.object_key_of(case),
                "status": "UNRESOLVED", "metrics": {}}
    try:
        metrics = eval_one(case, Path(job["cem"]), Path(job["scn"]),
                           Path(job["kin"]) if job["kin"] else None,
                           Path(job["mask"]) if job["mask"] else None, "SPIDER-CEM")
        status = "ok"
    except Exception as exc:  # noqa: BLE001
        metrics, status = {}, f"ERROR:{exc}"
    return {"case_id": case, "series": "full", "object_key": C.object_key_of(case),
            "status": status, "metrics": metrics}


def load_cache() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if CACHE.is_file():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["case_id"]] = r
    return out


def paper_cache() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for line in PAPER_CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("method") == "SPIDER-CEM":
                out[r["case_id"]] = {k: _finite(v) for k, v in r.get("metrics", {}).items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    C.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh and CACHE.is_file():
        CACHE.unlink()

    cache = load_cache()
    jobs = [j for j in build_jobs() if j["case_id"] not in cache]
    print(f"{len(cache)} cached, {len(jobs)} to compute (workers={args.workers})", flush=True)

    if jobs:
        with CACHE.open("a", encoding="utf-8") as fh, \
                mp.Pool(processes=max(1, args.workers)) as pool:
            for i, rec in enumerate(pool.imap_unordered(_worker, jobs), 1):
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"[{i:02d}/{len(jobs)}] {rec['case_id']:32s} {rec['status']}", flush=True)

    records = load_cache()

    # cross-check old metrics vs paper cache
    pc = paper_cache()
    print("\n=== cross-check recompute vs paper cache (|delta|>1e-3) ===")
    bad = 0
    for case, r in sorted(records.items()):
        if r["status"] != "ok" or case not in pc:
            continue
        for k in XCHECK_KEYS:
            a, b = _finite(r["metrics"].get(k)), pc[case].get(k, math.nan)
            if math.isfinite(a) and math.isfinite(b) and abs(a - b) > 1e-3:
                print(f"  {case:32s} {k:16s} recompute={a:.4f} paper={b:.4f} d={a-b:+.4f}")
                bad += 1
    print(f"cross-check mismatches: {bad}")

    # flat tsv
    header = ["case_id", "object_key", "status"] + KEYS
    lines = ["\t".join(header)]
    for case, r in sorted(records.items()):
        vals = [case, r["object_key"], r["status"]]
        vals += [("" if (k not in r["metrics"] or not math.isfinite(_finite(r["metrics"][k])))
                  else f"{_finite(r['metrics'][k]):.6g}") for k in KEYS]
        lines.append("\t".join(vals))
    OUT_TSV.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok = sum(1 for r in records.values() if r["status"] == "ok")
    print(f"\nwrote {C.rel(CACHE)} + {C.rel(OUT_TSV)} ({ok}/{len(records)} ok)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
