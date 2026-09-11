#!/usr/bin/env python3
"""Lightweight MPJPE-only pass for E214 (no CEM re-run, no geom/contact resampling).

MPJPE = mean per-joint Cartesian position error over ALL robot bodies (the whole
G1 kinematic subtree, not just root/eef), vs the fixed kinematic reference, in cm
(global / un-aligned, matching the eef/root pos-err convention).  It is computed
by the shared eval module (core_metrics._tracking_metrics ->
_table4_tracking_metrics, field track_mpjpe_cm_mean); this runner only does the
cheap forward-kinematics tracking pass, so it is seconds/case rather than the
~2min/case of the full evaluate_sequence.

Covers every series used by the ablation report:
  full            43 non-box023 selected rollouts (resolved via gen_paper_results)
  full__box023    7 box023 E173 full-stack rollouts (from the E214 manifest)
  A1..A4          200 ablation rollouts (from the E214 manifest)

Output (results/E214/eval/):
  e214_mpjpe.jsonl   one record per (case, series) with track_mpjpe_cm_mean

Usage:
    .venv/bin/python workspace/core4d/scripts/eval/runners/eval_E214_mpjpe.py --workers 8
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
from eval_E214_ablation import jobs_from_manifest  # noqa: E402

CACHE = C.EVAL_DIR / "e214_mpjpe.jsonl"


def build_jobs() -> list[dict[str, str]]:
    """(case, series) -> qpos/scene/kin paths for every rollout in the report."""
    jobs: list[dict[str, str]] = []
    # ablation rollouts + box023 full-stack, straight from the manifest resolver.
    for j in jobs_from_manifest():
        jobs.append({"case_id": j["case_id"], "series": j["series"],
                     "qpos": str(j["qpos"]), "scene": str(j["scene"]),
                     "kin": str(j["kin_ref"]) if j["kin_ref"] else ""})
    # 43 non-box023 full rollouts, resolved exactly like gen_paper_results.
    idx = G.SourceIndex()
    for case in C.load_cases():
        if C.object_key_of(case) == "box023":
            continue
        cfg = G.case_cfg(case)
        rl = G.find_rl_row(idx, case, cfg) or {}
        cem = G.resolve(rl.get(G.RL_CEM_NPZ, ""))
        scn = G.resolve(rl.get(G.RL_SCENE_ACT, ""))
        kin = G.resolve(rl.get(G.RL_TRAJECTORY, ""))
        if cem and scn:
            jobs.append({"case_id": case, "series": "full",
                         "qpos": str(cem), "scene": str(scn),
                         "kin": str(kin) if kin else ""})
    return jobs


def _worker(job: dict[str, str]) -> dict[str, Any]:
    # Heavy imports live inside the worker so each child loads them once.
    import mujoco  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415, F401
    from eval.core.core_metrics import EvalConfig, npz_qpos, _tracking_metrics  # noqa: PLC0415

    def _num(v: Any) -> float | None:
        return float(v) if (v is not None and math.isfinite(float(v))) else None

    case, series = job["case_id"], job["series"]
    rec = {"case_id": case, "series": series, "object_key": C.object_key_of(case)}
    try:
        if not job["kin"]:
            rec.update(status="NO_KIN", mpjpe=None, mpjpe_local=None)
            return rec
        model = mujoco.MjModel.from_xml_path(job["scene"])
        qpos, _ = npz_qpos(Path(job["qpos"]))
        t = _tracking_metrics(qpos, Path(job["kin"]), EvalConfig(), model=model)
        rec.update(status="ok",
                   mpjpe=_num(t.get("track_mpjpe_cm_mean")),           # MPJPE-G (global)
                   mpjpe_local=_num(t.get("track_mpjpe_local_cm_mean")))  # MPJPE-L (root-aligned)
    except Exception as exc:  # noqa: BLE001
        rec.update(status=f"ERROR:{exc}", mpjpe=None, mpjpe_local=None)
    return rec


def load_cache() -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    if CACHE.is_file():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[(r["case_id"], r["series"])] = r
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
    jobs = [j for j in build_jobs() if (j["case_id"], j["series"]) not in cache]
    print(f"{len(cache)} cached, {len(jobs)} to compute (workers={args.workers})", flush=True)

    if jobs:
        with CACHE.open("a", encoding="utf-8") as fh, \
                mp.Pool(processes=max(1, args.workers)) as pool:
            for i, rec in enumerate(pool.imap_unordered(_worker, jobs), 1):
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"[{i:03d}/{len(jobs)}] {rec['series']:22s} {rec['case_id']:30s} "
                      f"{rec['status']} G={rec['mpjpe']} L={rec.get('mpjpe_local')}", flush=True)

    records = load_cache()
    ok = sum(1 for r in records.values() if r.get("status") == "ok")
    print(f"\nwrote {C.rel(CACHE)} ({ok}/{len(records)} ok)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
