#!/usr/bin/env python3
"""Validate the SMPLX-GT reference alignment (Claim C1 for E214b / R301).

For every one of the 50 paper cases: resolve the kinematic reference, build the
scene-frame SMPLX reference, and report the object-fit residual (the fit target)
plus the mapped-pelvis-vs-robot-pelvis distance (an independent cross-check) and
the recovered human->scene scale. Writes a TSV + prints a summary. Pure analysis
(no physics sim); populates the per-case reference cache under results/E214/eval/.

Usage:
    .venv/bin/python workspace/core4d/scripts/eval/runners/validate_smplx_reference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
for _p in ("workspace/core4d/report/0908/code", "workspace/core4d/scripts",
           "workspace/core4d/scripts/eval/reports",
           "workspace/core4d/scripts/experiments/E214",
           "workspace/core4d/scripts/eval/runners"):
    sys.path.insert(0, str(REPO / _p))

import gen_paper_results as G  # noqa: E402
G.REPO = REPO
import e214_common as C  # noqa: E402
from eval_E214_ablation import jobs_from_manifest  # noqa: E402
from smplx_reference import build_smplx_reference  # noqa: E402

CACHE = C.EVAL_DIR / "smplx_ref"
OUT_TSV = C.EVAL_DIR / "smplx_reference_alignment.tsv"


def kin_ref_by_case() -> dict[str, Path]:
    """One kinematic-reference path per case (shared across all series)."""
    out: dict[str, Path] = {}
    for j in jobs_from_manifest():
        if j["case_id"] not in out and j.get("kin_ref"):
            out[j["case_id"]] = Path(j["kin_ref"])
    idx = G.SourceIndex()
    for case in C.load_cases():
        if case in out:
            continue
        rl = G.find_rl_row(idx, case, G.case_cfg(case)) or {}
        k = G.resolve(rl.get(G.RL_TRAJECTORY, ""))
        if k:
            out[case] = k
    return out


def main() -> int:
    C.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    kin = kin_ref_by_case()
    rows = []
    for case in C.load_cases():
        q = np.asarray(np.load(kin[case], allow_pickle=True)["qpos"], dtype=np.float64)
        r = build_smplx_reference(case, q, cache_dir=CACHE)
        rows.append((case, C.object_key_of(case), r.align_residual_m * 1000,
                     r.align_residual_max_m * 1000, r.pelvis_check_m * 1000,
                     r.scale, r.time_map[0], r.time_map[1], r.status))

    lines = ["case\tobject\tobj_res_mm\tobj_res_max_mm\tpelvis_mm\tscale\ttmap_a\ttmap_b\tstatus"]
    for c, o, rm, rx, pc, sc, a, b, st in rows:
        lines.append(f"{c}\t{o}\t{rm:.3f}\t{rx:.3f}\t{pc:.2f}\t{sc:.4f}\t{a}\t{b}\t{st}")
    OUT_TSV.write_text("\n".join(lines) + "\n", encoding="utf-8")

    res = np.array([r[2] for r in rows])
    pcs = np.array([r[4] for r in rows])
    noalign = [r[0] for r in rows if not r[8].startswith("ok")]
    print(f"wrote {C.rel(OUT_TSV)} ({len(rows)} cases)")
    print(f"obj residual mm : mean {res.mean():.2f}  median {np.median(res):.2f}  "
          f"p95 {np.percentile(res, 95):.2f}  max {res.max():.2f}")
    print(f"pelvis check mm : mean {pcs.mean():.1f}  median {np.median(pcs):.1f}  max {pcs.max():.1f}")
    print(f"scale range     : [{min(r[5] for r in rows):.3f}, {max(r[5] for r in rows):.3f}]")
    print(f"NO_GT_ALIGN     : {len(noalign)}  {noalign}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
