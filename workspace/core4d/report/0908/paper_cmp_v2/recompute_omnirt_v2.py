#!/usr/bin/env python3
"""Recompute OmniRetarget's 5 comparison metrics over the E214 50-case set.

The paper cache (_cache_method_metrics.jsonl) only stored OmniRetarget's 3mm
contact/penetration + geom, not the five metrics this comparison needs
(contact@5mm, pen@5mm, max-pen, obj speed, foot slip). We reuse
gen_paper_results' OmniRetarget replay/eval path verbatim -- same scene_act,
window, mask, and the shared eval.core.core_metrics.evaluate_sequence +
motion_health.run_health -- and only extend the requested metric set at runtime
(no edit to the core report file). OmniRetarget is a kinematic retargeter, so
this is a kinematic evaluation of its converted trajectory, identical protocol to
its published contact/penetration numbers.

Output: paper_cmp_v2/omnirt_metrics.jsonl  (one record per case; the 5 keys).

Usage:
    .venv/bin/python workspace/core4d/report/0908/paper_cmp_v2/recompute_omnirt_v2.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
for _p in ("workspace/core4d/report/0908/code", "workspace/core4d/scripts",
           "workspace/core4d/scripts/eval/reports",
           "workspace/core4d/scripts/experiments/E214"):
    sys.path.insert(0, str(REPO / _p))

import e214_common as C  # noqa: E402
import gen_paper_results as G  # noqa: E402
G.REPO = REPO  # gen_paper_results computes REPO one level short; fix to repo root.

OUT = Path(__file__).resolve().parent / "omnirt_metrics.jsonl"

# report key -> (field in core_metrics/run_health, src). The 3 physics keys are
# absent from gen_paper_results.METRICS, so inject their specs; obj_speed/foot_slip
# already exist there (as health, omni=False) -- we just add them to OMNI_KEYS.
EXTRA_SPECS = [
    {"key": "contact_5mm", "field": "hand_object_physics_contact_5mm_in_mask_frac",
     "src": "result", "kind": "pct", "dir": "higher", "omni": True, "label": "contact@5mm"},
    {"key": "pen_5mm", "field": "hand_object_physics_penetration_5mm_frame_frac",
     "src": "result", "kind": "pct", "dir": "lower", "omni": True, "label": "penetration@5mm"},
    {"key": "pen_max_mm", "field": "hand_object_physics_penetration_max_mm",
     "src": "result", "kind": "mm", "dir": "lower", "omni": True, "label": "max penetration mm"},
]
WANT = ["contact_5mm", "pen_5mm", "pen_max_mm", "obj_speed", "foot_slip"]


def main() -> int:
    for spec in EXTRA_SPECS:
        G.METRIC_BY_KEY[spec["key"]] = spec
    G.OMNI_KEYS = list(WANT)  # compute_omnirt evaluates exactly these keys

    cases = C.load_cases()
    idx = G.SourceIndex()
    records = []
    ok = 0
    with OUT.open("w", encoding="utf-8") as fh:
        for i, case in enumerate(cases, 1):
            rec = G.compute_omnirt(idx, case, {})
            metrics = {k: (float(v) if v is not None and _finite(v) else None)
                       for k, v in rec.get("metrics", {}).items()}
            out = {"case_id": case, "object_key": C.object_key_of(case),
                   "status": rec.get("status", "?"), "metrics": metrics,
                   "notes": rec.get("notes", [])}
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            fh.flush()
            if out["status"] == "ok":
                ok += 1
            print(f"[{i:02d}/{len(cases)}] {case:32s} {out['status']:14s} "
                  f"c5={_f(metrics.get('contact_5mm'))} p5={_f(metrics.get('pen_5mm'))} "
                  f"pmax={_f(metrics.get('pen_max_mm'))}", flush=True)
            records.append(out)
    print(f"\nwrote {OUT} ({ok}/{len(records)} ok)")
    return 0


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _f(v) -> str:
    return f"{float(v):.4g}" if _finite(v) else "—"


if __name__ == "__main__":
    raise SystemExit(main())
