#!/usr/bin/env python3
"""Compute OmniRetarget physics-rollout health metrics (ankle jerk / obj speed / foot slip).

`gen_paper_results.py` already replays each OmniRetarget kinematic trajectory under MuJoCo
position-servo control (E197/E109 protocol) and caches the resulting scene_act qpos at
`paper_results/omni_scene_act_qpos/<case>_omnirt_scene_act_qpos.npz`. It only *read out* the
4 contact metrics (METRICS with omni=True). Here we run the SAME `eval.core.motion_health.
run_health` on those cached replays to obtain the three dynamic-quality metrics, so they are
computed identically to SPIDER-CEM / SBTO (contrast GMR, whose values come from a kinematic
pipeline at a different scale).

Output: paper_cmp/omni_health.json = {case_id: {ankle_jerk, obj_speed, foot_slip}}.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = next(p for p in HERE.parents if (p / "workspace/core4d/scripts").is_dir())
sys.path.insert(0, str(REPO / "workspace/core4d/scripts"))

from eval.core.core_metrics import EvalConfig  # noqa: E402
from eval.core.motion_health import run_health  # noqa: E402

PAPER = REPO / "workspace/core4d/report/0908/paper_results"
CACHE = PAPER / "_cache_method_metrics.jsonl"
QPOS_DIR = PAPER / "omni_scene_act_qpos"
OUT = REPO / "workspace/core4d/report/0908/paper_cmp/omni_health.json"


def omni_records():
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("method") == "OmniRetarget" and rec.get("status") == "ok":
            yield rec


def main() -> int:
    cfg = EvalConfig()
    out: dict[str, dict[str, float]] = {}
    missing_qpos, missing_scene = [], []
    for rec in omni_records():
        case_id = rec["case_id"]
        qpos = QPOS_DIR / f"{case_id}_omnirt_scene_act_qpos.npz"
        scene = REPO / rec["scene_act"]
        if not qpos.is_file():
            missing_qpos.append(case_id)
            continue
        if not scene.is_file():
            missing_scene.append(case_id)
            continue
        h = run_health(qpos, scene, cfg)
        out[case_id] = {
            "ankle_jerk": float(h.get("ankle_jerk_p95", math.nan)),
            "obj_speed": float(h.get("obj_speed_max", math.nan)),
            "foot_slip": float(h.get("foot_slip_max_m", math.nan)),
        }
        print(f"  {case_id:32s} jerk={out[case_id]['ankle_jerk']:8.1f} "
              f"objspd={out[case_id]['obj_speed']:.3f} slip={out[case_id]['foot_slip']:.3f}",
              flush=True)

    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    n = len(out)
    print(f"\n[ok] wrote {OUT}  ({n} cases)")
    if missing_qpos:
        print(f"[warn] missing replay qpos ({len(missing_qpos)}): {missing_qpos}")
    if missing_scene:
        print(f"[warn] missing scene ({len(missing_scene)}): {missing_scene}")
    # overall means for a quick sanity check
    for k in ("ankle_jerk", "obj_speed", "foot_slip"):
        vals = [v[k] for v in out.values() if math.isfinite(v[k])]
        if vals:
            print(f"  OmniRetarget overall {k}: mean={sum(vals)/len(vals):.3f} (n={len(vals)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
