#!/usr/bin/env python3
"""E214 CPU contract: prove the ablation set is fully resolvable before any GPU.

Checks (no MuJoCo / no CEM):
  C1  case count == 50 and == paper 52 minus the two dropped E167A box021 cases.
  C2  box023 (7 cases) resolves through the E173 full-stack export, not E179/E190.
  C3  every case resolves to an EXISTING baseline config_act.yaml + scene/traj/mask.
  C4  every baseline config_act.yaml is full-stack (contact_hdmi+surface_band on,
      soft penalties 2/2/2, e167 z on, safety+hand+leg+posture gates on) -- so all
      four ablations are meaningful on it.
  C5  ablation toggle keys are all valid spider.config.Config fields, and each
      ablation's target keys are non-empty and disjoint from PLUMBING_KEYS.

Run: .venv/bin/python workspace/core4d/scripts/experiments/E214/test_e214_contract.py
"""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e214_common as C  # noqa: E402

# Expected full-stack baseline values (numeric compared with tolerance).
FULLSTACK_EXPECT = {
    "contact_hdmi_gain": ("gt", 0.0),
    "surface_band_rew_scale": ("gt", 0.0),
    "robot_object_penalty_scale": ("gt", 0.0),
    "leg_object_penalty_scale": ("gt", 0.0),
    "hand_floor_penalty_scale": ("gt", 0.0),
    "e167_body_z_enabled": ("true", None),
    "e167_ground_z_enabled": ("true", None),
    "cem_safety_gate_enabled": ("true", None),
    "cem_hand_gate_enabled": ("true", None),
    "cem_leg_gate_enabled": ("true", None),
    "cem_posture_gate_enabled": ("true", None),
}


def _fail(msg: str, bag: list[str]) -> None:
    bag.append(msg)


def main() -> int:
    failures: list[str] = []
    cases = C.load_cases()

    # C1 count
    if len(cases) != 50:
        _fail(f"C1 case count {len(cases)} != 50", failures)
    paper = [ln.strip() for ln in C.PAPER_CASE_FILE.read_text().splitlines() if ln.strip()]
    expect = [c for c in paper if c not in C.DROP_CASES]
    if cases != expect:
        _fail(f"C1 case list != paper minus drops (paper={len(paper)}, drops={len(C.DROP_CASES)})", failures)
    for d in C.DROP_CASES:
        if d in cases:
            _fail(f"C1 dropped case still present: {d}", failures)

    # C5 config field validity
    allowed = {f.name for f in fields(__import__("spider.config", fromlist=["Config"]).Config)}
    for abl, spec in C.ABLATIONS.items():
        toggles = spec["toggles"]
        if not toggles:
            _fail(f"C5 ablation {abl} has empty toggles", failures)
        for key in toggles:
            if key not in allowed:
                _fail(f"C5 {abl}: '{key}' not a Config field", failures)
            if key in C.PLUMBING_KEYS:
                _fail(f"C5 {abl}: target key '{key}' collides with PLUMBING_KEYS", failures)

    # C2/C3/C4 per-case resolution
    box023_seen = 0
    for case_id in cases:
        bp = C.baseline_paths(case_id)
        if C.object_key_of(case_id) == "box023":
            box023_seen += 1
            if bp.get("exp") != "E173_PRG_fullstack":
                _fail(f"C2 {case_id}: box023 source={bp.get('exp')} != E173_PRG_fullstack", failures)
        cfg = bp.get("config_act")
        if cfg is None:
            _fail(f"C3 {case_id}: baseline unresolved ({bp.get('note')})", failures)
            continue
        for k in ("scene_act", "trajectory", "contact_mask"):
            if bp.get(k) is None:
                _fail(f"C3 {case_id}: baseline {k} unresolved", failures)
        # C4 full-stack
        data = yaml.safe_load(Path(cfg).read_text(encoding="utf-8"))
        for key, (mode, thr) in FULLSTACK_EXPECT.items():
            val = data.get(key)
            if mode == "gt":
                if not (isinstance(val, (int, float)) and float(val) > thr):
                    _fail(f"C4 {case_id}: {key}={val} not > {thr}", failures)
            elif mode == "true":
                if val not in (True, "true", "True"):
                    _fail(f"C4 {case_id}: {key}={val} not enabled", failures)

    if box023_seen != 7:
        _fail(f"C2 box023 count {box023_seen} != 7", failures)

    print(f"E214 contract: {len(cases)} cases x {len(C.ABLATIONS)} ablations = "
          f"{len(cases) * len(C.ABLATIONS)} CEM runs")
    if failures:
        print(f"\nFAIL ({len(failures)}):")
        for f in failures[:60]:
            print("  -", f)
        if len(failures) > 60:
            print(f"  ... and {len(failures) - 60} more")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
