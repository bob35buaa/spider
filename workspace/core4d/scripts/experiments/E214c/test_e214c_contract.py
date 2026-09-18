#!/usr/bin/env python3
"""E214c CPU contract: prove the part-wise ablation set is fully resolvable + the
per-case toggles compute correctly, BEFORE any GPU is touched.

Checks (no MuJoCo / no CEM):
  C1  case count == 50 AND identical to E214's case set (same 50 = paper 52 - 2 box021).
  C2  box023 (7 cases) resolves through the E173 full-stack export.
  C3  every case resolves to an EXISTING baseline config_act.yaml + scene/traj/mask.
  C4  every baseline is full-stack (contact+surface_band on, soft penalties 2/2/2,
      e167 z on, safety+hand+leg+posture gates on) AND carries the part-wise hard-gate
      thresholds (hand min_sdf -10mm, leg +5mm, body -5mm) so the ablation is meaningful.
  C5  the 9 distinct toggle key names are valid spider.config.Config fields and disjoint
      from PLUMBING_KEYS.
  C6  per_case_toggles computes 6 keys per (case, ablation): the two TARGET gates x 3
      params, values == that case's baseline SOURCE-gate values; source gate untouched.
  C7  hydra_token formats every toggle key with a leading '+' (none are in default.yaml),
      the path E214 proved lands via load_config_path.

Run: .venv/bin/python workspace/core4d/scripts/experiments/E214c/test_e214c_contract.py
"""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e214c_common as C  # noqa: E402

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
# Part-wise hard-gate min_sdf expected in every baseline (the asymmetry we ablate).
PARTWISE_MIN_SDF = {
    "cem_hand_gate_min_sdf_m": -0.01,
    "cem_leg_gate_min_sdf_m": 0.005,
    "cem_safety_gate_min_sdf_m": -0.005,
}


def _fail(msg: str, bag: list[str]) -> None:
    bag.append(msg)


def _num_eq(a, b, tol=1e-9) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def main() -> int:
    failures: list[str] = []
    cases = C.load_cases()

    # C1 count + identity with E214's case set
    if len(cases) != 50:
        _fail(f"C1 case count {len(cases)} != 50", failures)
    e214_cases = [ln.strip() for ln in C.E214_CASE_FILE.read_text().splitlines() if ln.strip()]
    if cases != e214_cases:
        _fail(f"C1 case list != E214's (E214c={len(cases)}, E214={len(e214_cases)})", failures)
    paper = [ln.strip() for ln in C.PAPER_CASE_FILE.read_text().splitlines() if ln.strip()]
    expect = [c for c in paper if c not in C.DROP_CASES]
    if cases != expect:
        _fail(f"C1 case list != paper minus drops (paper={len(paper)}, drops={len(C.DROP_CASES)})", failures)

    # C5 toggle key names valid Config fields, disjoint from plumbing
    allowed = {f.name for f in fields(__import__("spider.config", fromlist=["Config"]).Config)}
    for key in C.ALL_TOGGLE_KEYS:
        if key not in allowed:
            _fail(f"C5 toggle key '{key}' not a Config field", failures)
        if key in C.PLUMBING_KEYS:
            _fail(f"C5 toggle key '{key}' collides with PLUMBING_KEYS", failures)
    if len(C.ALL_TOGGLE_KEYS) != 9:
        _fail(f"C5 expected 9 distinct toggle keys, got {len(C.ALL_TOGGLE_KEYS)}", failures)

    # C2/C3/C4/C6 per-case
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
        data = yaml.safe_load(Path(cfg).read_text(encoding="utf-8"))
        # C4 full-stack
        for key, (mode, thr) in FULLSTACK_EXPECT.items():
            val = data.get(key)
            if mode == "gt":
                if not (isinstance(val, (int, float)) and float(val) > thr):
                    _fail(f"C4 {case_id}: {key}={val} not > {thr}", failures)
            elif mode == "true":
                if val not in (True, "true", "True"):
                    _fail(f"C4 {case_id}: {key}={val} not enabled", failures)
        # C4 part-wise thresholds present + as expected
        for key, want in PARTWISE_MIN_SDF.items():
            if not _num_eq(data.get(key), want):
                _fail(f"C4 {case_id}: {key}={data.get(key)} != {want}", failures)
        # C6 per-case toggles for both ablations
        for abl in C.ABLATION_ORDER:
            spec = C.ABLATIONS[abl]
            try:
                tog = C.per_case_toggles(case_id, abl, cfg)
            except Exception as exc:  # noqa: BLE001
                _fail(f"C6 {case_id}/{abl}: per_case_toggles raised {exc}", failures)
                continue
            if len(tog) != 6:
                _fail(f"C6 {case_id}/{abl}: expected 6 toggles, got {len(tog)}", failures)
            src_prefix = C.GATE_PREFIX[spec["source_gate"]]
            # source gate must NOT be a toggle target
            if any(k.startswith(src_prefix + "_") for k in tog):
                _fail(f"C6 {case_id}/{abl}: source gate {src_prefix} appears in toggles", failures)
            # each target gate's 3 params present and == baseline source values
            for tgt in spec["target_gates"]:
                tp = C.GATE_PREFIX[tgt]
                for tk in C.THRESHOLD_KEYS:
                    tkey = f"{tp}_{tk}"
                    skey = f"{src_prefix}_{tk}"
                    if tkey not in tog:
                        _fail(f"C6 {case_id}/{abl}: missing toggle {tkey}", failures)
                    elif not _num_eq(tog[tkey], data.get(skey)):
                        _fail(f"C6 {case_id}/{abl}: {tkey}={tog[tkey]} != baseline {skey}={data.get(skey)}",
                              failures)
            # C7 hydra_token uses '+' (keys absent from default.yaml)
            for k in tog:
                tok = C.hydra_token(k, tog[k])
                if not tok.startswith(f"+{k}="):
                    _fail(f"C7 {case_id}/{abl}: hydra_token({k})='{tok}' missing '+' prefix", failures)

    if box023_seen != 7:
        _fail(f"C2 box023 count {box023_seen} != 7", failures)

    print(f"E214c contract: {len(cases)} cases x {len(C.ABLATIONS)} ablations = "
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
