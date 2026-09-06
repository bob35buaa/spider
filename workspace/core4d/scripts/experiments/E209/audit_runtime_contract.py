#!/usr/bin/env python3
"""E209 P5/P6 exit check: prove at RUNTIME that what ran is G1, not something else.

The override audit (P2) proves the *composed* config is single-variable. This
proves the *executed* config is, by reading back each run's `config_act.yaml` --
the file the process actually resolved -- and diffing it against E206's PRG
`config_act.yaml` for the same case.

Two failure modes this is the only guard against:

  * silently running G1A2 (gravcomp + the A2 hand-gate retune) instead of G1.
    A2 moves `cem_hand_gate_max_violation_pct` 0.10 -> 0.05 and
    `hard_floor_m` -0.020 -> -0.015; both are checked explicitly.
  * silently running the baseline (sidecar not picked up), which would make the
    whole comparison a no-op that still "passes" every other check.

Usage:
    .venv/bin/python workspace/core4d/scripts/experiments/E209/audit_runtime_contract.py \
        [--stage smoke|full] [--cases a,b]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E209"))

import e209_common as C  # noqa: E402

#: Runtime keys are classified three ways (same scheme as E206's
#: audit_runtime_arm_contract.py). Anything outside these three sets is a
#: confound and fails the audit.
#:
#: THE_VARIABLE -- the intervention itself, and `model_path`, which is just
#: `scene_name` resolved to a file. `model_path` is not ignored: it is asserted
#: below to end in exactly the expected scene basename on each side.
THE_VARIABLE = {"scene_name", "model_path"}

#: BY_DESIGN -- names the run, not the physics.
RUN_LOCAL_KEYS = {
    "output_dir", "video_path", "video_output_path", "run_name", "seed_everything",
}

#: STAGE_BUDGET -- differ only when the smoke stage deliberately shrinks the CEM
#: budget. `beta_traj` is derived from (num_samples, max_num_iterations), so it
#: moves with them. For stage=full these must match the baseline exactly, so the
#: set is empty there and any difference fails.
STAGE_BUDGET_KEYS = {"num_samples", "max_num_iterations", "beta_traj"}

#: Scalars that must hold at runtime regardless of what any YAML says.
REQUIRED_RUNTIME = {
    # A0 hand-gate (E163 default). A2 would be 0.05 / -0.015 -- see module docstring.
    "cem_hand_gate_enabled": True,
    "cem_hand_gate_min_sdf_m": -0.010,
    "cem_hand_gate_max_violation_pct": 0.10,
    "cem_hand_gate_hard_floor_m": -0.020,
    # PRG lower-body triple still on.
    "leg_object_penalty_scale": 2.0,
    "cem_leg_gate_enabled": True,
    # The object actuator gains that make `sag = m*g/kp` the thing gravcomp cancels.
    "init_pos_actuator_gain": 500.0,
    "init_rot_actuator_gain": 50.0,
}


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def same(a: object, b: object) -> bool:
    """Equality that treats NaN as equal to itself.

    Several gate fields default to NaN ("disabled"); plain `!=` reports every
    one of them as drift, which would bury a real confound in noise.
    """
    if isinstance(a, float) and isinstance(b, float):
        if a != a and b != b:  # both NaN
            return True
    return a == b


def audit_case(case_id: str, stage: str) -> list[str]:
    cand = load(C.config_act(case_id, stage))
    base = load(C.baseline_out_dir(case_id) / "config_act.yaml")
    fails: list[str] = []

    if cand.get("scene_name") != C.SCENE:
        fails.append(f"scene_name={cand.get('scene_name')!r} != {C.SCENE!r}")
    if base.get("scene_name") != C.BASE_SCENE:
        fails.append(f"baseline scene_name={base.get('scene_name')!r} != {C.BASE_SCENE!r}")

    # model_path is scene_name resolved to disk -- assert it, don't just excuse it.
    for label, cfg, want in (("E209", cand, C.SCENE), ("E206", base, C.BASE_SCENE)):
        mp = str(cfg.get("model_path", ""))
        if Path(mp).name != f"{want}.xml":
            fails.append(f"{label} model_path basename={Path(mp).name!r} != {want}.xml")

    for key, want in REQUIRED_RUNTIME.items():
        got = cand.get(key)
        if got is None or abs(float(got) - float(want)) > 1e-9:
            fails.append(f"{key}={got} != {want}")

    allowed = THE_VARIABLE | RUN_LOCAL_KEYS
    if stage != "full":
        allowed |= STAGE_BUDGET_KEYS
    else:
        # At full budget the CEM parameters must be bit-identical to the baseline.
        for key in ("num_samples", "max_num_iterations", "seed"):
            if key in base and not same(base.get(key), cand.get(key)):
                fails.append(f"full-stage budget drift {key}: {base.get(key)} -> {cand.get(key)}")

    drift = {
        k
        for k in set(base) | set(cand)
        if not same(base.get(k, "\0missing"), cand.get(k, "\0missing"))
    } - allowed
    if drift:
        fails.append(
            "runtime_drift:"
            + ",".join(f"{k}({base.get(k)}->{cand.get(k)})" for k in sorted(drift))
        )
    if cand.get("scene_name") == base.get("scene_name"):
        fails.append("scene_name identical to baseline -- gravcomp arm never engaged")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="full", choices=("full", "smoke"))
    ap.add_argument("--cases", default="")
    args = ap.parse_args()

    cases = [c.strip() for c in args.cases.split(",") if c.strip()] or [
        c for c in C.CASES if C.config_act(c, args.stage).is_file()
    ]
    if not cases:
        raise SystemExit(f"no {args.stage} config_act.yaml found yet")

    failures: dict[str, list[str]] = {}
    for case_id in cases:
        fails = audit_case(case_id, args.stage)
        if fails:
            failures[case_id] = fails
            print(f"  FAIL {case_id}: {'; '.join(fails)}")
        else:
            print(f"  {case_id:36s} PASS  runtime == E206 PRG + gravcomp only (A0 hand-gate)")

    out = C.S6_DIR / f"cem/{args.stage}/e209_runtime_contract.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"stage": args.stage, "checked": len(cases), "failures": failures},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if failures:
        raise SystemExit(f"runtime contract FAILED for {len(failures)}/{len(cases)} cases")
    print(f"\nruntime contract PASS: {len(cases)}/{len(cases)} ({args.stage}) -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
