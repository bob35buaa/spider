#!/usr/bin/env python3
"""E212 P6/P7 exit check: prove at RUNTIME that each arm ran its own g value.

The override audit (P3) proves the *composed* config is single-variable.  This
proves the *executed* one is, by reading back each run's ``config_act.yaml`` --
the file the process actually resolved -- and diffing it against E206's PRG
``config_act.yaml`` for the same case.

Like E211, this compiles the model the run actually loaded and reads
``MjModel.body_gravcomp`` for the object body.  For a
0/1 intervention "the scene name is different" is nearly enough; for a partial
sweep it is not -- a typo that wrote 0.6 as 0.06, or an ``ET`` write that lost
precision, would still produce a distinct scene name and a distinct sha256 and
would sail through every other guard.  ``body_gravcomp`` is the number the
integrator multiplies gravity by (``mujoco_warp/_src/passive.py:268``), so it is
the only reading that cannot be faked by naming.

Failure modes this is the only guard against:

  * silently running G1A2 (the A2 hand-gate retune) instead of the A0 baseline;
  * silently running the E206 baseline (sidecar not picked up), which would make
    the arm a no-op that still "passes" every other check;
  * silently running a different g than the arm claims.

Usage:
    .venv/bin/python workspace/core4d/scripts/experiments/E212/audit_runtime_contract.py \
        [--stage smoke|full] [--cases a,b] [--arms G06]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import yaml

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E212"))

import e212_common as C  # noqa: E402

#: Runtime keys are classified three ways (same scheme as E209's
#: audit_runtime_contract.py). Anything outside these three sets is a confound.
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
#: budget. `beta_traj` is derived from (num_samples, max_num_iterations).
STAGE_BUDGET_KEYS = {"num_samples", "max_num_iterations", "beta_traj"}

#: Scalars that must hold at runtime regardless of what any YAML says.
REQUIRED_RUNTIME = {
    # A0 hand-gate (E163 default). A2 would be 0.05 / -0.015.
    "cem_hand_gate_enabled": True,
    "cem_hand_gate_min_sdf_m": -0.010,
    "cem_hand_gate_max_violation_pct": 0.10,
    "cem_hand_gate_hard_floor_m": -0.020,
    # PRG lower-body triple still on.
    "leg_object_penalty_scale": 2.0,
    "cem_leg_gate_enabled": True,
    # The object actuator gains that make `sag = m*g/kp` the thing gravcomp scales.
    "init_pos_actuator_gain": 500.0,
    "init_rot_actuator_gain": 50.0,
}


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def same(a: object, b: object) -> bool:
    """Equality that treats NaN as equal to itself.

    Several gate fields default to NaN ("disabled"); plain `!=` reports every one
    of them as drift, which would bury a real confound in noise.
    """
    if isinstance(a, float) and isinstance(b, float):
        if a != a and b != b:  # both NaN
            return True
    return a == b


def compiled_object_gravcomp(model_path: Path) -> float:
    """The gravity multiplier the integrator will actually apply to the object."""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if bid < 0:
        raise ValueError(f"no body named 'object' in {model_path}")
    return float(model.body_gravcomp[bid])


def audit_case(case_id: str, arm: str, stage: str) -> list[str]:
    cand = load(C.config_act(case_id, arm, stage))
    base = load(C.prg_out_dir(case_id) / "config_act.yaml")
    want_scene = C.SCENE_BY_ARM[arm]
    want_g = float(C.gravcomp_str(arm))
    fails: list[str] = []

    if cand.get("scene_name") != want_scene:
        fails.append(f"scene_name={cand.get('scene_name')!r} != {want_scene!r}")
    if base.get("scene_name") != C.BASE_SCENE:
        fails.append(f"baseline scene_name={base.get('scene_name')!r} != {C.BASE_SCENE!r}")

    # model_path is scene_name resolved to disk -- assert it, don't just excuse it.
    for label, cfg, want in ((C.EXP, cand, want_scene), ("E206", base, C.BASE_SCENE)):
        mp = str(cfg.get("model_path", ""))
        if Path(mp).name != f"{want}.xml":
            fails.append(f"{label} model_path basename={Path(mp).name!r} != {want}.xml")

    # The reading that naming cannot fake: what the integrator multiplies by.
    mp = REPO / str(cand.get("model_path", ""))
    if mp.is_file():
        got_g = compiled_object_gravcomp(mp)
        if abs(got_g - want_g) > 1e-9:
            fails.append(f"compiled body_gravcomp={got_g} != {want_g}")
        base_mp = REPO / str(base.get("model_path", ""))
        if base_mp.is_file():
            base_g = compiled_object_gravcomp(base_mp)
            if base_g != 0.0:
                fails.append(f"baseline body_gravcomp={base_g} != 0.0")
    else:
        fails.append(f"model_path not readable: {mp}")

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
    ap.add_argument("--arms", default="")
    args = ap.parse_args()

    cases = [c.strip() for c in args.cases.split(",") if c.strip()] or list(C.CASES)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()] or list(C.ARM_ORDER)
    # Gate on the RESULT npz, not on config_act.yaml. The config is written at
    # process start (before the first CEM step), so keying off it counts
    # still-running rows as audited -- the audit would report "12/12 PASS" while
    # a run is mid-flight. The npz only appears when the rollout finished.
    pairs, pending = [], []
    for c in cases:
        for a in arms:
            if not C.config_act(c, a, args.stage).is_file():
                continue
            if C.result_npz(c, a, args.stage).is_file():
                pairs.append((c, a))
            else:
                pending.append(f"{c}/{a}")
    if pending:
        print(f"  skipping {len(pending)} run(s) still in flight: {', '.join(pending)}")
    if not pairs:
        raise SystemExit(f"no completed {args.stage} rollouts found yet")

    failures: dict[str, list[str]] = {}
    for case_id, arm in pairs:
        fails = audit_case(case_id, arm, args.stage)
        key = f"{case_id}/{arm}"
        if fails:
            failures[key] = fails
            print(f"  FAIL {key}: {'; '.join(fails)}")
        else:
            print(
                f"  {case_id:32s} {arm}  PASS  runtime == E206 PRG + gravcomp="
                f"{C.gravcomp_str(arm)} only (A0 hand-gate)"
            )

    out = C.S6_DIR / f"cem/{args.stage}/{C.EXP.lower()}_runtime_contract.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "stage": args.stage,
                "checked": len(pairs),
                "expected": C.EXPECTED_ROWS if args.stage == "full" else None,
                "pending": pending,
                "failures": failures,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if failures:
        raise SystemExit(f"runtime contract FAILED for {len(failures)}/{len(pairs)} runs")
    print(f"\nruntime contract PASS: {len(pairs)}/{len(pairs)} ({args.stage}) -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
