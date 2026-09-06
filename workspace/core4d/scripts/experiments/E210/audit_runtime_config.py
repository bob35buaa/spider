#!/usr/bin/env python3
"""E210 P6: re-read each rollout's `config_act.yaml` and assert what actually ran.

Composing the override proves what Hydra *would* resolve. This proves what the
process on the GPU actually used -- the two can diverge (stale override on disk,
a CLI arg the queue added, a task pointing somewhere else). E207's plan called
this "the only gate that can show we ran G1only rather than G1A2"; the same
applies here, with one extra axis: the task must be the *augmented* one, not the
orig it was seeded from.

Nine assertions per rollout:
  1. scene_name == the E210 gravcomp sidecar
  2-4. hand gate == E163 default (A0)  -- separates E210 from an aug x G1A2 arm
  5. leg_object_penalty_scale == 2.0   -- PRG still on
  6. cem_leg_gate_enabled == true      -- PRG still on
  7. init_pos_actuator_gain == 500     -- the kp that gravcomp compensates
  8. init_rot_actuator_gain == 50
  9. task == the augmented task, and the resolved model_path lives in it

Usage:
    .venv/bin/python .../E210/audit_runtime_config.py --stage smoke
    .venv/bin/python .../E210/audit_runtime_config.py --stage full --require-all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e210_common as C  # noqa: E402

EXPECTED_SCALARS = {
    "leg_object_penalty_scale": 2.0,
    "init_pos_actuator_gain": 500.0,
    "init_rot_actuator_gain": 50.0,
}


def audit_row(row: dict[str, str]) -> tuple[list[str], dict[str, object]]:
    cfg_path = C.repo_path(row["config_act"])
    if not cfg_path.is_file():
        return [f"missing config_act: {row['config_act']}"], {}
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    fails: list[str] = []

    if cfg.get("scene_name") != C.SCENE_NAME:
        fails.append(f"scene_name={cfg.get('scene_name')!r}!={C.SCENE_NAME}")

    # A0 hand gate. If any of these took E198's A2 values (0.05 / -0.015) the
    # run is a different arm and the E207 comparison is void.
    for key, want in C.E163_HAND_GATE.items():
        got = cfg.get(key)
        if got is None or abs(float(got) - want) > 1e-9:
            fails.append(f"{key}={got}!=A0({want})")

    for key, want in EXPECTED_SCALARS.items():
        got = cfg.get(key)
        if got is None or abs(float(got) - want) > 1e-9:
            fails.append(f"{key}={got}!={want}")
    if not bool(cfg.get("cem_leg_gate_enabled", False)):
        fails.append("cem_leg_gate_enabled!=true")

    if cfg.get("task") != row["target_task"]:
        fails.append(f"task={cfg.get('task')!r}!={row['target_task']}")
    model_path = str(cfg.get("model_path", ""))
    if row["target_task"] not in model_path:
        fails.append(f"model_path {model_path!r} is not inside {row['target_task']}")

    observed = {
        "scene_name": cfg.get("scene_name"),
        "task": cfg.get("task"),
        **{k: cfg.get(k) for k in C.E163_HAND_GATE},
        **{k: cfg.get(k) for k in EXPECTED_SCALARS},
        "cem_leg_gate_enabled": cfg.get("cem_leg_gate_enabled"),
        "num_samples": cfg.get("num_samples"),
        "max_num_iterations": cfg.get("max_num_iterations"),
        "seed": cfg.get("seed"),
    }
    return fails, observed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("full", "smoke"), default="full")
    ap.add_argument("--require-all", action="store_true",
                    help="fail if any manifest row has no config_act yet")
    args = ap.parse_args()

    manifest = C.FULL_MANIFEST if args.stage == "full" else C.SMOKE_MANIFEST
    rows = C.read_tsv(manifest)
    if not rows:
        raise SystemExit(f"no rows in {C.rel(manifest)}")

    results, failures, skipped = [], {}, []
    for row in rows:
        if not args.require_all and not C.repo_path(row["config_act"]).is_file():
            skipped.append(row["variant"])
            continue
        fails, observed = audit_row(row)
        results.append({"variant": row["variant"], "ok": not fails,
                        "fails": fails, "observed": observed})
        if fails:
            failures[row["variant"]] = fails

    for r in results:
        mark = "PASS" if r["ok"] else "FAIL"
        print(f"  {mark} {r['variant']}")
        if not r["ok"]:
            for f in r["fails"]:
                print(f"        {f}")
    if skipped:
        print(f"  (skipped {len(skipped)} rows with no config_act yet)")

    out = C.RESULTS / f"preflight/e210_runtime_config_audit_{args.stage}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    C.write_json(out, {
        "created_at": C.now(), "stage": args.stage, "manifest": C.rel(manifest),
        "audited": len(results), "skipped": skipped,
        "failures": failures, "results": results,
    })
    if failures:
        raise SystemExit(f"runtime config audit FAILED for {len(failures)} rollouts")
    print(f"\nruntime audit PASS: {len(results)} rollout(s), 9/9 assertions each "
          f"(scene=gravcomp, hand-gate=A0, PRG on, kp=500/50, aug task) -> {C.rel(out)}")
    print(json.dumps(results[0]["observed"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
