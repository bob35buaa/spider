# E128 Holosoma Runtime Startup Preflight Plan

Date: 2026-06-03

## Context

E127 proved that E126 paired fragment exports satisfy the static Box021 handbox partner motion contract. It did not prove IsaacSim can start the Holosoma scene, load the Box021 object, and drive partner hands using these absolute motion paths.

E128 performs the narrowest runtime gate before any PPO smoke or full RL training: bounded Holosoma replay startup for the two E126 paired fragment motions. It uses a no-debug replay probe to avoid headless visualization marker drawing, while preserving the Holosoma loader, object config, command config, and simulator startup path. This is still fragment/reward-wiring evidence only; source rows remain `FRAGMENT_HOLDOUT_ONLY`.

## Claims

| claim | success evidence |
|---|---|
| C1: Holosoma runtime can start with the E126 motion path | E128 no-debug replay probe exits with code 0 before timeout for each paired export and emits startup markers |
| C2: Box021 object and partner-hand config are used | command includes Box021 URDF and existing Box021 handbox partner experiment alias |
| C3: no training is launched | runner invokes `replay.py`, not `train_agent.py`, and summary reports `training_launched=false` |
| C4: fragment labels still block RL evidence | summary reports `rl_smoke_allowed_rows=0` and `rl_ready_rows=0` |

## Inputs

- E126 paired exports under `workspace/core4d/results/E126/holosoma_fragment_adapter_preflight/exports/`
- E127 static contract summary and manifest
- E128 no-debug replay probe:
  `workspace/core4d/scripts/E128/holosoma_replay_no_debug_probe.py`

## Outputs

```text
workspace/core4d/results/E128/holosoma_runtime_startup_preflight/
  e128_runtime_startup_manifest.tsv
  e128_runtime_startup_summary.json
  e128_runtime_startup_summary.md
  logs/*.log
```

## Success Criteria

- 2/2 startup commands exit 0 before timeout.
- Each log contains the expected E126 motion path.
- Each log is free of `Traceback`, `AttributeError`, `RuntimeError`, `AssertionError`, and `Exception`.
- Each log confirms `has_object=True` and `has_partner=True`.
- Each log emits `E128_PROBE_DONE` after the bounded startup loop.
- No training artifacts or checkpoints are created by E128.
- `rl_smoke_allowed_rows=0`, `rl_ready_rows=0`, `training_launched=false`.

## Command Shape

```bash
bash workspace/core4d/scripts/eval/eval_E128_holosoma_runtime_startup.sh
```

Environment knobs:

- `GPU_ID=0`
- `E128_TIMEOUT_SECONDS=300`
- `E128_MAX_STEPS=20`
- `HOLOSOMA_ROOT=/home/ubuntu/Workspace/holosoma`

## Non-Goals

- Do not run PPO smoke or full Holosoma RL.
- Do not mark `box021_035_p1/p2` as train-ready evidence.
- Do not use holdout fragment startup success to claim the main `box021_029_p2` gate is solved.
