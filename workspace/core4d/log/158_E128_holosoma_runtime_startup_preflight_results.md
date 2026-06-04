# E128 Holosoma Runtime Startup Preflight Results

Date: 2026-06-03

## Scope

E128 followed `workspace/core4d/plan/137_E128_holosoma_runtime_startup_preflight_plan.md`.
It did not launch CEM, PPO, Holosoma training, or checkpoint creation. The goal was
only to prove that Holosoma/IsaacSim can start with the E126 paired fragment exports,
load the Box021 object, register partner hands, and step a bounded no-debug replay
probe.

## Inputs

- `workspace/core4d/results/E126/holosoma_fragment_adapter_preflight/exports/E126_box021_035_p1_with_partner_box021_035_p2_mj_w_obj_w_partner.npz`
- `workspace/core4d/results/E126/holosoma_fragment_adapter_preflight/exports/E126_box021_035_p2_with_partner_box021_035_p1_mj_w_obj_w_partner.npz`
- Holosoma experiment alias:
  `exp:g1-29dof-wbt-w-object-r135-box021-handbox-exp0601-v4-3`
- Box021 object URDF:
  `/home/ubuntu/Workspace/holosoma/src/holosoma_retargeting/holosoma_retargeting/models/Box021/Box021.urdf`

## Result

Output root:

```text
workspace/core4d/results/E128/holosoma_runtime_startup_preflight/
```

Summary:

| metric | value |
|---|---:|
| rows | 2 |
| startup pass rows | 2 |
| max sim steps per row | 20 |
| timeout seconds | 300 |
| RL smoke allowed rows | 0 |
| RL-ready rows | 0 |
| training launched | false |
| status | pass |

Both rows returned code 0 before timeout. The logs contain the expected E126 motion
path, object registration, partner-hand registration, and probe markers:

- `box021_035_p1`: `has_object=True`, `has_partner=True`, `has_object_contact=False`,
  `time_step_total=414`, `E128_PROBE_DONE steps=20 done=False`
- `box021_035_p2`: `has_object=True`, `has_partner=True`, `has_object_contact=False`,
  `time_step_total=414`, `E128_PROBE_DONE steps=20 done=False`

The bounded loop intentionally stops before full replay completion, so `done=False`
is expected. No `Traceback`, `AttributeError`, `RuntimeError`, `AssertionError`, or
`Exception` signatures were found in the startup logs.

## Interpretation

E128 closes the runtime startup/config gap identified after E127: the E126 paired
fragment exports are not only structurally shaped like Holosoma motion inputs, they
can also start inside Holosoma/IsaacSim with the Box021 object and partner hands.

This remains pre-training evidence only. The source rows are still
`FRAGMENT_HOLDOUT_ONLY`; `rl_smoke_allowed_rows=0`, `rl_ready_rows=0`, and
`training_launched=false`. E128 does not solve the main `box021_029_p2` release gate
and should not be treated as CEM or RL success evidence.

## Verification

- `.venv/bin/python -m py_compile workspace/core4d/scripts/E128/holosoma_replay_no_debug_probe.py`
- `bash -n workspace/core4d/scripts/eval/eval_E128_holosoma_runtime_startup.sh`
- `bash workspace/core4d/scripts/eval/eval_E128_holosoma_runtime_startup.sh`
- `rg --no-ignore -a -n "E128_PROBE|Traceback|AttributeError|RuntimeError|AssertionError|Exception|Loading motion file|Registered individual object" workspace/core4d/results/E128/holosoma_runtime_startup_preflight/logs`

