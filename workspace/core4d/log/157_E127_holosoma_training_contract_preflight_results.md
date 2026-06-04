# E127 Holosoma Training-Contract Preflight Results

Date: 2026-06-03

## Goal

E127 validates whether the E126 paired fragment exports satisfy the static Holosoma Box021 partner training contract before any IsaacSim startup, PPO smoke, or full training. The upstream rows remain `FRAGMENT_HOLDOUT_ONLY`, so passing E127 must not unlock RL training.

## Implementation

Added:

- `workspace/core4d/plan/136_E127_holosoma_training_contract_preflight_plan.md`
- `workspace/core4d/scripts/E127/check_holosoma_training_contract.py`
- `workspace/core4d/scripts/eval/eval_E127_holosoma_training_contract.sh`

The checker reads E126 `e126_adapter_manifest.tsv`, validates both paired `_mj_w_obj_w_partner.npz` files, and checks the known Holosoma Box021 partner runtime contract:

- config key: `g1_29dof_wbt_w_object_r135_box021_handbox_exp0601_v4_3`
- CLI alias used by existing train scripts: `exp:g1-29dof-wbt-w-object-r135-box021-handbox-exp0601-v4-3`
- Box021 object URDF exists
- G1 handbox robot URDF exists
- partner-hand URDF exists

Subagent note: a read-only sidecar audit independently recommended the same next gate: loader/config/startup preflight rather than PPO/RL training. It also identified `object_contact` as optional in Holosoma's loader and warned that missing `object_contact` is only acceptable for the current Box021 handbox partner config, not for a future ref-contact-mask reward.

## Outputs

```text
workspace/core4d/results/E127/holosoma_training_contract_preflight/
  e127_training_contract_manifest.tsv
  e127_training_contract_summary.json
  e127_training_contract_summary.md
```

Summary:

- rows: 2
- structural pass rows: 2
- RL smoke allowed rows: 0
- RL-ready rows: 0
- training launched: false
- status: pass

Rows:

| case | partner | source decision | frames | fps | partner shape | object contact | rl smoke |
|---|---|---|---:|---:|---|---|---|
| `box021_035_p1` | `box021_035_p2` | `FRAGMENT_HOLDOUT_ONLY` | 214 | 50.0 | `214x2x3 / 214x2x4` | absent | blocked |
| `box021_035_p2` | `box021_035_p1` | `FRAGMENT_HOLDOUT_ONLY` | 214 | 50.0 | `214x2x3 / 214x2x4` | absent | blocked |

Both rows have:

- `joint_pos=214x36`
- `joint_vel=214x35`
- `body_pos_w=214x52x3`
- `numeric_nan_count=0`
- `numeric_nonfinite_count=0`
- `body_names_count=52`
- `joint_names_count=29`
- config registered: true
- CLI alias present in existing train scripts: true
- object/handbox/partner URDF checks: true

## Decision

E127 passes as a static training-contract preflight. It proves the E126 artifacts are structurally compatible with the existing Box021 handbox partner motion contract, but it deliberately keeps `rl_smoke_allowed=false` because the source rows are still fragment-only.

The next valid step, if we need to probe Holosoma runtime wiring, is a bounded IsaacSim loader/startup preflight with `num_envs=1` and a timeout. That should still be logged as fragment/reward-wiring evidence, not RL-ready evidence. The main `box021_029_p2` remains blocked.

## Validation

Passed:

```text
.venv/bin/python -m py_compile workspace/core4d/scripts/E127/check_holosoma_training_contract.py
bash -n workspace/core4d/scripts/eval/eval_E127_holosoma_training_contract.sh
bash workspace/core4d/scripts/eval/eval_E127_holosoma_training_contract.sh
```

No CEM, IsaacSim startup, PPO smoke, or Holosoma RL training was launched.
