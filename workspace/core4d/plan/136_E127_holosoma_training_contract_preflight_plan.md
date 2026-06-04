# E127 Holosoma Training-Contract Preflight Plan

Date: 2026-06-03

## Context

E125 found no main-case RL-ready row. E126 converted only the same-sequence `box021_035_p1/p2` holdout fragments into Holosoma-style paired motion files with `partner_hand_*` fields.

Before any Holosoma RL smoke or full training, E127 validates the static training contract used by existing Box021 partner runs (`R135/R136/R135o/R136o`) against the E126 artifacts. This is a no-GPU preflight and must preserve the upstream `FRAGMENT_HOLDOUT_ONLY` label.

## Claims

| claim | success evidence |
|---|---|
| C1: E126 artifacts still block RL training | manifest rows keep `source_decision=FRAGMENT_HOLDOUT_ONLY` and `rl_train_allowed=false` |
| C2: paired NPZ files satisfy the Holosoma motion schema used by Box021 partner scripts | required keys exist, numeric arrays are finite, and frame-aligned shapes pass |
| C3: expected Holosoma runtime assets are resolvable | Box021 object URDF, handbox robot URDF, partner-hand URDF exist, and the known Box021 partner experiment config is registered in Holosoma |
| C4: no training is launched | summary reports `training_launched=false` and `rl_smoke_allowed=false` |

## Inputs

- `workspace/core4d/results/E126/holosoma_fragment_adapter_preflight/e126_adapter_manifest.tsv`
- `workspace/core4d/results/E126/holosoma_fragment_adapter_preflight/exports/*_w_partner.npz`
- Holosoma config source under `/home/ubuntu/Workspace/holosoma/src/holosoma/holosoma/config_values/`
- Holosoma Box021 URDF:
  `/home/ubuntu/Workspace/holosoma/src/holosoma_retargeting/holosoma_retargeting/models/Box021/Box021.urdf`

## Outputs

```text
workspace/core4d/results/E127/holosoma_training_contract_preflight/
  e127_training_contract_manifest.tsv
  e127_training_contract_summary.json
  e127_training_contract_summary.md
```

## Success Criteria

- 2/2 E126 paired rows pass structural validation.
- `rl_train_allowed=false` for every row.
- `rl_smoke_allowed=false` for every row because the source label is fragment-only.
- Required Holosoma config/URDF checks pass.
- No IsaacSim, CEM, or RL training process is launched.

## Non-Goals

- Do not run Holosoma RL smoke or full training.
- Do not mark `box021_035_p1/p2` as RL-ready.
- Do not use fragment-only success to claim the main `box021_029_p2` gate is solved.
