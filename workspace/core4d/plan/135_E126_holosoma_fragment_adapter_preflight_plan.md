# E126 Holosoma Fragment Adapter Preflight Plan

Date: 2026-06-03

## Context

E125 produced no main-case RL-ready row. It did, however, identify two clean holdout fragments:

- `box021_035_p1 / E121_box021_035_p1_terminal_soft_surface`
- `box021_035_p2 / E121_box021_035_p2_terminal_soft_surface`

Both are `FRAGMENT_HOLDOUT_ONLY`, not release candidates. They are useful because they are the two people from the same Box021 sequence and both satisfy the strict hand-support geometry gate. E126 turns this into a downstream adapter preflight: can the fragment pair be converted into Holosoma-style motion files with partner hand fields?

## Claims

| claim | success evidence |
|---|---|
| C1: E125 fragment labels are preserved | output manifest keeps `FRAGMENT_HOLDOUT_ONLY` and `rl_train_allowed=false` |
| C2: qpos43 inputs can be converted by Holosoma's converter | per-person `_mj_w_obj.npz` files exist and pass required-key/shape checks |
| C3: partner hand fields can be added for the `035` pair | paired output has `partner_hand_pos_w (T,2,3)` and `partner_hand_quat_w (T,2,4)` |
| C4: the adapter does not claim release readiness | summary reports `rl_ready_rows=0` and `training_launched=false` |

## Inputs

- `workspace/core4d/results/E125/rl_hand_support_preflight/selected_preflight.tsv`
- E125 converter inputs under `workspace/core4d/results/E125/rl_hand_support_preflight/converter_inputs/`
- Holosoma converter:
  `/home/ubuntu/Workspace/holosoma/src/holosoma_retargeting/holosoma_retargeting/data_conversion/convert_data_format_mj.py`

## Outputs

```text
workspace/core4d/results/E126/holosoma_fragment_adapter_preflight/
  e126_adapter_manifest.tsv
  e126_adapter_summary.json
  e126_adapter_summary.md
  converter_inputs/*.npz
  exports/*_mj_w_obj.npz
  exports/*_mj_w_obj_w_partner.npz
  logs/convert.log
```

## Non-Goals

- Do not launch Holosoma RL training.
- Do not mark E125 fragments as RL-ready release rows.
- Do not use this as evidence that the main `box021_029_p2` case is solved.
