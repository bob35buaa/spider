# E141 Box021 Semantic Ref-Mask Reward Probe Plan

## Context

E139 proved that E138 semantic `object_contact` masks can reach Holosoma runtime as `motion_command.ref_object_contact` in a Box021 partner env. E140 then found the blocker: existing Box021 R135/R138 reward configs do not consume that reference mask, while Box023 R099/R109/R110 already have ref-mask reward terms.

E141 closes that config/runtime gap with one minimal Box021 reward variant and a bounded local no-PPO probe.

## Claims

1. A Box021 reward config can be derived from R138 and replace only the hand contact terms with `RefMaskedHandObjectContactReward` and `RefMaskedTwoHandObjectContactReward`.
2. A Box021 experiment alias can be derived from R138 with only the reward config changed, leaving termination unchanged for first smoke.
3. The E139 partner-injected semantic motions can start the new env and produce finite ref-mask reward log keys without launching PPO, CEM, or remote jobs.

## Scope

- Edit Holosoma reward/experiment config only where required for the new E141 variant.
- Add a local probe script under `workspace/core4d/scripts/E141/`.
- Use the E139 partner-injected semantic motion artifacts as input.
- Write results under `workspace/core4d/results/E141/box021_semantic_ref_mask_reward_probe/`.

## Implementation

1. Add reward config:
   `g1_29dof_wbt_reward_w_object_e141_box021_semantic_refmask_v4_3`
   derived from `g1_29dof_wbt_reward_w_object_r138_box021_r095_loadpath_v4_3`.
2. Replace:
   - `hand_object_contact` with `RefMaskedHandObjectContactReward`;
   - `two_hand_object_contact` with `RefMaskedTwoHandObjectContactReward`;
   - set regular contact persistence weights to `0.0` for the first probe.
3. Add experiment config:
   `g1_29dof_wbt_w_object_e141_box021_semantic_refmask_v4_3`
   derived from `g1_29dof_wbt_w_object_r138_box021_r095_loadpath_partner_v4_3`, changing reward only.
4. Expose the new experiment alias in `holosoma/config_values/experiment.py`.
5. Add and run:
   `workspace/core4d/scripts/eval/eval_E141_box021_semantic_ref_mask_reward_probe.sh`.

## Success Criteria

- Holosoma imports the new reward and experiment config.
- The probe starts the new Box021 partner env for E139 motions without debug visualization.
- For each probed row, `has_object_contact=true`, `has_partner=true`, and `ref_contact_frac > 0`.
- The expected log keys are present and finite:
  - `r099/ref_contact_frac`
  - `r099/ref_masked_hand_contact_reward_mean`
  - `r099/ref_masked_two_hand_contact_reward_mean`
- PPO smoke is allowed only if at least one row has positive ref-mask reward signal in addition to positive ref-contact mask.

## Non-Goals

- Do not launch PPO or full RL training.
- Do not run CEM.
- Do not use remote GPUs.
- Do not add `LostHandContactTermination` in the first smoke.
- Do not claim RL readiness unless the probe demonstrates positive reward signal.

## Fixed Command

```bash
bash workspace/core4d/scripts/eval/eval_E141_box021_semantic_ref_mask_reward_probe.sh
```

## Expected Outputs

- `workspace/core4d/results/E141/box021_semantic_ref_mask_reward_probe/e141_reward_probe_rows.tsv`
- `workspace/core4d/results/E141/box021_semantic_ref_mask_reward_probe/e141_reward_probe_summary.json`
- `workspace/core4d/results/E141/box021_semantic_ref_mask_reward_probe/e141_reward_probe_summary.md`
