# E140 Holosoma Semantic Ref-Mask Reward Readiness Results

## Summary

E140 audited whether the semantic `object_contact` masks exposed in E139 are actually consumed by the current Holosoma Box021 reward/config stack.

Result: pass, with blocker identified.

Recommendation:

`needs_ref_mask_reward_config_variant`

E139 runtime artifacts are ready, but the current Box021 R135/R138 experiment configs do not include the existing ref-mask reward terms.

## Result Paths

| artifact | path |
|---|---|
| readiness TSV | `workspace/core4d/results/E140/holosoma_semantic_ref_mask_reward_readiness/e140_ref_mask_reward_readiness.tsv` |
| summary JSON | `workspace/core4d/results/E140/holosoma_semantic_ref_mask_reward_readiness/e140_ref_mask_reward_readiness_summary.json` |
| summary MD | `workspace/core4d/results/E140/holosoma_semantic_ref_mask_reward_readiness/e140_ref_mask_reward_readiness_summary.md` |

## Findings

Runtime consumers of `motion_command.ref_object_contact` exist:

- `RefMaskedHandObjectContactReward`
- `RefMaskedTwoHandObjectContactReward`
- `LostHandContactTermination`

Existing reward configs using ref-mask reward are Box023 configs:

- `g1_29dof_wbt_reward_w_object_r099_box023_reference_contact_mask_v4_3`
- `g1_29dof_wbt_reward_w_object_r109_box023_r095_refmask_from_scratch_v4_3`
- `g1_29dof_wbt_reward_w_object_r110_box023_r095_r097_refmask_from_scratch_v4_3`

Box021 target configs are not ready:

| experiment | reward config | status |
|---|---|---|
| `g1_29dof_wbt_w_object_r135_box021_handbox_exp0601_v4_3` | `g1_29dof_wbt_reward_w_object_r119_box021_handbox_omnirt_v4_3` | `missing_ref_mask_reward` |
| `g1_29dof_wbt_w_object_r138_box021_r095_loadpath_partner_v4_3` | `g1_29dof_wbt_reward_w_object_r138_box021_r095_loadpath_v4_3` | `missing_ref_mask_reward` |

E139 artifact gate is ready:

- E139 summary status: pass.
- E139 semantic ref-mask runtime-ready rows: 4.
- E139 partner-injection pass rows: 4.

## Claims Verification

| Claim | Result |
|---|---|
| C1: reward/termination consumers of `ref_object_contact` are identified | pass |
| C2: R135/R138 Box021 configs are audited | pass |
| C3: E140 determines config readiness without training | pass |
| C4: no CEM/PPO/training/remote launch | pass |

## Verification

- `.venv/bin/python -m py_compile workspace/core4d/scripts/E140/audit_holosoma_semantic_ref_mask_reward_readiness.py`
- `bash -n workspace/core4d/scripts/eval/eval_E140_holosoma_semantic_ref_mask_reward_readiness.sh`
- `bash workspace/core4d/scripts/eval/eval_E140_holosoma_semantic_ref_mask_reward_readiness.sh`

## Next Step

Create a Box021 semantic ref-mask reward config variant derived from R138, then run a bounded no-PPO reward/replay probe on the E139 partner-injected semantic motions.

The minimal config direction is:

- replace regular `hand_object_contact` with `RefMaskedHandObjectContactReward`;
- replace regular `two_hand_object_contact` with `RefMaskedTwoHandObjectContactReward`;
- set non-ref-masked contact persistence rewards to zero or remove them for the first probe;
- use Box021 params (`_box021_contact_gate_params`, `_box021_two_hand_gate_params`);
- do not enable `LostHandContactTermination` in the first smoke.

E140 does not prove PPO readiness. `rl_ready_rows=0` remains correct.
