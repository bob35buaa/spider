# E130 Holosoma Reward-Side Inspection Plan

Date: 2026-06-03

## Scope

E130 is a no-training, no-CEM inspection branch after E128 and E129.

Inputs:

- E126 paired Holosoma fragment exports:
  - `workspace/core4d/results/E126/holosoma_fragment_adapter_preflight/exports/E126_box021_035_p1_with_partner_box021_035_p2_mj_w_obj_w_partner.npz`
  - `workspace/core4d/results/E126/holosoma_fragment_adapter_preflight/exports/E126_box021_035_p2_with_partner_box021_035_p1_mj_w_obj_w_partner.npz`
- Holosoma reward/config source:
  - `/home/ubuntu/Workspace/holosoma/src/holosoma/holosoma/config_values/wbt/g1/reward.py`
  - `/home/ubuntu/Workspace/holosoma/src/holosoma/holosoma/config_values/wbt/g1/experiment.py`
- E127/E128 contract/runtime startup evidence.

## Questions

1. Which Box021 Holosoma reward/config terms are relevant to handbox object support?
2. Do the E126 paired fragment exports contain the fields needed by those reward families?
3. Which reward-side checks can be approximated offline from NPZ geometry without PPO or IsaacSim contact sensors?
4. Which reward families must remain blocked until `object_contact` masks or simulator contact forces are available?

## Method

- Statically inspect the Holosoma reward/config definitions for:
  - `g1_29dof_wbt_reward_w_object_r119_box021_handbox_omnirt_v4_3`
  - `g1_29dof_wbt_reward_w_object_r138_box021_r095_loadpath_v4_3`
  - the R135/R138 experiment config mappings.
- Load the two E126 paired fragment exports and summarize required fields.
- Compute an offline object-box signed surface-distance proxy for:
  - actor hand fallback links: `left_rubber_hand_link`, `right_rubber_hand_link`
  - partner hand positions: `partner_hand_pos_w[:, 0/1]`
- Explicitly mark ref-masked and force/contact-gated rewards as unsupported when `object_contact` or simulator contact force evidence is missing.

## Outputs

```text
workspace/core4d/results/E130/holosoma_reward_side_inspection/
  e130_reward_terms.tsv
  e130_motion_reward_proxy.tsv
  e130_reward_side_summary.json
  e130_reward_side_summary.md
```

## Success Criteria

- The script runs without launching CEM, PPO, checkpoint creation, or remote jobs.
- Both paired E126 exports are audited.
- R119/R138 Box021 reward-side terms and R135/R138 config mappings are reported.
- The summary distinguishes offline proximity evidence from unsupported contact-mask/contact-force evidence.
- `rl_ready_rows=0`, `training_launched=false`, and `cem_launched=false`.

## Stop Conditions

- Do not treat E126 fragments as main-case release evidence.
- Do not launch full Holosoma PPO from E126/E128 artifacts.
- Do not claim ref-masked rewards are available unless `object_contact` exists with the expected two-hand frame mask.
