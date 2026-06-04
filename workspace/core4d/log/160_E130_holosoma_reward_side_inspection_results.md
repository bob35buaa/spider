# E130 Holosoma Reward-Side Inspection Results

Date: 2026-06-03

## Scope

E130 followed `workspace/core4d/plan/139_E130_holosoma_reward_side_inspection_plan.md`.
It did not launch CEM, PPO, Holosoma training, checkpoint creation, or remote jobs.

The run audited the two E126 paired `box021_035_p1/p2` fragment exports and the
Holosoma Box021 R135/R138 reward/config source. The fragments remain
`FRAGMENT_HOLDOUT_ONLY` and are not main `box021_029_p2` release evidence.

Subagent `Locke` performed a read-only Holosoma reward/config audit. Its conclusion
matched the local result: E130 can validate reward wiring and offline proximity
proxies, but contact-mask, contact-force, support-force behavior, and policy
quality require `object_contact`, simulator contact sensors, bounded replay, or
PPO and must not be inferred from NPZ geometry alone.

## Outputs

```text
workspace/core4d/results/E130/holosoma_reward_side_inspection/
  e130_reward_terms.tsv
  e130_motion_reward_proxy.tsv
  e130_reward_side_summary.json
  e130_reward_side_summary.md
```

## Result

| metric | value |
|---|---:|
| reward/config rows | 17 |
| motion proxy rows | 8 |
| paired exports audited | 2 |
| offline proximity reward terms | 2 |
| blocked contact/force terms | 13 |
| ref-mask allowed rows | 0 |
| RL-ready rows | 0 |
| training launched | false |
| CEM launched | false |
| status | pass |

Reward/config wiring confirmed:

- R135 `g1_29dof_wbt_w_object_r135_box021_handbox_exp0601_v4_3` maps to
  `g1_29dof_wbt_reward_w_object_r119_box021_handbox_omnirt_v4_3`.
- R138 `g1_29dof_wbt_w_object_r138_box021_r095_loadpath_partner_v4_3` maps to
  `g1_29dof_wbt_reward_w_object_r138_box021_r095_loadpath_v4_3`.
- Box021 reward geometry uses half extents `[0.195495, 0.15155, 0.25004]` and
  Holosoma handbox bodies `left_handbox_link/right_handbox_link`.

Motion-export checks:

- Both paired E126 exports contain object pose and partner hand pose.
- Both exports lack `object_contact`, so ref-mask rewards remain blocked.
- E126 body names do not include Holosoma handbox links, so the actor-side offline
  proxy uses `left_rubber_hand_link/right_rubber_hand_link` fallback links.

Selected offline proxy rows:

| export | sample | clean near 5cm | longest clean run |
|---|---|---:|---:|
| p1 with p2 partner | actor left rubber hand | 0.0% | 0 |
| p1 with p2 partner | actor right rubber hand | 4.2% | 9 |
| p1 with p2 partner | partner left hand | 35.0% | 69 |
| p1 with p2 partner | partner right hand | 20.1% | 43 |
| p2 with p1 partner | actor left rubber hand | 27.6% | 27 |
| p2 with p1 partner | actor right rubber hand | 47.2% | 39 |
| p2 with p1 partner | partner left hand | 17.8% | 10 |
| p2 with p1 partner | partner right hand | 34.1% | 24 |

## Interpretation

E130 closes the reward-side inspection branch enough to say the current Box021
R135/R138 Holosoma reward wiring is inspectable and the E126 paired exports can
support an offline surface-distance sanity check. It does not clear the fragments
for RL. The missing `object_contact` mask is a hard blocker for ref-masked contact
reward claims, and NPZ-only geometry cannot validate simulator contact-force,
support-force, lower-body contact, or learned carry behavior.

The next valid branch is either:

1. Add an `object_contact (T,2)` mask contract to the fragment/export path before
   any ref-mask reward conclusion.
2. Run a bounded no-PPO R135/R138 replay reward/evaluator probe, preserving the
   same no-training guardrails.
3. Return to the main `box021_029_p2` constrained-teacher path; E130 fragment
   evidence cannot unblock the main release gate.

## Verification

- `.venv/bin/python -m py_compile workspace/core4d/scripts/E130/inspect_holosoma_reward_side.py`
- `bash -n workspace/core4d/scripts/eval/eval_E130_holosoma_reward_side_inspection.sh`
- `bash workspace/core4d/scripts/eval/eval_E130_holosoma_reward_side_inspection.sh`
