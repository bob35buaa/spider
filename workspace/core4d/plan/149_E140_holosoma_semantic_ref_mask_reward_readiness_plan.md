# E140 Holosoma Semantic Ref-Mask Reward Readiness Plan

## Context

E139 proved that E138 semantic `object_contact` masks can be exposed through Holosoma runtime `motion_command.ref_object_contact` in the R135 Box021 partner env. The next decision is whether the current Holosoma Box021 reward/config stack already consumes those masks, or whether a new semantic-mask reward/config variant is required before any PPO smoke or full training.

## Claims

1. Holosoma contains reward or termination terms that read `motion_command.ref_object_contact`.
2. The current R135/R138 Box021 configs can be audited for whether those terms are present.
3. E140 can determine whether semantic ref-mask reward is config-ready without launching PPO, CEM, or remote jobs.

## Scope

Read-only/static audit plus local checker outputs:

- reward/termination term inventory for `ref_object_contact`;
- named config inventory for terms using those classes;
- R135/R138 Box021 reward/config status;
- E139 artifact compatibility summary;
- recommendation for the next step.

## Implementation

1. Add `workspace/core4d/scripts/E140/audit_holosoma_semantic_ref_mask_reward_readiness.py`.
2. Inspect Holosoma source files:
   - `holosoma/managers/reward/terms/wbt.py`
   - `holosoma/managers/termination/terms/wbt.py`
   - `holosoma/config_values/wbt/g1/reward.py`
   - `holosoma/config_values/wbt/g1/termination.py`
   - `holosoma/config_values/wbt/g1/experiment.py`
   - `holosoma/config_values/experiment.py`
3. Inspect E139 runtime summary and partner-injection manifest.
4. Write:
   - `e140_ref_mask_reward_readiness.tsv`;
   - `e140_ref_mask_reward_readiness_summary.json`;
   - `e140_ref_mask_reward_readiness_summary.md`.
5. Add fixed eval entry:
   `workspace/core4d/scripts/eval/eval_E140_holosoma_semantic_ref_mask_reward_readiness.sh`.

## Success Criteria

- The checker identifies all local reward/termination classes that consume `ref_object_contact`.
- The checker reports whether R135 and R138 Box021 reward configs include those terms.
- The checker reports whether E139 semantic runtime artifacts are present and passed.
- The checker emits an explicit next-step recommendation:
  - `existing_config_ready_for_reward_probe`, or
  - `needs_ref_mask_reward_config_variant`.
- No PPO, CEM, remote jobs, or checkpoint creation.

## Non-Goals

- Do not edit Holosoma reward/config files in E140.
- Do not launch PPO smoke/full.
- Do not run CEM.
- Do not claim RL readiness from static/config evidence alone.

## Fixed Command

```bash
bash workspace/core4d/scripts/eval/eval_E140_holosoma_semantic_ref_mask_reward_readiness.sh
```

## Expected Outputs

- `workspace/core4d/results/E140/holosoma_semantic_ref_mask_reward_readiness/e140_ref_mask_reward_readiness.tsv`
- `workspace/core4d/results/E140/holosoma_semantic_ref_mask_reward_readiness/e140_ref_mask_reward_readiness_summary.json`
- `workspace/core4d/results/E140/holosoma_semantic_ref_mask_reward_readiness/e140_ref_mask_reward_readiness_summary.md`
