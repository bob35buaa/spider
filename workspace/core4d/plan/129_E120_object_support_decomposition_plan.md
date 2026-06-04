# E120 Object Support Decomposition Plan

Date: 2026-06-03

## Context

`contact_improvement_plan.md` targets high true hand-object contact with low deep penetration, stable posture, and lower-body/object strict pass. E115-E119 have now isolated the main blocker on Box021:

- E115/E117 can remove lower-body/object interference, but hand-object contact collapses.
- E116 surface target plus upright guard can reduce lower-body interference, but the main case loses sustained contact.
- E118 carry corridor recovers contact only by allowing body/lower-body support and penetration regressions.
- E119 combines posture/bodyguard and carry corridor, but the main case still has no Pareto row: ref variants remain pelvis/posture failures at 48-49% physics contact, while the surface variant collapses contact and increases lower-body/object interference.

The next experiment should stop tuning scalar combinations of existing knobs. E120 introduces an explicit decomposition:

```text
hand support contact is rewarded; non-hand object support is penalized
```

This should make the carry state less gameable than E118/E119, where CEM can satisfy contact/object terms through torso, pelvis, arms, or legs instead of a valid hand-supported carry.

## Claims

### Claim A: near-zero hand SDF support improves contact without requiring penetration

Rewarding hand geoms for near-zero signed distance to the object during raw-contact windows should recover true contact better than target-distance-only rewards.

Measured by main `box021_029_p2`:

- physics contact improves over E119 ref/bodyguard (49.3%) and preferably over E116 surface/upright (46.7%);
- `hand_geom_deep_penetration_2cm` delta stays <= +3pp;
- visual keyframes show hand support rather than hand abandonment.

### Claim B: explicit non-hand support penalty prevents body/leg support shortcuts

Penalizing near/support SDF from torso, pelvis, upper arms, and lower-body geoms should prevent the object from being supported by the body while allowing the hands to stay near the object.

Measured by:

- `box021_029_p2` lower-body/object interference <= E113 level (8.0%), ideally <= 5.0%;
- no visual body/leg support;
- `pelvis_ok=True` or at least no E119-style posture collapse.

### Claim C: staged activation avoids E115/E117 contact collapse

A staged variant that first acquires hand support and then increases non-hand support penalty should avoid the E115/E117 pattern where lower-body penalties immediately destroy contact.

Measured by:

- staged variant keeps higher contact than direct-support variant when both lower-body penalties are active;
- no strict guard regression beyond existing E113/E119 blockers.

## Implementation

Add a small reward extension in the existing object SDF block:

- `hand_support_rew_scale`
- `hand_support_sigma`
- `hand_support_margin_m`
- `hand_support_gate_source`
- `hand_support_start_eval_time`
- `hand_support_end_eval_time`
- `hand_support_geom_names`
- `hand_support_geom_ids`
- `nonhand_support_penalty_scale`
- `nonhand_support_penalty_margin_m`
- `nonhand_support_penalty_gate_source`
- `nonhand_support_penalty_start_eval_time`
- `nonhand_support_penalty_end_eval_time`
- `nonhand_support_penalty_geom_names`
- `nonhand_support_penalty_geom_ids`

Reward sketch:

```text
hand_support_rew = scale * gate * exp(-abs(hand_sdf) / sigma)
nonhand_support_penalty = -scale * gate * max(margin - nonhand_sdf, 0)
```

Keep existing `hand_object_deep_penalty` for deep hand penetration. E120 should not alter eval gates or historical metrics.

Expected code files:

- `spider/config.py`
- `spider/simulators/mjwp.py`

Expected experiment artifacts:

- builder: `workspace/core4d/scripts/E120/build_object_support_decomposition_manifest.py`
- variants: `workspace/core4d/scripts/E120/variants.tsv`
- preflight: `workspace/core4d/results/E120/preflight/phaseA_preflight.tsv`
- train: `workspace/core4d/scripts/train/train_E120_object_support_decomp.sh`
- remote: `workspace/core4d/scripts/run_E120_remote.sh`
- pull: `workspace/core4d/scripts/pull_E120_remote_results.sh`
- eval: `workspace/core4d/scripts/eval/eval_E120_object_support_decomp.py`
- eval shell: `workspace/core4d/scripts/eval/eval_E120_object_support_decomp.sh`
- overrides: `examples/config/override/core4d_E120_*.yaml`

## Stage A Workset

Use the same minimal Box021 diagnostic workset as E119 plus one Box004 guard. The extra guard is needed because E115-E116 showed contact/lower-body changes can regress otherwise useful strict cases.

| case | role | split |
|---|---|---|
| `box021_029_p2` | main lower-body-aware contact case | `local-gpu0` |
| `box021_035_p2` | companion lower-body repair | `remote-gpu0` |
| `box021_035_p1` | strict/contact-margin guard | `remote-gpu1` |
| `box004_083_p2` | strict/contact guard against contact collapse | `local-gpu0` |

## Variants

All variants inherit E113 `hold_band` and keep the E119 posture/bodyguard baseline, but replace the scalar bodyguard-only idea with explicit hand/non-hand support decomposition.

| ablation | target | intent |
|---|---|---|
| `support_ref_direct` | `ref_fk`, gain 5.0 | Direct hand near-zero support reward + non-hand support penalty during contact window. |
| `support_ref_staged` | `ref_fk`, gain 5.0 | Same as direct, but non-hand support penalty activates later/lighter to avoid E115/E117 contact collapse. |
| `support_surface_direct` | E100 external surface target, gain 3.0 | Combine surface target with hand support reward and non-hand support penalty. |

Shared knobs:

- `hand_support_rew_scale=3.0`
- `hand_support_sigma=0.015`
- `hand_support_margin_m=0.01`
- `hand_support_gate_source=contact_mask_time_window`
- `hand_object_deep_penalty_scale=5.0`
- `nonhand_support_penalty_scale=1.5` direct / `0.8` staged
- `nonhand_support_penalty_margin_m=0.02`
- `nonhand_support_penalty_gate_source=contact_mask_time_window`
- staged non-hand penalty starts later than hand support
- E119 posture/body terms remain: `ctrl_ref_guard`, `stability_penalty`, `task_body_rew`, object pose rewards, hand-floor guard

## Success Gate

E120 Phase A is smoke first (`SMOKE_MAX_NUM_ITERATIONS=4`).

Launch full CEM only if smoke has one of:

1. `box021_029_p2` release candidate; or
2. `box021_029_p2` clear Pareto signal:
   - physics contact >= 57%;
   - lower-body/object interference <= 5%;
   - non-hand object support fraction <= 5%;
   - deep penetration delta <= +3pp;
   - `pelvis_ok=True` or visual posture clearly better than E119 ref variants;
   - object tracking remains pass.

Do not launch full if smoke repeats:

- E115/E117: lower-body fixed but contact collapses;
- E118: contact recovered through body/lower-body support or penetration;
- E119: contact present but pelvis/posture remains invalid.

## Verification Before Training

- `python -m py_compile` for the E120 builder/evaluator and touched core files.
- `bash -n` for train/remote/pull/eval shell scripts.
- preflight `all_preflight_ok=True`.
- split list returns 3 local, 3 remote-gpu0, 3 remote-gpu1 variants.
- local and remote `eval_E120_object_support_decomp.sh smoke --allow-missing` must run without fake metrics.
- `git diff --check` on E120 files and touched docs.

## Expected Decision

E120 is still diagnostic. If no main-case Pareto row appears, the next step should shift from CEM scalar reward design to a staged/curriculum or RL hand-support policy objective rather than adding more CEM penalties.
