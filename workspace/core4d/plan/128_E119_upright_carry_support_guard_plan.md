# E119 Upright Carry Support-Guard Plan

Date: 2026-06-03

## Context

`contact_improvement_plan.md` requires improving true hand-object contact without increasing deep penetration, lower-body/object interference, or posture failures. E115-E118 narrowed the blocker:

- E115 uniform `leg_object_penalty` can remove lower-body/object interference, but hand-object contact collapses.
- E116 `surface_upright_safety` is the best main-case posture signal: `box021_029_p2` lower-body goes 8.0% -> 0.0% and pelvis regression is small, but physics contact is only 46.7% and deep penetration increases +4.0pp.
- E117 phase/state-gated lower-body penalty still collapses contact and pelvis.
- E118 carry corridor recovers some contact with ref targets, but CEM uses collapse/body/object contact: main ref variants reach 50.7-56.0% physics contact but lower-body rises to 25.3-33.3% and deep penetration rises +10.7-13.3pp.

E119 tests whether the existing knobs can combine the two partial positives:

```text
E116 posture/safety guard + E118 carry/contact corridor
```

No core reward code should be changed in E119 Phase A. If this combination fails, the evidence supports adding a new less-gameable object/body support reward or staged curriculum instead of another scalar sweep.

## Claims

### Claim A: posture-first carry can avoid E118 collapse

Adding `ctrl_ref_guard_scale`, `stability_penalty_scale`, `task_body_rew_scale`, upper-body/pelvis `robot_object_penalty`, and `hand_object_deep_penalty` to the E118 carry corridor should prevent the main case from satisfying contact by falling backward or supporting the box on the body.

Measured by:

- `box021_029_p2` pelvis delta must be no worse than E116 `surface_upright_safety`.
- `test_leg_interference_frac` must stay <= E113 lower-body level (8.0%) and preferably 0.0%.
- visual keyframes must not show body/leg support replacing hand support.

### Claim B: contact can improve beyond E116 while preserving the posture guard

At least one E119 main-case variant should improve physics hand-object contact over E116 `surface_upright_safety` (46.7%) without exceeding the E113 lower-body/deep-penetration gates.

Measured by:

- `box021_029_p2` physics contact improves by at least +8pp over E107 baseline or is clearly above E116 while preserving gates.
- `delta_deep_penetration_2cm <= +3pp`.
- strict gate is PASS or classified as a real Pareto improvement rather than `contact_good_lowerbody_fail`, `penetration_fail`, or `lowerbody_fixed_contact_fail`.

### Claim C: the recipe does not break strict guards

The strict guard cases should not gain contact by introducing deep penetration, lower-body/object interference, or pelvis collapse.

Measured by:

- guard rows either PASS/WORK or fail for contact only;
- no guard row should require accepting `penetration_fail` or `contact_good_lowerbody_fail` as release.

## Stage A Workset

Use the minimal Box021 diagnostic workset first. Box004 guards are Stage B only if Stage A produces a Pareto row.

| case | role | split |
|---|---|---|
| `box021_029_p2` | main lower-body-aware contact case | `local-gpu0` |
| `box021_035_p2` | companion lower-body repair | `remote-gpu0` |
| `box021_035_p1` | strict/contact-margin guard | `remote-gpu1` |

## Variants

All variants inherit the E113 `hold_band` source override and set every inherited E116/E118 knob explicitly.

| ablation | target | intent |
|---|---|---|
| `corridor_ref_pose_bodyguard` | `ref_fk`, gain 5.0 | Test whether posture/safety guard can keep E118 ref-contact from collapsing into body/lower-body support. |
| `corridor_ref_pose_bodygate` | `ref_fk`, gain 5.0 + light lower-body pressure | Same as bodyguard, plus light `leg_object_penalty_scale=0.4` gated by `contact_mask_time_window` and upper-body CEM safety gate. |
| `corridor_surface_pose_bodyguard` | E100 external surface target, gain 3.0 | Start from E116 posture signal and add carry corridor/object orientation coupling. |

Shared posture/body-support guard:

- `robot_object_penalty_scale=2.0` on upper-body/pelvis/arms, excluding hands.
- `hand_object_deep_penalty_scale=5.0`.
- `hand_floor_penalty_scale=1.0`.
- `ctrl_ref_guard_scale=0.8`, active from 0.6s to 3.0s.
- `stability_penalty_scale=1.0`, threshold 0.60m.
- `task_body_rew_scale=1.5` on pelvis/torso/ankles/wrists.
- `task_obj_use_exp=true`, `task_obj_pos_rew_scale=0.5`, `task_obj_rot_rew_scale=0.6`.
- `carry_corridor_rew_scale=4.0`, gated by `contact_mask_time_window`.
- no uniform leg penalty except the explicit light `bodygate` variant.

## Stage B Expansion

Only if Stage A has a main-case Pareto row, add:

- `box004_082_p1`
- `box004_083_p2`

Stage B should run only the winning 1-2 Stage A variants, not another full 5 case x 3 sweep.

## Success Gate

E119 Phase A is smoke first (`SMOKE_MAX_NUM_ITERATIONS=4`).

Launch full CEM only if smoke has one of:

1. `box021_029_p2` release candidate; or
2. `box021_029_p2` clear Pareto signal over E116/E118:
   - physics contact >= E116 `surface_upright_safety` + 8pp, or >= E113 hold-band contact trend without lower-body regression;
   - lower-body <= 8.0%;
   - deep penetration delta <= +3pp;
   - pelvis not worse than E116 upright by visual/metric inspection.

Do not launch full if smoke repeats any of:

- E116 pattern: posture okay but hands abandon the box.
- E118 pattern: contact recovered through falling/body or lower-body support.
- E115/E117 pattern: lower-body fixed but contact collapses.

## Fixed Entrypoints

Planned artifacts:

- builder: `workspace/core4d/scripts/E119/build_upright_carry_manifest.py`
- variants: `workspace/core4d/scripts/E119/variants.tsv`
- preflight: `workspace/core4d/results/E119/preflight/phaseA_preflight.tsv`
- train: `workspace/core4d/scripts/train/train_E119_upright_carry.sh`
- remote: `workspace/core4d/scripts/run_E119_remote.sh`
- pull: `workspace/core4d/scripts/pull_E119_remote_results.sh`
- eval: `workspace/core4d/scripts/eval/eval_E119_upright_carry.py`
- eval shell: `workspace/core4d/scripts/eval/eval_E119_upright_carry.sh`
- overrides: `examples/config/override/core4d_E119_*.yaml`

## Verification Before Training

- `python -m py_compile` for the E119 builder/evaluator.
- `bash -n` for train/remote/pull/eval shell scripts.
- preflight `all_preflight_ok=True`.
- split list returns 3 local, 3 remote-gpu0, 3 remote-gpu1 variants.
- `eval_E119_upright_carry.sh smoke --allow-missing` writes missing outputs without fake metrics.
- `git diff --check` on E119 files and touched docs.

## Expected Decision

E119 is still a diagnostic, not a promised release recipe. The expected useful outcome is to determine whether existing posture/safety/carry knobs can produce a valid hand-supported carry state. If not, the next step should be a new reward/constraint that explicitly penalizes object support from non-hand body geoms or a staged carry curriculum.
