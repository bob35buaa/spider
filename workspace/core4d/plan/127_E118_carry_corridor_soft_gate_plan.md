# E118 Carry Corridor + Soft-Gated Lower-Body Plan

Date: 2026-06-03

## Context

E115, E116, and E117 all show the same failure from different directions:

- A uniform lower-body/object penalty can remove lower-body interference, but the hands abandon the object.
- External surface targets and upright/body guards can improve some cases, but still trade contact against penetration or lower-body support.
- A hard phase/state gate for the lower-body penalty is too brittle: on `box021_029_p2`, all E117 variants reduce lower-body 8.0% -> 0.0%, but physics contact collapses to 8-12% and pelvis stability regresses.

The next experiment therefore changes the reward structure. E118 should not add another independent scalar penalty. It should add a coupled carry corridor that is only valuable when hand contact, object pose, pelvis posture, and lower-body clearance are simultaneously plausible.

## Claim

If the optimizer is rewarded for a valid carry state rather than separately rewarded/penalized for contact, object clearance, posture, and lower-body avoidance, then it should be harder for CEM to satisfy one term by breaking another.

Operational claim:

> A soft carry corridor can preserve E113 hand-object contact while reducing lower-body/object interference, without creating the E115/E117 hand-abandonment or the E116 penetration/upright trade-off.

## Code Change

Add backward-compatible config fields:

- `carry_corridor_rew_scale`
- `carry_corridor_gate_source`
  - `contact_mask`
  - `time_window`
  - `contact_mask_time_window`
- `carry_corridor_start_eval_time`
- `carry_corridor_end_eval_time`
- `carry_corridor_hand_target_threshold_m`
- `carry_corridor_hand_sigma`
- `carry_corridor_clearance_min_m`
- `carry_corridor_clearance_max_m`
- `carry_corridor_clearance_sigma`
- `carry_corridor_pelvis_min_m`
- `carry_corridor_pelvis_sigma`
- `carry_corridor_rot_sigma`
- `carry_corridor_leg_margin_m`
- `carry_corridor_leg_sigma`
- `carry_corridor_leg_geom_names`
- `carry_corridor_leg_geom_ids`

Implementation site:

- `spider/config.py`
- `spider/simulators/mjwp.py`

Reward sketch:

```text
gate = contact/time gate
hand_score = exp(-min_hand_target_distance / hand_sigma)
clearance_score = exp(-distance_to_clearance_band / clearance_sigma)
pelvis_score = exp(-max(0, pelvis_min - pelvis_z) / pelvis_sigma)
rot_score = exp(-object_rot_error_to_ref / rot_sigma)
leg_score = exp(-max(0, leg_margin - leg_sdf) / leg_sigma)

carry_corridor_rew = scale * gate * hand_score * clearance_score * pelvis_score * rot_score * leg_score
```

This is intentionally a soft product rather than another hard gate. It only becomes large when the whole carry state is coherent, but it does not zero out exploration as aggressively as E117's binary hand-target gate.

Defaults keep existing behavior unchanged (`carry_corridor_rew_scale=0.0`).

## Workset

Use the same 5-case diagnostic workset as E115-E117:

| case | role |
|---|---|
| `box021_029_p2` | main blocked candidate; E113 contact gain but 8.0% lower-body interference |
| `box021_035_p2` | companion lower-body repair |
| `box021_035_p1` | strict Box021 guard |
| `box004_082_p1` | strict Box004 guard |
| `box004_083_p2` | strict Box004 guard |

## Ablations

| ablation | intent |
|---|---|
| `corridor_ref_soft` | E113 hold-band + dynamic ref-FK contact target + carry corridor, no explicit leg penalty |
| `corridor_ref_leglight` | `corridor_ref_soft` + light E117 leg penalty gated by contact mask |
| `corridor_surface_soft` | E116 E100 external surface target + carry corridor, no explicit leg penalty |

Rationale:

- `corridor_ref_soft` tests whether a coupled state reward can avoid hand abandonment without applying a separate lower-body penalty.
- `corridor_ref_leglight` tests whether a small lower-body term becomes usable once the corridor rewards coherent carry states.
- `corridor_surface_soft` tests whether object-local surface targets help when they are not paired with the previous hard upright/safety stack.

## Success Criteria

Main case `box021_029_p2`:

- lower-body strict pass and `leg_box_interference_frac <= 5%`;
- preserve useful contact: physics contact or SDF contact improves at least +8pp over baseline, and should not collapse relative to E113;
- `hand_geom_deep_penetration_2cm` delta `<= +3pp`;
- `pelvis_ok=True`, no fall/seat/collapse gate failure;
- object error does not regress;
- video confirms hands remain the primary support and there is no lower-body object support.

Guard cases:

- no strict WORK guard regresses to strict FAIL because of penetration, lower-body, or object-error regression;
- no Box004 guard repeats the E117 pelvis collapse;
- high-contact rows with deep penetration are not accepted as positive.

## Execution Policy

Start with smoke only.

Do not launch full CEM unless smoke has at least one release candidate or a clearly Pareto-improving row on `box021_029_p2`.

If full CEM is justified:

- run local GPU0 + `spider-remote` GPU0/GPU1 in parallel;
- do not kill unrelated processes;
- verify available memory and run on top if there is enough free VRAM;
- pull remote results before final evaluation.

## Evaluation

Reuse the E115/E117 evaluator structure:

- compare against baseline cache and E113 hold-band;
- output `method_metrics.csv`, `pareto_decisions.tsv`, `release_candidates.tsv`, `guard_failures.tsv`, and summary markdown;
- inspect keyframes/videos for at least:
  - `box021_029_p2` best row;
  - any high-contact companion row;
  - any guard row with pelvis or penetration regression.

## Decision Logic

- If `corridor_ref_soft` improves contact but lower-body remains high, the corridor needs stronger leg component or support-state definition.
- If `corridor_ref_leglight` fixes lower-body without contact collapse, expand this route to full CEM.
- If `corridor_surface_soft` is better than ref target, future work should combine object-local surface target with a soft state corridor, not E116's hard upright/safety stack.
- If all rows fail, the next step should be a stronger pose prior or learned/full-body prior; pure CEM scalar rewards are likely exhausted for this case family.
