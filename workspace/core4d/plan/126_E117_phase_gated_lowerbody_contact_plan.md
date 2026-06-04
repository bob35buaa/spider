# E117 Phase/State-Gated Lower-Body Contact Plan

Date: 2026-06-03

## Motivation

E115 proved that a uniform `leg_object_penalty` can remove lower-body/object interference but often does so by sacrificing hand-object contact. E116 proved that external surface targets and upright/safety guards are not sufficient to produce a release candidate: high contact variants still use lower-body support, while posture-safe variants lose too much hand contact.

E117 therefore changes the method rather than sweeping another scalar weight. The lower-body/object penalty is only enabled when a contact phase is active, or when the current hand target is already close enough. The hypothesis is:

> lower-body cleanup should happen after the optimizer has established hand contact, not while the optimizer is still searching for contact.

## Code Change

New backward-compatible config fields:

- `leg_object_penalty_gate_source`
  - `always`: historical E115 behavior.
  - `contact_mask`: enable penalty only on raw-contact frames.
  - `time_window`: enable penalty only inside `leg_object_penalty_start_eval_time` to `leg_object_penalty_end_eval_time`.
  - `contact_mask_time_window`: require both raw-contact phase and time window.
  - `hand_target`: enable penalty only when either hand reaches its contact target.
  - `contact_mask_and_hand_target`: require raw-contact phase and target success.
- `leg_object_penalty_start_eval_time`
- `leg_object_penalty_end_eval_time`
- `leg_object_penalty_hand_target_threshold_m`

Implementation sites:

- `spider/config.py`
- `spider/simulators/mjwp.py`

The default `always` preserves old behavior for all existing experiments.

## Workset

Use the same narrow diagnostic workset as E115/E116:

| case | role |
|---|---|
| `box021_029_p2` | main blocked candidate: E113 had contact gain but 8.0% lower-body interference |
| `box021_035_p2` | companion lower-body repair case |
| `box021_035_p1` | strict Box021 guard |
| `box004_082_p1` | strict Box004 guard |
| `box004_083_p2` | strict Box004 guard |

## Ablations

| ablation | intent |
|---|---|
| `mask_gate_s2` | E113 hold-band + `leg_object_penalty_scale=2`, gated only by raw-contact frames |
| `mask_time_gate_s2` | same penalty, gated by raw-contact frames and a post-approach time window `[0.6s, 3.0s]` |
| `handtarget_gate_s2` | dynamic ref-FK hand target + penalty only when raw contact is active and either hand is within `8cm` of target |

All variants keep E113 hold/contact evidence and do not relax penetration metrics.

## Success Criteria

Main case `box021_029_p2`:

- lower-body strict pass and `leg_box_interference_frac <= 5%`;
- preserve useful contact: `physics_contact` or SDF contact improves at least `+8pp` over baseline;
- `hand_geom_deep_penetration_2cm` delta `<= +3pp`;
- pelvis/fall and object error do not regress;
- video confirms no lower-body support and no fake hand contact.

Guard cases:

- no strict WORK guard regresses to FAIL;
- no deep-penetration or object-error regression;
- contact should not collapse relative to E113.

## Execution Policy

Start with smoke only. Do not launch full CEM unless smoke has at least one release candidate or a clearly Pareto-improving row on `box021_029_p2`.

Full CEM, if justified, should use local GPU0 plus `spider-remote` GPU0/GPU1, without killing unrelated processes.

## Decision Logic

- If `mask_gate_s2` still loses hand contact, pure phase gating is insufficient.
- If `mask_time_gate_s2` improves over `mask_gate_s2`, the approach/carry split matters and later variants should use a learned or data-derived phase boundary.
- If `handtarget_gate_s2` preserves E113 contact while reducing lower-body interference, expand this state-gated route.
- If all rows fail, the next method should add a carry corridor or anti-tip/object-orientation gate rather than further leg-penalty sweeps.
