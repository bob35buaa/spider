# E115 Lower-Body-Aware Contact Diagnostic Plan

Date: 2026-06-03

## Context

`contact_improvement_plan.md` has now produced:

- E110/E111: contact metric and data evidence chain.
- E112: contact-aware CEM can recover contact on anchors.
- E113: expanded hold-band full CEM produced 0 release candidates.
- E114: RL handoff gate correctly exported 0 rows and routed blocked rows.

The most informative E114 row is:

```text
box021_029_p2: physics contact +20.0pp, deep penetration +0.0pp, leg interference 8.0%, strict FAIL
```

This is the clearest current evidence that contact and penetration can improve together, while lower-body/object interference blocks RL handoff.

E115 therefore should not launch RL and should not expand to more cases. It should run a small diagnostic that asks whether existing lower-body/object penalty controls can preserve the contact gain while bringing leg interference below the strict 5% threshold.

## Claims

### C1: Lower-body-aware contact is the next bottleneck

If a lower-body-aware variant reduces `box021_029_p2` leg interference from 8.0% to <=5.0% while keeping physics-contact gain >=8pp and deep penetration delta <=3pp, then the next contact-improvement branch should focus on lower-body-aware contact, not raw target variants.

### C2: Guard cases must not regress

Any variant that fixes `box021_029_p2` but breaks existing strict WORK rows is not releaseable. Guard rows:

- `box021_035_p1` strict WORK / contact margin fail
- `box004_082_p1` strict WORK / contact margin fail
- `box004_083_p2` strict WORK / contact margin fail

For guards, require:

- `hold_strict_status == WORK`
- `delta_deep_penetration_2cm <= 3pp`
- leg interference <=5%
- object mean error within baseline +1cm or <=2.5cm

### C3: Raw-surface/handbox target variants remain secondary

Raw-surface or handbox-surface target variants should be resumed only if lower-body-aware contact fails to preserve E113's contact gain, or if videos show that ref-FK contact target is visibly wrong on the selected rows.

## Case Set

Phase A diagnostic set:

| role | case | reason |
|---|---|---|
| main | `box021_029_p2` | E114 `lowerbody_aware_contact`; physics contact +20pp but leg interference 8.0% |
| companion | `box021_035_p2` | E114 `lowerbody_repair`; contact gains near threshold but leg interference 9.8% |
| strict guard | `box021_035_p1` | strict WORK with low leg interference; must remain stable |
| strict guard | `box004_082_p1` | strict WORK guard from box004 |
| strict guard | `box004_083_p2` | strict WORK guard from box004 |

Do not include `box004_083_p1` in Phase A; it has lower-body failure but no contact gain, so it is a lower priority after the contact-aware rows.

## Variants

Start from the E113 `hold_band` overrides and add only existing config fields.

| variant | change | purpose |
|---|---|---|
| `hold_band_ref` | E113 output, no rerun | baseline for comparison |
| `leg_penalty_s2` | `leg_object_penalty_scale=2.0`, `leg_object_penalty_margin_m=0.02` | light lower-body avoidance |
| `leg_penalty_s4` | `leg_object_penalty_scale=4.0`, `leg_object_penalty_margin_m=0.02` | stronger lower-body avoidance |
| `leg_penalty_s2_contact_gain8` | `leg_penalty_s2` + `contact_hdmi_gain=8.0` | recover contact if leg penalty makes hands back off |

Use the standard lower-body geom list:

```yaml
leg_object_penalty_geom_names:
  - left_hip_collision
  - right_hip_collision
  - left_thigh_collision
  - right_thigh_collision
  - left_shin_collision
  - right_shin_collision
  - left_linkage_brace_collision
  - right_linkage_brace_collision
  - lf0
  - lf1
  - lf2
  - lf3
  - rf0
  - rf1
  - rf2
  - rf3
```

Keep existing E113 safety settings:

- `contact_hdmi_target_source=ref_fk`
- raw contact mask enabled
- `hold_contact_rew_scale=1.0`
- CEM safety gate inherited from `core4d_E089A_box021_person1_upperobj`
- no RL export

## Full CEM Execution

Only run full CEM after manifest/preflight/smoke pass.

Recommended split for 5 cases x 3 runnable variants = 15 full runs:

- local GPU0: strict guards `box004_082_p1`, `box004_083_p2`
- remote GPU0: main `box021_029_p2` variants
- remote GPU1: companion + `box021_035_p1` guard variants

Follow `.codex/skills/experiment-planning-zh/remote-execution.md`:

- sync scripts/overrides/preflight/task dirs to `spider-remote`
- do not kill existing processes
- if enough free memory remains, run on top of other programs
- pull remote results before local eval

## Outputs

```text
workspace/core4d/scripts/E115/
  build_lowerbody_contact_manifest.py
  variants.tsv

workspace/core4d/results/E115/
  preflight/
  cem/smoke/
  cem/full/
    full_eval_summary.md
    pareto_decisions.tsv
    release_candidates.tsv
    lowerbody_aware_contact.tsv
```

Fixed scripts:

```text
workspace/core4d/scripts/train/train_E115_lowerbody_contact.sh
workspace/core4d/scripts/run_E115_remote.sh
workspace/core4d/scripts/pull_E115_remote_results.sh
workspace/core4d/scripts/eval/eval_E115_lowerbody_contact.sh
workspace/core4d/scripts/eval/eval_E115_lowerbody_contact.py
```

## Success Criteria

E115 produces a release candidate only if all are true:

- main or companion row has physics contact delta >=8pp or contact fraction delta >=8pp;
- deep penetration delta <=3pp;
- lower-body interference <=5%;
- `hold_strict_status == WORK`;
- object mean error <= max(baseline+1cm, 2.5cm);
- guard rows remain strict WORK;
- videos show no hand-back false contact or lower-body compensation.

If no variant meets this:

- if contact gain is preserved but lower-body still fails, next step should use stricter safety gate / posture schedule;
- if lower-body improves but contact disappears, next step should run raw_surface/handbox target variants;
- if both fail, do not expand case count or launch RL.

## First Implementation Step

Implement manifest/preflight and smoke only. Do not start full CEM until:

- all expected masks/baselines/tasks are present;
- remote sync has been checked;
- smoke writes NPZ/MP4 and evaluator can classify missing/full outputs without fake metrics.
