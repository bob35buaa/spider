# E132 Holosoma MotionLoader Object-Contact Probe Plan

Date: 2026-06-03

## Context

E131 generated separate proxy-tagged Holosoma exports with `object_contact (T,2)`.
The next question is whether Holosoma's actual `MotionLoader` reads these masks
as object-contact data, instead of falling back to the all-false missing-mask
path observed in E128/E130.

E132 is a bounded runtime contract probe. It does not launch IsaacSim stepping,
CEM, PPO, checkpoint creation, or remote jobs.

## Claims

1. The original E126 paired exports still load with `has_object_contact=False`.
2. The E131 proxy exports load through Holosoma `MotionLoader` with
   `has_object_contact=True`.
3. E131 proxy `object_contact` active fractions observed through `MotionLoader`
   match the saved proxy manifest.
4. This is only structural runtime readiness. Semantic raw-contact readiness,
   RL readiness, and main `box021_029_p2` release readiness remain false.

## Method

- Run under Holosoma's `scripts/source_isaacsim_setup.sh` environment, matching
  E128 dependency setup.
- Instantiate `holosoma.managers.command.terms.wbt.MotionLoader` on CPU.
- Pass the full `body_names` and `joint_names` from each NPZ to avoid robot config
  dependency.
- Compare four rows:
  - E126 p1 paired export
  - E126 p2 paired export
  - E131 p1 proxy export
  - E131 p2 proxy export
- Write loader flags, shape, dtype, active fractions, and longest runs.

## Outputs

```text
workspace/core4d/results/E132/holosoma_motionloader_object_contact_probe/
  e132_motionloader_object_contact_manifest.tsv
  e132_motionloader_object_contact_summary.json
  e132_motionloader_object_contact_summary.md
```

## Success Criteria

- 2/2 E126 rows have `has_object_contact=false`.
- 2/2 E131 rows have `has_object_contact=true`.
- 2/2 E131 rows have `object_contact_shape=214x2`.
- 2/2 E131 rows have at least one active contact frame.
- `structural_ref_mask_runtime_ready_rows=2`.
- `semantic_ref_mask_ready_rows=0`, `rl_ready_rows=0`,
  `training_launched=false`, `cem_launched=false`.

## Guardrails

- Do not treat geometry proxies as raw-contact ground truth.
- Do not start PPO, CEM, or Holosoma training.
- Do not mutate E126/E131 exports.
- Do not mark fragments as main release evidence.
